import json

import numpy as np
import pytest

from libs.datasets.epic_kitchens import EpicKitchensDataset
from tools.audit_rq2_vlm_benchmark import audit_configs, audit_so_uda_pair
from tools.rq2_evaluate import selected_checkpoint_state
from tools.rq2_formal import (
    FROZEN_VIDEO_SET_SHA256, build_target_manifest, check_formal_config,
    check_formal_features, check_frozen_runtime_values, check_target_manifest,
)


def test_unlabeled_target_manifest_and_random_crop_have_no_gt(tmp_path):
    annotated = tmp_path / "annotated.json"
    annotated.write_text(json.dumps({"database": {
        "P01_01": {"subset": "source_train", "duration": 8, "fps": 30,
                   "annotations": [{"label": "take", "label_id": 0, "segment": [1, 4]}]},
        "P01_02": {"subset": "target_train", "duration": 8, "fps": 30,
                   "annotations": [{"label": "take", "label_id": 0, "segment": [2, 6]}]},
    }}), encoding="utf-8")
    manifest = tmp_path / "unlabeled.json"
    assert build_target_manifest(annotated, ["target_train"], manifest) == 1
    meta = check_target_manifest(manifest, ["target_train"], annotated)
    assert meta["videos"] == 1
    assert "annotations" not in manifest.read_text(encoding="utf-8")
    np.savez(tmp_path / "P01_02.npz", feats=np.ones((20, 4), dtype=np.float32))
    dataset = EpicKitchensDataset(
        is_training=True, split=["target_train"], feat_folder=str(tmp_path),
        json_file=str(manifest), feat_stride=16, num_frames=32, default_fps=30,
        downsample_rate=1, max_seq_len=4, trunc_thresh=0.3,
        crop_ratio=None, input_dim=4, num_classes=10, file_prefix=None,
        file_ext=".npz", force_upsampling=False,
        action_subsampling="10_majority_classes", unlabeled=True,
    )
    assert len(dataset) == 1
    for _ in range(5):
        item = dataset[0]
        assert item["feats"].shape == (4, 4)
        assert item["segments"].shape == (0, 2)
        assert item["labels"].shape == (0,)


def test_manifest_with_gt_is_rejected(tmp_path):
    path = tmp_path / "target.json"
    path.write_text(json.dumps({
        "schema": "rq2-unlabeled-target-v1", "target_splits": ["target_train"],
        "database": {"P01_02": {"subset": "target_train", "annotations": []}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="target manifest contains labels"):
        check_target_manifest(path, ["target_train"])


def test_formal_regime_rejects_target_supervision():
    cfg = {"dataset_name": "epic", "dataset": {"action_subsampling": "10_majority_classes"},
           "diagnostic": {"supervised_upper_bound": "source_only",
                          "oracle_target_task": "joint"}}
    with pytest.raises(ValueError, match="oracle"):
        check_formal_config(cfg, "so")
    cfg["diagnostic"]["oracle_target_task"] = "off"
    assert check_formal_config(cfg, "so") is None
    with pytest.raises(ValueError, match="training mode"):
        check_formal_config(cfg, "uda")


def test_config_audit_allows_protocol_leaf_but_not_detector_changes(tmp_path):
    base = tmp_path / "base.yaml"
    other = tmp_path / "other.yaml"
    base.write_text("dataset:\n  feat_folder: /a\n  input_dim: 512\n"
                    "rq2_feature_protocol:\n  protocol_sha256: aaa\n"
                    "model:\n  fpn_dim: 512\n", encoding="utf-8")
    other.write_text("dataset:\n  feat_folder: /b\n  input_dim: 768\n"
                     "rq2_feature_protocol:\n  protocol_sha256: bbb\n"
                     "model:\n  fpn_dim: 512\n", encoding="utf-8")
    assert audit_configs([base, other])["configs"] == 2
    other.write_text(other.read_text().replace("fpn_dim: 512", "fpn_dim: 256"),
                     encoding="utf-8")
    with pytest.raises(RuntimeError, match="model.fpn_dim"):
        audit_configs([base, other])


def test_checkpoint_selection_matches_training_evaluation():
    ckpt = {"state_dict": {"student": 1}, "ema_state_dict": {"ema": 2}}
    cfg = {"train_cfg": {"ema_active": True}}
    assert selected_checkpoint_state(ckpt, cfg) == ("ema_state_dict", {"ema": 2})
    cfg["train_cfg"]["evaluate_student"] = True
    assert selected_checkpoint_state(ckpt, cfg) == ("state_dict", {"student": 1})
    with pytest.raises(ValueError, match="lacks selected"):
        selected_checkpoint_state({}, cfg)


def test_runtime_config_only_allows_feature_interface_and_seed_changes():
    frozen = {"init_rand_seed": 123, "loader": {"batch_size": 2},
              "dataset": {"feat_folder": "/slowfast", "input_dim": 2304,
                          "max_seq_len": 2304}}
    candidate = {"init_rand_seed": 7, "loader": {"batch_size": 2},
                 "dataset": {"feat_folder": "/vlm", "input_dim": 768,
                             "max_seq_len": 2304}}
    check_frozen_runtime_values(frozen, candidate)
    candidate["loader"]["batch_size"] = 4
    with pytest.raises(ValueError, match="loader.batch_size"):
        check_frozen_runtime_values(frozen, candidate)


def test_formal_features_require_full_set_and_provenance(monkeypatch):
    protocol = {
        "protocol_sha256": "abc", "video_set_count": 626,
        "video_set_sha256": FROZEN_VIDEO_SET_SHA256,
        "provenance": {"extractor_sha256": "code", "packages": {"torch": "x"}},
    }
    monkeypatch.setattr(
        "tools.rq2_dense_protocol.validate_training_protocol",
        lambda *_args: protocol,
    )
    monkeypatch.setattr(
        "tools.rq2_formal.audit_features",
        lambda *_args: {"protocol_sha256": "abc"},
    )
    cfg = {"dataset": {"feat_folder": "/features", "input_dim": 768}}
    assert check_formal_features(cfg, "/features/protocol.json", "/slowfast") is protocol
    protocol["video_set_count"] = 625
    with pytest.raises(ValueError, match="complete frozen"):
        check_formal_features(cfg, "/features/protocol.json", "/slowfast")


def test_so_uda_pair_audit_only_allows_regime_and_manifest(tmp_path):
    common = {"seed": 6, "model": {"fpn_dim": 512}}
    source_only = {
        **common, "diagnostic": {"supervised_upper_bound": "source_only"},
        "rq2_formal": {"regime": "so", "target_manifest": None},
    }
    uda = {
        **common, "diagnostic": {"supervised_upper_bound": "off"},
        "rq2_formal": {"regime": "uda", "target_manifest": {"sha256": "abc"}},
    }
    so_path, uda_path = tmp_path / "so.json", tmp_path / "uda.json"
    so_path.write_text(json.dumps(source_only), encoding="utf-8")
    uda_path.write_text(json.dumps(uda), encoding="utf-8")
    audit_so_uda_pair(so_path, uda_path)
    uda["model"]["fpn_dim"] = 256
    uda_path.write_text(json.dumps(uda), encoding="utf-8")
    with pytest.raises(RuntimeError, match="model.fpn_dim"):
        audit_so_uda_pair(so_path, uda_path)
