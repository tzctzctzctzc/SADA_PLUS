import json

import numpy as np
import pytest

from libs.datasets.epic_kitchens import EpicKitchensDataset
from tools.audit_rq2_vlm_benchmark import audit_configs, audit_so_uda_pair
from tools import rq2_evaluate
from tools.rq2_evaluate import (
    aggregate_pair_summaries, checkpoint_config_for_audit, load_checkpoint,
    selected_checkpoint_state, summarize_so_uda,
)
from tools.rq2_formal import (
    FROZEN_VIDEO_SET_SHA256, build_formal_run_signature, build_target_manifest,
    check_formal_config, check_formal_features, check_frozen_runtime_values,
    check_target_manifest, validate_formal_resume, validate_formal_runtime,
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


def test_target_manifest_must_match_the_complete_unlabeled_split(tmp_path):
    annotated = tmp_path / "annotated.json"
    annotated.write_text(json.dumps({"database": {
        "P01_01": {"subset": "target_train", "duration": 8, "fps": 30,
                   "annotations": [{"label": "take", "label_id": 0,
                                    "segment": [1, 4]}]},
        "P01_02": {"subset": "target_train", "duration": 9, "fps": 30,
                   "annotations": []},
    }}), encoding="utf-8")
    manifest = tmp_path / "unlabeled.json"
    build_target_manifest(annotated, ["target_train"], manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["database"].pop("P01_02")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        check_target_manifest(manifest, ["target_train"], annotated)


def test_dataset_label_map_does_not_scan_other_domains(tmp_path):
    annotated = tmp_path / "annotated.json"
    annotated.write_text(json.dumps({"database": {
        "P01_01": {"subset": "source_train", "duration": 8, "fps": 30,
                   "annotations": [{"label": "take", "label_id": 0,
                                    "segment": [1, 4]}]},
        "P01_02": {"subset": "target_train", "duration": 8, "fps": 30,
                   "annotations": [{"label": "target_only", "label_id": 1,
                                    "segment": [2, 6]}]},
    }}), encoding="utf-8")
    np.savez(tmp_path / "P01_01.npz", feats=np.ones((20, 4), dtype=np.float32))
    dataset = EpicKitchensDataset(
        is_training=False, split=["source_train"], feat_folder=str(tmp_path),
        json_file=str(annotated), feat_stride=16, num_frames=32, default_fps=30,
        downsample_rate=1, max_seq_len=4, trunc_thresh=0.3,
        crop_ratio=None, input_dim=4, num_classes=2, file_prefix=None,
        file_ext=".npz", force_upsampling=False, action_subsampling="all",
    )
    assert dataset.label_dict == {"take": 0}


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
    ckpt["selection_provenance"] = {
        "split": "source_validation", "metric": "validation_loss",
        "metric_value": 1.0, "selected_epoch": 5,
        "state_key": "ema_state_dict",
    }
    ckpt["epoch"] = 5
    cfg["train_cfg"]["evaluate_student"] = True
    assert selected_checkpoint_state(ckpt, cfg, True) == (
        "ema_state_dict", {"ema": 2}
    )


def test_checkpoint_load_supports_torch_without_weights_only(monkeypatch):
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        if "weights_only" in kwargs:
            raise TypeError("load() got an unexpected keyword argument 'weights_only'")
        return {"path": path}

    monkeypatch.setattr(rq2_evaluate.torch, "load", fake_load)
    assert load_checkpoint("checkpoint.pt") == {"path": "checkpoint.pt"}
    assert calls == [
        {"map_location": "cpu", "weights_only": False},
        {"map_location": "cpu"},
    ]


def test_legacy_source_only_checkpoint_has_narrow_compatibility():
    frozen = {
        "dataset_name": "epic",
        "dataset": {"action_subsampling": "10_majority_classes"},
        "train_cfg": {"test_freq": 5},
    }
    saved = {
        **frozen,
        "dataset": dict(frozen["dataset"]),
        "train_cfg": {"test_freq": -1},
        "diagnostic": {"supervised_upper_bound": "source_only"},
    }
    audited, schema = checkpoint_config_for_audit(saved, frozen)
    assert schema == "legacy-source-only"
    assert audited["train_cfg"]["test_freq"] == 5
    assert saved["train_cfg"]["test_freq"] == -1
    saved["train_cfg"]["test_freq"] = 5
    with pytest.raises(ValueError, match="periodic target evaluation"):
        checkpoint_config_for_audit(saved, frozen)


def test_formal_checkpoint_config_is_reaudited():
    frozen = {
        "dataset_name": "epic",
        "dataset": {"action_subsampling": "10_majority_classes"},
        "train_cfg": {"test_freq": 5},
    }
    saved = {
        **frozen,
        "dataset": dict(frozen["dataset"]),
        "train_cfg": dict(frozen["train_cfg"]),
        "diagnostic": {
            "supervised_upper_bound": "source_only",
            "oracle_target_task": "joint",
        },
        "rq2_formal": {"regime": "so", "target_manifest": None},
    }
    with pytest.raises(ValueError, match="forbids target GT"):
        checkpoint_config_for_audit(saved, frozen)


def test_runtime_config_only_allows_feature_interface_and_seed_changes():
    frozen = {"init_rand_seed": 123, "loader": {"batch_size": 2},
              "dataset": {"feat_folder": "/slowfast", "input_dim": 2304,
                          "max_seq_len": 2304}}
    candidate = {"init_rand_seed": 7, "loader": {"batch_size": 2},
                 "dataset": {"feat_folder": "/vlm", "input_dim": 768,
                             "max_seq_len": 2304}}
    check_frozen_runtime_values(frozen, candidate)
    candidate["dataset"]["crop_alignment"] = 1
    check_frozen_runtime_values(frozen, candidate)
    candidate["dataset"]["crop_alignment"] = 16
    with pytest.raises(ValueError, match="dataset.crop_alignment"):
        check_frozen_runtime_values(frozen, candidate)
    candidate["dataset"]["crop_alignment"] = 1
    candidate["loader"]["batch_size"] = 4
    with pytest.raises(ValueError, match="loader.batch_size"):
        check_frozen_runtime_values(frozen, candidate)
    candidate["loader"]["new_behavior"] = True
    candidate["loader"]["batch_size"] = 2
    with pytest.raises(ValueError, match="unexpected runtime field"):
        check_frozen_runtime_values(frozen, candidate)
    nested_frozen = {"model": {"loss_weights": {"task": 1.0}}}
    nested_candidate = {
        "model": {"loss_weights": {"task": 1.0, "hidden_override": 2.0}}
    }
    with pytest.raises(ValueError, match="model.loss_weights.hidden_override"):
        check_frozen_runtime_values(nested_frozen, nested_candidate)


def test_formal_features_require_full_set_and_provenance(monkeypatch):
    digest = "a" * 64
    protocol = {
        "protocol_sha256": "abc", "video_set_count": 626,
        "video_set_sha256": FROZEN_VIDEO_SET_SHA256,
        "checkpoint_sha256": {"model.pt": digest},
        "provenance": {
            "extractor_sha256": digest,
            "repository": {"revision": "b" * 40, "tracked_diff_sha256": digest},
            "external_repository": {
                "revision": "c" * 40, "tracked_diff_sha256": digest,
            },
            "batch_tokens": 8, "device_map": "none", "attention": "sdpa",
            "dtype": "bfloat16", "packages": {"torch": "2.0"},
        },
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


def test_formal_config_rejects_behavior_changing_diagnostics():
    cfg = {
        "dataset_name": "epic",
        "dataset": {"action_subsampling": "10_majority_classes"},
        "train_cfg": {"evaluate_student": False, "test_target_val_only": False},
        "diagnostic": {"supervised_upper_bound": "source_only"},
    }
    check_formal_config(cfg, "so")
    for key, value in (
        ("stop_after_epoch", 1),
        ("target_alignment", "action"),
        ("conditional_paired_support", True),
        ("class_support", True),
    ):
        changed = json.loads(json.dumps(cfg))
        changed["diagnostic"][key] = value
        with pytest.raises(ValueError, match="requires diagnostic"):
            check_formal_config(changed, "so")
    cfg["train_cfg"]["evaluate_student"] = True
    with pytest.raises(ValueError, match="EMA"):
        check_formal_config(cfg, "so")


def test_formal_runtime_and_resume_require_exact_signature():
    cfg = {
        "dataset_name": "epic",
        "dataset": {"action_subsampling": "10_majority_classes"},
        "train_cfg": {"evaluate_student": False, "test_target_val_only": False},
        "diagnostic": {"supervised_upper_bound": "source_only"},
        "rq2_formal": {
            "regime": "so", "shift": "S1", "target_manifest": None,
            "training_budget": "source-batches-per-epoch-v1",
            "rng_protocol": "independent-source-target-generators-v1",
        },
        "seed": 6,
    }
    signature = build_formal_run_signature(cfg)
    cfg["rq2_formal"]["run_signature"] = signature
    assert validate_formal_runtime(cfg, signature) == signature
    validate_formal_resume({"rq2_run_signature": signature}, signature)
    cfg["train_cfg"]["test_freq"] = -1
    with pytest.raises(ValueError, match="changed after audit"):
        validate_formal_runtime(cfg, signature)
    with pytest.raises(ValueError, match="different run signature"):
        validate_formal_resume({"rq2_run_signature": {}}, signature)


def test_resume_operational_path_does_not_change_run_signature():
    cfg = {
        "seed": 6,
        "rq2_formal": {"regime": "so", "shift": "S1"},
        "benchmark_audit": {
            "training_initialization": "random", "initialization_inputs": {},
            "checkpoint_selection_metric": "source_validation_loss",
        },
    }
    initial = build_formal_run_signature(cfg)
    cfg["benchmark_audit"]["training_initialization"] = "checkpoint"
    cfg["benchmark_audit"]["initialization_inputs"] = {
        "resume": "/run/saved_ckpt/latest_ckpt/checkpoint-epoch5.pt"
    }
    assert build_formal_run_signature(cfg) == initial


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


def test_so_uda_pair_audit_accepts_distinct_valid_run_signatures(tmp_path):
    common = {"seed": 6, "model": {"fpn_dim": 512}, "opt": {"eta_min": 5e-5}}
    source_only = {
        **common, "name": "so", "unique_id": "run_so", "gpu": 0,
        "config_path": "/checkout_so/S1.yaml",
        "diagnostic": {"supervised_upper_bound": "source_only"},
        "rq2_formal": {"regime": "so", "target_manifest": None},
    }
    uda = {
        **common, "name": "uda", "unique_id": "run_uda", "gpu": 1,
        "config_path": "/checkout_uda/S1.yaml",
        "diagnostic": {"supervised_upper_bound": "off"},
        "rq2_formal": {"regime": "uda", "target_manifest": {"sha256": "abc"}},
    }
    for cfg in (source_only, uda):
        cfg["rq2_formal"]["run_signature"] = build_formal_run_signature(cfg)
    so_path, uda_path = tmp_path / "so.json", tmp_path / "uda.json"
    so_path.write_text(json.dumps(source_only), encoding="utf-8")
    uda_path.write_text(json.dumps(uda), encoding="utf-8")
    audit_so_uda_pair(so_path, uda_path)
    uda["rq2_formal"]["run_signature"]["seed"] = 7
    uda_path.write_text(json.dumps(uda), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid run signature"):
        audit_so_uda_pair(so_path, uda_path)


def test_so_uda_summary_computes_required_metrics_and_hashes(tmp_path, monkeypatch):
    common = {
        "split": ["validation"], "seed": 6, "shift": "S1",
        "config_sha256": "config", "feature_protocol_sha256": "feature",
        "annotation_sha256": "annotation", "map_by_tiou": {"0.1": 0.1},
        "ap_by_class": {"0": 0.1}, "predictions": [],
    }
    payloads = [
        {**common, "domain": "source", "checkpoint_sha256": "so", "map_0.1_to_0.5": 0.50},
        {**common, "domain": "target", "checkpoint_sha256": "so", "map_0.1_to_0.5": 0.30},
        {**common, "domain": "source", "checkpoint_sha256": "uda", "map_0.1_to_0.5": 0.48},
        {**common, "domain": "target", "checkpoint_sha256": "uda", "map_0.1_to_0.5": 0.36},
    ]
    paths = []
    for index, payload in enumerate(payloads):
        path = tmp_path / str(index) / "result.json"
        path.parent.mkdir()
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    monkeypatch.setattr(
        rq2_evaluate, "_paired_bootstrap_ci",
        lambda *_args, **_kwargs: {
            "unit": "video_id", "repeats": 10, "seed": 1,
            "videos": 2, "ci_95": [0.0, 0.1],
        },
    )
    result = summarize_so_uda(*paths, "annotations.json", tmp_path / "pair.json")
    assert result["metrics"]["Gap_SO"] == pytest.approx(-0.20)
    assert result["metrics"]["Retention_SO"] == pytest.approx(60.0)
    assert result["metrics"]["Gain_target"] == pytest.approx(0.06)
    assert result["metrics"]["Change_source"] == pytest.approx(-0.02)
    assert result["metrics"]["Recovery_gap"] == pytest.approx(0.08)
    assert set(result["inputs"]) == {"SO_source", "SO_target", "UDA_source", "UDA_target"}


def test_pair_aggregate_requires_complete_seed_grid(tmp_path):
    paths = []
    for seed in (6, 7, 8):
        payload = {
            "schema": "rq2-so-uda-pair-v1", "seed": seed, "shift": "S1",
            "feature_protocol_sha256": "feature",
            "metrics": {"T_SO": seed / 100, "T_UDA": seed / 100 + 0.1},
            "map_by_tiou": {"SO_source": {"0.1": 0.1}},
            "ap_by_class": {"SO_source": {"0": 0.1}},
        }
        path = tmp_path / "pair-{}.json".format(seed)
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    partial = aggregate_pair_summaries(
        paths, tmp_path / "partial.json", require_complete=False
    )
    assert partial["per_shift"]["S1"]["seeds"] == [6, 7, 8]
    with pytest.raises(ValueError, match="all six shifts"):
        aggregate_pair_summaries(paths, tmp_path / "full.json")
