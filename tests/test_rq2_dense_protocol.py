import numpy as np
import json
from pathlib import Path

from tools.audit_rq2_vlm_benchmark import audit_shift_registry
from tools.extract_rq2_dense_vlm_features import (
    required_video_ids_from_registry,
)

from tools.rq2_dense_protocol import (
    expected_centers_seconds,
    sampled_source_indices,
    temporal_length,
    protocol_payload,
    protocol_sha256,
    sha256_normalized_text,
    validate_training_protocol,
    video_set_sha256,
    atomic_savez,
    validate_feature_file,
)


def test_temporal_length_matches_slowfast_grid():
    assert temporal_length(33745, 50.0) == 1265
    assert temporal_length(10689, 50.0) == 400


def test_sampling_is_uniform_within_the_shared_receptive_field():
    indices = sampled_source_indices(0, 8, 30.0, 100)
    assert indices.tolist() == [0, 4, 9, 13, 18, 22, 27, 31]
    assert np.all(np.diff(indices) > 0)


def test_centers_match_sada_coordinate_conversion():
    assert expected_centers_seconds(3).tolist() == [16 / 30, 32 / 30, 48 / 30]


def test_training_protocol_binds_folder_and_dimension(tmp_path):
    payload = protocol_payload("egovideo_hod", {"checkpoint.pt": "abc"})
    video_ids = ["P01_01"]
    payload["video_selection"] = "shift_registry_and_annotation"
    payload["video_set_count"] = len(video_ids)
    payload["video_set_sha256"] = video_set_sha256(video_ids)
    payload["protocol_sha256"] = protocol_sha256(payload)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "video_ids.txt").write_text("P01_01\n", encoding="ascii")
    validated = validate_training_protocol(protocol_path, tmp_path, 512)
    assert validated["model_name"] == "egovideo_hod"


def test_training_protocol_rejects_wrong_dimension(tmp_path):
    payload = protocol_payload("egovideo_hod", {"checkpoint.pt": "abc"})
    payload["protocol_sha256"] = protocol_sha256(payload)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_training_protocol(protocol_path, tmp_path, 768)
    except ValueError as exc:
        assert "input_dim" in str(exc)
    else:
        raise AssertionError("wrong feature dimension was accepted")


def test_feature_validation_rejects_temporal_collapse(tmp_path):
    path = tmp_path / "collapsed.npz"
    atomic_savez(
        path,
        feats=np.ones((2, 512), dtype=np.float16),
        protocol_sha256=np.asarray("expected"),
    )
    assert not validate_feature_file(path, "egovideo_hod", 2, "expected")


def test_qwen_vision_tower_is_not_formal_training_interface(tmp_path):
    payload = protocol_payload("qwen3_vl_8b", {"weights": "abc"})
    payload["protocol_sha256"] = protocol_sha256(payload)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_training_protocol(protocol_path, tmp_path, 4096)
    except ValueError as exc:
        assert "not eligible" in str(exc)
    else:
        raise AssertionError("Qwen vision-tower ablation entered the formal table")


def test_shift_registry_matches_frozen_configs():
    repository_root = Path(__file__).resolve().parents[1]
    result = audit_shift_registry(
        repository_root / "configs" / "rq2" / "shift_registry.yaml"
    )
    assert list(result["shifts"]) == ["S1", "S2", "S3", "S4", "S5", "S6"]


def test_config_hash_is_portable_across_line_endings(tmp_path):
    lf = tmp_path / "lf.yaml"
    crlf = tmp_path / "crlf.yaml"
    lf.write_bytes(b"a: 1\nb: 2\n")
    crlf.write_bytes(b"a: 1\r\nb: 2\r\n")
    assert sha256_normalized_text(lf) == sha256_normalized_text(crlf)


def test_rq2_video_set_comes_from_frozen_splits_and_annotations(tmp_path):
    config_dir = tmp_path / "configs" / "rq2"
    config_dir.mkdir(parents=True)
    annotation_dir = tmp_path / "data"
    annotation_dir.mkdir()
    annotation = annotation_dir / "annotations.json"
    annotation.write_text(json.dumps({"database": {
        "P01_01": {"subset": "source_train"},
        "P01_02": {"subset": "target_val"},
        "P99_99": {"subset": "unused"},
    }}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "src_train_split: [source_train]\n"
        "src_val_split: [source_val]\n"
        "tgt_train_split: [target_train]\n"
        "tgt_val_split: [target_val]\n"
        "dataset:\n  json_file: data/annotations.json\n",
        encoding="utf-8",
    )
    registry = config_dir / "shift_registry.yaml"
    registry.write_text(
        "repository_root: ../..\n"
        "shifts:\n  S1:\n    config: config.yaml\n",
        encoding="utf-8",
    )
    video_ids = required_video_ids_from_registry(
        registry
    )
    assert video_ids == ["P01_01", "P01_02"]
    assert len(video_set_sha256(video_ids)) == 64
