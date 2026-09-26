import json

import pytest

from tools.run_rq2_matched_control import (
    checked_evaluation, check_new_protocol, idle_gpu_uuid, matched_feature_args,
)
from tools.rq2_dense_protocol import sha256_file


def test_new_queue_rejects_legacy_so_before_training():
    cfg = {"rq2_formal": {"training_budget": "source-batches-per-epoch-v1"}}
    with pytest.raises(ValueError, match="legacy SO/UDA"):
        check_new_protocol(cfg)


def test_vlm_requires_frozen_feature_root():
    with pytest.raises(ValueError, match="requires --feature-root"):
        matched_feature_args("qwen3_vl_8b", None)
    with pytest.raises(ValueError, match="frozen reference"):
        matched_feature_args("slowfast", "/alternate/features")
    assert matched_feature_args("slowfast", None) == []


def test_vlm_rejects_other_model_with_same_dimension(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.run_rq2_matched_control.validate_training_protocol",
                        lambda *_args: {"model_name": "timelens2_8b"})
    with pytest.raises(ValueError, match="another model"):
        matched_feature_args("qwen3_vl_8b", tmp_path)


@pytest.mark.parametrize("model,dim", [
    ("egovideo_hod", 512), ("internvideo2_clip_1b", 768),
    ("qwen3_vl_8b", 4096), ("timelens2_8b", 4096),
])
def test_vlm_binds_registered_dimension_and_protocol(tmp_path, monkeypatch, model, dim):
    checked = []

    def validate(path, folder, input_dim):
        checked.append((path, folder, input_dim))
        return {"model_name": model}

    monkeypatch.setattr("tools.run_rq2_matched_control.validate_training_protocol", validate)
    args = matched_feature_args(model, tmp_path)
    assert checked == [(tmp_path / "protocol.json", tmp_path, dim)]
    assert args == ["--feature_folder", str(tmp_path), "--input_dim", str(dim),
                    "--feature-protocol", str(tmp_path / "protocol.json")]


def test_gpu_gate_rejects_low_memory_but_running_external_process(monkeypatch):
    responses = iter(["GPU-test, 4, 0", "GPU-test, 12345"])
    monkeypatch.setattr("tools.run_rq2_matched_control.subprocess.check_output",
                        lambda *_args, **_kwargs: next(responses))
    with pytest.raises(RuntimeError, match="external jobs"):
        idle_gpu_uuid(3)


def test_evaluation_gate_requires_matching_policy_and_hashes(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"test")
    policy = {"training_budget": "budget", "source_optimization": "optimization",
              "epoch_budget": {}, "rng_protocol": "rng", "data_provenance": {"hash": "data"}}
    cfg = {"seed": 6, "rq2_formal": {"shift": "S1", **policy}}
    manifest = {"seed": 6, "shift": "S1", "checkpoint_sha256": sha256_file(checkpoint),
                "selection_provenance": {"split": "source_validation"},
                "training_protocol": policy, "outputs": {}}
    for name in ("source_val.json", "target_val.json"):
        path = tmp_path / name
        path.write_text(json.dumps({"training_protocol": policy, "map_0.1_to_0.5": 0.1}))
        manifest["outputs"][name] = sha256_file(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    checked_evaluation(tmp_path, cfg, checkpoint)
    (tmp_path / "target_val.json").write_text("{}")
    with pytest.raises(ValueError, match="output hash"):
        checked_evaluation(tmp_path, cfg, checkpoint)
