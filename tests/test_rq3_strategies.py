import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

# 与既有target_adapter测试一样隔离纯辅助模块，避免导入Linux NMS/完整训练器。
# 使用真实losses_aux和真实torch，不模拟训练数值或梯度。
PACKAGE = types.ModuleType("rq3_cpu_test")
PACKAGE.__path__ = [str(Path(__file__).resolve().parents[1] / "libs/models/SADA")]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location("rq3_cpu_test.rq3_auxiliary", Path(PACKAGE.__path__[0]) / "rq3_auxiliary.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RQ3Auxiliary, candidate_routes = MODULE.RQ3Auxiliary, MODULE.candidate_routes
from tools.rq3_protocol import TARGET_CLASS_NAMES
from tools.build_rq3_cache import build_cache, paired_text_probabilities, probabilities_from_logits
from tools.rq3_protocol import FRAME_PROTOCOL, SCHEMA, audit_baseline, content_hash, load_cache
from tools.run_rq3_matrix import matrix


def inputs():
    return {
        "videos": [{"metadata": {"video_id": "P01_01", "crop_start": 4}}],
        "points": [torch.tensor([[0., 0, 4, 1], [1, 0, 4, 1]])],
        "target_masks": [torch.ones(1, 1, 2, dtype=torch.bool)],
        "target_logits": [torch.tensor([[[2., -2], [-2., 2]]], requires_grad=True)],
        "target_features": [torch.tensor([[[0., 0], [1., 1]]], requires_grad=True)],
        "source_features": [torch.tensor([[[1., 0], [0., 1]]], requires_grad=True)],
        "source_masks": [torch.ones(1, 1, 2, dtype=torch.bool)],
        "source_labels": [torch.tensor([[[1., 0], [0., 1]]])],
        "source_loss": torch.tensor(100., requires_grad=True),
        "active": True, "alpha": 0.7,
    }


def auxiliary(strategy, labels=(0, 0), budget=0.05):
    cache = {"P01_01": [{"absolute_feature_index": 4 + i, "source_fpn_levels": [0], "class_id": label}
                         for i, label in enumerate(labels)]}
    return RQ3Auxiliary(strategy, (0,), (2,), 2, cache, budget)


def test_three_versions_add_no_parameters_or_initialization_rng():
    state = torch.random.get_rng_state().clone()
    for strategy in ("hard", "verify", "prototype"):
        assert list(auxiliary(strategy).parameters()) == []
    assert torch.equal(state, torch.random.get_rng_state())


def test_hard_and_verify_use_same_eligible_denominator_and_only_target_head_gradient():
    data = inputs()
    hard = auxiliary("hard")(**data)
    verify = auxiliary("verify")(**data)
    assert hard["eligible"] == verify["eligible"] == 2
    assert hard["selected"] == 2 and verify["selected"] == 1
    one = inputs()
    one["target_logits"][0] = one["target_logits"][0][:, :1]
    one["target_masks"][0] = one["target_masks"][0][..., :1]
    one["points"][0] = one["points"][0][:1]
    single = auxiliary("hard")(**one)
    assert torch.allclose(verify["raw_loss"], single["raw_loss"] / 2)
    verify["loss"].backward()
    assert data["target_logits"][0].grad[:, 0].abs().sum() > 0
    assert data["target_logits"][0].grad[:, 1].abs().sum() == 0
    assert data["source_features"][0].grad is None
    assert data["target_features"][0].grad is None
    assert data["source_loss"].grad == 0  # cap及scale不向source任务反传


def test_prototype_updates_from_source_gt_and_only_target_feature_gradient():
    data = inputs()
    model = auxiliary("prototype")
    output = model(**data)
    assert output["selected"] == 2 and output["raw_loss"] > 0
    assert torch.equal(model.prototypes[0], torch.eye(2))
    output["loss"].backward()
    assert data["target_features"][0].grad.abs().sum() > 0
    assert data["source_features"][0].grad is None
    assert data["target_logits"][0].grad is None
    restored = auxiliary("prototype")
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored.prototypes, model.prototypes)
    assert torch.equal(restored.ready, model.ready)


@pytest.mark.parametrize("strategy", ["hard", "verify", "prototype"])
def test_warmup_has_no_auxiliary_loss(strategy):
    data = inputs()
    data["active"] = False
    output = auxiliary(strategy)(**data)
    assert output["loss"] == 0 and output["selected"] == 0


@pytest.mark.parametrize("strategy", ["hard", "verify", "prototype"])
def test_auxiliary_cap_and_detached_scale(strategy):
    data = inputs()
    data["source_loss"] = torch.tensor(0.001, requires_grad=True)
    output = auxiliary(strategy)(**data)
    assert output["loss"] <= 0.05 * data["source_loss"] + 1e-9
    assert not output["scale"].requires_grad
    assert torch.isfinite(output["gradient_norm"]) and output["gradient_norm"] > 0


def test_prototype_missing_class_has_zero_loss_without_changing_denominator():
    data = inputs()
    data["source_labels"][0].zero_()
    output = auxiliary("prototype")(**data)
    assert output["eligible"] == 2 and output["selected"] == 0 and output["loss"] == 0


def test_routes_use_absolute_crop_coordinates_and_exclude_padding_and_other_levels():
    data = inputs()
    data["target_masks"][0][..., 1] = False
    cache = {"P01_01": [
        {"absolute_feature_index": 4, "source_fpn_levels": [0], "class_id": 0},
        {"absolute_feature_index": 5, "source_fpn_levels": [0], "class_id": 1},
        {"absolute_feature_index": 3, "source_fpn_levels": [0], "class_id": 0},
        {"absolute_feature_index": 4, "source_fpn_levels": [1], "class_id": 1},
    ]}
    routes = candidate_routes(data["videos"], data["points"], data["target_masks"],
                              data["target_logits"], cache, (0,))
    assert routes == [[(0, 0, 0, 0., 0)]]


def cache_payload():
    return {"schema": SCHEMA, "teacher_model": "qwen3_vl_8b", "shift": "S1", "seed": 6,
            "labels": list(TARGET_CLASS_NAMES), "target_gt_used": False,
            "target_video_ids": ["P01_01"], "teacher_selection": "source-validation-only",
            "feature_lengths": {"P01_01": 8},
            "candidate_sha256": "a" * 64, "frames_sha256": "b" * 64, "teacher_checkpoint_sha256": "c" * 64,
            "input_protocol": {"frame_protocol": FRAME_PROTOCOL, "input_frames": 8, "class_blind": True,
                               "readout": "closed-class-single-token-log-likelihood", "checkpoint_sha256": {"model": "d" * 64},
                               "prompt_sha256": "e" * 64, "processor_sha256": "f" * 64},
            "entries": [{"video_id": "P01_01", "absolute_feature_index": 4, "source_fpn_levels": [0],
                         "class_id": 0, "class_probabilities": [1.] + [0.] * 9}]}


def write_payload(tmp_path, payload):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cache_accepts_valid_candidate_semantics(tmp_path):
    path = write_payload(tmp_path, cache_payload())
    _, cache = load_cache(path, "qwen3_vl_8b", "S1", ["P01_01"])
    assert cache["P01_01"][0]["class_id"] == 0


@pytest.mark.parametrize("change", ["target_gt", "hidden_readout", "another_model", "members", "duplicate", "nan",
                                    "prompt_hash", "processor_hash", "empty"])
def test_cache_rejects_leakage_or_identity_conflicts(tmp_path, change):
    payload = cache_payload()
    if change == "target_gt":
        payload["entries"][0]["gt_class"] = 0
    elif change == "hidden_readout":
        payload["input_protocol"]["readout"] = "neutral_prompt_multimodal_last_token"
    elif change == "another_model":
        payload["teacher_model"] = "egovideo_hod"
    elif change == "members":
        payload["target_video_ids"] = ["outside"]
    elif change == "duplicate":
        payload["entries"].append(copy.deepcopy(payload["entries"][0]))
    elif change in ("prompt_hash", "processor_hash"):
        payload["input_protocol"][change.replace("_hash", "_sha256")] = "unknown"
    elif change == "empty":
        payload["entries"] = []
    else:
        payload["entries"][0]["class_probabilities"][0] = float("nan")
    with pytest.raises(ValueError):
        load_cache(write_payload(tmp_path, payload), "qwen3_vl_8b", "S1", ["P01_01"])


def test_matrix_is_54_unique_runs_single_seed_single_card():
    entries = matrix()
    assert len(entries) == len(set(entries)) == 54
    assert {entry[0] for entry in entries} == {"egovideo_hod", "internvideo2_clip_1b", "qwen3_vl_8b"}
    assert {entry[1] for entry in entries} == {"hard", "verify", "prototype"}
    assert {entry[2] for entry in entries} == {"S" + str(index) for index in range(1, 7)}


def test_baseline_equivalence_allows_only_rq3_root():
    baseline = {"rq2_formal": {"regime": "uda", "shift": "S1"}, "seed": 6,
                "dataset": {"input_dim": 2304}, "model": {"loss_weights": {"weight_downstream": 0.2}}}
    candidate = copy.deepcopy(baseline)
    candidate["rq3_auxiliary"] = {"strategy": "prototype"}
    assert audit_baseline(baseline, candidate)["only_auxiliary_changed"]
    candidate["model"]["loss_weights"]["weight_downstream"] = 1.0
    with pytest.raises(ValueError, match="beyond"):
        audit_baseline(baseline, candidate)


def test_off_configure_preserves_exact_config():
    from tools.rq3_protocol import configure
    cfg = {"model": {"input_dim": 2304}}
    original = copy.deepcopy(cfg)
    configure(cfg, "off", None, None, None)
    assert cfg == original


@pytest.mark.parametrize("strategy", ["hard", "verify", "prototype"])
def test_runtime_binding_checks_actual_seed_members_and_cache_hash(tmp_path, strategy, monkeypatch):
    from tools.rq3_protocol import configure, validate_config
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({"database": {"P01_01": {"subset": "target_train"}}}))
    cfg = {"init_rand_seed": 6, "tgt_train_split": ["target_train"],
           "dataset": {"input_dim": 2304, "json_file": str(annotation), "feat_folder": str(tmp_path), "file_ext": ".npz"},
           "opt": {"epochs": 50}, "loader": {"batch_size": 2}, "rq2_formal": {"shift": "S1"}}
    np.savez(tmp_path / "P01_01.npz", feats=np.zeros((8, 2304), dtype=np.float32))
    payload = cache_payload()
    bind_test_interface(tmp_path, payload, monkeypatch)
    path = write_payload(tmp_path, payload)
    configure(cfg, strategy, path, "qwen3_vl_8b", "S1")
    validate_config(cfg, "uda")
    cfg["init_rand_seed"] = 7
    with pytest.raises(ValueError, match="seed6"):
        validate_config(cfg, "uda")
    cfg["init_rand_seed"] = 6
    payload = cache_payload()
    payload["entries"][0]["absolute_feature_index"] += 1
    write_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="changed after audit"):
        validate_config(cfg, "uda")


def test_semantic_readout_never_uses_qwen_hidden_feature_as_scores():
    assert np.argmax(probabilities_from_logits(np.arange(10))) == 9
    probs = paired_text_probabilities(np.eye(10)[:2], np.eye(10), 10)
    assert np.argmax(probs, axis=-1).tolist() == [0, 1]
    with pytest.raises(ValueError, match="ten-class"):
        probabilities_from_logits(np.zeros(4096))


def test_builder_requires_same_exact_frames_and_no_gt():
    protocol = cache_payload()["input_protocol"]
    manifest = {"schema": "rq3-common-candidates-v1", "target_gt_used": False, "shift": "S1", "seed": 6,
                "teacher_selection": "source-validation-only", "teacher_checkpoint_sha256": "c" * 64,
                "frame_protocol": FRAME_PROTOCOL, "target_video_ids": ["P01_01"],
                "feature_lengths": {"P01_01": 8},
                "candidates": [{"candidate_id": "P01_01-4", "video_id": "P01_01", "absolute_feature_index": 4,
                                "source_fpn_levels": [0], "proposal_start_sec": 1., "proposal_end_sec": 2.,
                                "frames": list(range(8)), "frame_sha256": ["d" * 64] * 8}]}
    predictions = [{"candidate_id": "P01_01-4", "frame_sha256": content_hash(["d" * 64] * 8),
                    "class_probabilities": [1.] + [0.] * 9}]
    result = build_cache(manifest, predictions, protocol, "qwen3_vl_8b", "S1", "c" * 64)
    assert result["entries"][0]["class_id"] == 0
    predictions[0]["frame_sha256"] = "other"
    with pytest.raises(ValueError, match="different proposal frames"):
        build_cache(manifest, predictions, protocol, "qwen3_vl_8b", "S1", "c" * 64)


def bind_test_interface(tmp_path, payload, monkeypatch):
    """仅测试身份门禁，用真实小文件注册测试身份；不模拟官方模型或产出实验结果。"""
    from tools import rq3_protocol as protocol
    from tools.rq2_dense_protocol import protocol_sha256, sha256_file
    weight = tmp_path / "test_weights"
    weight.write_bytes(b"identity-gate-test-only")
    frozen = {"schema": "rq2-dense-vlm-feature-v1", "model_name": "qwen3_vl_8b",
              "checkpoint_sha256": {"model": sha256_file(weight)}}
    signature = protocol_sha256(frozen)
    frozen["protocol_sha256"] = signature
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen))
    monkeypatch.setitem(protocol.FROZEN_PROTOCOLS, "qwen3_vl_8b", signature)
    payload["input_protocol"]["checkpoint_sha256"] = frozen["checkpoint_sha256"]
    rows = [{"candidate_id": "P01_01-p0000004", "frame_sha256": "0" * 64,
             "class_probabilities": payload["entries"][0]["class_probabilities"]}]
    predictions = tmp_path / "identity_predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    payload["predictions_sha256"] = content_hash(rows)
    audit = {"schema": "rq3-vlm-interface-audit-v1", "teacher_model": "qwen3_vl_8b", "smoke_status": "passed",
             "frozen_protocol_path": str(frozen_path), "frozen_protocol_sha256": signature,
             "checkpoint_files": {"model": str(weight)},
             "processor_files": {str(weight): sha256_file(weight)},
             "predictions_path": str(predictions), "predictions_sha256": content_hash(rows),
             "common_manifest_sha256": payload["candidate_sha256"],
             **{key: payload["input_protocol"][key] for key in ("checkpoint_sha256", "prompt_sha256", "processor_sha256", "readout")}}
    path = tmp_path / "test_identity_audit.json"
    path.write_text(json.dumps(audit))
    payload["input_protocol"].update(interface_audit_path=str(path), interface_audit_sha256=sha256_file(path))


def test_verify_cap_uses_unfiltered_hard_reference_without_gradient_reweighting():
    data = inputs()
    data["source_loss"] = torch.tensor(.001, requires_grad=True)
    hard = auxiliary("hard")(**data)
    verify = auxiliary("verify")(**data)
    assert torch.equal(hard["scale"], verify["scale"])
    grad_h = torch.autograd.grad(hard["loss"], data["target_logits"], retain_graph=True)[0]
    grad_v = torch.autograd.grad(verify["loss"], data["target_logits"])[0]
    assert torch.equal(grad_h[:, 0], grad_v[:, 0])
    assert grad_v[:, 1].abs().sum() == 0
    assert verify["loss"] < hard["loss"]


@pytest.mark.parametrize("crop", [0, 1, 2, 3])
def test_high_level_candidate_survives_crop_phase_by_linear_interpolation(crop):
    cache = {"P01_01": [{"absolute_feature_index": 4, "source_fpn_levels": [1], "class_id": 0}]}
    grid = torch.tensor([[0., 0., 4., 2.], [2., 0., 4., 2.], [4., 0., 4., 2.]])
    values = torch.tensor([[[0.], [2.], [4.]]], requires_grad=True)
    videos = [{"metadata": {"video_id": "P01_01", "crop_start": crop, "crop_feat_len": 6}}]
    queries = candidate_routes(videos, [grid, grid], [torch.ones(1, 1, 3)] * 2, [values, values], cache, [1])[1]
    assert len(queries) == 1
    sampled, _ = MODULE.query_values(values, queries)
    assert sampled.item() == 4 - crop
    sampled.sum().backward()
    assert values.grad.sum().item() == 1


@pytest.mark.parametrize("change", ["inactive_level", "out_of_bounds"])
def test_cache_rejects_inactive_or_out_of_bounds_candidates(tmp_path, change):
    payload = cache_payload()
    if change == "inactive_level":
        payload["entries"][0]["source_fpn_levels"] = [3]
    else:
        payload["entries"][0]["absolute_feature_index"] = 8
    with pytest.raises(ValueError):
        load_cache(write_payload(tmp_path, payload))


def test_whole_semantic_cache_snapshot_catches_probability_only_changes(tmp_path):
    from tools.run_rq3_matrix import check_cache_snapshot
    from tools.rq2_dense_protocol import sha256_file
    path = write_payload(tmp_path, cache_payload())
    snapshot = {"cache.json": sha256_file(path)}
    check_cache_snapshot(tmp_path, snapshot)
    payload = cache_payload()
    payload["entries"][0].update(class_id=1, class_probabilities=[0., 1.] + [0.] * 8)
    write_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="between strategies"):
        check_cache_snapshot(tmp_path, snapshot)


def test_zero_coverage_cannot_be_announced_as_effective_auxiliary_training(tmp_path):
    from tools.run_rq3_matrix import supervision_audit
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"epoch": 50, "t_rq3_eligible": 0, "t_rq3_selected": 0, "t_rq3_loss": 0}))
    assert not supervision_audit(path)["effective"]


def test_official_identity_requires_registered_checkpoint_not_hex_format(tmp_path, monkeypatch):
    from tools.rq3_protocol import validate_interface_audit
    payload = cache_payload()
    bind_test_interface(tmp_path, payload, monkeypatch)
    validate_interface_audit(payload["input_protocol"], "qwen3_vl_8b")
    payload["input_protocol"]["checkpoint_sha256"] = {"unregistered_model": "0" * 64}
    with pytest.raises(ValueError, match="approved"):
        validate_interface_audit(payload["input_protocol"], "qwen3_vl_8b")


def test_auxiliary_reaches_real_optimizer_parameters_without_source_or_teacher_gradient():
    data = inputs()
    head = torch.nn.Conv1d(2, 2, 1)
    target = data["target_features"][0]
    data["target_logits"] = [head(target).transpose(1, 2)]
    before = head.weight.detach().clone()
    optimizer = torch.optim.SGD(head.parameters(), lr=.1)
    output = auxiliary("hard")(**data)
    optimizer.zero_grad()
    (.2 * output["loss"]).backward()
    optimizer.step()
    assert not torch.equal(head.weight, before)
    assert data["source_features"][0].grad is None


def test_proposal_midpoint_grid_and_true_interface_shape_contract():
    from tools.prepare_rq3_candidates import proposal_frame_indices
    from tools.score_rq3_vlm_candidates import check_scores, pixel_hash
    assert proposal_frame_indices(0, 8, 1) == list(range(8))
    assert len(set(proposal_frame_indices(0, .01, 30))) < 8
    assert check_scores(np.full(10, .1)) == pytest.approx([.1] * 10)
    with pytest.raises(ValueError, match="ten-class"):
        check_scores(np.zeros(4096))
    with pytest.raises(ValueError):
        check_scores([float('nan')] * 10)
    assert pixel_hash(np.zeros((2, 3, 3), dtype=np.uint8)) != pixel_hash(np.zeros((3, 2, 3), dtype=np.uint8))


def test_cached_probabilities_cannot_be_resigned_after_official_inference(tmp_path, monkeypatch):
    from tools.rq3_protocol import audit_cache_predictions, validate_interface_audit
    payload = cache_payload()
    bind_test_interface(tmp_path, payload, monkeypatch)
    audit = validate_interface_audit(payload["input_protocol"], "qwen3_vl_8b")
    audit_cache_predictions(payload, audit)
    payload["entries"][0]["class_probabilities"] = [0., 1.] + [0.] * 8
    with pytest.raises(ValueError, match="edited"):
        audit_cache_predictions(payload, audit)


def test_unknown_extra_intern_text_weights_are_rejected_before_model_loading(tmp_path):
    from tools.score_rq3_vlm_candidates import checkpoint_files
    from tools.rq2_dense_protocol import sha256_file
    path = tmp_path / "vision.pt"
    path.write_bytes(b"unit identity only")
    settings = {"checkpoint_files": {path.name: str(path)}, "text_checkpoint": str(path)}
    with pytest.raises(ValueError, match="unknown"):
        checkpoint_files("internvideo2_clip_1b", settings, {"checkpoint_sha256": {path.name: sha256_file(path)}})
