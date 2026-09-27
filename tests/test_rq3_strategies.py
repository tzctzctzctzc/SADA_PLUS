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
    assert routes[0][0].tolist() == [[True, False]]
    assert routes[0][1][0, 0] == 0


def cache_payload():
    return {"schema": SCHEMA, "teacher_model": "qwen3_vl_8b", "shift": "S1", "seed": 6,
            "labels": list(TARGET_CLASS_NAMES), "target_gt_used": False,
            "target_video_ids": ["P01_01"], "teacher_selection": "source-validation-only",
            "candidate_sha256": "a" * 64, "frames_sha256": "b" * 64, "teacher_checkpoint_sha256": "c" * 64,
            "input_protocol": {"frame_protocol": FRAME_PROTOCOL, "input_frames": 8, "class_blind": True,
                               "readout": "closed-class-log-likelihood", "checkpoint_sha256": {"model": "d" * 64},
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
def test_runtime_binding_checks_actual_seed_members_and_cache_hash(tmp_path, strategy):
    from tools.rq3_protocol import configure, validate_config
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({"database": {"P01_01": {"subset": "target_train"}}}))
    cfg = {"init_rand_seed": 6, "tgt_train_split": ["target_train"],
           "dataset": {"input_dim": 2304, "json_file": str(annotation)},
           "opt": {"epochs": 50}, "loader": {"batch_size": 2}, "rq2_formal": {"shift": "S1"}}
    path = write_payload(tmp_path, cache_payload())
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
