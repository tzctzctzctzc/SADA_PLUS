import json
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

for package_name in ("libs", "libs.models", "libs.models.SADA"):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
ROOT = Path(__file__).parents[1]
for module_name in ("anchor_validity", "vlm_routing"):
    qualified = "libs.models.SADA.{}".format(module_name)
    spec = importlib.util.spec_from_file_location(
        qualified, ROOT / "libs/models/SADA/{}.py".format(module_name)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)

vlm_routing = sys.modules["libs.models.SADA.vlm_routing"]
data_spec = importlib.util.spec_from_file_location(
    "vlm_test_data_utils", ROOT / "libs/datasets/data_utils.py"
)
data_utils = importlib.util.module_from_spec(data_spec)
data_spec.loader.exec_module(data_utils)
build_online_vlm_routing = vlm_routing.build_online_vlm_routing
apply_binary_band_oracle = vlm_routing.apply_binary_band_oracle
load_semantic_cache = vlm_routing.load_semantic_cache
target_labels_from_full_routing = vlm_routing.target_labels_from_full_routing
build_vlm_signature = vlm_routing.build_vlm_signature
validate_vlm_resume = vlm_routing.validate_vlm_resume
build_vlm_transition = vlm_routing.build_vlm_transition
validate_vlm_transition = vlm_routing.validate_vlm_transition
validate_sada_continuation_checkpoint = (
    vlm_routing.validate_sada_continuation_checkpoint
)
validate_semantic_selection_provenance = (
    vlm_routing.validate_semantic_selection_provenance
)
restore_training_trajectory = vlm_routing.restore_training_trajectory


def semantic_payload(entries):
    return {
        "semantic_cache_schema_version": "offline-vlm-semantic-evidence-v1",
        "source": "offline_vlm_candidate_blind_semantics",
        "key_fields": ["video_id", "absolute_feature_index"],
        "prompt_sha256": "f" * 64,
        "prompt_version": "candidate-blind-epic-semantic-gate-v3",
        "model": "test-vlm",
        "wire_api": "responses",
        "selection_teacher_provenance": {
            "selection_teacher_checkpoint_sha256": "a" * 64,
            "selection_teacher_state_sha256": "b" * 64,
            "selection_teacher_weights": "ema",
            "selection_beta": 0.1,
            "selection_alpha": 0.7,
        },
        "entries": entries,
    }


def bidirectional_payload(entries):
    return {
        "semantic_cache_schema_version": (
            "offline-vlm-bidirectional-semantic-evidence-v2"
        ),
        "source": "offline_vlm_candidate_blind_bidirectional_semantics",
        "key_fields": ["prompt_role", "video_id", "absolute_feature_index"],
        "selection_teacher_provenance": semantic_payload([])[
            "selection_teacher_provenance"
        ],
        "prompt_protocols": {
            "low_rescue": {
                "prompt_version": "candidate-blind-epic-semantic-gate-v3",
                "prompt_sha256": "a" * 64,
                "model": "test-vlm",
                "wire_api": "responses",
            },
            "high_veto": {
                "prompt_version": "candidate-blind-epic-semantic-veto-v1",
                "prompt_sha256": "b" * 64,
                "model": "test-vlm",
                "wire_api": "responses",
            },
        },
        "entries": entries,
    }


def inputs(crop_start=7):
    points = [torch.tensor([
        [0.0, 0.0, 0.0, 2.0],
        [2.0, 0.0, 0.0, 2.0],
        [4.0, 0.0, 0.0, 2.0],
    ])]
    masks = [torch.tensor([[True, True, True]])]
    logits = [torch.tensor([[[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]])]
    offsets = [torch.ones(1, 3, 2)]
    fpn = [torch.zeros(1, 4, 3)]
    videos = [{"metadata": {
        "video_id": "video",
        "crop_start": crop_start,
        "feat_stride": 1,
        "feat_num_frames": 0,
        "fps": 10,
        "duration": 100,
    }}]
    return videos, points, masks, logits, offsets, fpn


def route(
    cache,
    mode="semantic_only",
    validity_model=None,
    threshold=None,
    teacher_logits=None,
):
    videos, points, masks, logits, offsets, fpn = inputs()
    if teacher_logits is not None:
        logits = teacher_logits
    return build_online_vlm_routing(
        cache,
        validity_model,
        threshold,
        mode,
        videos,
        points,
        masks,
        logits,
        offsets,
        fpn,
        beta=0.1,
        alpha=0.7,
    )[0]


def test_semantic_cache_rejects_old_dual_gate_schema(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "routing_schema_version": "semantic-structural-exact-anchor-routing-v3",
        "entries": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported VLM semantic cache schema"):
        load_semantic_cache(path)


def test_semantic_cache_rejects_teacher_or_structure_evidence(tmp_path):
    path = tmp_path / "mixed.json"
    entry = {
        "video_id": "video",
        "absolute_feature_index": 11,
        "absolute_center_time": 1.1,
        "semantic_label": "TARGET_ACTION",
        "semantic_pass": True,
        "teacher_top1_class": 3,
    }
    path.write_text(
        json.dumps(semantic_payload([entry])), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="online routing evidence"):
        load_semantic_cache(path)


def test_absolute_index_lookup_never_projects_to_nearest_crop_anchor(tmp_path):
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps(semantic_payload([
        {
            "video_id": "video",
            "absolute_feature_index": 10,
            "absolute_center_time": 1.0,
            "semantic_label": "TARGET_ACTION",
            "semantic_pass": True,
        },
        {
            "video_id": "video",
            "absolute_feature_index": 11,
            "absolute_center_time": 1.1,
            "semantic_label": "TARGET_ACTION",
            "semantic_pass": True,
        },
    ])), encoding="utf-8")
    cache, count, digest, payload = load_semantic_cache(path)
    assert count == 2
    assert len(digest) == 64
    validate_semantic_selection_provenance(
        payload,
        {
            "routing_teacher_checkpoint_sha256": "a" * 64,
            "routing_teacher_state_sha256": "b" * 64,
            "routing_teacher_weights": "ema",
        },
        0.1,
        0.7,
    )
    routes = route(cache)
    # crop_start=7 and local lattice 0,2,4 gives exact absolute indices 7,9,11.
    # Cached index 10 must not be rounded or broadcast onto either 9 or 11.
    assert routes["semantic_known"].tolist() == [[False, False, True]]
    assert routes["action"].tolist() == [[False, False, True]]
    assert routes["ignore"].tolist() == [[True, True, False]]


def test_bidirectional_cache_isolates_low_and_high_at_same_absolute_key(tmp_path):
    path = tmp_path / "bidirectional.json"
    common = {
        "video_id": "video",
        "absolute_feature_index": 11,
        "absolute_center_time": 1.1,
    }
    path.write_text(
        json.dumps(
            bidirectional_payload(
                [
                    {
                        **common,
                        "prompt_role": "low_rescue",
                        "semantic_label": "TARGET_ACTION",
                        "semantic_pass": True,
                    },
                    {
                        **common,
                        "prompt_role": "high_veto",
                        "semantic_label": "TRUE_BACKGROUND",
                        "semantic_pass": False,
                    },
                ]
            )
        ),
        encoding="utf-8",
    )
    cache, count, _, _ = load_semantic_cache(path)
    assert count == 2
    assert cache[("low_rescue", "video", 11)]["semantic_pass"] is True
    assert cache[("high_veto", "video", 11)]["semantic_label"] == "TRUE_BACKGROUND"


def test_semantic_cache_selection_teacher_must_match_routing_teacher(tmp_path):
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps(semantic_payload([])), encoding="utf-8")
    _, _, _, payload = load_semantic_cache(path)
    with pytest.raises(ValueError, match="differs from routing teacher"):
        validate_semantic_selection_provenance(
            payload,
            {
                "routing_teacher_checkpoint_sha256": "c" * 64,
                "routing_teacher_state_sha256": "b" * 64,
                "routing_teacher_weights": "ema",
            },
            0.1,
            0.7,
        )


def test_online_route_uses_current_teacher_top1_not_cached_class():
    cache = {
        ("video", 11): {
            "semantic_pass": True,
            "semantic_label": "TARGET_ACTION",
            "absolute_center_time": 1.1,
        }
    }
    routes = route(cache)
    assert routes["pseudo_class"].tolist() == [[1, 1, 1]]


@pytest.mark.parametrize(
    "label, keep",
    [
        ("TARGET_ACTION", True),
        ("UNCERTAIN", True),
        ("OTHER_ACTIVITY", False),
        ("TRUE_BACKGROUND", False),
    ],
)
def test_high_veto_is_conservative_and_routes_clear_negatives_to_ignore(label, keep):
    cache = {
        ("high_veto", "video", 7): {
            "prompt_role": "high_veto",
            "semantic_pass": label == "TARGET_ACTION",
            "semantic_label": label,
            "absolute_center_time": 0.7,
        }
    }
    logits = [torch.tensor([[[2.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]])]
    result = route(cache, "bidirectional_semantic", teacher_logits=logits)
    assert bool(result["high_confidence"][0, 0]) is True
    assert bool(result["action"][0, 0]) is keep
    assert bool(result["ignore"][0, 0]) is (not keep)
    assert bool(result["background"][0, 0]) is False
    assert int(result["pseudo_class"][0, 0]) == 0


def test_missing_high_veto_evidence_keeps_frozen_teacher_action():
    logits = [torch.tensor([[[2.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]])]
    result = route({}, "bidirectional_semantic", teacher_logits=logits)
    assert bool(result["high_semantic_known"][0, 0]) is False
    assert bool(result["high_action"][0, 0]) is True
    assert bool(result["action"][0, 0]) is True


def test_low_and_high_evidence_never_cross_prompt_roles():
    high_only = {
        ("high_veto", "video", 11): {
            "prompt_role": "high_veto",
            "semantic_pass": True,
            "semantic_label": "TARGET_ACTION",
            "absolute_center_time": 1.1,
        }
    }
    mid_result = route(high_only, "bidirectional_semantic")
    assert bool(mid_result["semantic_known"][0, 2]) is False
    assert bool(mid_result["action"][0, 2]) is False

    low_only = {
        ("low_rescue", "video", 7): {
            "prompt_role": "low_rescue",
            "semantic_pass": False,
            "semantic_label": "TRUE_BACKGROUND",
            "absolute_center_time": 0.7,
        }
    }
    logits = [torch.tensor([[[2.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]])]
    high_result = route(
        low_only, "bidirectional_semantic", teacher_logits=logits
    )
    assert bool(high_result["high_semantic_known"][0, 0]) is False
    assert bool(high_result["action"][0, 0]) is True


def test_joint_gate_uses_online_validity_and_missing_cache_is_ignore():
    cache = {
        ("video", 9): {
            "semantic_pass": True,
            "semantic_label": "TARGET_ACTION",
            "absolute_center_time": 0.9,
        }
    }
    validity_model = {
        "model_type": "logistic",
        "feature_group": "structure_pure",
        "mean": torch.zeros(9),
        "std": torch.ones(9),
        "weight": torch.zeros(9),
        "bias": -10.0,
    }
    routes = route(cache, "semantic_and_validity", validity_model, 0.8)
    assert routes["semantic_known"].tolist() == [[False, True, False]]
    assert routes["action"].tolist() == [[False, False, False]]
    assert routes["ignore"].tolist() == [[True, True, True]]


def test_validity_only_routes_current_mid_confidence_without_semantic_cache():
    validity_model = {
        "model_type": "logistic",
        "feature_group": "structure_pure",
        "mean": torch.zeros(9),
        "std": torch.ones(9),
        "weight": torch.zeros(9),
        "bias": 10.0,
    }
    routes = route({}, "validity_only", validity_model, 0.8)
    assert routes["matched"].tolist() == [[True, True, True]]
    assert routes["action"].tolist() == [[True, True, True]]


def test_mid_confidence_ignore_control_never_rescues():
    routes = route({}, "mid_confidence_ignore")
    assert routes["action"].tolist() == [[False, False, False]]
    assert routes["ignore"].tolist() == [[True, True, True]]


def test_full_route_has_high_action_mid_ignore_and_low_background():
    videos, points, masks, _, offsets, fpn = inputs()
    logits = [torch.tensor([[
        [2.0, 0.0],
        [-1.0, 0.0],
        [-3.0, -4.0],
    ]])]
    routes = build_online_vlm_routing(
        {}, None, None, "mid_confidence_ignore",
        videos, points, masks, logits, offsets, fpn,
        beta=0.1, alpha=0.7,
    )[0]
    assert routes["action"].tolist() == [[True, False, False]]
    assert routes["ignore"].tolist() == [[False, True, False]]
    assert routes["background"].tolist() == [[False, False, True]]


def test_student_logits_cannot_change_frozen_teacher_labels():
    routes = route({
        ("video", 11): {
            "semantic_pass": True,
            "semantic_label": "TARGET_ACTION",
            "absolute_center_time": 1.1,
        }
    })
    valid_mask = torch.tensor([[True, True, True]])
    first = target_labels_from_full_routing(
        routes, valid_mask, num_classes=2, dtype=torch.float32
    )
    student_high_teacher_mid = torch.tensor([[[9.0, -9.0]] * 3])
    student_mid_teacher_high = torch.tensor([[[0.0, -1.0]] * 3])
    assert not torch.equal(student_high_teacher_mid, student_mid_teacher_high)
    second = target_labels_from_full_routing(
        routes, valid_mask, num_classes=2, dtype=torch.float32
    )
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_binary_band_oracle_isolates_high_cleanup_and_mid_rescue():
    # Use a five-anchor input directly so every confidence/GT combination is covered.
    valid = torch.ones(1, 5, dtype=torch.bool)
    logits = [torch.tensor([[
        [3.0, -3.0],   # high foreground
        [3.0, -3.0],   # high background
        [0.0, -3.0],   # mid foreground
        [0.0, -3.0],   # mid background
        [-3.0, -3.0],  # low background
    ]])]
    routes = build_online_vlm_routing(
        {}, None, None, 'mid_confidence_ignore',
        [{"metadata": {
            "video_id": "video", "crop_start": 0, "feat_stride": 1,
            "feat_num_frames": 0, "fps": 10, "duration": 100,
        }}],
        [torch.tensor([[float(i), 0.0, 0.0, 1.0] for i in range(5)])],
        [valid], logits, [torch.ones(1, 5, 2)],
        [torch.zeros(1, 4, 5)], beta=0.1, alpha=0.7,
    )[0]
    gt = torch.tensor([[
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]])

    control = apply_binary_band_oracle(routes, gt, 'control')
    high = apply_binary_band_oracle(routes, gt, 'high_clean')
    mid = apply_binary_band_oracle(routes, gt, 'mid_rescue')
    both = apply_binary_band_oracle(routes, gt, 'both')

    assert control['action'].tolist() == [[True, True, False, False, False]]
    assert control['ignore'].tolist() == [[False, False, True, True, False]]
    assert high['action'].tolist() == [[True, False, False, False, False]]
    assert high['background'].tolist() == [[False, True, False, False, True]]
    assert mid['action'].tolist() == [[True, True, True, False, False]]
    assert mid['background'].tolist() == [[False, False, False, True, True]]
    assert both['action'].tolist() == [[True, False, True, False, False]]
    assert both['background'].tolist() == [[False, True, False, True, True]]
    assert not both['ignore'].any()


def test_semantic_timestamp_must_match_current_feature_configuration():
    cache = {
        ("video", 11): {
            "semantic_pass": True,
            "semantic_label": "TARGET_ACTION",
            "absolute_center_time": 99.0,
        }
    }
    with pytest.raises(ValueError, match="timestamp differs"):
        route(cache)


def test_semantic_timestamp_accepts_legacy_float32_roundoff():
    videos, points, masks, logits, offsets, fpn = inputs(crop_start=747)
    videos[0]["metadata"].update({
        "feat_stride": 16,
        "feat_num_frames": 32,
        "fps": 30,
        "duration": 1000,
    })
    expected = (747 * 16 + 0.5 * 32) / 30
    legacy_float32 = torch.tensor(expected, dtype=torch.float32).item()
    routes = build_online_vlm_routing(
        {
            ("video", 747): {
                "semantic_pass": True,
                "semantic_label": "TARGET_ACTION",
                "absolute_center_time": legacy_float32,
            }
        },
        None,
        None,
        "semantic_only",
        videos,
        points,
        masks,
        logits,
        offsets,
        fpn,
        beta=0.1,
        alpha=0.7,
    )[0]
    assert routes["semantic_known"].tolist() == [[True, False, False]]
    assert routes["action"].tolist() == [[True, False, False]]


def test_real_crop_metadata_preserves_feature_receptive_field_for_router():
    class FixedRng:
        def randint(self, lower, upper):
            assert (lower, upper) == (0, 5)
            return 3

    video = data_utils.truncate_feats(
        {
            "feats": torch.zeros(4, 10),
            "segments": torch.tensor([[0.0, 10.0]]),
            "labels": torch.tensor([1]),
            "metadata": {
                "video_id": "video",
                "feat_stride": 16,
                "feat_num_frames": 32,
                "fps": 30,
                "duration": 10.0,
            },
        },
        max_seq_len=5,
        trunc_thresh=0.3,
        rng=FixedRng(),
    )
    assert video["metadata"]["feat_num_frames"] == 32
    assert video["metadata"]["crop_feat_len"] == 5
    absolute_index = video["metadata"]["crop_start"]
    center_time = (absolute_index * 16 + 0.5 * 32) / 30
    routes = build_online_vlm_routing(
        {
            ("video", absolute_index): {
                "semantic_pass": True,
                "semantic_label": "TARGET_ACTION",
                "absolute_center_time": center_time,
            }
        },
        None,
        None,
        "semantic_only",
        [video],
        [torch.tensor([[0.0, 0.0, 0.0, 1.0]])],
        [torch.tensor([[True]])],
        [torch.tensor([[[-1.0, 0.0]]])],
        [torch.ones(1, 1, 2)],
        [torch.zeros(1, 4, 1)],
        beta=0.1,
        alpha=0.7,
    )[0]
    assert routes["semantic_known"].tolist() == [[True]]
    assert routes["action"].tolist() == [[True]]


def test_resume_signature_rejects_changed_mode_or_artifact():
    prompt_protocols = {
        "low_rescue": {"prompt_sha256": "d" * 64},
        "high_veto": {"prompt_sha256": "e" * 64},
    }
    signature = build_vlm_signature(
        "a" * 64,
        "b" * 64,
        "semantic_and_validity",
        0.1,
        0.7,
        1,
        "c" * 64,
        "ema",
        prompt_protocols,
    )
    assert signature["schema_version"] == "frozen-routing-teacher-v4"
    assert signature["evaluation_weights"] == "student_ema"
    assert signature["semantic_prompt_protocols"] == prompt_protocols
    validate_vlm_resume({"vlm_routing": signature}, signature)
    changed = dict(signature)
    changed["routing_mode"] = "validity_only"
    with pytest.raises(ValueError, match="signature differs"):
        validate_vlm_resume({"vlm_routing": signature}, changed)
    changed_protocol = dict(signature)
    changed_protocol["semantic_prompt_protocols"] = {
        **prompt_protocols,
        "high_veto": {"prompt_sha256": "f" * 64},
    }
    with pytest.raises(ValueError, match="signature differs"):
        validate_vlm_resume({"vlm_routing": signature}, changed_protocol)
    with pytest.raises(ValueError, match="no frozen VLM routing signature"):
        validate_vlm_resume({}, signature)


def continuation_config():
    return {
        "dataset_name": "epic",
        "src_train_split": ["source_train"],
        "src_val_split": ["source_val"],
        "tgt_train_split": ["target_train"],
        "tgt_val_split": ["target_val"],
        "dataset": {"max_seq_len": 2304},
        "loader": {"batch_size": 2},
        "model": {"pl_threshold": 0.7},
        "opt": {"type": "AdamW", "epochs": 10},
        "train_cfg": {"warmup_adv": 5, "ema_active": True},
    }


def ordinary_sada_checkpoint():
    return {
        "epoch": 5,
        "state_dict": {},
        "ema_state_dict": {},
        "optimizer_gen_source": {},
        "optimizer_gen_source_alignment": {},
        "optimizer_disc": {},
        "config": continuation_config(),
    }


def test_sada_to_vlm_transition_is_distinct_from_strict_vlm_resume():
    checkpoint = ordinary_sada_checkpoint()
    current = continuation_config()
    current["dataset"]["crop_alignment"] = 1
    current["dataset"]["stage1_context"] = False
    assert validate_sada_continuation_checkpoint(checkpoint, current) == 5
    transition = build_vlm_transition(
        "a" * 64,
        5,
        "b" * 64,
        "e" * 64,
        "c" * 64,
        "d" * 64,
        "ema",
        500,
    )
    assert validate_vlm_transition({"vlm_transition": transition}) == transition
    assert transition["schema_version"] == "sada-to-vlm-continuation-v2"
    assert transition["student_ema_state_sha256"] == "e" * 64
    with pytest.raises(ValueError, match="teacher differs"):
        validate_vlm_transition(
            {"vlm_transition": transition},
            expected_teacher_checkpoint_sha256="e" * 64,
        )
    with pytest.raises(ValueError, match="ordinary SADA checkpoint"):
        validate_sada_continuation_checkpoint(
            {**checkpoint, "vlm_routing": {}}, current
        )


def test_sada_continuation_rejects_nonrouting_training_protocol_change():
    checkpoint = ordinary_sada_checkpoint()
    changed = continuation_config()
    changed["opt"] = {"type": "AdamW", "epochs": 20}
    with pytest.raises(ValueError, match="outside routing"):
        validate_sada_continuation_checkpoint(checkpoint, changed)


def test_true_continuation_restores_all_optimizers_and_scheduler_phase():
    parameters = [torch.nn.Parameter(torch.tensor([1.0])) for _ in range(3)]
    old_optimizers = [
        torch.optim.SGD([parameter], lr=lr, momentum=0.9)
        for parameter, lr in zip(parameters, (0.1, 0.2, 0.3))
    ]
    checkpoint = ordinary_sada_checkpoint()
    checkpoint.update({
        "optimizer_gen_source": old_optimizers[0].state_dict(),
        "optimizer_gen_source_alignment": old_optimizers[1].state_dict(),
        "optimizer_disc": old_optimizers[2].state_dict(),
    })
    new_optimizers = [
        torch.optim.SGD([torch.nn.Parameter(torch.tensor([0.0]))], lr=1.0)
        for _ in range(3)
    ]

    class Scheduler:
        pass

    schedulers = [Scheduler(), Scheduler(), Scheduler()]
    completed_steps = restore_training_trajectory(
        checkpoint,
        20,
        *new_optimizers,
        *schedulers,
    )
    assert completed_steps == 100
    assert [optimizer.param_groups[0]["lr"] for optimizer in new_optimizers] == [
        0.1,
        0.2,
        0.3,
    ]
    assert [scheduler.last_epoch for scheduler in schedulers] == [100, 100, 100]
    assert [scheduler._step_count for scheduler in schedulers] == [101, 101, 101]


def test_vlm_transition_hash_rejects_tampering():
    transition = build_vlm_transition(
        "a" * 64, 5, "b" * 64, "e" * 64, "c" * 64, "d" * 64, "ema", 500
    )
    transition["parent_epoch"] = 4
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_vlm_transition({"vlm_transition": transition})
