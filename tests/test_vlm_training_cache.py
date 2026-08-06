import copy

import pytest

from tools.build_vlm_training_cache import build_entries, build_payload
from tools.export_vlm_training_candidates import limit_queries


def manifest_row(candidate_id="exact", top1=3, center=12.5):
    return {
        "candidate_id": candidate_id,
        "video_id": "video",
        "fpn_level": 2,
        "anchor_index": 7,
        "global_anchor_index": 123,
        "absolute_center_time": center,
        "center_sec": center,
        "teacher_top1_class": top1,
        "member_anchors": [
            {
                "fpn_level": 2,
                "anchor_index": 7,
                "global_anchor_index": 123,
                "absolute_center_time": center,
                "teacher_top1_class": top1,
            }
        ],
    }


def prediction_row(manifest, prediction="TARGET_FOREGROUND"):
    is_action = prediction == "TARGET_FOREGROUND"
    fields = {
        "TARGET_FOREGROUND": ("YES", "YES"),
        "OTHER_ACTIVITY": ("YES", "NO"),
        "TRUE_BACKGROUND": ("NO", "NO"),
        "UNCERTAIN": ("UNCERTAIN", "UNCERTAIN"),
    }
    center_interaction, target_signature = fields[prediction]
    return {
        "candidate_id": manifest["candidate_id"],
        "routing_schema_version": "candidate-blind-exact-anchor-routing-v1",
        "prompt_sha256": "a" * 64,
        "model": "qwen-test",
        "wire_api": "chat",
        "status": "ok",
        "prediction": prediction,
        "route": "action" if is_action else "ignore",
        "action_rescue": is_action,
        "pseudo_label": manifest["teacher_top1_class"] if is_action else None,
        "center_interaction": center_interaction,
        "target_signature": target_signature,
        "model_reported_label": prediction,
        "parse_error": False,
        "protocol_error": False,
        "consistency_error": False,
        "local_gate_reason": None,
        "video_id": manifest["video_id"],
        "fpn_level": manifest["fpn_level"],
        "anchor_index": manifest["anchor_index"],
        "global_anchor_index": manifest["global_anchor_index"],
        "absolute_center_time": manifest["absolute_center_time"],
        "center_sec": manifest["absolute_center_time"],
        "teacher_top1_class": manifest["teacher_top1_class"],
    }


def test_query_budget_keeps_separated_highest_scores_per_video():
    rows = [
        {
            "candidate_id": "q{}".format(index),
            "video_id": "video",
            "center_sec": center,
            "sada_score_max": score,
        }
        for index, (center, score) in enumerate(
            [(1.0, 0.49), (1.4, 0.48), (3.0, 0.40), (5.0, 0.30)]
        )
    ]
    selected = limit_queries(rows, max_per_video=2, min_gap_sec=1.0)
    assert {row["center_sec"] for row in selected} == {1.0, 3.0}


def test_new_protocol_action_and_ignore_are_serialized_one_to_one():
    action_manifest = manifest_row("action")
    ignore_manifest = manifest_row("ignore", top1=2, center=8.0)
    ignore_manifest["anchor_index"] = 4
    ignore_manifest["global_anchor_index"] = 88
    ignore_manifest["member_anchors"][0].update(
        {"anchor_index": 4, "global_anchor_index": 88}
    )
    predictions = [
        prediction_row(action_manifest),
        prediction_row(ignore_manifest, "TRUE_BACKGROUND"),
    ]

    entries = build_entries([action_manifest, ignore_manifest], predictions)

    assert entries[0]["decision"] == "action"
    assert entries[0]["class_id"] == action_manifest["teacher_top1_class"]
    assert entries[1]["decision"] == "ignore"
    assert "class_id" not in entries[1]
    payload = build_payload([action_manifest, ignore_manifest], predictions)
    assert payload["counts"] == {"action": 1, "ignore": 1, "background": 0}
    assert payload["routing_schema_version"] == (
        "candidate-blind-exact-anchor-routing-v1"
    )


@pytest.mark.parametrize(
    "prediction", ["OTHER_ACTIVITY", "TRUE_BACKGROUND", "UNCERTAIN"]
)
def test_all_valid_non_confirm_results_route_to_ignore(prediction):
    manifest = manifest_row()
    result = prediction_row(manifest, prediction)
    assert build_entries([manifest], [result])[0]["decision"] == "ignore"


def test_local_uncertain_is_a_valid_ignore_without_model_label():
    manifest = manifest_row()
    result = prediction_row(manifest, "UNCERTAIN")
    result["model_reported_label"] = None
    result["local_gate_reason"] = "insufficient_pre_or_post_context"
    assert build_entries([manifest], [result])[0]["decision"] == "ignore"


def test_legacy_confirmation_label_is_rejected():
    manifest = manifest_row()
    result = prediction_row(manifest)
    result["prediction"] = "CONFIRM_KEY_ACTION_EVIDENCE"
    with pytest.raises(ValueError, match="disagrees with gate fields"):
        build_entries([manifest], [result])


@pytest.mark.parametrize(
    "field,value",
    [
        ("pseudo_label", 4),
        ("route", "ignore"),
        ("action_rescue", False),
        ("center_interaction", "NO"),
        ("parse_error", True),
        ("protocol_error", True),
        ("consistency_error", True),
    ],
)
def test_inconsistent_action_protocol_is_rejected(field, value):
    manifest = manifest_row()
    result = prediction_row(manifest)
    result[field] = value
    with pytest.raises(ValueError):
        build_entries([manifest], [result])


def test_missing_protocol_field_is_rejected_instead_of_ignored():
    manifest = manifest_row()
    result = prediction_row(manifest, "UNCERTAIN")
    result.pop("protocol_error")
    with pytest.raises(ValueError, match="missing a routing protocol field"):
        build_entries([manifest], [result])


def test_absolute_time_alias_and_member_must_match():
    manifest = manifest_row()
    bad_alias = copy.deepcopy(manifest)
    bad_alias["center_sec"] += 0.1
    with pytest.raises(ValueError, match="center_sec differs"):
        build_entries([bad_alias], [prediction_row(manifest)])

    bad_member = copy.deepcopy(manifest)
    bad_member["member_anchors"][0]["absolute_center_time"] += 0.1
    with pytest.raises(ValueError, match="member does not match"):
        build_entries([bad_member], [prediction_row(manifest)])


def test_mixed_prompt_or_model_is_rejected():
    first = manifest_row("first")
    second = manifest_row("second", center=14.0)
    second["global_anchor_index"] = 124
    second["anchor_index"] = 8
    second["member_anchors"][0].update(
        {"global_anchor_index": 124, "anchor_index": 8}
    )
    first_prediction = prediction_row(first)
    second_prediction = prediction_row(second, "UNCERTAIN")
    second_prediction["model"] = "another-model"
    with pytest.raises(ValueError, match="mixes Prompt, model, or wire API"):
        build_entries([first, second], [first_prediction, second_prediction])


def test_prediction_coverage_must_equal_manifest():
    first = manifest_row("first")
    second = manifest_row("second", center=14.0)
    second["global_anchor_index"] = 124
    second["anchor_index"] = 8
    second["member_anchors"][0].update(
        {"global_anchor_index": 124, "anchor_index": 8}
    )
    with pytest.raises(ValueError, match="coverage differs"):
        build_entries([first, second], [prediction_row(first)])
