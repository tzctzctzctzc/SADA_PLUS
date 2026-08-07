import copy

import pytest

from tools.build_vlm_training_cache import build_entries, build_payload
from tools.export_vlm_training_candidates import limit_queries


def structure(score=0.9, threshold=0.8):
    return {
        "structure_probe_schema_version": "source-supervised-anchor-validity-v1",
        "structure_probe_sha256": "b" * 64,
        "structure_score": score,
        "structure_threshold": threshold,
        "structural_pass": score >= threshold,
        "structure_feature_values": [0.1] * 13,
    }


def manifest_row(candidate_id="exact", top1=3, center=12.5, score=0.9):
    gate = structure(score)
    return {
        "candidate_id": candidate_id,
        "video_id": "video",
        "fpn_level": 2,
        "anchor_index": 7,
        "global_anchor_index": 123,
        "absolute_center_time": center,
        "center_sec": center,
        "teacher_top1_class": top1,
        **gate,
        "member_anchors": [{
            "fpn_level": 2,
            "anchor_index": 7,
            "global_anchor_index": 123,
            "absolute_center_time": center,
            "teacher_top1_class": top1,
            **gate,
        }],
    }


def prediction_row(manifest, prediction="TARGET_ACTION"):
    semantic = prediction == "TARGET_ACTION"
    structural = manifest["structural_pass"]
    routes = {
        "semantic_only": "action" if semantic else "ignore",
        "structure_only": "action" if structural else "ignore",
        "semantic_and_structure": "action" if semantic and structural else "ignore",
    }
    joint = routes["semantic_and_structure"] == "action"
    return {
        "candidate_id": manifest["candidate_id"],
        "routing_schema_version": "semantic-structural-exact-anchor-routing-v3",
        "prompt_sha256": "a" * 64,
        "model": "qwen-test",
        "wire_api": "chat",
        "status": "ok",
        "prediction": prediction,
        "semantic_pass": semantic,
        "structural_pass": structural,
        "ablation_routes": routes,
        "route": routes["semantic_and_structure"],
        "action_rescue": joint,
        "pseudo_label": manifest["teacher_top1_class"] if joint else None,
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
        **{key: manifest[key] for key in structure()},
    }


def test_query_budget_keeps_separated_highest_scores_per_video():
    rows = [
        {"candidate_id": "q{}".format(index), "video_id": "video", "center_sec": center, "sada_score_max": score}
        for index, (center, score) in enumerate([(1.0, 0.49), (1.4, 0.48), (3.0, 0.40), (5.0, 0.30)])
    ]
    selected = limit_queries(rows, max_per_video=2, min_gap_sec=1.0)
    assert {row["center_sec"] for row in selected} == {1.0, 3.0}


def test_three_ablation_routes_are_derived_from_the_same_two_gates():
    both = manifest_row("both", score=0.9)
    semantic_only = manifest_row("semantic", center=14.0, score=0.2)
    semantic_only["anchor_index"] = 8
    semantic_only["global_anchor_index"] = 124
    semantic_only["member_anchors"][0].update({"anchor_index": 8, "global_anchor_index": 124})
    predictions = [prediction_row(both), prediction_row(semantic_only)]

    assert [row["decision"] for row in build_entries([both, semantic_only], predictions, "semantic_only")] == ["action", "action"]
    assert [row["decision"] for row in build_entries([both, semantic_only], predictions, "structure_only")] == ["action", "ignore"]
    joint = build_entries([both, semantic_only], predictions)
    assert [row["decision"] for row in joint] == ["action", "ignore"]
    assert joint[0]["class_id"] == both["teacher_top1_class"]


def test_payload_records_probe_and_ablation_provenance():
    manifest = manifest_row()
    payload = build_payload([manifest], [prediction_row(manifest)])
    assert payload["routing_schema_version"] == "semantic-structural-exact-anchor-routing-v3"
    assert payload["routing_mode"] == "semantic_and_structure"
    assert payload["structure_probe_sha256"] == "b" * 64
    assert payload["counts"] == {"action": 1, "ignore": 0, "background": 0}


@pytest.mark.parametrize("prediction", ["OTHER_ACTIVITY", "TRUE_BACKGROUND", "UNCERTAIN"])
def test_valid_non_target_semantics_are_ignore(prediction):
    manifest = manifest_row()
    assert build_entries([manifest], [prediction_row(manifest, prediction)])[0]["decision"] == "ignore"


def test_structure_failure_does_not_erase_semantic_only_ablation():
    manifest = manifest_row(score=0.2)
    result = prediction_row(manifest)
    assert build_entries([manifest], [result], "semantic_only")[0]["decision"] == "action"
    assert build_entries([manifest], [result], "structure_only")[0]["decision"] == "ignore"
    assert build_entries([manifest], [result])[0]["decision"] == "ignore"


def test_local_uncertain_is_valid_and_structure_only_remains_available():
    manifest = manifest_row()
    result = prediction_row(manifest, "UNCERTAIN")
    result["model_reported_label"] = None
    result["local_gate_reason"] = "insufficient_pre_or_post_context"
    assert build_entries([manifest], [result])[0]["decision"] == "ignore"
    assert build_entries([manifest], [result], "structure_only")[0]["decision"] == "action"


def test_legacy_protocol_and_silent_inconsistency_are_rejected():
    manifest = manifest_row()
    result = prediction_row(manifest)
    result["routing_schema_version"] = "candidate-blind-exact-anchor-routing-v2"
    with pytest.raises(ValueError, match="unsupported routing_schema_version"):
        build_entries([manifest], [result])

    result = prediction_row(manifest)
    result["semantic_pass"] = False
    with pytest.raises(ValueError, match="semantic_pass"):
        build_entries([manifest], [result])


def test_source_probe_score_and_member_are_strictly_validated():
    manifest = manifest_row()
    bad_score = copy.deepcopy(manifest)
    bad_score["structural_pass"] = False
    with pytest.raises(ValueError, match="structural_pass disagrees"):
        build_entries([bad_score], [prediction_row(manifest)])

    bad_member = copy.deepcopy(manifest)
    bad_member["member_anchors"][0]["structure_score"] = 0.1
    bad_member["member_anchors"][0]["structural_pass"] = False
    with pytest.raises(ValueError, match="member does not match"):
        build_entries([bad_member], [prediction_row(manifest)])


def test_prediction_coverage_must_equal_manifest():
    first = manifest_row("first")
    second = manifest_row("second", center=14.0)
    second["anchor_index"] = 8
    second["global_anchor_index"] = 124
    second["member_anchors"][0].update({"anchor_index": 8, "global_anchor_index": 124})
    with pytest.raises(ValueError, match="coverage differs"):
        build_entries([first, second], [prediction_row(first)])
