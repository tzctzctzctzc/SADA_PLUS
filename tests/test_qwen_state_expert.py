import math

from tools.run_qwen3_vl_state_expert import (
    EXPERT_LABELS,
    build_prompt,
    cross_validated_accuracy,
    fuse_logits,
    normalize_logits,
    option_mapping,
    should_route,
    source_accuracy,
)


def baseline_row(candidate_id="p0", class_id=0, teacher=0, gt=0):
    probabilities = [0.01] * 10
    probabilities[class_id] = 0.91
    logits = [math.log(value) for value in probabilities]
    return {
        "candidate_id": candidate_id,
        "video_id": "v0",
        "teacher_top1_class": teacher,
        "gt_assigned_class_ids": [gt],
        "class_probabilities": probabilities,
        "class_logits": logits,
    }


def test_prompt_mapping_is_deterministic_and_complete():
    assert option_mapping("proposal-1") == option_mapping("proposal-1")
    assert set(option_mapping("proposal-1")) == set(EXPERT_LABELS)
    prompt = build_prompt("proposal-1", 8)
    assert "PUT" in prompt and "INSERT" in prompt and "POUR" in prompt
    assert "0 to 3" in prompt


def test_normalize_logits_respects_shuffled_mapping():
    mapping = ("OTHER", "PUT", "POUR", "INSERT")
    probabilities, prediction = normalize_logits([5.0, 1.0, 0.0, -1.0], mapping)
    assert prediction == "OTHER"
    assert abs(sum(probabilities.values()) - 1.0) < 1e-7


def test_route_uses_teacher_or_qwen_topk():
    assert should_route(baseline_row(class_id=1), topk=1)
    assert should_route(baseline_row(class_id=0, teacher=5), topk=1)
    assert not should_route(baseline_row(class_id=0, teacher=0), topk=1)


def test_expert_likelihood_ratio_can_correct_insert():
    row = baseline_row(class_id=1, teacher=1, gt=5)
    row["class_logits"][5] = row["class_logits"][1] - 0.2
    expert = {"PUT": 0.05, "INSERT": 0.80, "POUR": 0.05, "OTHER": 0.10}
    fused = fuse_logits(row, expert, alpha=1.0)
    assert max(range(10), key=fused.__getitem__) == 5


def test_source_accuracy_reports_routed_and_overall_gain():
    rows = [
        baseline_row("p0", class_id=1, teacher=1, gt=5),
        baseline_row("p1", class_id=0, teacher=0, gt=0),
    ]
    expert = {
        "p0": {
            "expert_probabilities": {
                "PUT": 0.01, "INSERT": 0.95, "POUR": 0.01, "OTHER": 0.03
            }
        }
    }
    report = source_accuracy(rows, expert, alpha=1.0)
    assert report["routed_known_count"] == 1
    assert report["routed_accuracy"] == 1.0
    assert report["baseline_routed_accuracy"] == 0.0
    assert report["overall_accuracy"] == 1.0


def test_cross_validation_selects_weights_without_using_held_out_video():
    rows = [
        baseline_row("p0", class_id=1, teacher=1, gt=5),
        {**baseline_row("p1", class_id=1, teacher=1, gt=5), "video_id": "v1"},
    ]
    expert = {
        key: {
            "expert_probabilities": {
                "PUT": 0.01, "INSERT": 0.95, "POUR": 0.01, "OTHER": 0.03
            }
        }
        for key in ("p0", "p1")
    }
    report = cross_validated_accuracy(rows, expert, [0.0, 1.0])
    assert report["routed_accuracy"] == 1.0
    assert report["baseline_routed_accuracy"] == 0.0
    assert set(report["fold_selected_alphas"]) == {"v0", "v1"}
