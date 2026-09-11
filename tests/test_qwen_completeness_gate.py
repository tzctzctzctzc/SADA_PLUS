from tools.run_qwen3_vl_completeness_gate import (
    OUTCOMES,
    build_prompt,
    normalize_logits,
    outcome_mapping,
)


def test_completeness_mapping_is_deterministic_and_complete():
    first = outcome_mapping("candidate-1")
    assert first == outcome_mapping("candidate-1")
    assert set(first) == set(OUTCOMES)


def test_completeness_prompt_marks_temporal_regions():
    prompt = build_prompt("candidate-2")
    assert "Frames 1-4 are BEFORE" in prompt
    assert "frames 5-12 are INSIDE" in prompt
    assert "frames 13-16 are AFTER" in prompt
    assert "exactly one digit from 0 to 4" in prompt


def test_completeness_logits_follow_rotated_mapping():
    mapping = outcome_mapping("candidate-3")
    probabilities, prediction = normalize_logits([0, 0, 8, 0, 0], mapping)
    assert prediction == mapping[2]
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
