from tools.run_qwen3_vl_open_set_gate import (
    OUTCOMES,
    build_prompt,
    normalize_logits,
    outcome_mapping,
)


def test_open_set_mapping_is_deterministic():
    mapping = outcome_mapping("p-1")
    assert mapping == outcome_mapping("p-1")
    assert set(mapping) == set(OUTCOMES)


def test_open_set_prompt_defines_closed_world_rejection():
    prompt = build_prompt("p-2", 8)
    assert "TAKE, PUT, WASH" in prompt
    assert "OTHER_ACTION" in prompt
    assert "BACKGROUND_OR_UNCERTAIN" in prompt


def test_open_set_logits_respect_mapping():
    mapping = outcome_mapping("p-3")
    probabilities, prediction = normalize_logits([0, 7, 0], mapping)
    assert prediction == mapping[1]
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
