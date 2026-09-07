from types import SimpleNamespace

from tools.run_qwen_api_classification import (
    PROMPT_VERSION,
    WIRE_API,
    build_api_content,
    parse_api_response,
    parse_digit_output,
    prediction_is_current,
)
from tools.run_qwen3_vl_classification import build_prompt, prompt_sha256


def test_parse_digit_response_accepts_strict_and_explanatory_content():
    assert parse_digit_output("5") == 5
    assert parse_digit_output("The answer is 5.") == 5


def test_api_content_keeps_all_frames_as_separate_images(tmp_path):
    paths = []
    for index in range(1, 3):
        path = tmp_path / "frame_{:02d}.jpg".format(index)
        path.write_bytes(b"jpeg")
        paths.append(path)
    content = build_api_content(
        paths, "answer with one digit"
    )
    assert [item["type"] for item in content] == [
        "text", "image_url", "text", "image_url", "text"
    ]
    assert content[0]["text"] == "FRAME 1"
    assert content[2]["text"] == "FRAME 2"
    assert content[-1]["text"] == "answer with one digit"


def test_api_response_maps_digit_using_candidate_specific_mapping():
    candidate_id = "proposal-1"
    result = parse_api_response(
        {
            "model": "qwen3.8-max",
            "choices": [{"message": {"content": "5"}}],
            "usage": {"total_tokens": 10},
        },
        candidate_id,
    )
    assert result["prediction"] == result["digit_to_label"][5]
    assert result["confidence_available"] is False
    assert result["usage"]["total_tokens"] == 10


def test_resume_contract_requires_prompt_and_wire_match():
    candidate_id = "proposal-2"
    args = SimpleNamespace(model="qwen3.8-max")
    row = {
        "candidate_id": candidate_id,
        "status": "ok",
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(candidate_id),
        "wire_api": WIRE_API,
    }
    assert prediction_is_current(row, args)
    assert not prediction_is_current({**row, "wire_api": "old"}, args)
    assert not prediction_is_current({**row, "prompt_sha256": "old"}, args)
