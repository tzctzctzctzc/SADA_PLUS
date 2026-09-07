import base64
import io

import pytest
from PIL import Image

from tools.run_qwen3_vl_classification import digit_to_label, LABEL_TO_ID
from tools.run_responses_api_classification import (
    image_data_url,
    parse_api_response,
    parse_json_output,
    response_text,
)


def test_image_payload_respects_pixel_budget(tmp_path):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (1920, 1080), "white").save(path)
    data_url = image_data_url(path, 448 * 448)
    payload = base64.b64decode(data_url.split(",", 1)[1])
    with Image.open(io.BytesIO(payload)) as resized:
        assert resized.width * resized.height <= 448 * 448 + 1000


def test_response_text_reads_nested_responses_output():
    body = {
        "output": [
            {"content": [{"type": "output_text", "text": '{"digit": 2}'}]}
        ]
    }
    assert response_text(body) == '{"digit": 2}'


def test_parse_json_output_accepts_json_fence():
    assert parse_json_output('```json\n{"digit": 1}\n```') == {"digit": 1}


def test_probabilities_are_mapped_back_to_dataset_order():
    candidate_id = "proposal-1"
    probabilities = [0.01] * 10
    probabilities[4] = 0.91
    body = {
        "model": "gpt-5.6-luna",
        "output_text": '{"digit":4,"probabilities":' + str(probabilities) + "}",
    }
    result = parse_api_response(body, candidate_id)
    expected_label = digit_to_label(candidate_id)[4]
    assert result["prediction"] == expected_label
    assert result["class_id"] == LABEL_TO_ID[expected_label]
    assert sum(result["class_probabilities"]) == pytest.approx(1.0)
    assert result["confidence_semantics"].startswith("model_self_reported")


def test_declared_digit_must_match_probability_argmax():
    body = {
        "output_text": '{"digit":0,"probabilities":[0,1,0,0,0,0,0,0,0,0]}'
    }
    with pytest.raises(ValueError, match="argmax"):
        parse_api_response(body, "proposal-2")
