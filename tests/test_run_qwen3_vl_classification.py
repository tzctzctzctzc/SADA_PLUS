from pathlib import Path
from types import SimpleNamespace

from tools.run_qwen3_vl_classification import (
    DIGITS,
    LABELS,
    build_messages,
    build_prompt,
    digit_to_label,
    normalize_digit_logits,
    prediction_is_current,
    prompt_sha256,
    resolve_digit_token_ids,
)
from tools.run_qwen_vl_matrix import protocol


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [100 + int(text)]


def test_digit_mapping_is_deterministic_and_complete():
    first = digit_to_label("proposal-1")
    second = digit_to_label("proposal-1")
    assert first == second
    assert set(first) == set(LABELS)
    assert len(first) == len(DIGITS)


def test_prompt_has_all_digit_label_options():
    prompt = build_prompt("proposal-1")
    mapping = digit_to_label("proposal-1")
    for digit, label in zip(DIGITS, mapping):
        assert "{}: {}".format(digit, label) in prompt
    assert "exactly one digit" in prompt


def test_prompt_reflects_dense_frame_count():
    prompt = build_prompt("proposal-1", frame_count=16)
    assert "The 16 images above" in prompt
    assert prompt_sha256("proposal-1", frame_count=16) != prompt_sha256(
        "proposal-1", frame_count=8
    )


def test_build_messages_keeps_eight_images_separate_and_ordered(tmp_path):
    paths = []
    for index in range(1, 9):
        path = tmp_path / "frame_{:02d}.jpg".format(index)
        path.write_bytes(b"jpeg")
        paths.append(path)
    messages = build_messages(paths, "prompt", 100, 200)
    images = [item for item in messages[0]["content"] if item["type"] == "image"]
    assert len(images) == 8
    assert images[0]["image"] == paths[0].resolve().as_uri()
    assert images[-1]["image"] == paths[-1].resolve().as_uri()
    assert all(item["min_pixels"] == 100 for item in images)
    assert all(item["max_pixels"] == 200 for item in images)


def test_digit_logits_are_mapped_back_to_dataset_class_order():
    mapping = digit_to_label("proposal-2")
    winning_digit = 3
    logits = [-2.0] * 10
    logits[winning_digit] = 4.0
    result = normalize_digit_logits(logits, mapping)
    assert result["prediction"] == mapping[winning_digit]
    assert result["class_id"] == LABELS.index(mapping[winning_digit])
    assert abs(sum(result["class_probabilities"]) - 1.0) < 1e-8
    assert result["confidence"] > 0.97


def test_digit_tokens_must_be_single_and_unique():
    assert resolve_digit_token_ids(FakeTokenizer()) == tuple(range(100, 110))


def test_resume_requires_matching_model_and_prompt_contract():
    args = SimpleNamespace(model_path="Qwen/Qwen3-VL-8B-Instruct")
    row = {
        "candidate_id": "proposal-3",
        "status": "ok",
        "model": args.model_path,
        "prompt_version": "qwen3-vl-8b-ordered-images-digit-logits-v1",
        "prompt_sha256": prompt_sha256("proposal-3"),
        "wire_api": "qwen3-vl-next-token-classification",
    }
    assert prediction_is_current(row, args)
    assert not prediction_is_current({**row, "prompt_sha256": "stale"}, args)
    assert not prediction_is_current({**row, "model": "other"}, args)


def test_qwen3_8b_and_32b_use_the_same_wire_protocol():
    version_8b, wire_8b, _ = protocol("qwen3-8b")
    version_32b, wire_32b, _ = protocol("qwen3-32b")
    assert version_8b.startswith("qwen3-vl-8b-")
    assert version_32b.startswith("qwen3-vl-32b-")
    assert wire_8b == wire_32b == "qwen3-vl-next-token-classification"


def test_dense_frame_protocol_is_versioned():
    version, _, _ = protocol("qwen3-32b", frame_count=16)
    assert version == "qwen3-vl-32b-ordered-images-16f-digit-logits-v1"
