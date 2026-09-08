import copy

import pytest

from tools.build_consensus_proposal_semantic_cache import build_consensus


def _payload(rows, model):
    return {
        "source": "offline_vlm_proposal_semantics",
        "prompt_protocol": {"model": model},
        "entries": rows,
        "successful_predictions": len(rows),
    }


def test_consensus_keeps_only_confident_matching_predictions():
    rows = [
        {
            "candidate_id": "a",
            "class_id": 0,
            "semantic_label": "TAKE",
            "confidence": 0.8,
            "class_probabilities": [0.8, 0.2],
        },
        {
            "candidate_id": "b",
            "class_id": 1,
            "semantic_label": "PUT",
            "confidence": 0.9,
            "class_probabilities": [0.1, 0.9],
        },
    ]
    other = copy.deepcopy(rows)
    other[0]["class_probabilities"] = [0.6, 0.4]
    other[0]["confidence"] = 0.6
    other[1]["class_id"] = 0

    result = build_consensus(
        [_payload(rows, "8b"), _payload(other, "32b")],
        min_confidence=0.5,
    )

    assert [entry["candidate_id"] for entry in result["entries"]] == ["a"]
    assert result["entries"][0]["class_probabilities"] == pytest.approx(
        [0.7, 0.3]
    )
    assert result["entries"][0]["consensus_models"] == 2
    assert result["consensus"]["retained_candidates"] == 1
