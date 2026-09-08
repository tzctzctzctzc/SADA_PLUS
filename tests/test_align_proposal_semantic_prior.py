import pytest

from tools.align_proposal_semantic_prior import (
    align_prior,
    source_class_prior,
)


def test_source_prior_uses_only_requested_training_subsets():
    annotations = {"database": {
        "source_train": {
            "subset": "ek55_white_training",
            "annotations": [{"label_id": 0}, {"label_id": 0}, {"label_id": 1}],
        },
        "source_val": {
            "subset": "ek55_white_validation",
            "annotations": [{"label_id": 1}],
        },
        "target_train": {
            "subset": "ek100_white_training",
            "annotations": [{"label_id": 1}],
        },
    }}

    assert source_class_prior(annotations, "ek55_", 2) == pytest.approx(
        [2 / 3, 1 / 3]
    )


def test_prior_alignment_reduces_overrepresented_pseudo_class():
    payload = {"entries": [
        {
            "candidate_id": "a",
            "class_id": 0,
            "semantic_label": "TAKE",
            "class_probabilities": [0.9, 0.1],
            "confidence": 0.9,
        },
        {
            "candidate_id": "b",
            "class_id": 0,
            "semantic_label": "TAKE",
            "class_probabilities": [0.6, 0.4],
            "confidence": 0.6,
        },
    ]}

    result = align_prior(payload, [0.2, 0.8])

    assert result["entries"][1]["class_id"] == 1
    assert result["entries"][1]["semantic_label"] == "PUT"
    assert result["prior_alignment"]["selection_uses_target_gt"] is False
