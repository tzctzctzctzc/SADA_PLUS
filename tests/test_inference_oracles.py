import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_inference_oracles.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_inference_oracles", MODULE_PATH)
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def fixtures():
    predictions = pd.DataFrame(
        [
            ["v1", 0.0, 4.0, 1, 0.9],
            ["v1", 10.0, 13.0, 0, 0.8],
            ["v1", 30.0, 31.0, 0, 0.7],
        ],
        columns=oracle.PRED_COLUMNS,
    )
    ground_truth = pd.DataFrame(
        [
            ["v1", 0.0, 5.0, 0],
            ["v1", 10.0, 15.0, 0],
        ],
        columns=["video-id", "t-start", "t-end", "label"],
    )
    return predictions, ground_truth


def test_class_oracle_relabels_only_overlapping_predictions():
    predictions, ground_truth = fixtures()
    result = oracle.apply_match_oracle(
        predictions,
        ground_truth,
        class_aware=False,
        min_iou=0.1,
        replace_label=True,
    )
    assert result["label"].tolist() == [0, 0, 0]
    assert result["score"].tolist() == predictions["score"].tolist()


def test_boundary_oracle_requires_correct_class_and_keeps_invalid_rows():
    predictions, ground_truth = fixtures()
    result = oracle.apply_match_oracle(
        predictions,
        ground_truth,
        class_aware=True,
        min_iou=0.1,
        replace_boundary=True,
    )
    assert result.loc[0, ["t-start", "t-end"]].tolist() == [0.0, 4.0]
    assert result.loc[1, ["t-start", "t-end"]].tolist() == [10.0, 15.0]
    assert result.loc[2, ["t-start", "t-end"]].tolist() == [30.0, 31.0]


def test_validity_oracle_drops_wrong_class_and_background():
    predictions, ground_truth = fixtures()
    result = oracle.apply_match_oracle(
        predictions,
        ground_truth,
        class_aware=True,
        min_iou=0.1,
        drop_invalid=True,
    )
    assert len(result) == 1
    assert result.iloc[0]["t-start"] == 10.0


def test_proposal_coverage_separates_semantic_and_temporal_recall():
    predictions, ground_truth = fixtures()
    agnostic = oracle.proposal_coverage(
        predictions, ground_truth, [0.5], class_aware=False
    )
    aware = oracle.proposal_coverage(
        predictions, ground_truth, [0.5], class_aware=True
    )
    assert agnostic["recall"]["0.5"] == 1.0
    assert aware["recall"]["0.5"] == 0.5


def test_proposal_ceiling_emits_one_exact_row_per_covered_gt():
    predictions, ground_truth = fixtures()
    result = oracle.proposal_ceiling(
        predictions, ground_truth, min_iou=0.5, class_aware=False
    )
    assert len(result) == 2
    assert result[["t-start", "t-end"]].values.tolist() == [
        [0.0, 5.0],
        [10.0, 15.0],
    ]


def test_detection_error_taxonomy_is_score_aware():
    ground_truth = pd.DataFrame(
        [
            ["v1", 0.0, 5.0, 0],
            ["v1", 10.0, 15.0, 1],
        ],
        columns=["video-id", "t-start", "t-end", "label"],
    )
    predictions = pd.DataFrame(
        [
            ["v1", 0.0, 5.0, 0, 0.9],
            ["v1", 0.0, 5.0, 0, 0.8],
            ["v1", 10.0, 15.0, 0, 0.7],
            ["v1", 3.0, 6.0, 0, 0.6],
            ["v1", 8.0, 12.0, 0, 0.5],
            ["v1", 30.0, 31.0, 0, 0.4],
        ],
        columns=oracle.PRED_COLUMNS,
    )
    assert oracle.detection_error_labels(
        predictions, ground_truth, tiou_threshold=0.5, min_overlap=0.1
    ).tolist() == [
        "true_positive",
        "double_detection",
        "wrong_label",
        "localization",
        "confusion",
        "background",
    ]
