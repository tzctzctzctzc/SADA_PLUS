import torch

from tools.audit_selective_alignment_routes import summarize_selector_records


def test_selector_summary_separates_error_removal_and_correct_retention():
    records = {
        "ordinary_valid": torch.ones(6, dtype=torch.bool),
        "strict_valid": torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool),
        "score": torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1, 0.9]),
        "gt_action": torch.tensor([1, 0, 1, 1, 0, 0], dtype=torch.bool),
        "correct_class": torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.bool),
        "reliable_action": torch.tensor(
            [1, 0, 0, 0, 0, 0], dtype=torch.bool
        ),
        "stable_background": torch.tensor(
            [0, 0, 0, 0, 1, 0], dtype=torch.bool
        ),
        "class_consistent": torch.tensor(
            [1, 0, 1, 0, 0, 0], dtype=torch.bool
        ),
        "action_weight": torch.tensor([0.9, 0, 0, 0, 0, 0.0]),
        "background_weight": torch.tensor([0, 0, 0, 0, 0.8, 0.0]),
    }

    summary = summarize_selector_records(records, alpha=0.6)

    assert summary["groups"]["baseline_action"]["n"] == 4
    assert summary["groups"]["reliable_action"]["n"] == 1
    assert summary["groups"]["edge_excluded"]["n"] == 1
    assert summary["error_handling"]["false_action_n"] == 2
    assert summary["error_handling"]["false_action_removal_recall"] == 1.0
    assert summary["error_handling"]["correct_action_retention"] == 1.0
    assert summary["error_handling"]["missed_action_removal_recall"] == 1.0
    assert summary["error_handling"]["true_background_retention"] == 1.0
    assert len(summary["action_weight_quantiles"]) == 1
