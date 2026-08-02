import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).parents[1] / "tools/audit_low_confidence_topk_consensus.py"
)
SPEC = importlib.util.spec_from_file_location("low_confidence_topk_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class LowConfidenceTopkAuditTest(unittest.TestCase):
    def test_best_positive_rank_supports_multi_hot_labels(self):
        probabilities = torch.tensor([
            [0.8, 0.7, 0.6, 0.5],
            [0.9, 0.8, 0.7, 0.6],
        ])
        labels = torch.tensor([
            [False, True, False, True],
            [False, False, True, True],
        ])
        rank, score = audit.best_positive_rank(probabilities, labels)
        self.assertTrue(torch.equal(rank, torch.tensor([2, 3])))
        self.assertTrue(torch.allclose(score, torch.tensor([0.7, 0.7])))

    def test_majority_candidate_breaks_tie_by_mean_score(self):
        probabilities = torch.tensor([
            [[0.8, 0.2, 0.1]],
            [[0.3, 0.7, 0.1]],
            [[0.6, 0.4, 0.1]],
            [[0.1, 0.9, 0.0]],
        ])
        candidate, support, scores = audit.majority_candidate(probabilities)
        self.assertEqual(candidate.item(), 1)
        self.assertEqual(support.item(), 2)
        self.assertTrue(torch.allclose(scores[:, 0], torch.tensor([0.2, 0.7, 0.4, 0.9])))


if __name__ == "__main__":
    unittest.main()
