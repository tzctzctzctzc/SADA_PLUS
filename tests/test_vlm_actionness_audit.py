import unittest

from tools.score_vlm_actionness_audit import summarize


class VlmActionnessAuditTest(unittest.TestCase):
    def test_metrics_keep_recovery_and_false_promotion_separate(self):
        rows = [
            {
                "prediction": "A",
                "gt_is_foreground": True,
                "sada_top1_is_correct": True,
            },
            {
                "prediction": "B",
                "gt_is_foreground": True,
                "sada_top1_is_correct": True,
            },
            {
                "prediction": "A",
                "gt_is_foreground": False,
                "sada_top1_is_correct": False,
            },
            {
                "prediction": "U",
                "gt_is_foreground": False,
                "sada_top1_is_correct": False,
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["candidate_foreground_recall"], 0.5)
        self.assertEqual(result["background_false_promotion_rate"], 0.5)
        self.assertEqual(result["promoted_action_precision"], 0.5)
        self.assertEqual(result["correct_action_routing_coverage"], 0.5)
        self.assertEqual(result["uncertain_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
