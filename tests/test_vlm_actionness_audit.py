import unittest

from tools.score_vlm_actionness_audit import summarize, summarize_prompt_branch


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

    def test_low_and_high_prompt_metrics_use_different_decision_policies(self):
        low = [
            {"prediction": "TARGET_ACTION", "gt_is_foreground": True},
            {"prediction": "UNCERTAIN", "gt_is_foreground": True},
            {"prediction": "TARGET_ACTION", "gt_is_foreground": False},
        ]
        low_result = summarize_prompt_branch(low, "low_rescue")
        self.assertEqual(low_result["true_action_rescue_recall"], 0.5)
        self.assertEqual(low_result["semantic_rescue_precision"], 0.5)
        self.assertEqual(low_result["pass_coverage"], 2 / 3)

        high = [
            {"prediction": "TARGET_ACTION", "gt_is_foreground": True},
            {"prediction": "UNCERTAIN", "gt_is_foreground": True},
            {"prediction": "OTHER_ACTIVITY", "gt_is_foreground": True},
            {
                "prediction": "OTHER_ACTIVITY",
                "gt_is_foreground": False,
                "center_all_gt_class_ids": [20],
            },
            {"prediction": "TRUE_BACKGROUND", "gt_is_foreground": False},
            {"prediction": "UNCERTAIN", "gt_is_foreground": False},
        ]
        high_result = summarize_prompt_branch(high, "high_veto")
        self.assertEqual(high_result["true_action_retention"], 2 / 3)
        self.assertEqual(high_result["background_non_target_rejection"], 2 / 3)
        self.assertEqual(high_result["veto_precision"], 2 / 3)
        self.assertEqual(high_result["veto_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
