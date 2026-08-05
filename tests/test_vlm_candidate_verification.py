import unittest

from tools.score_vlm_candidate_verification import summarize, summarize_real


class VlmCandidateVerificationTest(unittest.TestCase):
    def test_candidate_recall_and_false_confirms_are_separate(self):
        rows = [
            {"probe_group": "candidate_action", "prediction": "CONFIRM_TARGET", "candidate_label_id": 0},
            {"probe_group": "candidate_action", "prediction": "UNCERTAIN", "candidate_label_id": 0},
            {"probe_group": "other_target_action", "prediction": "OTHER_ACTIVITY", "candidate_label_id": 0},
            {"probe_group": "other_action", "prediction": "CONFIRM_TARGET", "candidate_label_id": 0},
            {"probe_group": "idle_background", "prediction": "TRUE_BACKGROUND", "candidate_label_id": 0},
        ]
        result = summarize(rows)
        self.assertEqual(result["candidate_confirm_recall"], 0.5)
        self.assertEqual(result["confirmed_precision"], 0.5)
        self.assertEqual(result["non_candidate_false_confirm_rate"], 1 / 3)
        self.assertEqual(result["overall_uncertain_rate"], 0.2)
        self.assertEqual(
            result["by_probe_group"]["other_action"]["confirm_rate"], 1.0
        )
        self.assertEqual(result["by_candidate_class"]["0"]["n"], 5)

    def test_real_sada_audit_separates_wrong_class_and_background(self):
        rows = [
            {
                "prediction": "CONFIRM_TARGET",
                "gt_is_foreground": True,
                "sada_top1_is_correct": True,
                "sada_top1_class": 0,
            },
            {
                "prediction": "OTHER_ACTIVITY",
                "gt_is_foreground": True,
                "sada_top1_is_correct": True,
                "sada_top1_class": 0,
            },
            {
                "prediction": "CONFIRM_TARGET",
                "gt_is_foreground": True,
                "sada_top1_is_correct": False,
                "sada_top1_class": 1,
            },
            {
                "prediction": "TRUE_BACKGROUND",
                "gt_is_foreground": False,
                "sada_top1_is_correct": False,
                "sada_top1_class": 1,
            },
        ]
        result = summarize_real(
            rows,
            {
                "gt_foreground_anchors": 10,
                "original_selected_correct_action_anchors": 2,
            },
        )
        self.assertEqual(result["candidate_correct_confirm_recall"], 0.5)
        self.assertEqual(result["confirmed_correct_action_precision"], 0.5)
        self.assertEqual(result["wrong_action_class_false_confirm_rate"], 1.0)
        self.assertEqual(result["background_false_confirm_rate"], 0.0)
        self.assertEqual(result["background_routing_recall"], 1.0)
        self.assertEqual(result["correct_action_routing_coverage_before_vlm"], 0.2)
        self.assertEqual(result["correct_action_routing_coverage_after_vlm"], 0.3)


if __name__ == "__main__":
    unittest.main()
