import unittest

from tools.score_vlm_candidate_verification import summarize


class VlmCandidateVerificationTest(unittest.TestCase):
    def test_candidate_recall_and_false_confirms_are_separate(self):
        rows = [
            {"probe_group": "candidate_action", "prediction": "CONFIRM_TARGET"},
            {"probe_group": "candidate_action", "prediction": "UNCERTAIN"},
            {"probe_group": "other_target_action", "prediction": "OTHER_ACTIVITY"},
            {"probe_group": "other_action", "prediction": "CONFIRM_TARGET"},
            {"probe_group": "idle_background", "prediction": "TRUE_BACKGROUND"},
        ]
        result = summarize(rows)
        self.assertEqual(result["candidate_confirm_recall"], 0.5)
        self.assertEqual(result["confirmed_precision"], 0.5)
        self.assertEqual(result["non_candidate_false_confirm_rate"], 1 / 3)
        self.assertEqual(result["overall_uncertain_rate"], 0.2)
        self.assertEqual(
            result["by_probe_group"]["other_action"]["confirm_rate"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
