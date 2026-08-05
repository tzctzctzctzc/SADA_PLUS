import unittest

from tools.build_vlm_candidate_selectivity_manifest import build_rows
from tools.score_vlm_candidate_selectivity import summarize


class VlmCandidateSelectivityTest(unittest.TestCase):
    def test_four_unique_candidates_reuse_one_frame_cache(self):
        anchors = [
            {
                "candidate_id": "video-l0-000001",
                "sada_top1_class": 0,
                "sada_top2_class": 1,
                "center_all_gt_class_ids": [0],
            }
        ]
        names = {index: "class-{}".format(index) for index in range(40)}
        rows = build_rows(anchors, names, seed=123)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row["candidate_label_id"] for row in rows}), 4)
        self.assertEqual({row["frame_cache_id"] for row in rows}, {"video-l0-000001"})
        random_row = next(
            row for row in rows if row["candidate_role"] == "random_wrong"
        )
        self.assertFalse(random_row["candidate_is_center_gt"])

    def test_selectivity_metrics_detect_prompt_acquiescence(self):
        base = {
            "anchor_id": "a",
            "sada_top1_class": 0,
        }
        rows = [
            {
                **base,
                "candidate_role": "top1",
                "candidate_label_id": 0,
                "candidate_is_center_gt": True,
                "prediction": "CONFIRM_TARGET",
            },
            {
                **base,
                "candidate_role": "top2",
                "candidate_label_id": 1,
                "candidate_is_center_gt": False,
                "prediction": "CONFIRM_TARGET",
            },
            {
                **base,
                "candidate_role": "near_confusion",
                "candidate_label_id": 12,
                "candidate_is_center_gt": False,
                "prediction": "OTHER_ACTIVITY",
            },
            {
                **base,
                "candidate_role": "random_wrong",
                "candidate_label_id": 5,
                "candidate_is_center_gt": False,
                "prediction": "OTHER_ACTIVITY",
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["multi_confirmation_rate"], 1.0)
        self.assertEqual(result["candidate_selectivity_rate"], 0.0)
        self.assertEqual(result["wrong_candidate_confirmation_rate"], 1 / 3)
        self.assertEqual(result["exclusive_confirmation_count"], 0)


if __name__ == "__main__":
    unittest.main()
