import unittest

from tools.build_vlm_listwise_manifest import build_rows
from tools.score_vlm_listwise import summarize


class VlmListwiseTest(unittest.TestCase):
    def test_six_permutations_are_exactly_position_balanced(self):
        anchors = [
            {
                "candidate_id": "anchor-{}".format(index),
                "sada_top1_class": index % 10,
                "sada_top2_class": (index + 1) % 10,
                "center_all_gt_class_ids": [index % 10],
            }
            for index in range(60)
        ]
        names = {index: "class-{}".format(index) for index in range(40)}
        rows = build_rows(anchors, names, seed=123)
        counts = {
            (role, letter): 0
            for role in ("top1", "top2", "near_confusion")
            for letter in "ABC"
        }
        for row in rows:
            self.assertEqual(len({
                option["class_id"] for option in row["option_mapping"].values()
            }), 3)
            for letter, option in row["option_mapping"].items():
                counts[(option["role"], letter)] += 1
        self.assertEqual(set(counts.values()), {20})

    def test_partial_cycle_does_not_require_multiple_of_six(self):
        anchors = [
            {
                "candidate_id": "anchor-{}".format(index),
                "sada_top1_class": index % 10,
                "sada_top2_class": (index + 1) % 10,
            }
            for index in range(7)
        ]
        names = {index: "class-{}".format(index) for index in range(40)}
        rows = build_rows(anchors, names, seed=123)
        self.assertEqual(len(rows), 7)

    def test_top1_filter_metrics(self):
        rows = [
            {
                "prediction": "B",
                "sada_top1_class": 0,
                "center_all_gt_class_ids": [0],
                "option_mapping": {
                    "A": {"role": "top2", "class_id": 1},
                    "B": {"role": "top1", "class_id": 0},
                    "C": {"role": "near_confusion", "class_id": 12},
                },
            },
            {
                "prediction": "A",
                "sada_top1_class": 0,
                "center_all_gt_class_ids": [1],
                "option_mapping": {
                    "A": {"role": "top2", "class_id": 1},
                    "B": {"role": "top1", "class_id": 0},
                    "C": {"role": "near_confusion", "class_id": 12},
                },
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["candidate_pool_top1_correct_rate"], 0.5)
        self.assertEqual(result["selected_top1_precision"], 1.0)
        self.assertEqual(result["correct_top1_retention"], 1.0)
        self.assertEqual(result["wrong_top1_win_rate"], 0.0)
        self.assertEqual(result["selected_any_candidate_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
