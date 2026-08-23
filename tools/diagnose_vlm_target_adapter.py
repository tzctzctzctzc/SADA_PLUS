import argparse
import collections
import json
import random
from pathlib import Path


KNOWN = (
    "TAKE", "PUT", "WASH", "OPEN", "CLOSE", "INSERT", "TURN_ON",
    "CUT", "TURN_OFF", "POUR",
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def join_rows(manifest_path, prediction_paths):
    manifest_rows = read_jsonl(manifest_path)
    manifest = {row["candidate_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("duplicate manifest candidate")
    prediction_rows = [
        row
        for path in prediction_paths
        for row in read_jsonl(path)
        if row.get("status") == "ok"
    ]
    predictions = {row["candidate_id"]: row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("duplicate successful prediction")
    missing = set(manifest) - set(predictions)
    if missing:
        raise ValueError("predictions are incomplete")
    return [
        {**row, "prediction": predictions[candidate_id]["prediction"]}
        for candidate_id, row in manifest.items()
    ]


def attach_gt_views(rows, database):
    enriched = []
    for raw in rows:
        row = dict(raw)
        start = float(row["proposal_start_sec"])
        end = float(row["proposal_end_sec"])
        overlap_by_class = collections.defaultdict(float)
        interval_annotation_count = 0
        for annotation in database[row["video_id"]]["annotations"]:
            segment = annotation["segment"]
            overlap = max(
                0.0,
                min(end, float(segment[1])) - max(start, float(segment[0])),
            )
            if overlap > 0:
                interval_annotation_count += 1
                overlap_by_class[int(annotation["label_id"])] += overlap
        dominant_overlap = max(overlap_by_class.values(), default=0.0)
        row["gt_anchor"] = sorted(set(
            int(class_id) for class_id in row["gt_assigned_class_ids"]
            if int(class_id) < len(KNOWN)
        ))
        row["gt_center"] = sorted(set(
            int(class_id) for class_id in row["center_all_gt_class_ids"]
            if int(class_id) < len(KNOWN)
        ))
        row["gt_interval_any"] = sorted(
            class_id for class_id in overlap_by_class
            if class_id < len(KNOWN)
        )
        row["gt_interval_all"] = sorted(overlap_by_class)
        row["interval_annotation_count"] = interval_annotation_count
        row["gt_interval_dominant"] = sorted(
            class_id
            for class_id, overlap in overlap_by_class.items()
            if class_id < len(KNOWN)
            and abs(overlap - dominant_overlap) < 1e-9
        )
        row["proposal_duration_sec"] = end - start
        enriched.append(row)
    return enriched


def vlm_class(row):
    return KNOWN.index(row["prediction"]) if row["prediction"] in KNOWN else None


def is_consensus(row):
    return vlm_class(row) == int(row["teacher_top1_class"])


def precision(rows, class_fn, gt_key):
    eligible = [row for row in rows if class_fn(row) is not None]
    if not eligible:
        return None
    return sum(class_fn(row) in row[gt_key] for row in eligible) / len(eligible)


def same_size_score_control(rows, count):
    return sorted(
        rows, key=lambda row: row["teacher_score"], reverse=True
    )[:count]


def class_score_control(rows, consensus):
    counts = collections.Counter(
        int(row["teacher_top1_class"]) for row in consensus
    )
    selected = []
    for class_id, count in sorted(counts.items()):
        class_rows = [
            row for row in rows
            if int(row["teacher_top1_class"]) == class_id
        ]
        selected.extend(same_size_score_control(class_rows, count))
    return selected


def stratified_random_control(
    rows, consensus, gt_key, samples=10000, seed=20260823
):
    rng = random.Random(seed)
    counts = collections.Counter(
        int(row["teacher_top1_class"]) for row in consensus
    )
    pools = {
        class_id: [
            row for row in rows
            if int(row["teacher_top1_class"]) == class_id
        ]
        for class_id in counts
    }
    observed = precision(
        consensus, lambda row: int(row["teacher_top1_class"]), gt_key
    )
    values = []
    for _ in range(samples):
        selected = []
        for class_id, count in counts.items():
            selected.extend(rng.sample(pools[class_id], count))
        values.append(precision(
            selected, lambda row: int(row["teacher_top1_class"]), gt_key
        ))
    return {
        "samples": samples,
        "mean_precision": sum(values) / len(values),
        "p_precision_at_least_consensus": (
            sum(value >= observed for value in values) / len(values)
        ),
    }


def leave_one_video_out(rows, gt_key):
    teacher = lambda row: int(row["teacher_top1_class"])
    gains = []
    for video_id in sorted({row["video_id"] for row in rows}):
        retained = [row for row in rows if row["video_id"] != video_id]
        consensus = [row for row in retained if is_consensus(row)]
        control = class_score_control(retained, consensus)
        if not consensus or not control:
            continue
        gain = (
            precision(consensus, teacher, gt_key)
            - precision(control, teacher, gt_key)
        )
        gains.append({"held_out_video": video_id, "gain": gain})
    values = [row["gain"] for row in gains]
    if not values:
        return {
            "video_count": 0,
            "nonpositive_count": 0,
            "min_gain": None,
            "max_gain": None,
            "min_gain_held_out_video": None,
            "max_gain_held_out_video": None,
        }
    return {
        "video_count": len(gains),
        "nonpositive_count": sum(value <= 0 for value in values),
        "min_gain": min(values),
        "max_gain": max(values),
        "min_gain_held_out_video": min(
            gains, key=lambda row: row["gain"]
        )["held_out_video"],
        "max_gain_held_out_video": max(
            gains, key=lambda row: row["gain"]
        )["held_out_video"],
    }


def view_summary(rows, gt_key, random_samples):
    consensus = [row for row in rows if is_consensus(row)]
    disagreement = [
        row for row in rows
        if vlm_class(row) is not None and not is_consensus(row)
    ]
    top_score = same_size_score_control(rows, len(consensus))
    class_top_score = class_score_control(rows, consensus)
    teacher = lambda row: int(row["teacher_top1_class"])
    teacher_all = precision(rows, teacher, gt_key)
    consensus_precision = precision(consensus, teacher, gt_key)
    top_score_precision = precision(top_score, teacher, gt_key)
    class_top_score_precision = precision(class_top_score, teacher, gt_key)
    return {
        "teacher_all_precision": teacher_all,
        "vlm_known_precision": precision(rows, vlm_class, gt_key),
        "consensus_count": len(consensus),
        "consensus_coverage": len(consensus) / len(rows),
        "consensus_precision": consensus_precision,
        "consensus_gain_vs_teacher_all": consensus_precision - teacher_all,
        "same_size_top_score_precision": top_score_precision,
        "consensus_gain_vs_same_size_top_score": (
            consensus_precision - top_score_precision
        ),
        "class_matched_top_score_precision": class_top_score_precision,
        "consensus_gain_vs_class_matched_top_score": (
            consensus_precision - class_top_score_precision
        ),
        "disagreement_count": len(disagreement),
        "disagreement_teacher_precision": precision(
            disagreement, teacher, gt_key
        ),
        "disagreement_vlm_precision": precision(
            disagreement, vlm_class, gt_key
        ),
        "class_stratified_random_control": stratified_random_control(
            rows, consensus, gt_key, samples=random_samples
        ),
        "leave_one_video_out_vs_class_matched_top_score": (
            leave_one_video_out(rows, gt_key)
        ),
    }


def group_summary(rows, gt_key):
    teacher = lambda row: int(row["teacher_top1_class"])
    consensus = [row for row in rows if is_consensus(row)]
    return {
        "count": len(rows),
        "teacher_precision": precision(rows, teacher, gt_key),
        "consensus_count": len(consensus),
        "consensus_precision": precision(consensus, teacher, gt_key),
        "mean_teacher_score": (
            sum(float(row["teacher_score"]) for row in rows) / len(rows)
            if rows else None
        ),
    }


def grouped_diagnostics(rows, gt_key):
    by_class = {}
    for class_id, class_name in enumerate(KNOWN):
        selected = [
            row for row in rows
            if int(row["teacher_top1_class"]) == class_id
        ]
        by_class[class_name] = group_summary(selected, gt_key)
    by_level = {
        str(level): group_summary([
            row for row in rows if row["source_fpn_levels"] == [level]
        ], gt_key)
        for level in (0, 1, 2)
    }
    by_video = {
        video_id: group_summary([
            row for row in rows if row["video_id"] == video_id
        ], gt_key)
        for video_id in sorted({row["video_id"] for row in rows})
    }
    duration_ranges = (
        ("le_2", 0.0, 2.0),
        ("2_to_4", 2.0, 4.0),
        ("4_to_8", 4.0, 8.0),
        ("gt_8", 8.0, float("inf")),
    )
    by_duration = {
        name: group_summary([
            row for row in rows
            if lower < row["proposal_duration_sec"] <= upper
        ], gt_key)
        for name, lower, upper in duration_ranges
    }
    score_ranges = (
        ("0.30_to_0.35", 0.30, 0.35),
        ("0.35_to_0.40", 0.35, 0.40),
        ("0.40_to_0.45", 0.40, 0.45),
        ("gt_0.45", 0.45, float("inf")),
    )
    by_score = {
        name: group_summary([
            row for row in rows
            if lower <= float(row["teacher_score"]) < upper
        ], gt_key)
        for name, lower, upper in score_ranges
    }
    return {
        "by_teacher_class": by_class,
        "by_fpn_level": by_level,
        "by_video": by_video,
        "by_proposal_duration": by_duration,
        "by_teacher_score": by_score,
    }


def representative_candidates(rows, limit=8):
    def take(predicate):
        return [
            row["candidate_id"]
            for row in sorted(
                (row for row in rows if predicate(row)),
                key=lambda row: row["teacher_score"],
                reverse=True,
            )[:limit]
        ]

    teacher = lambda row: int(row["teacher_top1_class"])
    return {
        "consensus_interval_correct": take(
            lambda row: is_consensus(row)
            and teacher(row) in row["gt_interval_dominant"]
        ),
        "consensus_interval_wrong": take(
            lambda row: is_consensus(row)
            and teacher(row) not in row["gt_interval_any"]
        ),
        "vlm_corrects_teacher_at_center": take(
            lambda row: vlm_class(row) is not None
            and not is_consensus(row)
            and vlm_class(row) in row["gt_center"]
            and teacher(row) not in row["gt_center"]
        ),
        "teacher_correct_vlm_wrong_at_center": take(
            lambda row: vlm_class(row) is not None
            and not is_consensus(row)
            and teacher(row) in row["gt_center"]
            and vlm_class(row) not in row["gt_center"]
        ),
        "anchor_metric_mismatch": take(
            lambda row: is_consensus(row)
            and teacher(row) not in row["gt_anchor"]
            and teacher(row) in row["gt_interval_dominant"]
        ),
    }


def diagnose(rows, random_samples=10000):
    gt_views = (
        "gt_anchor",
        "gt_center",
        "gt_interval_any",
        "gt_interval_dominant",
    )
    return {
        "schema_version": "vlm-target-adapter-diagnosis-v1",
        "row_count": len(rows),
        "video_count": len({row["video_id"] for row in rows}),
        "mean_teacher_score": sum(
            float(row["teacher_score"]) for row in rows
        ) / len(rows),
        "mean_consensus_teacher_score": sum(
            float(row["teacher_score"]) for row in rows if is_consensus(row)
        ) / sum(is_consensus(row) for row in rows),
        "gt_view_agreement": {
            "anchor_equals_center": sum(
                row["gt_anchor"] == row["gt_center"] for row in rows
            ) / len(rows),
            "anchor_equals_interval_dominant": sum(
                row["gt_anchor"] == row["gt_interval_dominant"]
                for row in rows
            ) / len(rows),
        },
        "proposal_overlap": {
            "no_annotation_count": sum(
                row["interval_annotation_count"] == 0 for row in rows
            ),
            "multiple_annotation_count": sum(
                row["interval_annotation_count"] >= 2 for row in rows
            ),
            "multiple_class_count": sum(
                len(row["gt_interval_all"]) >= 2 for row in rows
            ),
            "known_and_unknown_class_count": sum(
                any(class_id < len(KNOWN) for class_id in row["gt_interval_all"])
                and any(
                    class_id >= len(KNOWN)
                    for class_id in row["gt_interval_all"]
                )
                for row in rows
            ),
            "only_unknown_class_count": sum(
                bool(row["gt_interval_all"])
                and all(
                    class_id >= len(KNOWN)
                    for class_id in row["gt_interval_all"]
                )
                for row in rows
            ),
        },
        "views": {
            gt_key.removeprefix("gt_"): view_summary(
                rows, gt_key, random_samples
            )
            for gt_key in gt_views
        },
        "diagnostics": {
            gt_key.removeprefix("gt_"): grouped_diagnostics(rows, gt_key)
            for gt_key in gt_views
        },
        "representative_candidates": representative_candidates(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True, nargs="+")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-samples", type=int, default=10000)
    args = parser.parse_args()

    rows = join_rows(args.manifest, args.predictions)
    database = json.loads(
        Path(args.annotations).read_text(encoding="utf-8")
    )["database"]
    result = diagnose(
        attach_gt_views(rows, database),
        random_samples=args.random_samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["views"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
