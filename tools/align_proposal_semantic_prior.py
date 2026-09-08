import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.build_proposal_semantic_cache import KNOWN_LABELS


def source_class_prior(annotation_payload, subset_prefix, num_classes):
    counts = Counter()
    for video in annotation_payload["database"].values():
        subset = video["subset"]
        if not subset.startswith(subset_prefix) or not subset.endswith("training"):
            continue
        counts.update(
            annotation["label_id"]
            for annotation in video["annotations"]
            if annotation["label_id"] < num_classes
        )
    total = sum(counts.values())
    if total == 0:
        raise ValueError("source split contains no selected classes")
    return [counts[index] / total for index in range(num_classes)]


def align_prior(payload, source_prior, strength=1.0):
    rows = [
        entry for entry in payload["entries"]
        if entry.get("class_probabilities") is not None
    ]
    if not rows or len(source_prior) != len(rows[0]["class_probabilities"]):
        raise ValueError("source prior and proposal vocabulary differ")
    num_classes = len(source_prior)
    pseudo_prior = [
        sum(entry["class_probabilities"][index] for entry in rows) / len(rows)
        for index in range(num_classes)
    ]
    scales = [
        (source_prior[index] / max(pseudo_prior[index], 1e-12)) ** strength
        for index in range(num_classes)
    ]
    result = copy.deepcopy(payload)
    for entry in result["entries"]:
        probabilities = entry.get("class_probabilities")
        if probabilities is None:
            continue
        corrected = [
            probabilities[index] * scales[index]
            for index in range(num_classes)
        ]
        normalizer = sum(corrected)
        corrected = [value / normalizer for value in corrected]
        class_id = max(range(num_classes), key=corrected.__getitem__)
        entry["class_probabilities"] = corrected
        entry["class_id"] = class_id
        entry["semantic_label"] = KNOWN_LABELS[class_id]
        entry["confidence"] = corrected[class_id]
    result["prior_alignment"] = {
        "strength": float(strength),
        "source_prior": source_prior,
        "pseudo_prior_before": pseudo_prior,
        "selection_uses_target_gt": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--source-subset-prefix", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache_path = Path(args.cache)
    annotation_path = Path(args.annotations)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    num_classes = len(payload["entries"][0]["class_probabilities"])
    prior = source_class_prior(
        annotations, args.source_subset_prefix, num_classes
    )
    result = align_prior(payload, prior, args.strength)
    result["prior_alignment"].update({
        "source_subset_prefix": args.source_subset_prefix,
        "source_annotations_sha256": hashlib.sha256(
            annotation_path.read_bytes()
        ).hexdigest(),
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["prior_alignment"], indent=2))


if __name__ == "__main__":
    main()
