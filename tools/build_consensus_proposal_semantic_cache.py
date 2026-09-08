import argparse
import copy
import json
from pathlib import Path


def _entries_by_id(payload):
    return {entry["candidate_id"]: entry for entry in payload["entries"]}


def build_consensus(payloads, min_confidence=0.5):
    if len(payloads) < 2:
        raise ValueError("consensus requires at least two caches")
    indexed = [_entries_by_id(payload) for payload in payloads]
    shared_ids = set(indexed[0])
    for entries in indexed[1:]:
        shared_ids &= set(entries)

    retained = []
    for candidate_id in sorted(shared_ids):
        rows = [entries[candidate_id] for entries in indexed]
        class_ids = [row.get("class_id") for row in rows]
        if (
            any(class_id is None for class_id in class_ids)
            or len(set(class_ids)) != 1
            or any(float(row.get("confidence", 0.0)) < min_confidence for row in rows)
            or any(row.get("class_probabilities") is None for row in rows)
        ):
            continue
        probability_rows = [row["class_probabilities"] for row in rows]
        if len({len(probabilities) for probabilities in probability_rows}) != 1:
            raise ValueError("consensus caches use different class vocabularies")
        mean_probabilities = [
            sum(values) / len(values) for values in zip(*probability_rows)
        ]
        normalizer = sum(mean_probabilities)
        mean_probabilities = [value / normalizer for value in mean_probabilities]
        entry = copy.deepcopy(rows[0])
        entry["class_probabilities"] = mean_probabilities
        entry["confidence"] = max(mean_probabilities)
        entry["consensus_models"] = len(rows)
        retained.append(entry)

    result = copy.deepcopy(payloads[0])
    result["source"] = "offline_vlm_consensus_proposal_semantics"
    result["entries"] = retained
    result["successful_predictions"] = len(retained)
    result["consensus"] = {
        "models": [payload.get("prompt_protocol", {}) for payload in payloads],
        "min_confidence": float(min_confidence),
        "shared_candidates": len(shared_ids),
        "retained_candidates": len(retained),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payloads = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.inputs
    ]
    result = build_consensus(payloads, args.min_confidence)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["consensus"], indent=2))


if __name__ == "__main__":
    main()
