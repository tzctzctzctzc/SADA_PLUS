"""Convert frozen visual/text embeddings into RQ1 zero-shot predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.rq1_feature_store import SCHEMA, sha256_file
from tools.run_clip_target_adapter import LABELS, read_jsonl


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--feature-prefix", required=True)
    parser.add_argument("--text-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    feature_rows = read_jsonl(args.feature_manifest)
    eval_rows = read_jsonl(args.eval_manifest)
    row_index = {row["candidate_id"]: index for index, row in enumerate(feature_rows)}
    if len(row_index) != len(feature_rows):
        raise ValueError("feature manifest contains duplicate candidate IDs")

    prefix = Path(args.feature_prefix)
    feature_meta = json.loads(
        Path(str(prefix) + ".meta.json").read_text(encoding="utf-8")
    )
    if feature_meta.get("schema") != SCHEMA:
        raise ValueError("unsupported feature schema")
    if feature_meta.get("manifest_sha256") != sha256_file(args.feature_manifest):
        raise ValueError("feature manifest hash differs from the feature store")
    features = np.load(str(prefix) + ".features.npy", mmap_mode="r")
    status = np.load(str(prefix) + ".status.npy", mmap_mode="r")
    if len(features) != len(feature_rows) or not np.all(status == 1):
        raise ValueError("feature store is incomplete")

    text_path = Path(args.text_features)
    text = normalize_rows(np.load(text_path))
    if text.shape != (len(LABELS), features.shape[1]):
        raise ValueError("text feature shape mismatch")
    text_meta = json.loads(
        Path(str(text_path) + ".meta.json").read_text(encoding="utf-8")
    )
    if text_meta.get("labels") != list(LABELS):
        raise ValueError("text labels differ from the RQ1 label order")
    logit_scale = float(text_meta["logit_scale"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for shift in SHIFT_CONFIGS:
            shift_rows = [
                row for row in eval_rows
                if any(role["shift"] == shift for role in row["memberships"])
            ]
            visual = normalize_rows(
                features[[row_index[row["candidate_id"]] for row in shift_rows]]
            )
            similarities = visual @ text.T
            logits = logit_scale * similarities
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            for row, probs, cosine in zip(shift_rows, probabilities, similarities):
                order = np.argsort(probs)[::-1]
                top1, top2 = map(int, order[:2])
                stream.write(json.dumps({
                    "candidate_id": row["candidate_id"],
                    "video_id": row["video_id"],
                    "shift": shift,
                    "status": "ok",
                    "model": args.model_name,
                    "source_supervision": False,
                    "class_id": top1,
                    "prediction": LABELS[top1],
                    "confidence": float(probs[top1]),
                    "margin": float(probs[top1] - probs[top2]),
                    "class_probabilities": probs.tolist(),
                    "class_similarities": cosine.tolist(),
                    "logit_scale": logit_scale,
                }, sort_keys=True) + "\n")
            print("{} evaluate={}".format(shift, len(shift_rows)), flush=True)
    Path(str(output) + ".protocol.json").write_text(json.dumps({
        "model": args.model_name,
        "frozen_feature_model": feature_meta["model"],
        "feature_pooling": feature_meta["pooling"],
        "frame_count": feature_meta["frame_count"],
        "source_supervision": False,
        "target_supervision": False,
        "text_feature_metadata": text_meta,
    }, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
