"""把共同候选与真实VLM类别预测绑定；不执行教师推理，不制造伪预测。"""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.rq2_dense_protocol import sha256_file
from tools.rq3_protocol import FRAME_PROTOCOL, MODELS, SCHEMA, content_hash, validate_cache
from tools.run_clip_target_adapter import LABELS as TARGET_CLASS_NAMES


def probabilities_from_logits(logits, temperature=1.0):
    values = np.asarray(logits, dtype=np.float64)
    if values.shape[-1] != 10 or not np.isfinite(values).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("RQ3 requires finite ten-class logits and a positive source-fixed temperature")
    values = values / temperature
    weights = np.exp(values - values.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def paired_text_probabilities(video, text, scale):
    """输入必须来自该模型官方配对编码器；不训练target分类头。"""
    video, text = np.asarray(video, dtype=np.float64), np.asarray(text, dtype=np.float64)
    if video.ndim != 2 or text.ndim != 2 or text.shape[0] != 10 or video.shape[1] != text.shape[1]:
        raise ValueError("official video/text embeddings have incompatible dimensions")
    if not np.isfinite(video).all() or not np.isfinite(text).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("official video/text scores are nonfinite")
    if np.any(np.linalg.norm(video, axis=-1) <= 1e-8) or np.any(np.linalg.norm(text, axis=-1) <= 1e-8):
        raise ValueError("official video/text embeddings collapsed")
    video = video / np.linalg.norm(video, axis=-1, keepdims=True)
    text = text / np.linalg.norm(text, axis=-1, keepdims=True)
    return probabilities_from_logits(scale * video @ text.T)


def build_cache(manifest, predictions, protocol, model, shift, teacher_hash):
    """manifest同一份用于三个VLM和策略；预测输入禁止带TAL类别/分数。"""
    if manifest.get("schema") != "rq3-common-candidates-v1":
        raise ValueError("RQ3 requires a frozen common candidate manifest")
    if (manifest.get("target_gt_used") is not False or manifest.get("shift") != shift
            or manifest.get("seed") != 6 or manifest.get("teacher_selection") != "source-validation-only"
            or manifest.get("teacher_checkpoint_sha256") != teacher_hash):
        raise ValueError("RQ3 candidate selection has mismatched SO teacher or target GT")
    if manifest.get("frame_protocol") != FRAME_PROTOCOL:
        raise ValueError("RQ3 candidate frame protocol differs")
    predicted = {row["candidate_id"]: row for row in predictions}
    if protocol.get("interface_audit_path"):
        audit = json.loads(Path(protocol["interface_audit_path"]).read_text())
        if audit["common_manifest_sha256"] != content_hash(manifest) or audit["predictions_sha256"] != content_hash(predictions):
            raise ValueError("RQ3 predictions/manifest differ from real VLM interface audit")
    candidates = manifest["candidates"]
    if len(predicted) != len(predictions) or len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate RQ3 candidate/prediction")
    if set(predicted) != {row["candidate_id"] for row in candidates}:
        raise ValueError("RQ3 predictions do not cover the exact common candidate set")
    entries, frame_rows = [], []
    for row in candidates:
        if (set(row) != {"candidate_id", "video_id", "absolute_feature_index", "source_fpn_levels",
                         "proposal_start_sec", "proposal_end_sec", "frames", "frame_sha256"}
                or len(row["frames"]) != 8 or len(set(row["frames"])) != 8
                or any(type(frame) is not int or frame < 0 for frame in row["frames"])
                or not 0 <= row["proposal_start_sec"] < row["proposal_end_sec"]):
            raise ValueError("RQ3 candidate has GT, invalid bounds or non-exact eight-frame input")
        prediction = predicted[row["candidate_id"]]
        if set(prediction) != {"candidate_id", "frame_sha256", "class_probabilities"}:
            raise ValueError("RQ3 prediction must contain only candidate identity, frames and class probabilities")
        frame_identity = content_hash(row["frame_sha256"])
        if (len(row["frame_sha256"]) != 8 or prediction["frame_sha256"] != frame_identity
                or any(not isinstance(digest, str) or len(digest) != 64
                       or any(c not in "0123456789abcdef" for c in digest)
                       for digest in row["frame_sha256"])):
            raise ValueError("RQ3 VLM scored different proposal frames")
        probs = prediction["class_probabilities"]
        entries.append({"video_id": row["video_id"], "absolute_feature_index": row["absolute_feature_index"],
                        "source_fpn_levels": row["source_fpn_levels"],
                        "class_id": max(range(10), key=lambda index: probs[index]),
                        "class_probabilities": probs})
        frame_rows.append({"candidate_id": row["candidate_id"], "frames": row["frames"], "sha256": row["frame_sha256"]})
    payload = {"schema": SCHEMA, "teacher_model": model, "shift": shift, "seed": 6,
            "labels": list(TARGET_CLASS_NAMES), "target_gt_used": False,
            "target_video_ids": manifest["target_video_ids"], "input_protocol": protocol,
            "feature_lengths": manifest["feature_lengths"],
            "teacher_selection": "source-validation-only", "teacher_checkpoint_sha256": teacher_hash,
            "candidate_sha256": content_hash(manifest), "frames_sha256": content_hash(frame_rows),
            "predictions_sha256": content_hash(predictions),
            "entries": entries}
    validate_cache(payload, model, shift)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--input-protocol", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--shift", choices=["S" + str(i) for i in range(1, 7)], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tools.rq3_protocol import validate_interface_audit
    protocol = json.loads(args.input_protocol.read_text(encoding="utf-8"))
    validate_interface_audit(protocol, args.model)
    payload = build_cache(json.loads(args.manifest.read_text(encoding="utf-8")),
                          [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()],
                          protocol, args.model, args.shift,
                          sha256_file(args.teacher_checkpoint))
    # 内存核验通过才落盘，不留下已知不合格的正式缓存。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
    print("RQ3 class cache verified; entries={}".format(len(payload["entries"])))


if __name__ == "__main__":
    main()
