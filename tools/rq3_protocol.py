"""RQ3缓存身份与基线等价检查；VLM特征不得替换SlowFast学生输入。"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from tools.rq2_dense_protocol import sha256_file
from tools.run_clip_target_adapter import LABELS as TARGET_CLASS_NAMES


SCHEMA = "rq3-candidate-class-cache-v1"
MODELS = ("egovideo_hod", "internvideo2_clip_1b", "qwen3_vl_8b")
STRATEGIES = ("hard", "verify", "prototype")
FRAME_PROTOCOL = "proposal-interior-eight-midpoints-v1"


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def members_for(cfg, domain):
    database = json.loads(Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8"))["database"]
    splits = {value.lower() for value in cfg[domain + "_train_split"]}
    return sorted(key for key, value in database.items() if value["subset"].lower() in splits)


def load_cache(path, model=None, shift=None, members=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_cache(payload, model, shift, members)


def validate_cache(payload, model=None, shift=None, members=None):
    if payload.get("schema") != SCHEMA or payload.get("teacher_model") not in MODELS:
        raise ValueError("invalid RQ3 class-cache schema/model")
    if (model and payload["teacher_model"] != model) or (shift and payload["shift"] != shift):
        raise ValueError("RQ3 cache belongs to another model/shift")
    if (payload.get("labels") != list(TARGET_CLASS_NAMES) or payload.get("seed") != 6
            or payload.get("shift") not in {"S" + str(i) for i in range(1, 7)}):
        raise ValueError("RQ3 cache class order or seed differs")
    protocol = payload["input_protocol"]
    if protocol.get("frame_protocol") != FRAME_PROTOCOL or protocol.get("input_frames") != 8:
        raise ValueError("RQ3 cache must use the common eight-frame proposal protocol")
    if protocol.get("class_blind") is not True or payload.get("target_gt_used") is not False:
        raise ValueError("RQ3 cache is not class-blind/target-GT-free")
    if protocol.get("readout") != ("closed-class-log-likelihood" if payload["teacher_model"] == "qwen3_vl_8b"
                                  else "official-paired-video-text-cosine"):
        raise ValueError("RQ3 requires the model's audited semantic readout, not dense hidden features")
    if payload.get("target_video_ids") != sorted(set(payload["target_video_ids"])):
        raise ValueError("RQ3 cache has duplicate or unordered target members")
    if members is not None and payload["target_video_ids"] != sorted(members):
        raise ValueError("RQ3 cache target membership differs from training")
    for key in ("candidate_sha256", "frames_sha256", "teacher_checkpoint_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("RQ3 cache lacks provenance: " + key)
    if payload.get("teacher_selection") != "source-validation-only":
        raise ValueError("RQ3 proposal teacher must be source-selected SO")
    weights = protocol.get("checkpoint_sha256")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("RQ3 VLM checkpoint hashes are missing")
    for digest in weights.values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("RQ3 VLM checkpoint hash is invalid")
    for key in ("prompt_sha256", "processor_sha256"):
        digest = protocol.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("RQ3 prompt/processor identity is invalid: " + key)
    if not payload.get("entries"):
        raise ValueError("RQ3 cache has no semantic candidates")
    by_video, seen = {}, set()
    for entry in payload["entries"]:
        if set(entry) != {"video_id", "absolute_feature_index", "source_fpn_levels", "class_id", "class_probabilities"}:
            raise ValueError("RQ3 cache entries cannot contain target GT or boundary targets")
        index, levels, probs = entry["absolute_feature_index"], entry["source_fpn_levels"], entry["class_probabilities"]
        if type(index) is not int or index < 0 or not levels or len(set(levels)) != len(levels):
            raise ValueError("RQ3 cache coordinate/levels are invalid")
        if any(type(level) is not int or level not in range(6) for level in levels):
            raise ValueError("RQ3 cache FPN level is invalid")
        if len(probs) != 10 or any(not math.isfinite(value) or value < 0 for value in probs) or abs(sum(probs) - 1) > 1e-5:
            raise ValueError("RQ3 class probabilities must be finite normalized ten-class scores")
        if entry["class_id"] != max(range(10), key=lambda i: probs[i]):
            raise ValueError("RQ3 class id differs from top1")
        if entry["video_id"] not in payload["target_video_ids"]:
            raise ValueError("RQ3 candidate is outside target train")
        key = (entry["video_id"], index)
        if key in seen:
            raise ValueError("duplicate RQ3 candidate coordinate")
        seen.add(key)
        by_video.setdefault(entry["video_id"], []).append(entry)
    return payload, by_video


def configure(cfg, strategy, cache_path, model, shift):
    if strategy == "off":
        if cache_path or model:
            raise ValueError("off RQ3 must not load a teacher/cache")
        return
    if strategy not in STRATEGIES or model not in MODELS:
        raise ValueError("RQ3 strategy/model is invalid")
    if int(cfg["dataset"]["input_dim"]) != 2304 or int(cfg["init_rand_seed"]) != 6:
        raise ValueError("RQ3 requires SlowFast input and seed6")
    payload, _ = load_cache(cache_path, model, shift, members_for(cfg, "tgt"))
    cfg["rq3_auxiliary"] = {
        "schema": "rq3-three-strategies-v1", "strategy": strategy, "teacher_model": model,
        "cache_path": str(Path(cache_path).resolve()), "cache_sha256": sha256_file(cache_path),
        "candidate_sha256": payload["candidate_sha256"], "frames_sha256": payload["frames_sha256"],
        "teacher_checkpoint_sha256": payload["teacher_checkpoint_sha256"],
        "input_protocol": payload["input_protocol"], "levels": [0, 1, 2],
        "budget": 0.05, "prototype_momentum": 0.9, "normalizer": "common-eligible-count",
        "activation": "sada-adaptation-phase", "target_gt_used": False,
    }


def validate_config(cfg, regime):
    auxiliary = cfg.get("rq3_auxiliary")
    if not auxiliary:
        return
    if (regime != "uda" or auxiliary.get("schema") != "rq3-three-strategies-v1"
            or auxiliary.get("strategy") not in STRATEGIES or int(cfg["init_rand_seed"]) != 6
            or int(cfg["dataset"]["input_dim"]) != 2304
            or int(cfg["opt"]["epochs"]) != 50 or int(cfg["loader"]["batch_size"]) != 2):
        raise ValueError("RQ3 requires original matched SADA-UDA, SlowFast, seed6/50epoch/batch2")
    fixed = {"levels": [0, 1, 2], "budget": 0.05, "prototype_momentum": 0.9,
             "normalizer": "common-eligible-count", "activation": "sada-adaptation-phase", "target_gt_used": False}
    if any(auxiliary.get(key) != value for key, value in fixed.items()):
        raise ValueError("RQ3 common strategy protocol changed")
    path = auxiliary["cache_path"]
    if sha256_file(path) != auxiliary["cache_sha256"]:
        raise ValueError("RQ3 cache changed after audit")
    payload, _ = load_cache(path, auxiliary["teacher_model"], cfg["rq2_formal"]["shift"]
                            if cfg.get("rq2_formal") else None, members_for(cfg, "tgt"))
    for key in ("candidate_sha256", "frames_sha256", "teacher_checkpoint_sha256", "input_protocol"):
        if auxiliary[key] != payload[key]:
            raise ValueError("RQ3 cache provenance differs: " + key)


def baseline_signature(cfg):
    from tools.rq2_formal import build_formal_run_signature
    plain = deepcopy(getattr(cfg, "config", cfg))
    plain.pop("rq3_auxiliary", None)
    return build_formal_run_signature(plain)


def audit_baseline(baseline, candidate):
    if baseline.get("rq3_auxiliary") or baseline.get("rq2_formal", {}).get("regime") != "uda":
        raise ValueError("RQ3 baseline must be the completed no-VLM matched UDA")
    if baseline_signature(baseline) != baseline_signature(candidate):
        raise ValueError("RQ3 changed the baseline beyond the auxiliary strategy")
    return {"baseline_signature": baseline_signature(baseline), "only_auxiliary_changed": True}
