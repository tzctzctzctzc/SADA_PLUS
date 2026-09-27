"""RQ3缓存身份与基线等价检查；VLM特征不得替换SlowFast学生输入。"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from tools.rq2_dense_protocol import sha256_file
from tools.run_clip_target_adapter import LABELS as TARGET_CLASS_NAMES


SCHEMA = "rq3-candidate-class-cache-v2"
MODELS = ("egovideo_hod", "internvideo2_clip_1b", "qwen3_vl_8b")
STRATEGIES = ("hard", "verify", "prototype")
FRAME_PROTOCOL = "proposal-interior-eight-midpoints-v1"
ACTIVE_LEVELS = (0, 1, 2)
# 官方OpenGVLab/InternVL，revision=851fb933056db175ce98036baec2dd6826cd2c26，LFS实际SHA。
# 文件及配对来源：https://huggingface.co/OpenGVLab/InternVL/blob/851fb933056db175ce98036baec2dd6826cd2c26/internvl_c_13b_224px.pth
INTERN_TEXT_SHA256 = "ce5ae86c0dce4fd0eec09577cf9f6aa1471898c619b8062505e098b58248ad74"
FROZEN_PROTOCOLS = {
    "egovideo_hod": "84156b8bb6e7032102d9dd87385226f51cb5630c4c0d7131cfee6f334c367f2d",
    "internvideo2_clip_1b": "e71b91d6d01718b943a806628e25e89730f01ffde4a3aed7635b9d31f3308f76",
    "qwen3_vl_8b": "e5887af04cdb04811a0c35d335411dfb7beb0ca1e30ee14f4a79974d0cc6fa3e",
}


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
    if protocol.get("readout") != ("closed-class-single-token-log-likelihood" if payload["teacher_model"] == "qwen3_vl_8b"
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
    lengths = payload.get("feature_lengths", {})
    if set(lengths) != set(payload["target_video_ids"]) or any(type(n) is not int or n <= 0 for n in lengths.values()):
        raise ValueError("RQ3 cache lacks full target reference lengths")
    by_video, seen = {}, set()
    for entry in payload["entries"]:
        if set(entry) != {"video_id", "absolute_feature_index", "source_fpn_levels", "class_id", "class_probabilities"}:
            raise ValueError("RQ3 cache entries cannot contain target GT or boundary targets")
        index, levels, probs = entry["absolute_feature_index"], entry["source_fpn_levels"], entry["class_probabilities"]
        if type(index) is not int or index < 0 or not levels or len(set(levels)) != len(levels):
            raise ValueError("RQ3 cache coordinate/levels are invalid")
        if any(type(level) is not int or level not in ACTIVE_LEVELS for level in levels):
            raise ValueError("RQ3 cache FPN level is invalid")
        if len(probs) != 10 or any(not math.isfinite(value) or value < 0 for value in probs) or abs(sum(probs) - 1) > 1e-5:
            raise ValueError("RQ3 class probabilities must be finite normalized ten-class scores")
        if entry["class_id"] != max(range(10), key=lambda i: probs[i]):
            raise ValueError("RQ3 class id differs from top1")
        if entry["video_id"] not in payload["target_video_ids"]:
            raise ValueError("RQ3 candidate is outside target train")
        if index >= lengths[entry["video_id"]]:
            raise ValueError("RQ3 candidate exceeds reference feature length")
        key = (entry["video_id"], index)
        if key in seen:
            raise ValueError("duplicate RQ3 candidate coordinate")
        seen.add(key)
        by_video.setdefault(entry["video_id"], []).append(entry)
    return payload, by_video


def validate_reference_lengths(payload, cfg):
    dataset = cfg["dataset"]
    if int(dataset.get("downsample_rate", 1)) != 1:
        raise ValueError("RQ3 absolute candidate coordinates require original reference grid")
    for video, expected in payload["feature_lengths"].items():
        path = Path(dataset["feat_folder"]) / ((dataset.get("file_prefix") or "") + video + dataset["file_ext"])
        with np.load(path, allow_pickle=False) as data:
            if data["feats"].shape != (expected, 2304):
                raise ValueError("RQ3 reference feature shape/length differs: " + video)


def validate_interface_audit(protocol, model, verify_files=True):
    """正式入口必须绑定真实接口smoke及已冻结的RQ2官方权重身份。"""
    if not protocol.get("interface_audit_path") or not protocol.get("interface_audit_sha256"):
        raise ValueError("RQ3 needs a passed real-model interface audit, not hash formatting alone")
    path = Path(protocol["interface_audit_path"])
    if sha256_file(path) != protocol["interface_audit_sha256"]:
        raise ValueError("RQ3 VLM interface audit changed")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (audit.get("schema") != "rq3-vlm-interface-audit-v1" or audit.get("teacher_model") != model
            or audit.get("smoke_status") != "passed"
            or audit.get("frozen_protocol_sha256") != FROZEN_PROTOCOLS[model]
            or audit.get("checkpoint_sha256") != protocol["checkpoint_sha256"]
            or audit.get("prompt_sha256") != protocol["prompt_sha256"]
            or audit.get("processor_sha256") != protocol["processor_sha256"]
            or audit.get("readout") != protocol["readout"]):
        raise ValueError("RQ3 teacher does not match the approved real-model interface")
    frozen = Path(audit["frozen_protocol_path"])
    from tools.rq2_dense_protocol import protocol_sha256
    original = json.loads(frozen.read_text(encoding="utf-8"))
    if protocol_sha256(original) != FROZEN_PROTOCOLS[model] or original["model_name"] != model:
        raise ValueError("RQ3 official checkpoint anchor is unknown")
    for name, digest in original["checkpoint_sha256"].items():
        if audit["checkpoint_sha256"].get(name) != digest:
            raise ValueError("RQ3 teacher differs from the audited RQ2 official checkpoint")
    if model == "internvideo2_clip_1b" and audit["checkpoint_sha256"].get("internvl_c_13b_224px.pth") != INTERN_TEXT_SHA256:
        raise ValueError("RQ3 InternVideo2 requires the official paired InternVL-C text checkpoint")
    if verify_files:
        if set(audit["checkpoint_files"]) != set(audit["checkpoint_sha256"]):
            raise ValueError("RQ3 official checkpoint file list is incomplete")
        for name, file in audit["checkpoint_files"].items():
            if sha256_file(file) != audit["checkpoint_sha256"][name]:
                raise ValueError("RQ3 actual official weight hash differs: " + name)
        if not audit.get("processor_files"):
            raise ValueError("RQ3 actual processor/tokenizer identity is missing")
        for file, digest in audit["processor_files"].items():
            if sha256_file(file) != digest:
                raise ValueError("RQ3 actual processor/tokenizer changed: " + file)
    return audit


def audit_cache_predictions(payload, audit):
    """概率内容必须来自该真实接口产物；不能把其他教师/编辑后的预测重新签名。"""
    rows = [json.loads(line) for line in Path(audit["predictions_path"]).read_text().splitlines() if line.strip()]
    if (content_hash(rows) != audit["predictions_sha256"]
            or payload.get("predictions_sha256") != audit["predictions_sha256"]
            or payload["candidate_sha256"] != audit["common_manifest_sha256"]):
        raise ValueError("RQ3 cached predictions differ from the real official interface output")
    predicted = {row["candidate_id"]: row["class_probabilities"] for row in rows}
    if len(predicted) != len(rows) or len(rows) != len(payload["entries"]):
        raise ValueError("RQ3 semantic cache candidate coverage differs")
    for entry in payload["entries"]:
        identity = "{}-p{:07d}".format(entry["video_id"], entry["absolute_feature_index"])
        if predicted.get(identity) != entry["class_probabilities"]:
            raise ValueError("RQ3 cached classes/probabilities were edited after official inference")


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
    interface = validate_interface_audit(payload["input_protocol"], model)
    audit_cache_predictions(payload, interface)
    validate_reference_lengths(payload, cfg)
    cfg["rq3_auxiliary"] = {
        "schema": "rq3-three-strategies-v2", "strategy": strategy, "teacher_model": model,
        "cache_path": str(Path(cache_path).resolve()), "cache_sha256": sha256_file(cache_path),
        "candidate_sha256": payload["candidate_sha256"], "frames_sha256": payload["frames_sha256"],
        "teacher_checkpoint_sha256": payload["teacher_checkpoint_sha256"],
        "input_protocol": payload["input_protocol"], "levels": [0, 1, 2],
        "budget": 0.05, "prototype_momentum": 0.9, "normalizer": "common-eligible-count",
        "activation": "sada-adaptation-phase", "target_gt_used": False,
        "coordinate_mapping": "continuous-fpn-linear-v1", "cap_reference": "unfiltered-hard-for-hard-verify-v1",
    }


def validate_config(cfg, regime):
    auxiliary = cfg.get("rq3_auxiliary")
    if not auxiliary:
        return
    if (regime != "uda" or auxiliary.get("schema") != "rq3-three-strategies-v2"
            or auxiliary.get("strategy") not in STRATEGIES or int(cfg["init_rand_seed"]) != 6
            or int(cfg["dataset"]["input_dim"]) != 2304
            or int(cfg["opt"]["epochs"]) != 50 or int(cfg["loader"]["batch_size"]) != 2):
        raise ValueError("RQ3 requires original matched SADA-UDA, SlowFast, seed6/50epoch/batch2")
    fixed = {"levels": [0, 1, 2], "budget": 0.05, "prototype_momentum": 0.9,
             "normalizer": "common-eligible-count", "activation": "sada-adaptation-phase", "target_gt_used": False,
             "coordinate_mapping": "continuous-fpn-linear-v1", "cap_reference": "unfiltered-hard-for-hard-verify-v1"}
    if any(auxiliary.get(key) != value for key, value in fixed.items()):
        raise ValueError("RQ3 common strategy protocol changed")
    path = auxiliary["cache_path"]
    if sha256_file(path) != auxiliary["cache_sha256"]:
        raise ValueError("RQ3 cache changed after audit")
    payload, _ = load_cache(path, auxiliary["teacher_model"], cfg["rq2_formal"]["shift"]
                            if cfg.get("rq2_formal") else None, members_for(cfg, "tgt"))
    interface = validate_interface_audit(payload["input_protocol"], auxiliary["teacher_model"])
    audit_cache_predictions(payload, interface)
    validate_reference_lengths(payload, cfg)
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
