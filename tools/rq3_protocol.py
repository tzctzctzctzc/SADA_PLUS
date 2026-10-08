"""RQ3缓存身份与基线等价检查；VLM特征不得替换SlowFast学生输入。"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from tools.rq2_dense_protocol import read_protocol, sha256_file
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


def read_frozen_protocol(path, model):
    if model not in FROZEN_PROTOCOLS:
        raise ValueError("unknown official RQ2 teacher model")
    payload = read_protocol(path)
    if payload["protocol_sha256"] != FROZEN_PROTOCOLS[model] or payload["model_name"] != model:
        raise ValueError("unknown official RQ2 teacher checkpoint anchor")
    return payload


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
    original = read_frozen_protocol(audit["frozen_protocol_path"], model)
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


def rebuild_verified_cache(cache_path, manifest_path, model, shift):
    """从缓存绑定的原始接口审计定位预测文件，校验后重建缓存用于迁移复用。"""
    cached, _ = load_cache(cache_path, model, shift)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (cached["candidate_sha256"] != content_hash(manifest)
            or cached["target_video_ids"] != manifest["target_video_ids"]
            or cached["feature_lengths"] != manifest["feature_lengths"]):
        raise ValueError("reused RQ3 cache differs from the selected candidate manifest")
    audit = validate_interface_audit(cached["input_protocol"], model)
    audit_cache_predictions(cached, audit)
    predictions = [json.loads(line) for line in Path(audit["predictions_path"]).read_text().splitlines()
                   if line.strip()]
    from tools.build_rq3_cache import build_cache
    rebuilt = build_cache(manifest, predictions, cached["input_protocol"], model, shift,
                          cached["teacher_checkpoint_sha256"])
    if rebuilt != cached:
        raise ValueError("reused RQ3 cache cannot be reproduced from its audited predictions")
    return rebuilt


REVISION = 'rq3-fixed-participation-time-matched-v5'
REVISED_FIXED = {
    'verify_scope': 'anchor-local-center-only-v1',
    'verify_action': 'reject-disagreed-foreground-only-no-relabel-no-background-conversion',
    'prototype_implementation': 'VLMProposalCoAdaptationBranch@24b5dfbe7cd445988d28e400f38a5240be8b1ba0',
    'prototype_source_weight': 1.0, 'prototype_alignment_weight': 0.1,
    'prototype_direct_weight': 0.0, 'prototype_target_start_epoch': 1,
    'prototype_hidden_dim': 256, 'prototype_embedding_dim': 128,
    'confidence_policy': 'fixed-top20-percent-per-shift-v1', 'participation_fraction': 0.2,
    'prototype_crop_policy': 'complete-proposal-only-no-clipping', 'prototype_momentum': 0.9,
    'prototype_max_proposals': 256, 'target_gt_used': False,
}


def proposal_evidence(payload, manifest_path, selected_coordinates=None):
    """把已核验概率与同一教师所见的原始proposal区间精确绑定，不重新推理。"""
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if (content_hash(manifest) != payload['candidate_sha256']
            or manifest.get('target_gt_used') is not False
            or manifest['target_video_ids'] != payload['target_video_ids']
            or manifest['feature_lengths'] != payload['feature_lengths']):
        raise ValueError('RQ3 proposal manifest identity differs')
    candidates = {(row['video_id'], row['absolute_feature_index']): row for row in manifest['candidates']}
    if len(candidates) != len(manifest['candidates']) or len(candidates) != len(payload['entries']):
        raise ValueError('RQ3 proposal coverage/uniqueness differs')
    selected = None if selected_coordinates is None else {tuple(key) for key in selected_coordinates}
    if selected is not None and (len(selected) != len(selected_coordinates) or not selected.issubset(candidates)):
        raise ValueError('RQ3 selected candidate identities differ')
    by_video = {}
    for entry in payload['entries']:
        candidate = candidates[(entry['video_id'], entry['absolute_feature_index'])]
        start, end = candidate['proposal_start_sec'], candidate['proposal_end_sec']
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError('RQ3 proposal interval invalid')
        if candidate['source_fpn_levels'] != entry['source_fpn_levels']:
            raise ValueError('RQ3 proposal FPN identity differs')
        row = dict(entry, proposal_start_sec=start, proposal_end_sec=end,
                   candidate_id=candidate['candidate_id'],
                   evidence_scope=manifest.get('evidence_scope', 'proposal-whole-v1'),
                   semantic_label=TARGET_CLASS_NAMES[entry['class_id']])
        if selected is not None:
            row['participation_selected'] = (entry['video_id'], entry['absolute_feature_index']) in selected
        by_video.setdefault(entry['video_id'], []).append(row)
    return by_video


def configure(cfg, strategy, cache_path, model, shift, revision='legacy', manifest_path=None, calibration_path=None):
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
    if revision == 'revised':
        if strategy not in ('verify', 'prototype') or not manifest_path or not calibration_path:
            raise ValueError('revised RQ3 requires Verify/Prototype, matched manifest and source calibration')
        from tools.rq3_confidence import load_calibration, select_budget, BUDGET_SCHEMA
        calibration = load_calibration(calibration_path, cfg, payload,
                                       json.loads(Path(manifest_path).read_text()), strategy)
        if calibration['schema'] != BUDGET_SCHEMA:
            raise ValueError('fixed participation requires new source diagnostics')
        selected = select_budget(payload['entries'])
        if not selected:
            raise ValueError('fixed participation has no unambiguous target candidates')
        proposal_evidence(payload, manifest_path)
        legacy = cfg['rq3_auxiliary']
        cfg['rq3_auxiliary'] = {key: legacy[key] for key in (
            'strategy', 'teacher_model', 'cache_path', 'cache_sha256', 'candidate_sha256',
            'frames_sha256', 'teacher_checkpoint_sha256', 'input_protocol')}
        cfg['rq3_auxiliary'].update(REVISED_FIXED, schema=REVISION,
                                   manifest_path=str(Path(manifest_path).resolve()),
                                   manifest_sha256=sha256_file(manifest_path),
                                   calibration_path=str(Path(calibration_path).resolve()),
                                   calibration_sha256=sha256_file(calibration_path),
                                   confidence_threshold=calibration['threshold'],
                                   selected_coordinates=selected)
    elif revision != 'legacy':
        raise ValueError('unknown RQ3 revision')
    elif calibration_path:
        raise ValueError('legacy/Hard cannot load revised confidence calibration')


def validate_config(cfg, regime):
    from tools.rq3_accelerated import valid_batch
    auxiliary = cfg.get("rq3_auxiliary")
    if not auxiliary:
        return
    revised = auxiliary.get('schema') == REVISION
    if (regime != "uda" or auxiliary.get("schema") not in ("rq3-three-strategies-v2", REVISION)
            or auxiliary.get("strategy") not in STRATEGIES or int(cfg["init_rand_seed"]) != 6
            or int(cfg["dataset"]["input_dim"]) != 2304
            or int(cfg["opt"]["epochs"]) != 50 or not valid_batch(cfg)):
        raise ValueError("RQ3 requires original matched SADA-UDA, SlowFast, seed6/50epoch/batch2")
    fixed = {"levels": [0, 1, 2], "budget": 0.05, "prototype_momentum": 0.9,
             "normalizer": "common-eligible-count", "activation": "sada-adaptation-phase", "target_gt_used": False,
             "coordinate_mapping": "continuous-fpn-linear-v1", "cap_reference": "unfiltered-hard-for-hard-verify-v1"}
    if revised:
        fixed = REVISED_FIXED
        if auxiliary['strategy'] not in ('verify', 'prototype'):
            raise ValueError('Hard must remain on the original protocol')
        if sha256_file(auxiliary['manifest_path']) != auxiliary['manifest_sha256']:
            raise ValueError('RQ3 original proposal manifest changed')
    if any(auxiliary.get(key) != value for key, value in fixed.items()):
        raise ValueError("RQ3 common strategy protocol changed")
    path = auxiliary["cache_path"]
    if sha256_file(path) != auxiliary["cache_sha256"]:
        raise ValueError("RQ3 cache changed after audit")
    payload, _ = load_cache(path, auxiliary["teacher_model"], cfg["rq2_formal"]["shift"]
                            if cfg.get("rq2_formal") else None, members_for(cfg, "tgt"))
    if revised:
        proposal_evidence(payload, auxiliary['manifest_path'])
        from tools.rq3_confidence import load_calibration, select_budget, BUDGET_SCHEMA
        if sha256_file(auxiliary['calibration_path']) != auxiliary['calibration_sha256']:
            raise ValueError('RQ3 source calibration changed')
        calibrated = load_calibration(auxiliary['calibration_path'], cfg, payload,
                                      json.loads(Path(auxiliary['manifest_path']).read_text()), auxiliary['strategy'])
        if auxiliary['confidence_threshold'] != calibrated['threshold']:
            raise ValueError('RQ3 threshold differs from source calibration')
        if (calibrated['schema'] != BUDGET_SCHEMA
                or auxiliary['selected_coordinates'] != select_budget(payload['entries'])):
            raise ValueError('RQ3 fixed participation selection changed')
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
    from tools.rq3_early_stop import enabled
    early = enabled(candidate)
    if baseline.get("rq3_auxiliary") or baseline.get("rq2_formal", {}).get("regime") != "uda":
        raise ValueError("RQ3 baseline must be the completed no-VLM matched UDA")
    acceleration = candidate.get('rq3_auxiliary', {}).get('acceleration')
    if acceleration:
        from tools.rq3_accelerated import reference_config
        normalized = reference_config(candidate, baseline)
    else:
        normalized = candidate
    if baseline_signature(baseline) != baseline_signature(normalized):
        raise ValueError("RQ3 changed the baseline beyond the auxiliary strategy")
    return {"baseline_signature": baseline_signature(baseline), "only_auxiliary_changed": not bool(acceleration or early),
            **({'stopping_budget_matched': False, 'early_stopping': candidate['rq3_auxiliary']['early_stopping']} if early else {}),
            **({'acceleration': acceleration, 'batch_matched': False} if acceleration else {})}
