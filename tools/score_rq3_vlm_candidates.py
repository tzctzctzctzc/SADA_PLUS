"""真实官方类别读出；默认只测一个共同候选，不启动学生训练。"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image

from tools.build_rq3_cache import paired_text_probabilities
from tools.rq2_dense_protocol import sha256_file
from tools.rq3_protocol import (FRAME_PROTOCOL, FROZEN_PROTOCOLS, INTERN_TEXT_SHA256,
                                MODELS, content_hash, read_frozen_protocol)
from tools.run_clip_target_adapter import ACTION_PHRASES


def pixel_hash(image):
    image = np.asarray(image, dtype=np.uint8)
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def check_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (10,) or not np.isfinite(scores).all() or (scores < 0).any() or abs(scores.sum() - 1) > 1e-5:
        raise ValueError("official VLM readout did not return normalized finite ten-class probabilities")
    return scores.tolist()


def checkpoint_files(model, settings, frozen):
    """逐文件真实SHA；不把软链接文件名、维度或调用者自报hash当官方身份。"""
    paths = {}
    for name in frozen["checkpoint_sha256"]:
        if model == "qwen3_vl_8b":
            path = Path(settings["model_path"]) / name
        else:
            path = Path(settings["checkpoint_files"][name])
        if sha256_file(path) != frozen["checkpoint_sha256"][name]:
            raise ValueError("actual official checkpoint differs: " + name)
        paths[name] = str(path.resolve())
    if model == "internvideo2_clip_1b":
        path = Path(settings["text_checkpoint"])
        if sha256_file(path) != INTERN_TEXT_SHA256:
            raise ValueError("InternVideo2 paired text weights are unknown; no generic CLIP substitution")
        paths["internvl_c_13b_224px.pth"] = str(path.resolve())
    return paths


def native_repository(settings, frozen, key):
    from tools.extract_rq2_dense_vlm_features import git_provenance
    root = Path(settings[key]).resolve()
    actual = git_provenance(root)
    expected = frozen["provenance"]["external_repository"]
    if not actual["revision"] or actual != expected or actual["tracked_diff_sha256"] != hashlib.sha256(b"").hexdigest():
        raise ValueError("official VLM repository differs from the audited RQ2 code")
    return root


def processor_files(model, settings):
    paths = [Path(__file__).resolve(), Path(__file__).with_name("run_qwen3_vl_classification.py").resolve(),
             Path(__file__).with_name("extract_rq2_dense_vlm_features.py").resolve(),
             Path(__file__).with_name("extract_rq1_egovideo_features.py").resolve()]
    directories = []
    if model == "qwen3_vl_8b":
        directories.append(Path(settings["model_path"]))
    elif model == "internvideo2_clip_1b":
        config_path = Path(settings["official_config"])
        paths.append(config_path)
        config = json.loads(config_path.read_text())["model"]
        directories.extend(Path(config[key]) for key in ("tokenizer_path",))
        directories.append(Path(config["text_encoder"]["llama_path"]))
    else:
        import clip
        paths.append(Path(clip.__file__).parent / "bpe_simple_vocab_16e6.txt.gz")
    for directory in directories:
        paths.extend(path for path in directory.iterdir() if path.is_file() and path.suffix in (".json", ".model", ".txt", ".jinja"))
    return {str(path.resolve()): sha256_file(path) for path in paths}


class AttributeConfig(dict):
    def __getattr__(self, key):
        return self[key]


def attribute_config(value):
    return AttributeConfig({k: attribute_config(v) for k, v in value.items()}) if isinstance(value, dict) else value


def load_teacher(model, settings, frozen):
    """只加载官方配对语义接口；dense隐藏向量从不作为十类logits。"""
    import torch
    if model == "egovideo_hod":
        from tools.extract_rq2_dense_vlm_features import EgoVideoEncoder
        import clip
        native_repository(settings, frozen, "egohod_root")
        encoder = EgoVideoEncoder(types.SimpleNamespace(**dict(settings, device="cuda:0")))
        with torch.inference_mode():
            text = encoder.model.encode_text(clip.tokenize(list(ACTION_PHRASES)).to("cuda:0")).float().cpu().numpy()
        scale = float(encoder.model.logit_scale.exp().detach().cpu())
        return lambda frames, paths, row: paired_text_probabilities(encoder.encode(frames, 1), text, scale)[0]
    if model == "internvideo2_clip_1b":
        root = native_repository(settings, frozen, "internvideo_root") / "multi_modality"
        sys.path.insert(0, str(root))
        package = types.ModuleType("_rq3_official_intern_models")
        package.__path__ = [str(root / "models")]
        sys.modules[package.__name__] = package
        official = importlib.import_module(package.__name__ + ".internvideo2_clip")

        class CheckedInternVideo(official.InternVideo2_CLIP):
            def load_state_dict(self, state, strict=True):
                result = super().load_state_dict(state, strict=False)
                # 官方LLaMA构造器从配置随机初始化，必须真正加载全部文本权重。
                missing = [key for key in result.missing_keys if not key.startswith("vision_encoder.fc_norm.")]
                if missing or result.unexpected_keys:
                    raise ValueError("official paired InternVideo2 load incomplete: " + str(result))
                return result

        config = attribute_config(json.loads(Path(settings["official_config"]).read_text()))
        if (config.model.vision_encoder.num_frames != 8 or config.model.vision_encoder.img_size != 224
                or config.model.vision_encoder.clip_embed_dim != 768
                or Path(config.model.text_ckpt_path).resolve() != Path(settings["text_checkpoint"]).resolve()):
            raise ValueError("InternVideo2 official config/frame/text identity differs")
        encoder = CheckedInternVideo(config).eval().to("cuda:0").bfloat16()
        with torch.inference_mode():
            text = encoder.encode_text(encoder.tokenizer(list(ACTION_PHRASES)).to("cuda:0")).float().cpu().numpy()
        scale = 1 / float(encoder.temp.detach().cpu())

        def score(frames, paths, row):
            tensors = [encoder.transform(torch.from_numpy(frame.copy()).permute(2, 0, 1)) for frame in frames]
            videos = torch.stack(tensors)[None].to("cuda:0", dtype=torch.bfloat16)
            with torch.inference_mode():
                video = encoder.encode_vision(videos).float().cpu().numpy()
            return paired_text_probabilities(video, text, scale)[0]
        return score
    from tools.run_qwen3_vl_classification import infer_one, load_qwen3
    args = types.SimpleNamespace(model_path=settings["model_path"], device="cuda:0", offline=True,
                                 dtype="bfloat16", attn_implementation="sdpa", device_map="none",
                                 frame_count=8, prompt_mode="baseline", input_mode="native_video",
                                 min_pixels=224 * 224, max_pixels=448 * 448, score_temperature=1.)
    runtime = load_qwen3(args)

    def score(frames, paths, row):
        # 帧均为proposal内部八等分的中点；相邻间隔是duration/8，不是duration/7。
        fps = 8 / (row["proposal_end_sec"] - row["proposal_start_sec"])
        result = infer_one(runtime, paths, row["candidate_id"], args, sample_fps=fps)
        return result["class_probabilities"]
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--settings", type=Path, required=True, help="官方模型本地路径与配对配置，不下载权重")
    parser.add_argument("--frozen-protocol", type=Path, required=True, help="已完成RQ2的protocol.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="完整缓存预测；默认只有单候选接口smoke")
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    from tools.run_rq2_matched_control import check_code, idle_gpu_uuid
    check_code(args.expected_head)
    frozen = read_frozen_protocol(args.frozen_protocol, args.model)
    settings = json.loads(args.settings.read_text())
    files = checkpoint_files(args.model, settings, frozen)
    # 每个构造器实际加载的文件也必须在已审计的列表内。
    if args.model == "egovideo_hod" and str(Path(settings["checkpoint"]).resolve()) not in files.values():
        raise ValueError("EgoVideo constructor would load unaudited weights")
    if args.model == "internvideo2_clip_1b":
        config = json.loads(Path(settings["official_config"]).read_text())["model"]
        if {str(Path(config[key]).resolve()) for key in ("vision_ckpt_path", "text_ckpt_path", "extra_ckpt_path")} != set(files.values()):
            raise ValueError("InternVideo2 constructor checkpoint list differs")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("target_gt_used") is not False or manifest.get("frame_protocol") != FRAME_PROTOCOL:
        raise ValueError("candidate manifest is not the class-blind common protocol")
    candidates = manifest["candidates"]
    if not candidates:
        raise ValueError("no real proposal available for interface smoke")
    from tools.run_qwen3_vl_classification import build_prompt
    prompts = ([build_prompt(row["candidate_id"], 8, "baseline", "native_video") for row in candidates]
               if args.model == "qwen3_vl_8b" else list(ACTION_PHRASES))
    prompt_digest = content_hash(prompts)
    assets = processor_files(args.model, settings)
    processor_digest = content_hash({"settings": settings, "processor_files": assets,
                                     "frozen_protocol": FROZEN_PROTOCOLS[args.model],
                                     "qwen_classifier": sha256_file(Path(__file__).with_name("run_qwen3_vl_classification.py"))})
    # 锁和实际空闲检查在模型加载前；只用一张物理GPU3。
    import fcntl
    with Path("/vepfs/vlms/SADA_PLUS/.rq2_gpu3.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ["CUDA_VISIBLE_DEVICES"] = idle_gpu_uuid(3)
        scorer = load_teacher(args.model, settings, frozen)
        predictions = []
        for row in candidates if args.full else candidates[:1]:
            paths = [args.frame_root / row["candidate_id"] / (str(index) + ".png") for index in row["frames"]]
            frames = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
            if len(frames) != 8 or [pixel_hash(frame) for frame in frames] != row["frame_sha256"]:
                raise ValueError("teacher did not receive the exact eight frozen RGB frames")
            scores = check_scores(scorer(frames, paths, row))
            predictions.append({"candidate_id": row["candidate_id"], "frame_sha256": content_hash(row["frame_sha256"]),
                                "class_probabilities": scores})
    checkpoint_files(args.model, settings, frozen)
    if processor_files(args.model, settings) != assets:
        raise ValueError("VLM processor/config changed during interface verification")
    args.output_root.mkdir(parents=True, exist_ok=False)
    audit = {"schema": "rq3-vlm-interface-audit-v1", "teacher_model": args.model, "smoke_status": "passed",
             "frozen_protocol_path": str(args.frozen_protocol.resolve()), "frozen_protocol_sha256": FROZEN_PROTOCOLS[args.model],
             "checkpoint_files": files, "checkpoint_sha256": {name: sha256_file(file) for name, file in files.items()},
             "readout": "closed-class-single-token-log-likelihood" if args.model == "qwen3_vl_8b" else "official-paired-video-text-cosine",
             "prompt_sha256": prompt_digest, "processor_sha256": processor_digest,
             "input_frames": 8, "class_blind": True, "frame_protocol": FRAME_PROTOCOL,
             "scored_candidates": len(predictions), "common_manifest_sha256": content_hash(manifest),
             "processor_files": assets, "predictions_sha256": content_hash(predictions),
             "predictions_path": str((args.output_root / "predictions.jsonl").resolve()),
             "single_token_semantics": args.model == "qwen3_vl_8b", "code_commit": args.expected_head}
    path = args.output_root / "interface_audit.json"
    path.write_text(json.dumps(audit, sort_keys=True))
    protocol = {key: audit[key] for key in ("readout", "checkpoint_sha256", "prompt_sha256", "processor_sha256",
                                           "input_frames", "class_blind", "frame_protocol")}
    protocol.update(interface_audit_path=str(path.resolve()), interface_audit_sha256=sha256_file(path))
    (args.output_root / "input_protocol.json").write_text(json.dumps(protocol, sort_keys=True))
    (args.output_root / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
    print("real official VLM interface passed: {} candidates; no student training".format(len(predictions)))


if __name__ == "__main__":
    main()
