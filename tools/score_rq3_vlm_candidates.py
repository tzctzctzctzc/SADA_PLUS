"""真实官方类别读出；默认只测一个共同候选，不启动学生训练。"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import platform

import numpy as np
from PIL import Image

from tools.build_rq3_cache import paired_text_probabilities
from tools.rq2_dense_protocol import sha256_file
from tools.rq3_protocol import (FRAME_PROTOCOL, FROZEN_PROTOCOLS, INTERN_TEXT_SHA256,
                                MODELS, content_hash, read_frozen_protocol)
from tools.run_clip_target_adapter import ACTION_PHRASES

CODE_ROOT = Path(__file__).resolve().parents[1]
OPENAI_CLIP_REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"


def check_code(expected_head):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=CODE_ROOT, text=True)
    if head != expected_head or status.strip():
        raise ValueError("RQ3 interface requires the audited clean code commit")


def idle_gpu_uuid(gpu):
    row = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    uuid, memory, utilization = (part.strip() for part in row)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader",
    ], text=True)
    if int(memory) >= 2048 or int(utilization) > 5 or any(uuid in line for line in processes.splitlines()):
        raise RuntimeError("selected physical GPU is occupied; external jobs are not interrupted")
    return uuid


def openai_clip_root(settings):
    root = Path(settings["clip_root"]).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if head != OPENAI_CLIP_REVISION or status.strip() or not (root / "clip" / "__init__.py").is_file():
        raise ValueError("EgoVideo requires the pinned clean OpenAI CLIP tokenizer source")
    return root


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
        import qwen_vl_utils
        qwen_utils_root = Path(qwen_vl_utils.__file__).resolve().parent
        paths.extend(path for path in qwen_utils_root.rglob("*.py") if path.is_file())
        directories.append(qwen_utils_root)
    elif model == "internvideo2_clip_1b":
        config_path = Path(settings["official_config"])
        paths.append(config_path)
        config = json.loads(config_path.read_text())["model"]
        directories.extend(Path(config[key]) for key in ("tokenizer_path",))
        directories.append(Path(config["text_encoder"]["llama_path"]))
    else:
        root = openai_clip_root(settings)
        paths.extend(path for path in (root / "clip").rglob("*.py") if path.is_file())
        paths.append(root / "clip" / "bpe_simple_vocab_16e6.txt.gz")
    for directory in directories:
        paths.extend(path for path in directory.iterdir() if path.is_file() and path.suffix in (".json", ".model", ".txt", ".jinja"))
    return {str(path.resolve()): sha256_file(path) for path in paths}


class AttributeConfig(dict):
    def __getattr__(self, key):
        return self[key]


def attribute_config(value):
    return AttributeConfig({k: attribute_config(v) for k, v in value.items()}) if isinstance(value, dict) else value


def runtime_versions():
    versions = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "transformers", "peft", "tokenizers", "timm", "sentencepiece"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def check_intern_runtime():
    # 新PEFT改变q/v_proj.weight键名，新Transformers移除了每层rotary缓存。
    # 使用官方旧接口的配套环境，不以strict=False吞掉缺失文本参数。
    versions = runtime_versions()
    expected = {"torch": "1.13.1", "transformers": "4.28.1", "peft": "0.3.0", "tokenizers": "0.13.3"}
    if any((versions.get(key) or "").split("+")[0] != value for key, value in expected.items()):
        raise ValueError("InternVideo2 needs its isolated legacy runtime: " + str(expected))


def import_internvideo_teacher(root):
    """保留官方相对导入层级，只加载所需模型，避免__init__导入无关6B/MobileCLIP。"""
    package_name = "_rq3_official_intern"
    for suffix, relative in (("", ""), (".models", "models"),
                             (".models.backbones", "models/backbones"),
                             (".models.backbones.internvideo2", "models/backbones/internvideo2"),
                             (".utils", "utils")):
        package = types.ModuleType(package_name + suffix)
        package.__path__ = [str(root / relative)]
        sys.modules[package.__name__] = package
    # 与已审计RQ2接口一致：不启用缺失的融合RMSNorm；禁止静默替换其运算。
    try:
        importlib.import_module("flash_attn.ops.rms_norm")
    except ModuleNotFoundError as error:
        if error.name not in ("dropout_layer_norm", "flash_attn", "flash_attn.ops.rms_norm"):
            raise
        import torch.nn as nn

        class UnavailableFusedRMSNorm(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError("RQ3 InternVideo requires non-fused RMSNorm")

        module = types.ModuleType("flash_attn.ops.rms_norm")
        module.DropoutAddRMSNorm = UnavailableFusedRMSNorm
        sys.modules[module.__name__] = module
        if error.name == "flash_attn":
            # 官方视觉文件无条件import这些类，但非融合路径从不实例化它们。
            # 这里只让import成立；若误启用任一融合分支，立即报错而非伪造计算。
            for name, attribute in ((package_name + ".models.backbones.internvideo2.flash_attention_class", "FlashAttention"),
                                    ("flash_attn.modules.mlp", "FusedMLP")):
                unavailable = types.ModuleType(name)
                setattr(unavailable, attribute, UnavailableFusedRMSNorm)
                sys.modules[name] = unavailable
    backbone_name = package_name + ".models.backbones.internvideo2"
    vision = importlib.import_module(backbone_name + ".internvideo2_clip_vision")
    text = importlib.import_module(backbone_name + ".internvideo2_clip_text")
    backbone = sys.modules[backbone_name]
    backbone.InternVideo2, backbone.LLaMA, backbone.Tokenizer = vision.InternVideo2, text.LLaMA, text.Tokenizer
    return importlib.import_module(package_name + ".models.internvideo2_clip")


def load_teacher(model, settings, frozen):
    """只加载官方配对语义接口；dense隐藏向量从不作为十类logits。"""
    import torch
    if model == "egovideo_hod":
        clip_root = openai_clip_root(settings)
        sys.path.insert(0, str(clip_root))
        import clip
        from tools.extract_rq2_dense_vlm_features import EgoVideoEncoder
        native_repository(settings, frozen, "egohod_root")
        encoder = EgoVideoEncoder(types.SimpleNamespace(**dict(settings, device="cuda:0")))
        with torch.inference_mode():
            text = encoder.model.encode_text(clip.tokenize(list(ACTION_PHRASES)).to("cuda:0")).float().cpu().numpy()
        scale = float(encoder.model.logit_scale.exp().detach().cpu())
        return lambda frames, paths, row: paired_text_probabilities(encoder.encode(frames, 1), text, scale)[0]
    if model == "internvideo2_clip_1b":
        check_intern_runtime()
        root = native_repository(settings, frozen, "internvideo_root") / "multi_modality"
        sys.path.insert(0, str(root))
        official = import_internvideo_teacher(root)

        class CheckedInternVideo(official.InternVideo2_CLIP):
            def load_state_dict(self, state, strict=True):
                result = super().load_state_dict(state, strict=False)
                # 官方LLaMA构造器从配置随机初始化，必须真正加载全部文本权重。
                missing = [key for key in result.missing_keys if not key.startswith("vision_encoder.fc_norm.")]
                if missing or result.unexpected_keys:
                    raise ValueError("official paired InternVideo2 load incomplete: " + str(result))
                return result

        config = attribute_config(json.loads(Path(settings["official_config"]).read_text()))
        if any(config.model.vision_encoder.get(key, False) for key in
               ("use_flash_attn", "use_fused_rmsnorm", "use_fused_mlp")):
            raise ValueError("RQ3 InternVideo uses the audited official non-fused vision path")
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
                                     "runtime_versions": runtime_versions(),
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
    audit["runtime_versions"] = runtime_versions()
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
