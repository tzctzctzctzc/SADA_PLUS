"""Extract frozen full-video VLM features on the shared RQ2 temporal grid."""

import argparse
import hashlib
import importlib.metadata
import importlib
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import yaml

from tools.rq2_dense_protocol import (
    FEAT_STRIDE,
    FEATURE_FPS,
    MODEL_SPECS,
    NEUTRAL_MULTIMODAL_PROMPT,
    RECEPTIVE_FIELD_FRAMES,
    TIMEPLE_NEUTRAL_PROMPT,
    atomic_savez,
    expected_centers_seconds,
    protocol_payload,
    protocol_sha256,
    reference_temporal_length_is_compatible,
    sampled_source_indices,
    sha256_file,
    sha256_normalized_text,
    temporal_length,
    validate_feature_file,
    video_set_sha256,
)


class SequentialFrameReader:
    def __init__(self, path):
        self.path = Path(path)
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError("cannot open video: {}".format(path))
        self.position = 0

    def close(self):
        self.capture.release()

    def _reopen_at(self, index):
        self.capture.release()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError("cannot reopen video: {}".format(self.path))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.position = index

    def read(self, indices):
        requested = np.asarray(indices, dtype=np.int64).reshape(-1)
        # Adjacent temporal tokens overlap. Decode each source frame once per
        # batch, then restore the requested order for the model input.
        unique_indices = np.unique(requested)
        cache = {}
        for index in unique_indices.tolist():
            index = int(index)
            if index < self.position:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                self.position = index
            while self.position < index:
                if not self.capture.grab():
                    # Some official EPIC-KITCHENS videos contain damaged
                    # packets between valid sampled frames. Seek directly to
                    # the requested frame instead of changing the sample.
                    self._reopen_at(index)
                    break
                self.position += 1
            ok, frame = self.capture.read()
            if not ok:
                # Long videos can occasionally leave OpenCV's decoder in a bad
                # state after repeated backward seeks. Reopen once at the exact
                # requested frame before treating the source video as corrupt.
                self._reopen_at(index)
                ok, frame = self.capture.read()
                if not ok:
                    raise RuntimeError("failed to decode frame {}".format(index))
            self.position += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cache[index] = rgb
        return [cache[int(index)] for index in requested.tolist()]


class EgoVideoEncoder:
    def __init__(self, args):
        import torch
        from PIL import Image
        from tools.extract_rq1_egovideo_features import image_transform, load_model

        self.torch = torch
        self.Image = Image
        self.device = args.device
        self.transform = image_transform()
        self.model = load_model(args.egohod_root, args.checkpoint, args.device)

    def encode(self, frames, batch_tokens):
        videos = []
        for offset in range(0, len(frames), 8):
            clip = [
                self.transform(self.Image.fromarray(frame))
                for frame in frames[offset:offset + 8]
            ]
            videos.append(self.torch.stack(clip, dim=1))
        batch = self.torch.stack(videos).to(
            self.device, dtype=self.torch.float16, non_blocking=True
        )
        with self.torch.inference_mode():
            features = self.model.encode_image(batch)[0].float()
        return normalized_numpy(features)


class InternVideoEncoder:
    def __init__(self, args):
        import torch

        module_root = (
            Path(args.internvideo_root).resolve()
            / "multi_modality" / "models" / "backbones" / "internvideo2"
        )
        package_name = "_rq2_internvideo2"
        package = types.ModuleType(package_name)
        package.__path__ = [str(module_root)]
        sys.modules[package_name] = package

        try:
            importlib.import_module("flash_attn.ops.rms_norm")
        except ModuleNotFoundError:
            import torch.nn as nn

            class UnavailableFusedRMSNorm(nn.Module):
                def __init__(self, *unused_args, **unused_kwargs):
                    super().__init__()
                    raise RuntimeError("fused RMSNorm is unavailable")

            fallback = types.ModuleType("flash_attn.ops.rms_norm")
            fallback.DropoutAddRMSNorm = UnavailableFusedRMSNorm
            sys.modules["flash_attn.ops.rms_norm"] = fallback

        vision_module = importlib.import_module(
            package_name + ".internvideo2_clip_vision"
        )
        position_module = importlib.import_module(package_name + ".pos_embed")
        InternVideo2 = vision_module.InternVideo2
        interpolate_pos_embed = position_module.interpolate_pos_embed

        self.torch = torch
        self.device = args.device
        self.model = InternVideo2(
            in_chans=3,
            patch_size=14,
            img_size=224,
            qkv_bias=False,
            drop_path_rate=0.3,
            head_drop_path_rate=0.0,
            embed_dim=1408,
            num_heads=16,
            mlp_ratio=48 / 11,
            init_values=0.1,
            qk_normalization=True,
            depth=40,
            use_flash_attn=False,
            use_fused_rmsnorm=False,
            use_fused_mlp=False,
            fused_mlp_heuristic=1,
            attn_pool_num_heads=16,
            clip_embed_dim=768,
            layerscale_no_force_fp32=True,
            num_frames=8,
            tubelet_size=1,
            sep_pos_embed=False,
            use_checkpoint=False,
            checkpoint_num=0,
        )
        stage = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=False)
        stage = stage.get("module", stage.get("model", stage))
        interpolate_pos_embed(stage, self.model, orig_t_size=4)
        state = {}
        model_keys = self.model.state_dict()
        for key, value in stage.items():
            if not key.startswith("vision_encoder."):
                continue
            name = key.removeprefix("vision_encoder.")
            if any(part in name for part in (
                "clip_decoder", "final_clip_decoder", "clip_pos_embed",
                "clip_img_pos_embed", "img_pos_embed",
            )):
                continue
            if name in model_keys and model_keys[name].shape == value.shape:
                state[name] = value
        extra = torch.load(args.clip_checkpoint, map_location="cpu", weights_only=False)
        extra = extra.get("module", extra.get("model", extra))
        for key, value in extra.items():
            if key.startswith("vision_encoder."):
                name = key.removeprefix("vision_encoder.")
                if name in model_keys and model_keys[name].shape == value.shape:
                    state[name] = value
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        critical_missing = [
            key for key in missing
            if not key.startswith("fc_norm.")
        ]
        if critical_missing or unexpected:
            raise RuntimeError(
                "InternVideo2 checkpoint load incomplete: missing={} unexpected={}".format(
                    critical_missing, unexpected
                )
            )
        self.model = self.model.eval().to(args.device).bfloat16()

    def encode(self, frames, batch_tokens):
        import torch.nn.functional as functional

        array = np.stack(frames).reshape(batch_tokens, 8, *frames[0].shape)
        tensor = self.torch.from_numpy(array).permute(0, 1, 4, 2, 3).float()
        tensor = tensor.reshape(batch_tokens * 8, 3, *tensor.shape[-2:])
        tensor = functional.interpolate(
            tensor, size=(224, 224), mode="bicubic", align_corners=False
        ).div_(255.0)
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        tensor = tensor.sub_(mean).div_(std)
        tensor = tensor.reshape(batch_tokens, 8, 3, 224, 224).permute(0, 2, 1, 3, 4)
        tensor = tensor.to(self.device, dtype=self.torch.bfloat16, non_blocking=True)
        with self.torch.inference_mode():
            features = self.model(tensor).float()
        return normalized_numpy(features)


def _video_messages(clips, prompt, min_pixels, max_pixels, assistant_text=None):
    messages = []
    sample_fps = (
        (MODEL_SPECS["qwen3_vl_8b"]["input_frames"] - 1)
        * FEATURE_FPS
        / (RECEPTIVE_FIELD_FRAMES - 1)
    )
    for clip in clips:
        user = {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": clip,
                    "sample_fps": sample_fps,
                    "raw_fps": sample_fps,
                    "min_pixels": int(min_pixels),
                    "max_pixels": int(max_pixels),
                },
                {"type": "text", "text": prompt},
            ],
        }
        item = [user]
        if assistant_text is not None:
            item.append({"role": "assistant", "content": assistant_text})
        messages.append(item)
    return messages


def _process_video_messages(runtime, messages, add_generation_prompt):
    texts = [
        runtime["processor"].apply_chat_template(
            item,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        for item in messages
    ]
    image_inputs, video_inputs, video_kwargs = runtime["process_vision_info"](
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if video_inputs is not None:
        video_inputs, video_metadata = zip(*video_inputs)
        video_inputs = list(video_inputs)
        video_metadata = list(video_metadata)
    else:
        video_metadata = None
    return runtime["processor"](
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadata,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs
    ).to(runtime["device"])


class MultimodalLastTokenEncoder:
    def __init__(self, args):
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not args.model_repository:
            raise ValueError("multimodal models require --model-repository")
        dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
        kwargs = {
            "dtype": dtype,
            "attn_implementation": args.attn_implementation,
            "local_files_only": args.offline,
        }
        if args.device_map != "none":
            kwargs["device_map"] = args.device_map
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, **kwargs
        )
        if args.device_map == "none":
            model.to(args.device)
        model.eval()
        self.torch = torch
        self.Image = Image
        self.runtime = {
            "torch": torch,
            "model": model,
            "processor": AutoProcessor.from_pretrained(
                args.model_path,
                local_files_only=args.offline,
                fix_mistral_regex=True,
            ),
            "process_vision_info": process_vision_info,
            "device": model.device,
        }

    def encode(self, frames, batch_tokens):
        runtime = self.runtime
        clips = [
            [self.Image.fromarray(frame) for frame in frames[offset:offset + 8]]
            for offset in range(0, len(frames), 8)
        ]
        messages = _video_messages(
            clips,
            NEUTRAL_MULTIMODAL_PROMPT,
            224 * 224,
            448 * 448,
        )
        inputs = _process_video_messages(runtime, messages, True)
        with runtime["torch"].inference_mode():
            output = runtime["model"](
                **inputs,
                use_cache=False,
                logits_to_keep=1,
                output_hidden_states=True,
                return_dict=True,
            )
        if not output.hidden_states:
            raise RuntimeError("model did not return multimodal hidden states")
        attention_mask = inputs["attention_mask"].bool()
        positions = runtime["torch"].arange(
            attention_mask.shape[1], device=attention_mask.device
        ).unsqueeze(0).expand_as(attention_mask)
        last_positions = positions.masked_fill(~attention_mask, -1).max(dim=1).values
        batch_indices = runtime["torch"].arange(
            batch_tokens, device=last_positions.device
        )
        features = output.hidden_states[-1][batch_indices, last_positions].float()
        return normalized_numpy(features)


class QwenVisionEncoder:
    def __init__(self, args):
        from PIL import Image
        from tools.run_qwen3_vl_classification import load_qwen3

        self.Image = Image
        self.runtime = load_qwen3(args)

    def encode(self, frames, batch_tokens):
        runtime = self.runtime
        images = [self.Image.fromarray(frame) for frame in frames]
        inputs = runtime["processor"].image_processor(
            images=images,
            min_pixels=224 * 224,
            max_pixels=448 * 448,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(runtime["device"])
        image_grid_thw = inputs["image_grid_thw"].to(runtime["device"])
        with runtime["torch"].inference_mode():
            embeddings, _ = runtime["model"].get_image_features(
                pixel_values, image_grid_thw
            )
        per_frame = runtime["torch"].stack(
            [embedding.float().mean(dim=0) for embedding in embeddings]
        )
        return normalized_numpy(per_frame.reshape(batch_tokens, 8, -1).mean(dim=1))


class TimePLEEncoder:
    def __init__(self, args):
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        if not args.timeple_root:
            raise ValueError("TimePLE requires --timeple-root")
        source_root = str(Path(args.timeple_root).resolve() / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        import timeple  # noqa: F401
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
        kwargs = {
            "dtype": dtype,
            "attn_implementation": args.attn_implementation,
            "local_files_only": args.offline,
        }
        if args.device_map != "none":
            kwargs["device_map"] = args.device_map
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, **kwargs
        )
        if args.device_map == "none":
            model.to(args.device)
        model.eval()
        if not getattr(model, "use_timeple_codec", False):
            raise RuntimeError("checkpoint does not enable the TimePLE codec")
        self.Image = Image
        self.runtime = {
            "torch": torch,
            "model": model,
            "processor": AutoProcessor.from_pretrained(
                args.model_path,
                local_files_only=args.offline,
                fix_mistral_regex=True,
            ),
            "process_vision_info": process_vision_info,
            "device": model.device,
        }

    def encode(self, frames, batch_tokens):
        runtime = self.runtime
        clips = [
            [self.Image.fromarray(frame) for frame in frames[offset:offset + 8]]
            for offset in range(0, len(frames), 8)
        ]
        messages = _video_messages(
            clips,
            TIMEPLE_NEUTRAL_PROMPT,
            224 * 224,
            448 * 448,
            assistant_text="<|TIMESPAN|>",
        )
        inputs = _process_video_messages(runtime, messages, False)
        with runtime["torch"].inference_mode():
            output = runtime["model"](
                **inputs,
                use_cache=False,
                logits_to_keep=1,
                compute_timeple_loss=False,
                output_hidden_states=True,
                return_dict=True,
            )
        with runtime["torch"].inference_mode():
            hidden = output.hidden_states[-1]
            token_id = int(runtime["model"].config.timespan_token_id)
            matches = inputs["input_ids"].eq(token_id) & inputs["attention_mask"].bool()
            if not runtime["torch"].all(matches.sum(dim=1) == 1):
                raise RuntimeError("expected exactly one TimePLE timespan token per clip")
            positions = matches.float().argmax(dim=1)
            rows = runtime["torch"].arange(batch_tokens, device=hidden.device)
            features = hidden[rows, positions]
            model = runtime["model"]
            if getattr(model, "use_timeple_interface_adapter", False):
                features = model.timeple_interface_adapter.forward_output(
                    features,
                    compute_diagnostics=False,
                ).adapted
            trunk = model.timeple_codec.decoder.trunk
            trunk_dtype = next(trunk.parameters()).dtype
            features = trunk(features.to(trunk_dtype)).float()
        return normalized_numpy(features)


def normalized_numpy(features):
    features = features.float()
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.detach().cpu().numpy().astype(np.float16)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--reference-feature-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shift-registry")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-tokens", type=int, default=8)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--checkpoint")
    parser.add_argument("--egohod-root")
    parser.add_argument("--internvideo-root")
    parser.add_argument("--stage2-checkpoint")
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model-repository")
    parser.add_argument("--timeple-root")
    parser.add_argument("--assignment-file")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device-map", choices=("none", "auto"), default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


def checkpoint_hashes(args):
    paths = []
    if args.model == "egovideo_hod":
        paths = [args.checkpoint]
    elif args.model == "internvideo2_clip_1b":
        paths = [args.stage2_checkpoint, args.clip_checkpoint]
    elif args.model in (
        "qwen3_vl_8b", "timelens2_8b", "timeple_8b",
        "qwen3_vl_8b_vision_tower",
    ):
        model_root = Path(args.model_path)
        if not model_root.is_dir():
            raise FileNotFoundError(
                "Qwen RQ2 extraction requires a local model directory: {}".format(
                    model_root
                )
            )
        paths = sorted(model_root.glob("*.safetensors"))
        paths += sorted(model_root.glob("*.json"))
        if not paths:
            raise FileNotFoundError("Qwen model directory contains no weights")
    for path in paths:
        if not path or not Path(path).is_file():
            raise FileNotFoundError("required checkpoint is missing: {}".format(path))
    return {Path(path).name: checkpoint_fingerprint(path) for path in paths}


def checkpoint_fingerprint(path):
    path = Path(path)
    if path.is_symlink():
        blob_name = Path(os.readlink(path)).name
        if re.fullmatch(r"[0-9a-f]{64}", blob_name):
            return blob_name
    return sha256_file(path)


def git_provenance(path):
    try:
        code_revision = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "-C", str(path), "diff", "--no-ext-diff", "--binary", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "tracked_diff_sha256": None}
    return {
        "revision": code_revision,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def extraction_provenance(args):
    packages = {}
    for name in ("torch", "opencv-python", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    runtime = {
        "egovideo_hod": {
            "device_map": "none",
            "attention": "model_native",
            "dtype": "float16",
        },
        "internvideo2_clip_1b": {
            "device_map": "none",
            "attention": "official_torch_unfused_ops",
            "dtype": "bfloat16",
        },
        "qwen3_vl_8b": {
            "device_map": args.device_map,
            "attention": args.attn_implementation,
            "dtype": args.dtype,
            "fix_mistral_regex": True,
        },
        "timelens2_8b": {
            "device_map": args.device_map,
            "attention": args.attn_implementation,
            "dtype": args.dtype,
            "fix_mistral_regex": True,
        },
        "timeple_8b": {
            "device_map": args.device_map,
            "attention": args.attn_implementation,
            "dtype": args.dtype,
            "fix_mistral_regex": True,
        },
        "qwen3_vl_8b_vision_tower": {
            "device_map": args.device_map,
            "attention": args.attn_implementation,
            "dtype": args.dtype,
        },
    }[args.model]
    return {
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
        "repository": git_provenance(Path(__file__).resolve().parents[1]),
        "external_repository": git_provenance(
            args.egohod_root or args.internvideo_root or args.model_repository
            or args.timeple_root or args.model_path
        ),
        "batch_tokens": args.batch_tokens,
        **runtime,
        "packages": packages,
    }


def video_selection_from_registry(path):
    registry_path = Path(path).resolve()
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    repository_root = (registry_path.parent / registry["repository_root"]).resolve()
    subsets = set()
    annotation_paths = set()
    config_hashes = {}
    for shift_id, entry in registry["shifts"].items():
        config_path = repository_root / entry["config"]
        actual_hash = sha256_normalized_text(config_path)
        if actual_hash != entry.get("config_sha256"):
            raise ValueError("{} config hash mismatch".format(shift_id))
        config_hashes[shift_id] = {
            "path": entry["config"],
            "sha256": actual_hash,
        }
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for key in ("src_train_split", "src_val_split", "tgt_train_split", "tgt_val_split"):
            subsets.update(str(value).lower() for value in config[key])
        annotation_paths.add(
            str((repository_root / config["dataset"]["json_file"]).resolve())
        )
    if len(annotation_paths) != 1:
        raise ValueError("RQ2 shifts must use one common annotation database")
    annotation_path = Path(next(iter(annotation_paths)))
    database = json.loads(annotation_path.read_text(encoding="utf-8"))["database"]
    video_ids = sorted(
        video_id
        for video_id, entry in database.items()
        if str(entry["subset"]).lower() in subsets
    )
    provenance = {
        "registry_sha256": sha256_normalized_text(registry_path),
        "configs": config_hashes,
        "annotation": {
            "path": str(annotation_path.relative_to(repository_root)).replace("\\", "/"),
            "sha256": sha256_file(annotation_path),
        },
    }
    return video_ids, provenance


def required_video_ids_from_registry(path):
    return video_selection_from_registry(path)[0]


def build_encoder(args):
    if args.model == "egovideo_hod":
        if not args.egohod_root:
            raise ValueError("EgoVideo requires --egohod-root")
        return EgoVideoEncoder(args)
    if args.model == "internvideo2_clip_1b":
        if not args.internvideo_root:
            raise ValueError("InternVideo2 requires --internvideo-root")
        return InternVideoEncoder(args)
    if args.model in ("qwen3_vl_8b", "timelens2_8b"):
        return MultimodalLastTokenEncoder(args)
    if args.model == "timeple_8b":
        return TimePLEEncoder(args)
    return QwenVisionEncoder(args)


def video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("cannot open video: {}".format(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError("invalid video metadata: {}".format(path))
    return frame_count, fps


def write_shared_protocol_files(output_root, video_ids, payload, signature):
    documents = {
        "video_ids.txt": "".join(video_id + "\n" for video_id in video_ids),
        "protocol.json": json.dumps(
            {**payload, "protocol_sha256": signature},
            indent=2,
            sort_keys=True,
        ) + "\n",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for name, content in documents.items():
        path = output_root / name
        if path.is_file() and path.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                "shared output already contains a different {}".format(name)
            )
        temporary = output_root / ".{}.{}.tmp".format(name, os.getpid())
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)


def assigned_video_ids(path, canonical_video_ids):
    if not path:
        return list(canonical_video_ids)
    requested = Path(path).read_text(encoding="ascii").splitlines()
    if not requested or requested != list(dict.fromkeys(requested)):
        raise ValueError("assignment file must contain unique video IDs")
    unknown = sorted(set(requested).difference(canonical_video_ids))
    if unknown:
        raise ValueError("assignment contains videos outside the protocol: {}".format(unknown))
    return requested


def main():
    args = parse_args()
    if args.batch_tokens <= 0 or args.shard_count <= 0:
        raise ValueError("batch tokens and shard count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    if args.video_id and args.assignment_file:
        raise ValueError("--video-id and --assignment-file cannot be combined")
    spec = MODEL_SPECS[args.model]
    reference_root = Path(args.reference_feature_root)
    if args.shift_registry:
        video_ids, selection_provenance = video_selection_from_registry(
            args.shift_registry
        )
        selection_source = "shift_registry_and_annotation"
    else:
        video_ids = sorted(path.stem for path in reference_root.glob("*.npz"))
        selection_provenance = None
        selection_source = "reference_feature_root"
    if args.video_id:
        requested = set(args.video_id)
        missing = sorted(requested.difference(video_ids))
        if missing:
            raise ValueError("video IDs absent from reference features: {}".format(missing))
        video_ids = [video_id for video_id in video_ids if video_id in requested]
    missing_references = [
        video_id for video_id in video_ids
        if not (reference_root / (video_id + ".npz")).is_file()
    ]
    if missing_references:
        raise FileNotFoundError(
            "selected videos missing reference features: {}".format(missing_references)
        )
    payload = protocol_payload(
        args.model, checkpoint_hashes(args), extraction_provenance(args)
    )
    payload["video_selection"] = selection_source
    if selection_provenance is not None:
        payload["video_selection_provenance"] = selection_provenance
    payload["video_set_count"] = len(video_ids)
    payload["video_set_sha256"] = video_set_sha256(video_ids)
    signature = protocol_sha256(payload)
    all_video_ids = list(video_ids)
    video_ids = assigned_video_ids(args.assignment_file, all_video_ids)
    video_ids = [
        video_id for index, video_id in enumerate(video_ids)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit_videos > 0:
        video_ids = video_ids[:args.limit_videos]
    if not video_ids:
        raise ValueError("no videos selected")
    output_root = Path(args.output_root)
    write_shared_protocol_files(
        output_root, all_video_ids, payload, signature
    )
    if args.protocol_only:
        print(signature, flush=True)
        return 0

    pending = []
    for video_id in video_ids:
        with np.load(reference_root / (video_id + ".npz")) as reference:
            expected_length = int(reference["feats"].shape[0])
        output = output_root / (video_id + ".npz")
        if not validate_feature_file(
            output, args.model, expected_length, signature
        ):
            pending.append((video_id, expected_length))
    if not pending:
        print("all selected videos already complete", flush=True)
        return 0
    encoder = build_encoder(args)
    started = time.time()
    for number, (video_id, expected_length) in enumerate(pending, 1):
        video_path = Path(args.video_root) / (video_id + ".MP4")
        if not video_path.is_file():
            raise FileNotFoundError("video is missing: {}".format(video_path))
        frame_count, source_fps = video_metadata(video_path)
        calculated = temporal_length(frame_count, source_fps)
        if not reference_temporal_length_is_compatible(
            calculated, expected_length
        ):
            raise RuntimeError(
                "{} temporal length mismatch: video={} reference={}".format(
                    video_id, calculated, expected_length
                )
            )
        chunks = []
        reader = SequentialFrameReader(video_path)
        try:
            for start in range(0, expected_length, args.batch_tokens):
                stop = min(start + args.batch_tokens, expected_length)
                index_rows = [
                    sampled_source_indices(
                        token, spec["input_frames"], source_fps, frame_count
                    )
                    for token in range(start, stop)
                ]
                frames = reader.read(np.stack(index_rows))
                chunks.append(encoder.encode(frames, stop - start))
        finally:
            reader.close()
        features = np.concatenate(chunks, axis=0)
        if features.shape != (expected_length, spec["output_dim"]):
            raise RuntimeError("unexpected feature shape: {}".format(features.shape))
        atomic_savez(
            output_root / (video_id + ".npz"),
            feats=features,
            centers_sec=expected_centers_seconds(expected_length).astype(np.float32),
            feature_fps=np.float32(FEATURE_FPS),
            feat_stride=np.int32(FEAT_STRIDE),
            receptive_field_frames=np.int32(RECEPTIVE_FIELD_FRAMES),
            source_fps=np.float32(source_fps),
            source_frame_count=np.int64(frame_count),
            protocol_sha256=np.asarray(signature),
        )
        elapsed = max(time.time() - started, 1e-9)
        print(
            "{}/{} {} tokens={} videos_per_hour={:.2f}".format(
                number, len(pending), video_id, expected_length,
                number / elapsed * 3600,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
