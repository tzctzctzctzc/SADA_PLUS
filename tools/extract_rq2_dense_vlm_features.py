"""Extract frozen full-video VLM features on the shared RQ2 temporal grid."""

import argparse
import importlib
import json
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
    RECEPTIVE_FIELD_FRAMES,
    atomic_savez,
    expected_centers_seconds,
    protocol_payload,
    protocol_sha256,
    reference_temporal_length_is_compatible,
    sampled_source_indices,
    sha256_file,
    temporal_length,
    validate_feature_file,
    video_set_sha256,
)


class SequentialFrameReader:
    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError("cannot open video: {}".format(path))
        self.position = 0

    def close(self):
        self.capture.release()

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
                    raise RuntimeError("failed to seek frame {}".format(index))
                self.position += 1
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


class QwenEncoder:
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
        features = per_frame.reshape(batch_tokens, 8, -1).mean(dim=1)
        return normalized_numpy(features)


def normalized_numpy(features):
    features = features.float()
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.cpu().numpy().astype(np.float16)


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
    elif args.model == "qwen3_vl_8b":
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
    return {Path(path).name: sha256_file(path) for path in paths}


def required_video_ids_from_registry(path):
    registry_path = Path(path).resolve()
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    repository_root = (registry_path.parent / registry["repository_root"]).resolve()
    subsets = set()
    annotation_paths = set()
    for entry in registry["shifts"].values():
        config_path = repository_root / entry["config"]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for key in ("src_train_split", "src_val_split", "tgt_train_split", "tgt_val_split"):
            subsets.update(str(value).lower() for value in config[key])
        annotation_paths.add(
            str((repository_root / config["dataset"]["json_file"]).resolve())
        )
    if len(annotation_paths) != 1:
        raise ValueError("RQ2 shifts must use one common annotation database")
    database = json.loads(
        Path(next(iter(annotation_paths))).read_text(encoding="utf-8")
    )["database"]
    return sorted(
        video_id
        for video_id, entry in database.items()
        if str(entry["subset"]).lower() in subsets
    )


def build_encoder(args):
    if args.model == "egovideo_hod":
        if not args.egohod_root:
            raise ValueError("EgoVideo requires --egohod-root")
        return EgoVideoEncoder(args)
    if args.model == "internvideo2_clip_1b":
        if not args.internvideo_root:
            raise ValueError("InternVideo2 requires --internvideo-root")
        return InternVideoEncoder(args)
    return QwenEncoder(args)


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


def main():
    args = parse_args()
    if args.batch_tokens <= 0 or args.shard_count <= 0:
        raise ValueError("batch tokens and shard count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    spec = MODEL_SPECS[args.model]
    reference_root = Path(args.reference_feature_root)
    if args.shift_registry:
        video_ids = required_video_ids_from_registry(args.shift_registry)
        selection_source = "shift_registry_and_annotation"
    else:
        video_ids = sorted(path.stem for path in reference_root.glob("*.npz"))
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
    payload = protocol_payload(args.model, checkpoint_hashes(args))
    payload["video_selection"] = selection_source
    payload["video_set_count"] = len(video_ids)
    payload["video_set_sha256"] = video_set_sha256(video_ids)
    signature = protocol_sha256(payload)
    all_video_ids = list(video_ids)
    video_ids = [
        video_id for index, video_id in enumerate(all_video_ids)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit_videos > 0:
        video_ids = video_ids[:args.limit_videos]
    if not video_ids:
        raise ValueError("no videos selected")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "video_ids.txt").write_text(
        "".join(video_id + "\n" for video_id in all_video_ids),
        encoding="ascii",
    )
    (output_root / "protocol.json").write_text(
        json.dumps({**payload, "protocol_sha256": signature}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

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
