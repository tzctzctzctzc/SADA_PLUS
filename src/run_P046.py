#!/usr/bin/env python3
"""Train and evaluate the three frozen P046 mechanism-pilot arms."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils.vision_process import process_vision_info

ROOT = Path("/vepfs/vlms/1")
RUN_DIR = ROOT / "runs/P046_self_normalized_role_corruption"
CONFIG_PATH = ROOT / "configs/P046_self_normalized_role_corruption.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pairwise_margin_loss(positive: torch.Tensor, negative: torch.Tensor, margin: float, temperature: float) -> torch.Tensor:
    """Smooth hinge: lower only when positive exceeds negative by the margin."""
    return temperature * F.softplus((margin - (positive - negative)) / temperature)


def pool_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last = attention_mask.shape[1] - attention_mask.flip(dims=[1]).argmax(dim=1) - 1
    row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return F.normalize(hidden_state[row, last].float(), p=2, dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["ordinary_retrieval", "generic_augmentation", "role_corruption"], default="role_corruption")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def load_official_classes(model_path: Path):
    source = model_path / "scripts/qwen3_vl_embedding.py"
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_official_P046", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Qwen3VLForEmbedding, module.Qwen3VLProcessor, sha256_file(source)


def main() -> None:
    args = parse_args()
    assert Path.cwd().resolve() == ROOT
    config = json.loads(CONFIG_PATH.read_text())
    train_cfg = config["training"]
    assert args.arm in train_cfg["arms"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu_uuid"]
    assert torch.cuda.device_count() == 1
    visible_uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    assert visible_uuid == config["gpu_uuid"]
    assert sha256_file(ROOT / config["data"]["query_path"]) == config["data"]["query_sha256"]
    if not args.audit_only:
        corruption_summary = json.loads((RUN_DIR / "corruption_audit_summary.json").read_text())
        assert corruption_summary["status"] == "pass"

    seed = train_cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    bundles = [json.loads(line) for line in (RUN_DIR / "bundles.jsonl").open() if line.strip()]
    train_rows = [row for row in bundles if row["split"] == "train"]
    validation_rows = [row for row in bundles if row["split"] == "validation"]
    assert len(train_rows) == config["data"]["counts"]["train"]
    assert len(validation_rows) == config["data"]["counts"]["validation"]
    image_manifest = json.loads((RUN_DIR / "image_manifest.json").read_text())
    assert image_manifest["complete"]
    image_entries = {row["idx"]: row for row in image_manifest["entries"]}

    @functools.lru_cache(maxsize=512)
    def load_image(product_id: str) -> Image.Image:
        entry = image_entries[product_id]
        path = ROOT / entry["path"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")

    model_path = ROOT / config["model"]["path"]
    model_class, processor_class, source_sha = load_official_classes(model_path)
    processor = processor_class.from_pretrained(model_path, padding_side="right", local_files_only=True)
    model = model_class.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=config["model"]["attn_implementation"],
        low_cpu_mem_usage=True,
    )
    lora = LoraConfig(
        r=train_cfg["lora_rank"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lora)
    if train_cfg["train_visual_merger"]:
        for name, parameter in model.named_parameters():
            if ".visual.merger." in name:
                parameter.requires_grad = True
    if train_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.to("cuda")
    model.train()

    min_pixels = config["model"]["min_pixels"]
    max_pixels = config["model"]["max_pixels"]

    def image_content(product_id: str, augment: bool = False) -> Any:
        if not augment:
            return "file://" + str((ROOT / image_entries[product_id]["path"]).resolve())
        image = load_image(product_id).copy()
        image = ImageEnhance.Brightness(image).enhance(1.08)
        return ImageEnhance.Contrast(image).enhance(0.92)

    def query_conversation(row: dict[str, Any], variant: str) -> list[dict[str, Any]]:
        source_ids = list(row["source_ids"])
        if variant == "corrupt":
            source_ids[row["corrupt_slot"]] = row["donor_id"]
        content: list[dict[str, Any]] = []
        for slot, product_id in enumerate(source_ids):
            content.append({"type": "text", "text": f"Product {slot + 1}:\n"})
            content.append(
                {
                    "type": "image",
                    "image": image_content(product_id, augment=variant == "augment" and slot == row["corrupt_slot"]),
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                }
            )
            content.append({"type": "text", "text": "\n"})
        content.append({"type": "text", "text": row["instruction"]})
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Represent the multi-reference product retrieval query. Respect the attribute assigned to each numbered product.",
                    }
                ],
            },
            {"role": "user", "content": content},
        ]

    def candidate_conversation(product_id: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Represent the candidate product image for retrieval."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_content(product_id),
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                ],
            },
        ]

    def prepare(conversations: list[list[dict[str, Any]]]) -> dict[str, torch.Tensor]:
        texts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images, video_inputs, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            videos, video_metadata = None, None
        inputs = processor(
            text=texts,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=config["model"]["max_length"],
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        return {key: value.to("cuda") for key, value in inputs.items()}

    def embed(conversations: list[list[dict[str, Any]]]) -> torch.Tensor:
        inputs = prepare(conversations)
        outputs = model(**inputs, use_cache=False)
        return pool_last(outputs.last_hidden_state, inputs["attention_mask"])

    lora_parameters = []
    merger_parameters = []
    unexpected_trainable = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_parameters.append(parameter)
        elif ".visual.merger." in name:
            merger_parameters.append(parameter)
        else:
            unexpected_trainable.append(name)
    assert lora_parameters and merger_parameters and not unexpected_trainable
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": train_cfg["learning_rate_lora"]},
            {"params": merger_parameters, "lr": train_cfg["learning_rate_visual_merger"]},
        ],
        weight_decay=train_cfg["weight_decay"],
    )

    def one_loss(row: dict[str, Any], arm: str, retain_embeddings: bool = False):
        query_kinds = ["original"]
        if arm == "generic_augmentation":
            query_kinds.append("augment")
        elif arm == "role_corruption":
            query_kinds.append("corrupt")
        query_embeddings = embed([query_conversation(row, kind) for kind in query_kinds])
        candidates = embed(
            [candidate_conversation(row["positive_target_id"]), candidate_conversation(row["negative_target_id"])]
        )
        if retain_embeddings:
            query_embeddings.retain_grad()
            candidates.retain_grad()
        positive = (query_embeddings[0] * candidates[0]).sum()
        negative = (query_embeddings[0] * candidates[1]).sum()
        rank_loss = pairwise_margin_loss(positive, negative, train_cfg["rank_margin"], train_cfg["temperature"])
        auxiliary = torch.zeros((), device=positive.device)
        if arm == "generic_augmentation":
            auxiliary = 1.0 - (query_embeddings[0] * query_embeddings[1]).sum()
        elif arm == "role_corruption":
            corrupt_positive = (query_embeddings[1] * candidates[0]).sum()
            auxiliary = pairwise_margin_loss(positive, corrupt_positive, train_cfg["role_margin"], train_cfg["temperature"])
        loss = rank_loss + train_cfg["aux_weight"] * auxiliary
        return loss, rank_loss, auxiliary, positive, negative, query_embeddings, candidates

    def evaluate(rows: list[dict[str, Any]]) -> dict[str, float]:
        model.eval()
        role_margins = []
        retrieval_margins = []
        with torch.no_grad():
            for row in rows:
                queries = embed([query_conversation(row, "original"), query_conversation(row, "corrupt")])
                candidates = embed(
                    [candidate_conversation(row["positive_target_id"]), candidate_conversation(row["negative_target_id"])]
                )
                original_positive = float((queries[0] * candidates[0]).sum())
                corrupt_positive = float((queries[1] * candidates[0]).sum())
                negative = float((queries[0] * candidates[1]).sum())
                role_margins.append(original_positive - corrupt_positive)
                retrieval_margins.append(original_positive - negative)
        model.train()
        return {
            "rows": len(rows),
            "role_pair_accuracy": sum(value > 0 for value in role_margins) / len(role_margins),
            "mean_role_margin": sum(role_margins) / len(role_margins),
            "retrieval_accuracy": sum(value > 0 for value in retrieval_margins) / len(retrieval_margins),
            "mean_retrieval_margin": sum(retrieval_margins) / len(retrieval_margins),
        }

    parameter_summary = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "lora": sum(parameter.numel() for parameter in lora_parameters),
        "visual_merger": sum(parameter.numel() for parameter in merger_parameters),
    }

    if args.audit_only:
        optimizer.zero_grad(set_to_none=True)
        loss, rank_loss, auxiliary, positive, negative, queries, candidates = one_loss(train_rows[0], "role_corruption", True)
        loss.backward()
        layer_pattern = re.compile(r"language_model\.layers\.(\d+)\.")
        all_layers = {
            int(match.group(1))
            for name, _ in model.named_modules()
            if (match := layer_pattern.search(name))
        }
        lora_gradient_layers = {
            int(match.group(1))
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            if (match := layer_pattern.search(name))
        }
        merger_nonzero = any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in merger_parameters
        )
        candidate_gradient = candidates.grad is not None and torch.count_nonzero(candidates.grad).item() > 0
        checks = {
            "physical_gpu_assertion": True,
            "all_language_layers_lora": bool(all_layers) and all_layers == lora_gradient_layers,
            "visual_merger_trainable": merger_nonzero,
            "candidate_branch_gradient": candidate_gradient,
            "finite_loss": bool(torch.isfinite(loss)),
            "positive_and_negative_are_distinct": train_rows[0]["positive_target_id"] != train_rows[0]["negative_target_id"],
        }
        summary = {
            "status": "pass" if all(checks.values()) else "fail",
            "classification": "faithful" if all(checks.values()) else "incorrect",
            "checks": checks,
            "gpu_uuid": config["gpu_uuid"],
            "model_revision": config["model"]["revision"],
            "official_source_sha256": source_sha,
            "parameter_summary": parameter_summary,
            "language_layers": len(all_layers),
            "language_layers_with_nonzero_lora_grad": len(lora_gradient_layers),
            "loss": float(loss.detach()),
            "rank_loss": float(rank_loss.detach()),
            "role_loss": float(auxiliary.detach()),
            "positive_score": float(positive.detach()),
            "negative_score": float(negative.detach()),
            "candidate_embedding_grad_norm": float(candidates.grad.float().norm()) if candidates.grad is not None else 0.0,
            "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        (RUN_DIR / "runtime_gradient_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["status"] != "pass":
            raise SystemExit(2)
        return

    started = time.time()
    before = evaluate(validation_rows)
    history = []
    order = sorted(train_rows, key=lambda row: row["selection_hash"])
    torch.cuda.reset_peak_memory_stats()
    for step in range(train_cfg["steps"]):
        if time.time() - started > train_cfg["max_gpu_hours_per_arm"] * 3600:
            raise RuntimeError("frozen per-arm GPU-hour limit exceeded")
        row = order[step % len(order)]
        optimizer.zero_grad(set_to_none=True)
        loss, rank_loss, auxiliary, positive, negative, _, _ = one_loss(row, args.arm)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "bundle_id": row["bundle_id"],
                "loss": float(loss.detach()),
                "rank_loss": float(rank_loss.detach()),
                "auxiliary_loss": float(auxiliary.detach()),
                "positive_score": float(positive.detach()),
                "negative_score": float(negative.detach()),
            }
        )
        print(json.dumps({"arm": args.arm, **history[-1]}), flush=True)

    after = evaluate(validation_rows)
    result = {
        "status": "complete",
        "arm": args.arm,
        "gpu_uuid": config["gpu_uuid"],
        "seed": seed,
        "steps": train_cfg["steps"],
        "model_revision": config["model"]["revision"],
        "official_source_sha256": source_sha,
        "parameter_summary": parameter_summary,
        "before": before,
        "after": after,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    (RUN_DIR / f"pilot_{args.arm}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
