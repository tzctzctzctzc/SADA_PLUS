#!/usr/bin/env python3
"""Train and evaluate the four frozen P048 large-pilot arms."""

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
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils.vision_process import process_vision_info

ROOT = Path("/vepfs/vlms/1")
RUN_DIR = ROOT / "runs/P048_visible_role_contrast"
CONFIG_PATH = ROOT / "configs/P048_visible_role_contrast.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pairwise_margin_loss(positive: torch.Tensor, negative: torch.Tensor, margin: float, temperature: float) -> torch.Tensor:
    return temperature * F.softplus((margin - (positive - negative)) / temperature)


def pool_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last = attention_mask.shape[1] - attention_mask.flip(dims=[1]).argmax(dim=1) - 1
    row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return F.normalize(hidden_state[row, last].float(), p=2, dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=["ordinary_retrieval", "paraphrase_invariance", "image_role_swap", "text_role_swap"],
        default="text_role_swap",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def load_official_classes(model_path: Path):
    source = model_path / "scripts/qwen3_vl_embedding.py"
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_official_P048", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Qwen3VLForEmbedding, module.Qwen3VLProcessor, sha256_file(source)


def retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    count = len(rows)
    return {
        "rows": count,
        "binding_pair_accuracy": sum(row["binding_correct"] for row in rows) / count,
        "mean_binding_margin": sum(row["binding_margin"] for row in rows) / count,
        "image_swap_pair_accuracy": sum(row["image_swap_correct"] for row in rows) / count,
        "mean_image_swap_margin": sum(row["image_swap_margin"] for row in rows) / count,
        "mean_paraphrase_cosine": sum(row["paraphrase_cosine"] for row in rows) / count,
        "map_at_10": sum(row["ap_at_10"] for row in rows) / count,
        "mrr": sum(1.0 / row["rank"] for row in rows) / count,
        "recall_at_1": sum(row["rank"] <= 1 for row in rows) / count,
        "recall_at_5": sum(row["rank"] <= 5 for row in rows) / count,
        "recall_at_10": sum(row["rank"] <= 10 for row in rows) / count,
        "hard_negative_violation_rate": sum(row["hard_negative_violation"] for row in rows) / count,
        "paraphrase_map_at_10": sum(row["paraphrase_ap_at_10"] for row in rows) / count,
        "paraphrase_mrr": sum(1.0 / row["paraphrase_rank"] for row in rows) / count,
        "paraphrase_recall_at_1": sum(row["paraphrase_rank"] <= 1 for row in rows) / count,
        "paraphrase_recall_at_5": sum(row["paraphrase_rank"] <= 5 for row in rows) / count,
        "paraphrase_recall_at_10": sum(row["paraphrase_rank"] <= 10 for row in rows) / count,
        "paraphrase_top1_agreement": sum(row["top1_id"] == row["paraphrase_top1_id"] for row in rows) / count,
    }


def main() -> None:
    args = parse_args()
    assert Path.cwd().resolve() == ROOT
    config = json.loads(CONFIG_PATH.read_text())
    train_cfg = config["training"]
    assert args.arm in train_cfg["arms"]
    assert json.loads((RUN_DIR / "implementation_audit_static.json").read_text())["classification"] == "faithful"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu_uuid"]
    assert torch.cuda.device_count() == 1
    assert "GPU-" + str(torch.cuda.get_device_properties(0).uuid) == config["gpu_uuid"]
    assert sha256_file(ROOT / config["data"]["query_path"]) == config["data"]["query_sha256"]
    if not args.audit_only:
        assert json.loads((RUN_DIR / "corruption_audit_summary.json").read_text())["status"] == "pass"
        assert json.loads((RUN_DIR / "runtime_gradient_audit.json").read_text())["classification"] == "faithful"

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

    @functools.lru_cache(maxsize=4096)
    def load_image(product_id: str) -> Image.Image:
        entry = image_entries[product_id]
        data = (ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")

    model_path = ROOT / config["model"]["path"]
    revision_metadata = model_path / ".cache/huggingface/download/config.json.metadata"
    assert revision_metadata.read_text().splitlines()[0] == config["model"]["revision"]
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

    def image_content(product_id: str) -> str:
        return "file://" + str((ROOT / image_entries[product_id]["path"]).resolve())

    def query_conversation(row: dict[str, Any], variant: str) -> list[dict[str, Any]]:
        source_ids = list(row["source_ids"])
        instruction_key = "original"
        if variant == "image_role_swap":
            source_ids.reverse()
        elif variant == "paraphrase":
            instruction_key = "paraphrase"
        elif variant == "text_role_swap":
            instruction_key = "text_role_swap"
        else:
            assert variant == "original"
        content: list[dict[str, Any]] = []
        for slot, product_id in enumerate(source_ids):
            content.extend(
                [
                    {"type": "text", "text": f"Product {slot + 1}:\n"},
                    {
                        "type": "image",
                        "image": image_content(product_id),
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    {"type": "text", "text": "\n"},
                ]
            )
        content.append({"type": "text", "text": row["instructions"][instruction_key]})
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
            {"role": "system", "content": [{"type": "text", "text": "Represent the candidate product image for retrieval."}]},
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
        query_variants = ["original", "paraphrase", "image_role_swap", "text_role_swap"]
        queries = embed([query_conversation(row, variant) for variant in query_variants])
        candidates = embed(
            [candidate_conversation(row["positive_target_id"]), candidate_conversation(row["negative_target_id"])]
        )
        if retain_embeddings:
            queries.retain_grad()
            candidates.retain_grad()
        scores = queries @ candidates[0]
        negative = (queries[0] * candidates[1]).sum()
        rank_loss = pairwise_margin_loss(scores[0], negative, train_cfg["rank_margin"], train_cfg["temperature"])
        auxiliary = torch.zeros((), device=scores.device)
        if arm == "paraphrase_invariance":
            auxiliary = 1.0 - (queries[0] * queries[1]).sum()
        elif arm == "image_role_swap":
            auxiliary = pairwise_margin_loss(scores[0], scores[2], train_cfg["role_margin"], train_cfg["temperature"])
        elif arm == "text_role_swap":
            auxiliary = pairwise_margin_loss(scores[0], scores[3], train_cfg["role_margin"], train_cfg["temperature"])
        loss = rank_loss + train_cfg["aux_weight"] * auxiliary
        return loss, rank_loss, auxiliary, scores[0], negative, queries, candidates

    def evaluate(rows: list[dict[str, Any]], label: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
        model.eval()
        gallery_ids = [row["positive_target_id"] for row in rows] + [row["negative_target_id"] for row in rows]
        assert len(gallery_ids) == len(set(gallery_ids))
        gallery_chunks = []
        batch_size = train_cfg["candidate_eval_batch_size"]
        with torch.no_grad():
            for start in range(0, len(gallery_ids), batch_size):
                gallery_chunks.append(embed([candidate_conversation(value) for value in gallery_ids[start : start + batch_size]]).cpu())
            gallery = torch.cat(gallery_chunks).float()
            gallery = F.normalize(gallery, p=2, dim=-1)
            index = {product_id: position for position, product_id in enumerate(gallery_ids)}
            per_query = []
            for number, row in enumerate(rows, 1):
                variants = ["original", "paraphrase", "image_role_swap", "text_role_swap"]
                queries = embed([query_conversation(row, variant) for variant in variants]).cpu().float()
                scores = queries @ gallery.T
                positive_index = index[row["positive_target_id"]]
                negative_index = index[row["negative_target_id"]]
                positive_scores = scores[:, positive_index]
                rank = int((scores[0] > positive_scores[0]).sum()) + 1
                paraphrase_rank = int((scores[1] > positive_scores[1]).sum()) + 1
                top1 = int(scores[0].argmax())
                paraphrase_top1 = int(scores[1].argmax())
                binding_margin = float(positive_scores[0] - positive_scores[3])
                image_margin = float(positive_scores[0] - positive_scores[2])
                per_query.append(
                    {
                        "bundle_id": row["bundle_id"],
                        "query_idx": row["query_idx"],
                        "target_class": row["target_class"],
                        "active_role": row["active_role"],
                        "active_role_key": row["active_role_key"],
                        "positive_target_id": row["positive_target_id"],
                        "negative_target_id": row["negative_target_id"],
                        "rank": rank,
                        "ap_at_10": 1.0 / rank if rank <= 10 else 0.0,
                        "paraphrase_rank": paraphrase_rank,
                        "paraphrase_ap_at_10": 1.0 / paraphrase_rank if paraphrase_rank <= 10 else 0.0,
                        "top1_id": gallery_ids[top1],
                        "paraphrase_top1_id": gallery_ids[paraphrase_top1],
                        "binding_margin": binding_margin,
                        "binding_correct": binding_margin > 0,
                        "image_swap_margin": image_margin,
                        "image_swap_correct": image_margin > 0,
                        "paraphrase_cosine": float((queries[0] * queries[1]).sum()),
                        "positive_score": float(positive_scores[0]),
                        "paraphrase_positive_score": float(positive_scores[1]),
                        "image_swap_positive_score": float(positive_scores[2]),
                        "text_swap_positive_score": float(positive_scores[3]),
                        "hard_negative_score": float(scores[0, negative_index]),
                        "hard_negative_violation": bool(scores[0, negative_index] >= positive_scores[0]),
                    }
                )
                if number % 32 == 0:
                    print(json.dumps({"event": "evaluation", "label": label, "queries": number, "total": len(rows)}), flush=True)
        model.train()
        return retrieval_summary(per_query), per_query

    parameter_summary = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "lora": sum(parameter.numel() for parameter in lora_parameters),
        "visual_merger": sum(parameter.numel() for parameter in merger_parameters),
    }

    if args.audit_only:
        optimizer.zero_grad(set_to_none=True)
        loss, rank_loss, auxiliary, positive, negative, queries, candidates = one_loss(train_rows[0], "text_role_swap", True)
        loss.backward()
        layer_pattern = re.compile(r"language_model\.layers\.(\d+)\.")
        all_layers = {
            int(match.group(1))
            for name, _ in model.named_modules()
            if (match := layer_pattern.search(name))
        }
        gradient_layers = {
            int(match.group(1))
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            if (match := layer_pattern.search(name))
        }
        merger_nonzero = any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0 for parameter in merger_parameters)
        candidate_gradient = candidates.grad is not None and torch.count_nonzero(candidates.grad).item() > 0
        text_swap_gradient = queries.grad is not None and torch.count_nonzero(queries.grad[3]).item() > 0
        checks = {
            "physical_gpu_assertion": True,
            "all_language_layers_lora": bool(all_layers) and all_layers == gradient_layers,
            "visual_merger_trainable": merger_nonzero,
            "candidate_branch_gradient": candidate_gradient,
            "text_swap_branch_gradient": text_swap_gradient,
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
            "language_layers_with_nonzero_lora_grad": len(gradient_layers),
            "loss": float(loss.detach()),
            "rank_loss": float(rank_loss.detach()),
            "role_loss": float(auxiliary.detach()),
            "positive_score": float(positive.detach()),
            "negative_score": float(negative.detach()),
            "candidate_embedding_grad_norm": float(candidates.grad.float().norm()) if candidates.grad is not None else 0.0,
            "text_swap_embedding_grad_norm": float(queries.grad[3].float().norm()) if queries.grad is not None else 0.0,
            "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        (RUN_DIR / "runtime_gradient_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["status"] != "pass":
            raise SystemExit(2)
        return

    started = time.time()
    before, before_rows = evaluate(validation_rows, f"{args.arm}:before")
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
        if (step + 1) % 10 == 0 or step == 0:
            print(json.dumps({"arm": args.arm, **history[-1]}), flush=True)

    after, after_rows = evaluate(validation_rows, f"{args.arm}:after")
    for name, rows in [(f"eval_before_{args.arm}.jsonl", before_rows), (f"eval_after_{args.arm}.jsonl", after_rows)]:
        with (RUN_DIR / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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
