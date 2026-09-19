#!/usr/bin/env python3
"""Train and evaluate the frozen-retriever P049 candidate-aware role verifier."""

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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils.vision_process import process_vision_info

ROOT = Path("/vepfs/vlms/1")
RUN_DIR = ROOT / "runs/P049_counterfactual_role_verifier"
CONFIG_PATH = ROOT / "configs/P049_counterfactual_role_verifier.json"


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
    rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return F.normalize(hidden_state[rows, last].float(), p=2, dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["ordinary_verifier", "positive_counterfactual", "dod_counterfactual"])
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--prepare-base", action="store_true")
    return parser.parse_args()


def load_official_classes(model_path: Path):
    source = model_path / "scripts/qwen3_vl_embedding.py"
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_official_P049", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Qwen3VLForEmbedding, module.Qwen3VLProcessor, sha256_file(source)


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    return {
        "rows": n,
        "map_at_10": sum(row["ap_at_10"] for row in rows) / n,
        "mrr": sum(1.0 / row["rank"] for row in rows) / n,
        "recall_at_1": sum(row["rank"] <= 1 for row in rows) / n,
        "recall_at_5": sum(row["rank"] <= 5 for row in rows) / n,
        "recall_at_10": sum(row["rank"] <= 10 for row in rows) / n,
        "paraphrase_map_at_10": sum(row["paraphrase_ap_at_10"] for row in rows) / n,
        "paraphrase_recall_at_1": sum(row["paraphrase_rank"] <= 1 for row in rows) / n,
        "paraphrase_recall_at_5": sum(row["paraphrase_rank"] <= 5 for row in rows) / n,
        "paraphrase_recall_at_10": sum(row["paraphrase_rank"] <= 10 for row in rows) / n,
        "paraphrase_top1_agreement": sum(row["top1_id"] == row["paraphrase_top1_id"] for row in rows) / n,
        "binding_pair_accuracy": sum(row["binding_margin"] > 0 for row in rows) / n,
        "mean_binding_margin": sum(row["binding_margin"] for row in rows) / n,
        "dod_pair_accuracy": sum(row["dod_margin"] > 0 for row in rows) / n,
        "mean_dod_margin": sum(row["dod_margin"] for row in rows) / n,
        "hard_negative_violation_rate": sum(row["hard_negative_violation"] for row in rows) / n,
    }


def zscore(values: list[float]) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    std = float(tensor.std(unbiased=False))
    if std < 1e-6:
        return [0.0 for _ in values]
    return ((tensor - tensor.mean()) / tensor.std(unbiased=False)).tolist()


def main() -> None:
    args = parse_args()
    assert sum([args.audit_only, args.prepare_base, args.arm is not None]) == 1
    assert Path.cwd().resolve() == ROOT
    config = json.loads(CONFIG_PATH.read_text())
    train_cfg = config["training"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu_uuid"]
    assert torch.cuda.device_count() == 1
    assert "GPU-" + str(torch.cuda.get_device_properties(0).uuid) == config["gpu_uuid"]
    assert json.loads((RUN_DIR / "implementation_audit_static.json").read_text())["classification"] == "faithful"
    prepare_summary = json.loads((RUN_DIR / "prepare_summary.json").read_text())
    assert prepare_summary["inherited_visual_gate"]["status"] == "pass"
    image_manifest = json.loads((RUN_DIR / "image_manifest.json").read_text())
    assert image_manifest["complete"]
    image_entries = {row["idx"]: row for row in image_manifest["entries"]}
    bundles = [json.loads(line) for line in (RUN_DIR / "bundles.jsonl").open() if line.strip()]
    train_rows = [row for row in bundles if row["split"] == "train"]
    validation_rows = [row for row in bundles if row["split"] == "validation"]
    assert len(train_rows) == config["data"]["counts"]["train"]
    assert len(validation_rows) == config["data"]["counts"]["validation"]
    seed = train_cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    @functools.lru_cache(maxsize=4096)
    def load_image(product_id: str) -> Image.Image:
        entry = image_entries[product_id]
        data = (ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")

    def image_content(product_id: str) -> str:
        return "file://" + str((ROOT / image_entries[product_id]["path"]).resolve())

    def query_conversation(row: dict[str, Any], variant: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for slot, product_id in enumerate(row["source_ids"]):
            content.extend([
                {"type": "text", "text": f"Product {slot + 1}:\n"},
                {"type": "image", "image": image_content(product_id), "min_pixels": config["model"]["min_pixels"], "max_pixels": config["model"]["max_pixels"]},
                {"type": "text", "text": "\n"},
            ])
        content.append({"type": "text", "text": row["instructions"][variant]})
        return [{"role": "system", "content": [{"type": "text", "text": "Represent the multi-reference product retrieval query. Respect which numbered product supplies each requested attribute."}]}, {"role": "user", "content": content}]

    def candidate_conversation(product_id: str) -> list[dict[str, Any]]:
        return [{"role": "system", "content": [{"type": "text", "text": "Represent the candidate product image for retrieval."}]}, {"role": "user", "content": [{"type": "image", "image": image_content(product_id), "min_pixels": config["model"]["min_pixels"], "max_pixels": config["model"]["max_pixels"]}]}]

    def joint_conversation(row: dict[str, Any], variant: str, candidate_id: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for slot, product_id in enumerate(row["source_ids"]):
            content.extend([
                {"type": "text", "text": f"Reference Product {slot + 1}:\n"},
                {"type": "image", "image": image_content(product_id), "min_pixels": config["model"]["min_pixels"], "max_pixels": config["model"]["max_pixels"]},
                {"type": "text", "text": "\n"},
            ])
        content.extend([
            {"type": "text", "text": f"Retrieval instruction:\n{row['instructions'][variant]}\nCandidate product:\n"},
            {"type": "image", "image": image_content(candidate_id), "min_pixels": config["model"]["min_pixels"], "max_pixels": config["model"]["max_pixels"]},
            {"type": "text", "text": "\nRepresent whether this candidate satisfies the instruction and the reference-to-attribute assignments."},
        ])
        return [{"role": "system", "content": [{"type": "text", "text": "You are a candidate-aware verifier for multi-reference composed product retrieval."}]}, {"role": "user", "content": content}]

    model_path = ROOT / config["model"]["path"]
    revision_metadata = model_path / ".cache/huggingface/download/config.json.metadata"
    assert revision_metadata.read_text().splitlines()[0] == config["model"]["revision"]
    model_class, processor_class, source_sha = load_official_classes(model_path)
    processor = processor_class.from_pretrained(model_path, padding_side="right", local_files_only=True)

    def prepare(conversations: list[list[dict[str, Any]]]) -> dict[str, torch.Tensor]:
        texts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images, video_inputs, video_kwargs = process_vision_info(conversations, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True)
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            videos, video_metadata = None, None
        inputs = processor(text=texts, images=images, videos=videos, video_metadata=video_metadata, truncation=True, max_length=config["model"]["max_length"], padding=True, do_resize=False, return_tensors="pt", **video_kwargs)
        return {key: value.to("cuda") for key, value in inputs.items()}

    def new_core_model():
        return model_class.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=config["model"]["attn_implementation"], low_cpu_mem_usage=True)

    if args.prepare_base:
        model = new_core_model().to("cuda").eval()
        model.config.use_cache = False

        def embed(conversations: list[list[dict[str, Any]]]) -> torch.Tensor:
            inputs = prepare(conversations)
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)
            return pool_last(outputs.last_hidden_state, inputs["attention_mask"])

        gallery_ids = [row["positive_target_id"] for row in validation_rows] + [row["negative_target_id"] for row in validation_rows]
        assert len(gallery_ids) == len(set(gallery_ids))
        chunks = []
        for start in range(0, len(gallery_ids), 8):
            chunks.append(embed([candidate_conversation(value) for value in gallery_ids[start:start + 8]]).cpu())
        gallery = torch.cat(chunks).float()
        index = {product_id: idx for idx, product_id in enumerate(gallery_ids)}
        cached_rows = []
        baseline_rows = []
        top_k = train_cfg["eval_top_k"]
        for number, row in enumerate(validation_rows, 1):
            variants = ["original", "paraphrase", "text_role_swap", "paraphrase_text_role_swap"]
            queries = embed([query_conversation(row, variant) for variant in variants]).cpu().float()
            scores = queries @ gallery.T
            positive_index, negative_index = index[row["positive_target_id"]], index[row["negative_target_id"]]
            original_top = torch.topk(scores[0], top_k)
            paraphrase_top = torch.topk(scores[1], top_k)
            rank = int((scores[0] > scores[0, positive_index]).sum()) + 1
            para_rank = int((scores[1] > scores[1, positive_index]).sum()) + 1
            cached_rows.append({
                "bundle_id": row["bundle_id"], "positive_target_id": row["positive_target_id"], "negative_target_id": row["negative_target_id"],
                "base_rank": rank, "base_paraphrase_rank": para_rank,
                "base_positive_score": float(scores[0, positive_index]), "base_negative_score": float(scores[0, negative_index]),
                "base_paraphrase_positive_score": float(scores[1, positive_index]), "base_paraphrase_negative_score": float(scores[1, negative_index]),
                "base_binding_margin": float(scores[0, positive_index] - scores[2, positive_index]),
                "base_paraphrase_binding_margin": float(scores[1, positive_index] - scores[3, positive_index]),
                "original_top_ids": [gallery_ids[idx] for idx in original_top.indices.tolist()], "original_top_scores": original_top.values.tolist(),
                "paraphrase_top_ids": [gallery_ids[idx] for idx in paraphrase_top.indices.tolist()], "paraphrase_top_scores": paraphrase_top.values.tolist(),
            })
            baseline_rows.append({
                "rank": rank, "ap_at_10": 1.0 / rank if rank <= 10 else 0.0,
                "paraphrase_rank": para_rank, "paraphrase_ap_at_10": 1.0 / para_rank if para_rank <= 10 else 0.0,
                "top1_id": gallery_ids[int(scores[0].argmax())], "paraphrase_top1_id": gallery_ids[int(scores[1].argmax())],
                "binding_margin": float(scores[0, positive_index] - scores[2, positive_index]), "dod_margin": 0.0,
                "hard_negative_violation": bool(scores[0, negative_index] >= scores[0, positive_index]),
            })
            if number % 32 == 0:
                print(json.dumps({"event": "base_cache", "queries": number, "total": len(validation_rows)}), flush=True)
        with (RUN_DIR / "base_validation.jsonl").open("w") as handle:
            for row in cached_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {"status": "pass", "gpu_uuid": config["gpu_uuid"], "model_revision": config["model"]["revision"], "official_source_sha256": source_sha, "gallery_size": len(gallery_ids), "top_k": top_k, "metrics": summarize(baseline_rows), "cache_sha256": sha256_file(RUN_DIR / "base_validation.jsonl")}
        (RUN_DIR / "base_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    core = new_core_model()
    lora = LoraConfig(r=train_cfg["lora_rank"], lora_alpha=train_cfg["lora_alpha"], lora_dropout=train_cfg["lora_dropout"], bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type=TaskType.FEATURE_EXTRACTION)
    model = get_peft_model(core, lora)
    if train_cfg["train_visual_merger"]:
        for name, parameter in model.named_parameters():
            if ".visual.merger." in name:
                parameter.requires_grad = True
    if train_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.to("cuda").train()
    text_config = getattr(model.config, "text_config", model.config)
    head = nn.Linear(text_config.hidden_size, 1, bias=True).to("cuda")

    lora_parameters, merger_parameters, unexpected = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_parameters.append(parameter)
        elif ".visual.merger." in name:
            merger_parameters.append(parameter)
        else:
            unexpected.append(name)
    assert lora_parameters and merger_parameters and not unexpected
    optimizer = torch.optim.AdamW([
        {"params": lora_parameters, "lr": train_cfg["learning_rate_lora"]},
        {"params": merger_parameters, "lr": train_cfg["learning_rate_visual_merger"]},
        {"params": head.parameters(), "lr": train_cfg["learning_rate_head"]},
    ], weight_decay=train_cfg["weight_decay"])

    def score_conversations(conversations: list[list[dict[str, Any]]]) -> torch.Tensor:
        inputs = prepare(conversations)
        outputs = model(**inputs, use_cache=False)
        return head(pool_last(outputs.last_hidden_state, inputs["attention_mask"])).squeeze(-1)

    def one_query_loss(row: dict[str, Any], batch: list[dict[str, Any]], arm: str, retain_scores: bool = False):
        candidate_ids = [row["positive_target_id"], row["negative_target_id"]]
        for other in batch:
            if other["bundle_id"] != row["bundle_id"]:
                candidate_ids.extend([other["positive_target_id"], other["negative_target_id"]])
        original = score_conversations([joint_conversation(row, "original", candidate_id) for candidate_id in candidate_ids])
        rank_loss = torch.stack([pairwise_margin_loss(original[0], value, train_cfg["rank_margin"], train_cfg["temperature"]) for value in original[1:]]).mean()
        zero = torch.zeros((), device="cuda")
        cf_loss, dod_loss, para_loss = zero, zero, zero
        swapped = torch.empty(0, device="cuda")
        paraphrase = torch.empty(0, device="cuda")
        if arm == "positive_counterfactual":
            swapped = score_conversations([joint_conversation(row, "text_role_swap", row["positive_target_id"])])
            cf_loss = pairwise_margin_loss(original[0], swapped[0], train_cfg["counterfactual_margin"], train_cfg["temperature"])
        elif arm == "dod_counterfactual":
            swapped = score_conversations([
                joint_conversation(row, "text_role_swap", row["positive_target_id"]),
                joint_conversation(row, "text_role_swap", row["negative_target_id"]),
            ])
            paraphrase = score_conversations([
                joint_conversation(row, "paraphrase", row["positive_target_id"]),
                joint_conversation(row, "paraphrase_text_role_swap", row["positive_target_id"]),
            ])
            positive_delta = original[0] - swapped[0]
            negative_delta = original[1] - swapped[1]
            paraphrase_delta = paraphrase[0] - paraphrase[1]
            cf_loss = pairwise_margin_loss(original[0], swapped[0], train_cfg["counterfactual_margin"], train_cfg["temperature"])
            dod_loss = pairwise_margin_loss(positive_delta, negative_delta, train_cfg["dod_margin"], train_cfg["temperature"])
            para_loss = F.smooth_l1_loss(positive_delta, paraphrase_delta)
        if retain_scores:
            original.retain_grad()
            if swapped.numel():
                swapped.retain_grad()
            if paraphrase.numel():
                paraphrase.retain_grad()
        loss = rank_loss + train_cfg["counterfactual_weight"] * cf_loss + train_cfg["dod_weight"] * dod_loss + train_cfg["paraphrase_weight"] * para_loss
        return loss, {"rank": rank_loss, "counterfactual": cf_loss, "dod": dod_loss, "paraphrase": para_loss}, {"original": original, "swapped": swapped, "paraphrase": paraphrase}

    parameter_summary = {"total": sum(parameter.numel() for parameter in model.parameters()) + sum(parameter.numel() for parameter in head.parameters()), "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) + sum(parameter.numel() for parameter in head.parameters()), "lora": sum(parameter.numel() for parameter in lora_parameters), "visual_merger": sum(parameter.numel() for parameter in merger_parameters), "head": sum(parameter.numel() for parameter in head.parameters())}

    if args.audit_only:
        optimizer.zero_grad(set_to_none=True)
        loss, parts, scores = one_query_loss(train_rows[0], train_rows[:2], "dod_counterfactual", True)
        loss.backward()
        pattern = re.compile(r"language_model\.layers\.(\d+)\.")
        all_layers = {int(match.group(1)) for name, _ in model.named_modules() if (match := pattern.search(name))}
        grad_layers = {int(match.group(1)) for name, parameter in model.named_parameters() if "lora_" in name and parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0 if (match := pattern.search(name))}
        checks = {
            "physical_gpu_assertion": True,
            "all_language_layers_lora": bool(all_layers) and all_layers == grad_layers,
            "visual_merger_trainable": any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0 for parameter in merger_parameters),
            "scalar_head_gradient": all(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0 for parameter in head.parameters()),
            "original_positive_and_negative_gradients": scores["original"].grad is not None and torch.count_nonzero(scores["original"].grad[:2]).item() == 2,
            "positive_and_negative_swap_gradients": scores["swapped"].grad is not None and torch.count_nonzero(scores["swapped"].grad).item() == 2,
            "paraphrase_branches_have_gradient": scores["paraphrase"].grad is not None and torch.count_nonzero(scores["paraphrase"].grad).item() == 2,
            "finite_loss": bool(torch.isfinite(loss)),
        }
        summary = {"status": "pass" if all(checks.values()) else "fail", "classification": "faithful" if all(checks.values()) else "incorrect", "checks": checks, "gpu_uuid": config["gpu_uuid"], "model_revision": config["model"]["revision"], "official_source_sha256": source_sha, "parameter_summary": parameter_summary, "language_layers": len(all_layers), "language_layers_with_nonzero_lora_grad": len(grad_layers), "loss": float(loss.detach()), "loss_parts": {key: float(value.detach()) for key, value in parts.items()}, "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated()}
        (RUN_DIR / "runtime_gradient_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["status"] != "pass":
            raise SystemExit(2)
        return

    assert json.loads((RUN_DIR / "runtime_gradient_audit.json").read_text())["classification"] == "faithful"
    assert (RUN_DIR / "base_validation.jsonl").exists()
    base_rows = {row["bundle_id"]: row for row in [json.loads(line) for line in (RUN_DIR / "base_validation.jsonl").open() if line.strip()]}

    @torch.no_grad()
    def score_eval(row: dict[str, Any], variant: str, candidate_ids: list[str]) -> dict[str, float]:
        model.eval()
        values = []
        for start in range(0, len(candidate_ids), train_cfg["eval_batch_size"]):
            conversations = [joint_conversation(row, variant, candidate_id) for candidate_id in candidate_ids[start:start + train_cfg["eval_batch_size"]]]
            values.extend(score_conversations(conversations).float().cpu().tolist())
        model.train()
        return dict(zip(candidate_ids, values))

    def rerank(base_ids: list[str], base_scores: list[float], auxiliary: dict[str, float], positive_id: str, base_rank: int) -> tuple[int, str, dict[str, float]]:
        combined = [left + right for left, right in zip(zscore(base_scores), zscore([auxiliary[value] for value in base_ids]))]
        final = dict(zip(base_ids, combined))
        ordered = sorted(base_ids, key=lambda value: (-final[value], value))
        rank = ordered.index(positive_id) + 1 if positive_id in ordered else base_rank
        return rank, ordered[0], final

    def evaluate() -> tuple[dict[str, float], list[dict[str, Any]]]:
        rows_out = []
        for number, row in enumerate(validation_rows, 1):
            base = base_rows[row["bundle_id"]]
            positive_id, negative_id = row["positive_target_id"], row["negative_target_id"]
            original_ids = list(dict.fromkeys(base["original_top_ids"] + [positive_id, negative_id]))
            para_ids = list(dict.fromkeys(base["paraphrase_top_ids"] + [positive_id, negative_id]))
            original = score_eval(row, "original", original_ids)
            swapped = score_eval(row, "text_role_swap", original_ids)
            paraphrase = score_eval(row, "paraphrase", para_ids)
            paraphrase_swapped = score_eval(row, "paraphrase_text_role_swap", para_ids)
            delta = {value: original[value] - swapped[value] for value in original_ids}
            para_delta = {value: paraphrase[value] - paraphrase_swapped[value] for value in para_ids}
            raw_rank, raw_top1, raw_final = rerank(base["original_top_ids"], base["original_top_scores"], original, positive_id, base["base_rank"])
            delta_rank, delta_top1, delta_final = rerank(base["original_top_ids"], base["original_top_scores"], delta, positive_id, base["base_rank"])
            raw_para_rank, raw_para_top1, _ = rerank(base["paraphrase_top_ids"], base["paraphrase_top_scores"], paraphrase, positive_id, base["base_paraphrase_rank"])
            delta_para_rank, delta_para_top1, _ = rerank(base["paraphrase_top_ids"], base["paraphrase_top_scores"], para_delta, positive_id, base["base_paraphrase_rank"])
            use_delta = args.arm != "ordinary_verifier"
            rank, top1 = (delta_rank, delta_top1) if use_delta else (raw_rank, raw_top1)
            para_rank, para_top1 = (delta_para_rank, delta_para_top1) if use_delta else (raw_para_rank, raw_para_top1)
            top_ids = base["original_top_ids"]
            final = delta_final if use_delta else raw_final
            if positive_id in top_ids and negative_id in top_ids:
                hard_violation = final[negative_id] >= final[positive_id]
            elif positive_id in top_ids:
                hard_violation = False
            elif negative_id in top_ids:
                hard_violation = True
            else:
                hard_violation = base["base_negative_score"] >= base["base_positive_score"]
            binding_margin = delta[positive_id]
            dod_margin = delta[positive_id] - delta[negative_id]
            rows_out.append({
                "bundle_id": row["bundle_id"], "target_class": row["target_class"], "active_role_key": row["active_role_key"],
                "positive_target_id": positive_id, "negative_target_id": negative_id,
                "rank": rank, "ap_at_10": 1.0 / rank if rank <= 10 else 0.0, "top1_id": top1,
                "paraphrase_rank": para_rank, "paraphrase_ap_at_10": 1.0 / para_rank if para_rank <= 10 else 0.0, "paraphrase_top1_id": para_top1,
                "base_rank": base["base_rank"], "base_paraphrase_rank": base["base_paraphrase_rank"],
                "raw_rerank_rank": raw_rank, "delta_rerank_rank": delta_rank,
                "raw_paraphrase_rank": raw_para_rank, "delta_paraphrase_rank": delta_para_rank,
                "binding_margin": binding_margin, "negative_binding_margin": delta[negative_id], "dod_margin": dod_margin,
                "paraphrase_binding_margin": para_delta[positive_id],
                "hard_negative_violation": bool(hard_violation),
                "original_positive_score": original[positive_id], "swap_positive_score": swapped[positive_id],
                "original_negative_score": original[negative_id], "swap_negative_score": swapped[negative_id],
            })
            if number % 16 == 0:
                print(json.dumps({"event": "verifier_evaluation", "arm": args.arm, "queries": number, "total": len(validation_rows)}), flush=True)
        return summarize(rows_out), rows_out

    started = time.time()
    history = []
    torch.cuda.reset_peak_memory_stats()
    steps = 0
    for epoch in range(train_cfg["epochs"]):
        order = list(train_rows)
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), train_cfg["batch_size"]):
            if time.time() - started > train_cfg["max_gpu_hours_per_arm"] * 3600:
                raise RuntimeError("frozen per-arm GPU-hour limit exceeded")
            batch = order[start:start + train_cfg["batch_size"]]
            assert len(batch) == train_cfg["batch_size"]
            optimizer.zero_grad(set_to_none=True)
            aggregate = {"rank": 0.0, "counterfactual": 0.0, "dod": 0.0, "paraphrase": 0.0, "loss": 0.0}
            for row in batch:
                loss, parts, _ = one_query_loss(row, batch, args.arm)
                (loss / len(batch)).backward()
                aggregate["loss"] += float(loss.detach()) / len(batch)
                for key, value in parts.items():
                    aggregate[key] += float(value.detach()) / len(batch)
            torch.nn.utils.clip_grad_norm_([parameter for parameter in list(model.parameters()) + list(head.parameters()) if parameter.requires_grad], 1.0)
            optimizer.step()
            steps += 1
            history.append({"step": steps, "epoch": epoch + 1, **aggregate})
            if steps == 1 or steps % 10 == 0:
                print(json.dumps({"arm": args.arm, **history[-1]}), flush=True)
    metrics, eval_rows = evaluate()
    with (RUN_DIR / f"eval_after_{args.arm}.jsonl").open("w") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {"status": "complete", "arm": args.arm, "gpu_uuid": config["gpu_uuid"], "seed": seed, "epochs": train_cfg["epochs"], "steps": steps, "model_revision": config["model"]["revision"], "official_source_sha256": source_sha, "parameter_summary": parameter_summary, "metrics": metrics, "history": history, "elapsed_seconds": time.time() - started, "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated()}
    (RUN_DIR / f"pilot_{args.arm}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
