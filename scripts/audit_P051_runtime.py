#!/usr/bin/env python3
"""GPU1 上验证真实 Qwen 计算、梯度、更新和检查点；不做效果 pilot。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils.vision_process import process_vision_info
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracebind import (TraceBindHead, assignment_set_nll, binding_advantage,
                       paired_dod_loss, teacher_ranking_loss)


def last_token(hidden, mask):
    last = mask.shape[1] - mask.flip(1).argmax(1) - 1
    return hidden[torch.arange(len(hidden), device=hidden.device), last].float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/P051_tracebind.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    assert Path.cwd().resolve() == Path(config["root"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu_uuid"]
    assert torch.cuda.device_count() == 1
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    gpu_map = dict(line.strip().split(", ") for line in inventory.strip().splitlines())
    assert gpu_map[str(config["gpu_index"])] == config["gpu_uuid"]
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert (uuid if uuid.startswith("GPU-") else "GPU-" + uuid) == config["gpu_uuid"]
    torch.manual_seed(config["training"]["seed"])
    torch.cuda.manual_seed_all(config["training"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    run = Path("runs/P051_tracebind")
    run.mkdir(parents=True, exist_ok=True)
    assert json.loads((run / "math_audit.json").read_text())["classification"] == "math_pass_runtime_and_data_pending"
    mc, tc = config["model"], config["training"]
    model_path = Path(mc["path"])
    assert (model_path / ".cache/huggingface/download/config.json.metadata").read_text().splitlines()[0] == mc["revision"]
    source = model_path / "scripts/qwen3_vl_embedding.py"
    spec = importlib.util.spec_from_file_location("official_tracebind_embedding", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor = module.Qwen3VLProcessor.from_pretrained(model_path, local_files_only=True, padding_side="right")
    manifest = json.loads(Path("runs/P049_counterfactual_role_verifier/image_manifest.json").read_text())
    images = {row["idx"]: row for row in manifest["entries"]}
    row = next(json.loads(line) for line in Path("runs/P049_counterfactual_role_verifier/bundles.jsonl").open())
    for product in row["product_ids"]:
        assert hashlib.sha256(Path(images[product]["path"]).read_bytes()).hexdigest() == images[product]["sha256"]

    def image(product):
        return {"type": "image", "image": "file://" + str(Path(images[product]["path"]).resolve()),
                "min_pixels": mc["min_pixels"], "max_pixels": mc["max_pixels"]}

    def conversation(content, system):
        return [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]

    def query(variant, order=None):
        order = list(range(2)) if order is None else order
        content = []
        for number, source_index in enumerate(order):
            content += [{"type": "text", "text": f"Product {number + 1}:\n"}, image(row["source_ids"][source_index])]
        instruction = row["instructions"][variant]
        if order != [0, 1]:
            instruction = instruction.replace("Product 1", "Product TEMP").replace("Product 2", "Product 1").replace("Product TEMP", "Product 2")
        content.append({"type": "text", "text": instruction})
        return conversation(content, "Represent the multi-reference product retrieval query. Respect the attribute assigned to each numbered product.")

    candidate_ids = [row["positive_target_id"], row["negative_target_id"]]
    candidates = [conversation([image(product)], "Represent the candidate product image for retrieval.") for product in candidate_ids]
    queries = [query("original"), query("paraphrase"), query("original", [1, 0])]
    probe_conversations = []
    for source_id in row["source_ids"]:
        for role in row["role_keys"]:
            probe_conversations.append(conversation([image(source_id), {"type": "text", "text": f"Represent only the visible {role} of this reference product for matching a candidate."}], "Extract the specified visual role from the reference image."))

    def prepare(conversations):
        text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        pics, videos, kwargs = process_vision_info(conversations, image_patch_size=16,
                                                  return_video_metadata=True, return_video_kwargs=True)
        assert videos is None
        inputs = processor(text=text, images=pics, padding=True, truncation=False,
                           do_resize=False, return_tensors="pt", **kwargs)
        assert inputs["input_ids"].shape[1] <= mc["max_length"], "不能截断图像/角色来让审计通过"
        return {key: value.to("cuda") for key, value in inputs.items()}

    inputs_c, inputs_q, inputs_p = prepare(candidates), prepare(queries), prepare(probe_conversations)
    core = module.Qwen3VLForEmbedding.from_pretrained(model_path, local_files_only=True,
                                                     torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                     attn_implementation=mc["attn_implementation"]).to("cuda")
    core.config.use_cache = False
    # 教师在任何 LoRA/merger 更新前编码，并存成冻结值；不使用不断变化的 disable_adapter 学生。
    core.eval()
    with torch.no_grad():
        teacher_g = F.normalize(last_token(core(**inputs_c, use_cache=False).last_hidden_state, inputs_c["attention_mask"]), dim=-1)
        teacher_q = F.normalize(last_token(core(**inputs_q, use_cache=False).last_hidden_state, inputs_q["attention_mask"]), dim=-1)
        teacher_scores = (teacher_q @ teacher_g.T).detach().clone()
    teacher_hash = hashlib.sha256(teacher_scores.cpu().numpy().tobytes()).hexdigest()
    lora = LoraConfig(r=tc["lora_rank"], lora_alpha=tc["lora_alpha"], lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      task_type=TaskType.FEATURE_EXTRACTION)
    model = get_peft_model(core, lora)
    merger = []
    for name, param in model.named_parameters():
        if ".visual.merger." in name:
            param.requires_grad = True
            merger.append(param)
    assert merger
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    hidden_dim = model.config.text_config.hidden_size
    head = TraceBindHead(hidden_dim, mc).to("cuda")
    lora_params = [param for name, param in model.named_parameters() if "lora_" in name and param.requires_grad]
    unexpected = [name for name, param in model.named_parameters() if param.requires_grad and "lora_" not in name and ".visual.merger." not in name]
    assert not unexpected
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": tc["learning_rate_lora"]},
                                  {"params": merger, "lr": tc["learning_rate_visual_merger"]},
                                  {"params": head.parameters(), "lr": tc["learning_rate_head"]}], weight_decay=0.0)
    trainables = {"model." + name: param for name, param in model.named_parameters() if param.requires_grad}
    trainables.update({"head." + name: param for name, param in head.named_parameters()})
    before = {name: param.detach().cpu().clone() for name, param in trainables.items()}
    merge_size = model.config.vision_config.spatial_merge_size

    def forward():
        out_c = model(**inputs_c, use_cache=False).last_hidden_state
        g = F.normalize(last_token(out_c, inputs_c["attention_mask"]), dim=-1)
        slots = []
        for index, grid in enumerate(inputs_c["image_grid_thw"]):
            t, height, width = grid.tolist()
            assert t == 1 and height % merge_size == width % merge_size == 0
            patches = out_c[index, inputs_c["input_ids"][index] == model.config.image_token_id]
            slots.append(head.evidence(patches, height // merge_size, width // merge_size))
        evidence = torch.stack(slots)
        out_q = model(**inputs_q, use_cache=False).last_hidden_state
        q = F.normalize(last_token(out_q, inputs_q["attention_mask"]), dim=-1)
        out_p = model(**inputs_p, use_cache=False).last_hidden_state
        probes = head.probes(last_token(out_p, inputs_p["attention_mask"])).reshape(2, 2, -1)
        scores = [head.score(value, g, probes, evidence) for value in q]
        return scores, (q, g, probes, evidence)

    checks = {"physical_gpu1_uuid_verified": True, "teacher_encoded_before_any_update": True,
              "all_three_forward_paths_exercised": True, "no_unexpected_trainable_parameters": True}
    losses, gradient_layers = [], set()
    started = time.monotonic()
    model.train()
    for step in range(tc["audit_optimizer_steps"]):
        optimizer.zero_grad(set_to_none=True)
        scores, features = forward()
        original, paraphrase, permuted = scores
        rank = sum(F.cross_entropy(score.total[None] / tc["temperature"], torch.tensor([0], device="cuda")) for score in scores) / len(scores)
        assign = assignment_set_nll(original.energies[0], [True, False], [True, True], mc["assignment_temperature"])
        positive_gap = binding_advantage(original.energies[0], [True, False], [False, True], mc["assignment_temperature"])
        negative_gap = binding_advantage(original.energies[1], [True, False], [False, True], mc["assignment_temperature"])
        dod = paired_dod_loss(positive_gap, negative_gap, tc["dod_margin"], tc["temperature"])
        consistency = F.smooth_l1_loss(original.total, paraphrase.total) + F.smooth_l1_loss(original.total, permuted.total)
        distill = sum(teacher_ranking_loss(score.utility, teacher_scores[i], tc["temperature"]) for i, score in enumerate(scores)) / len(scores)
        loss = rank + tc["assignment_weight"] * assign + tc["dod_weight"] * dod + tc["consistency_weight"] * consistency + tc["distillation_weight"] * distill
        loss.backward()
        pattern = re.compile(r"language_model\.layers\.(\d+)\.")
        gradient_layers.update(int(match.group(1)) for name, param in model.named_parameters()
                               if "lora_" in name and param.grad is not None and param.grad.abs().sum().item() > 0
                               if (match := pattern.search(name)))
        checks["merger_nonzero_gradient"] = all(param.grad is not None and param.grad.abs().sum().item() > 0 for param in merger)
        checks["every_head_parameter_nonzero_gradient"] = all(param.grad is not None and param.grad.abs().sum().item() > 0 for param in head.parameters())
        checks["finite_loss_and_gradients"] = bool(torch.isfinite(loss)) and all(param.grad is None or torch.isfinite(param.grad).all().item() for param in trainables.values())
        torch.nn.utils.clip_grad_norm_(list(trainables.values()), 1.0)
        optimizer.step()
        parts = {name: float(value.detach()) for name, value in [("rank", rank), ("assignment", assign), ("dod", dod), ("consistency", consistency), ("distillation", distill)]}
        losses.append(parts)
        print(json.dumps({"audit_step": step + 1, "loss": float(loss.detach()), "parts": parts}), flush=True)
    changed = [name for name, param in trainables.items() if not torch.equal(before[name], param.detach().cpu())]
    del before
    layers = model.config.text_config.num_hidden_layers
    checks["all_language_layers_have_lora_gradient"] = len(gradient_layers) == layers
    checks["lora_merger_and_all_heads_updated"] = (any("lora_" in name for name in changed)
                                                   and any(".visual.merger." in name for name in changed)
                                                   and all("head." + name in changed for name, _ in head.named_parameters()))
    checks["teacher_values_unchanged"] = hashlib.sha256(teacher_scores.cpu().numpy().tobytes()).hexdigest() == teacher_hash
    model.eval()
    with torch.no_grad():
        scores, (q, g, probes, evidence) = forward()
        direct = scores[0].total.detach().clone()
        indexed = torch.cat([head.score(q[0], g[i:i + 1], probes, evidence[i:i + 1]).total for i in range(len(g))])
        checks["indexed_direct_score_equality"] = bool(torch.allclose(direct, indexed, atol=1e-6))
        equivariant = head.score(q[0], g, probes.flip(0), evidence, intended=[1, 0])
        checks["structured_branch_source_permutation_equivariance"] = bool(torch.allclose(direct, equivariant.total, atol=1e-6))
    checkpoint = {"config": config, "model_revision": mc["revision"],
                  "official_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "trainable_model_state": {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad},
                  "head_state": {name: value.cpu() for name, value in head.state_dict().items()},
                  "scope": "runtime_debug_only_not_a_trained_method_checkpoint"}
    checkpoint_path = run / "runtime_debug_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    del checkpoint
    # 故意修改所有可训练参数后重载；检查保存路径是否覆盖 merger、LoRA 和新头。
    with torch.no_grad():
        for param in trainables.values():
            param.add_(0.01)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(loaded["trainable_model_state"], strict=False)
    head.load_state_dict(loaded["head_state"], strict=True)
    del loaded
    with torch.no_grad():
        restored, _ = forward()
        checks["checkpoint_roundtrip_scores_equal"] = bool(torch.allclose(direct, restored[0].total, atol=1e-6))
    report = {"classification": "runtime_paths_pass_data_and_full_training_pipeline_pending" if all(checks.values()) else "incorrect",
              "checks": checks, "is_efficacy_evidence": False, "gpu_uuid": config["gpu_uuid"],
              "language_layers": layers, "gradient_layers": sorted(gradient_layers),
              "trainable_parameters": sum(param.numel() for param in trainables.values()),
              "changed_parameter_tensors": len(changed), "audit_steps": len(losses), "losses": losses,
              "elapsed_seconds": time.monotonic() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "remaining": ["real correspondence dataset integration", "visually verified balanced task dataset", "matched multi-arm training and full-gallery evaluation"],
              "runtime_label_scope": "P049 images reused only to test computational paths; assignment/DoD labels here are not new verified research evidence"}
    (run / "runtime_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
