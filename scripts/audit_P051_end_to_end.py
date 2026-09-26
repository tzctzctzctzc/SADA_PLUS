#!/usr/bin/env python3
"""真实二/三参考与精细对应数据的完整计算路径审计，不是效果训练。"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracebind import correspondence_loss, tracebind_objective
from tracebind_encoder import TraceBindEncoder


def main():
    config = json.loads(Path("configs/P051_tracebind.json").read_text())
    assert Path.cwd().resolve() == Path(config["root"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu_uuid"] and torch.cuda.device_count() == 1
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert (uuid if uuid.startswith("GPU-") else "GPU-" + uuid) == config["gpu_uuid"]
    torch.manual_seed(config["training"]["seed"]); torch.cuda.manual_seed_all(config["training"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    run = Path("runs/P051_tracebind")
    rows = list(map(json.loads, (run / "orbits_byte_disjoint.jsonl").open()))
    train = [row for row in rows if row["split"] == "train"]
    chosen = [next(row for row in train if len(row["source_ids"]) == n) for n in [2, 3]]
    ambiguous = next((row for row in train if sum(row["valid_assignments"]) > 1), None)
    if ambiguous and ambiguous not in chosen:
        chosen.append(ambiguous)
    images = {entry["idx"]: entry for entry in json.loads((run / "orbits_image_manifest.json").read_text())["entries"]}
    csummary = json.loads((run / "correspondence_integrity_audit.json").read_text())
    assert csummary["status"] == "pass"
    pairs = list(itertools.islice(map(json.loads, (run / "correspondences_disjoint.jsonl").open()), 32))
    for pair in pairs:
        images[pair["uid"] + "-src"] = {"path": pair["source_image"], "sha256": pair["source_sha256"]}
        images[pair["uid"] + "-tgt"] = {"path": pair["target_image"], "sha256": pair["target_sha256"]}
    for product in {p for row in chosen for p in row["product_ids"]}:
        assert hashlib.sha256(Path(images[product]["path"]).read_bytes()).hexdigest() == images[product]["sha256"]
    encoder = TraceBindEncoder(config, images)
    encoder.model.eval()
    teacher = {}
    # 使用多个真实候选作为冻结邻域，核验蒸馏不是只看一对正负的空接口。
    neighbours_ids = list(dict.fromkeys(row[key] for row in train[:16] for key in ["positive_target_id", "negative_target_id"]))
    with torch.no_grad():
        neighbours = encoder.encode_images(neighbours_ids, local=False).detach()
        for row in chosen:
            g = encoder.encode_images([row["positive_target_id"], row["negative_target_id"]], local=False)
            q = torch.stack([encoder.encode_query(row), encoder.encode_query(row, "paraphrase"),
                             encoder.encode_query(row, order=reversed(range(len(row["source_ids"]))))])
            teacher[row["bundle_id"]] = (q.detach(), g.detach(), (q @ g.T).detach())
    frozen_hash = hashlib.sha256(neighbours.cpu().numpy().tobytes()).hexdigest()
    encoder.adapt(); optimizer = encoder.optimizer(); params = encoder.trainable_parameters()
    before = {name: param.detach().cpu().clone() for name, param in params.items()}
    checks, history, layer_grads, feature_norms = {}, [], set(), []
    for step in range(2):
        encoder.model.train(); optimizer.zero_grad(set_to_none=True)
        for row in chosen:
            scores, (q, g, probes, banks) = encoder.score_row(row)
            tq, tg, ts = teacher[row["bundle_id"]]
            probes.retain_grad(); banks.retain_grad(); g.retain_grad()
            for value in q:
                value.retain_grad()
            geometry = {"student_queries": q, "student_candidates": g, "teacher_queries": tq,
                        "teacher_candidates": tg, "teacher_neighbours": neighbours}
            loss, parts = tracebind_objective(scores, row, ts, config, geometry=geometry)
            assert torch.isfinite(loss), "实际目标函数出现非有限损失"
            (loss / len(chosen)).backward()
            feature_norms.append({"source_count": len(row["source_ids"]), "probe_gradient": float(probes.grad.norm()),
                                  "evidence_gradient": float(banks.grad.norm()), "gallery_global_gradient": float(g.grad.norm()),
                                  "query_global_gradients": [float(value.grad.norm()) for value in q]})
            history.append({"step": step, "bundle_id": row["bundle_id"], "valid_assignment_count": sum(row["valid_assignments"]),
                            "parts": {name: float(value.detach()) for name, value in parts.items()}})
        pattern = re.compile(r"language_model\.layers\.(\d+)\.")
        layer_grads.update(int(match.group(1)) for name, param in encoder.model.named_parameters()
                           if "lora_" in name and param.grad is not None and param.grad.abs().sum().item() > 0
                           if (match := pattern.search(name)))
        checks["merger_all_parameters_have_gradient"] = all(param.grad is not None and param.grad.abs().sum().item() > 0
                                                             for name, param in encoder.model.named_parameters() if ".visual.merger." in name)
        checks["heads_all_parameters_have_gradient"] = all(param.grad is not None and param.grad.abs().sum().item() > 0 for param in encoder.head.parameters())
        if step == 1:
            # 第一轮 LoRA B 为零时 A 的梯度可为零；第二轮仍必须逐张量检查。
            checks["all_lora_tensors_have_gradient_after_second_step"] = all(param.grad is not None and param.grad.abs().sum().item() > 0
                                                                            for name, param in encoder.model.named_parameters() if "lora_" in name)
        checks["finite_loss_and_all_gradients"] = all(torch.isfinite(param.grad).all().item() for param in params.values() if param.grad is not None)
        assert checks["finite_loss_and_all_gradients"]
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0); optimizer.step()
    changed = [name for name, param in params.items() if not torch.equal(before[name], param.detach().cpu())]
    del before
    changed_layers = {int(match.group(1)) for name in changed if "lora_" in name if (match := pattern.search(name))}
    checks["all_merger_and_head_parameters_update"] = all(name in changed for name in params if name.startswith("head.") or ".visual.merger." in name)
    checks["all_lora_tensors_update_after_two_steps"] = all(name in changed for name in params if "lora_" in name)
    checks["all_36_layers_gradient_and_update"] = len(layer_grads) == len(changed_layers) == encoder.model.config.text_config.num_hidden_layers
    checks["all_feature_branches_receive_gradient"] = all(row["probe_gradient"] > 0 and row["evidence_gradient"] > 0 and row["gallery_global_gradient"] > 0
                                                           and all(value > 0 for value in row["query_global_gradients"]) for row in feature_norms)
    checks["frozen_teacher_neighbourhood_unchanged"] = hashlib.sha256(neighbours.cpu().numpy().tobytes()).hexdigest() == frozen_hash
    # 真正进入精细对应损失，所有目标坐标均来自双方法一致且在前景掩码内的点。
    encoder.model.train(); optimizer.zero_grad(set_to_none=True)
    corr_info = None
    for pair in pairs:
        assert hashlib.sha256(Path(pair["point_path"]).read_bytes()).hexdigest() == pair["point_sha256"]
        products = [pair["uid"] + "-src", pair["uid"] + "-tgt"]
        _, _, patches, shapes = encoder.encode_images(products)
        points = np.load(pair["point_path"])
        sy, ty = points["source_yx"], points["target_yx"]
        sh, sw = shapes[0]; th, tw = shapes[1]
        si = np.floor(sy[:, 0] * sh / 64).astype(int) * sw + np.floor(sy[:, 1] * sw / 64).astype(int)
        ti = np.floor(ty[:, 0] * th / 64).astype(int) * tw + np.floor(ty[:, 1] * tw / 64).astype(int)
        positive = torch.zeros(sh * sw, th * tw, dtype=torch.bool, device="cuda")
        positive[torch.tensor(si, device="cuda"), torch.tensor(ti, device="cuda")] = True
        source_keep, target_keep = positive.any(1), positive.any(0)
        positive = positive[source_keep][:, target_keep]
        if positive.all():
            continue
        source_features = encoder.head.probes(patches[0][source_keep])
        target_features = F.normalize(encoder.head.evidence_projection(patches[1][target_keep]), dim=-1)
        source_features.retain_grad(); target_features.retain_grad()
        # 只使用已有对应的目标格作为对比池；临近正确格不被当作负例。
        target_indices = target_keep.nonzero().flatten()
        target_yx = torch.stack([target_indices // tw, target_indices % tw], -1)
        distances = (target_yx[:, None] - target_yx[None, :]).abs().amax(-1)
        ambiguous = (positive.float() @ (distances <= 1).float()) > 0
        allowed = positive | ~ambiguous
        if not (allowed & ~positive).any():
            continue
        corr_loss = correspondence_loss(source_features, target_features, positive, allowed)
        corr_loss.backward()
        checks["real_correspondence_both_features_have_gradient"] = bool(source_features.grad.norm() > 0 and target_features.grad.norm() > 0)
        checks["real_correspondence_reaches_merger_and_lora"] = any(param.grad is not None and param.grad.abs().sum() > 0 for name, param in encoder.model.named_parameters() if ".visual.merger." in name) and any(param.grad is not None and param.grad.abs().sum() > 0 for name, param in encoder.model.named_parameters() if "lora_" in name)
        corr_info = {"uid": pair["uid"], "loss": float(corr_loss.detach()), "trusted_points": pair["verified_points"],
                     "source_cells": len(positive), "target_cells": positive.shape[1], "multi_positive_cells": int((positive.sum(-1) > 1).sum())}
        break
    checks["real_correspondence_dataset_used"] = corr_info is not None
    encoder.model.eval()
    with torch.no_grad():
        scores, (q, g, probes, banks) = encoder.score_row(chosen[1])
        direct = scores[0].total.clone()
        indexed = torch.cat([encoder.head.score(q[0], g[i:i+1], probes, banks[i:i+1]).total for i in range(len(g))])
        checks["indexed_direct_equality"] = bool(torch.allclose(direct, indexed, atol=1e-6))
        order = torch.tensor([2, 0, 1], device="cuda")
        permuted = encoder.head.score(q[0], g, probes[order], banks, intended=order.argsort())
        checks["three_reference_structure_equivariance"] = bool(torch.allclose(direct, permuted.total, atol=1e-6))
        controls = {arm: encoder.head.score(q[0], g, probes, banks, arm=arm).total for arm in config["training"]["arms"]}
        checks["binding_changes_score_vs_role_sum"] = not torch.allclose(controls["tracebind"], controls["role_sum"], atol=1e-7)
        checks["role_sum_changes_score_vs_global"] = not torch.allclose(controls["role_sum"], controls["ordinary"], atol=1e-7)
        inference_row = {"source_ids": chosen[1]["source_ids"], "instructions": {"original": chosen[1]["instructions"]["original"]}}
        iq, ip = encoder.encode_query(inference_row), encoder.encode_probes(inference_row)
        label_free = encoder.head.score(iq, g, ip, banks).total
        checks["inference_needs_no_candidate_or_role_labels"] = bool(torch.allclose(label_free, direct, atol=1e-6))
    checkpoint = run / "runtime_v2_checkpoint.pt"
    encoder.save(checkpoint, optimizer, {"purpose": "runtime_audit_not_efficacy", "data_sha": hashlib.sha256((run / "orbits_byte_disjoint.jsonl").read_bytes()).hexdigest()})
    cold = TraceBindEncoder(config, images)
    cold.load(checkpoint); cold.model.eval()
    with torch.no_grad():
        restored, _ = cold.score_row(chosen[1])
        checks["fresh_model_checkpoint_roundtrip"] = bool(torch.allclose(direct, restored[0].total, atol=1e-6))
    cold_optimizer = cold.optimizer(); cold.load(checkpoint, cold_optimizer)
    checks["optimizer_state_restored"] = len(cold_optimizer.state) == len(optimizer.state) > 0
    report = {"classification": "complete_computational_paths_pass_training_driver_and_visual_data_gate_pending" if all(checks.values()) else "incorrect",
              "checks": {k: bool(v) for k, v in checks.items()}, "actual_source_counts": [len(row["source_ids"]) for row in chosen],
              "history": history, "feature_gradient_norms": feature_norms, "real_correspondence": corr_info,
              "trainable_parameters": sum(param.numel() for param in params.values()), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "source_hashes": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['src/tracebind.py','src/tracebind_encoder.py','scripts/audit_P051_end_to_end.py']},
              "is_efficacy_evidence": False, "remaining": ["visual relative quality gate and manual inspection", "matched full multi-arm driver and complete gallery metric integration"]}
    (run / "end_to_end_runtime_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
