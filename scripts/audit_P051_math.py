#!/usr/bin/env python3
"""执行真实打分/损失代码的数学审计；不视作方法效果实验。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracebind import (TraceBindHead, assignment_set_nll, binding_advantage,
                       paired_dod_loss, teacher_ranking_loss, correspondence_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/P051_tracebind.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    torch.manual_seed(config["training"]["seed"])
    head = TraceBindHead(24, config["model"])
    q = F.normalize(torch.randn(24), dim=-1)
    g = F.normalize(torch.randn(7, 24), dim=-1)
    probes = head.probes(torch.randn(3, 3, 24))
    evidence = torch.stack([head.evidence(torch.randn(6 * 8, 24), 6, 8) for _ in range(7)])
    score = head.score(q, g, probes, evidence)
    checks = {"six_assignments_for_three_roles": score.energies.shape == (7, 6),
              "posterior_normalization": bool(torch.allclose(score.assignment_log_probs.exp().sum(-1), torch.ones(7)))}
    # 来源行重排同时更新 intended mapping；角色列顺序仍固定。
    order = torch.tensor([2, 0, 1])
    inverse = order.argsort()
    permuted = head.score(q, g, probes[order], evidence, intended=inverse)
    checks["source_reindex_equivariance"] = bool(torch.allclose(score.total, permuted.total, atol=1e-6))
    reordered_roles = head.score(q, g, probes[:, order], evidence, intended=order)
    checks["role_reindex_equivariance"] = bool(torch.allclose(score.total, reordered_roles.total, atol=1e-6))
    indexed = torch.cat([head.score(q, g[k:k + 2], probes, evidence[k:k + 2]).total for k in range(0, 7, 2)])
    checks["chunked_index_equals_direct_score"] = bool(torch.allclose(score.total, indexed, atol=1e-6))
    two = head.score(q, g, probes[:2, :2], evidence)
    expected = F.logsigmoid((two.energies[:, 0] - two.energies[:, 1]) / head.assignment_temperature)
    checks["two_reference_exact_logsigmoid_identity"] = bool(torch.allclose(two.assignment_log_probs[:, 0], expected, atol=1e-6))
    e = torch.tensor([0.9, 0.8, -0.1], requires_grad=True)
    valid, supervised = [True, True, False], [True, True, True]
    set_loss = assignment_set_nll(e, valid, supervised, 0.1)
    set_loss.backward()
    checks["both_valid_assignments_get_reward"] = bool((e.grad[:2] < 0).all() and e.grad[2] > 0)
    unknown = torch.tensor([0.9, 99., 0.1], requires_grad=True)
    unknown_loss = assignment_set_nll(unknown, [True, False, False], [True, False, True], 0.1)
    unknown_loss.backward()
    checks["unknown_assignment_not_punished"] = unknown.grad[1].item() == 0
    ambiguous = assignment_set_nll(e.detach(), [True, True, True], [True, True, True], 0.1)
    checks["all_assignments_valid_zero_nll"] = ambiguous.item() == 0
    pos, neg = torch.tensor([0.8, 0.1], requires_grad=True), torch.tensor([0.4, 0.3], requires_grad=True)
    dod = paired_dod_loss(binding_advantage(pos, [True, False], [False, True], .1),
                          binding_advantage(neg, [True, False], [False, True], .1), .05, .07)
    dod.backward()
    checks["dod_four_gradient_signs"] = bool(pos.grad[0] < 0 and pos.grad[1] > 0 and neg.grad[0] > 0 and neg.grad[1] < 0)
    teacher = torch.randn(7, requires_grad=True)
    student = torch.randn(7, requires_grad=True)
    distill = teacher_ranking_loss(student, teacher, .07)
    distill.backward()
    checks["teacher_target_detached"] = teacher.grad is None and student.grad.abs().sum().item() > 0
    corr_src, corr_tgt = torch.randn(3, 24, requires_grad=True), torch.randn(4, 24, requires_grad=True)
    corr = correspondence_loss(corr_src, corr_tgt, [0, 1, 2], torch.ones(3, 4, dtype=torch.bool))
    corr.backward()
    checks["correspondence_both_sides_gradient"] = bool(corr_src.grad.abs().sum() > 0 and corr_tgt.grad.abs().sum() > 0)
    head.zero_grad()
    loss = F.cross_entropy(score.total.unsqueeze(0) / .07, torch.tensor([0]))
    loss.backward()
    checks["all_head_parameters_receive_gradient"] = all(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    checks["head_has_nonzero_support_and_binding"] = bool(F.softplus(head.log_support_weight) > 0 and F.softplus(head.log_binding_weight) > 0)
    report = {"classification": "math_pass_runtime_and_data_pending" if all(checks.values()) else "incorrect",
              "checks": checks, "source_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "src/tracebind.py").read_bytes()).hexdigest(),
              "is_efficacy_evidence": False, "loss": float(loss.detach())}
    out = Path("runs/P051_tracebind")
    out.mkdir(parents=True, exist_ok=True)
    (out / "math_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
