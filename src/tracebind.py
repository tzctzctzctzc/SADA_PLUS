"""TraceBind 的可索引计算；本文件不读取查询标签或候选元数据。

轴约定：probes[source, role, dim]，evidence[candidate, slot, dim]。
两个角色允许引用同一空间证据，不施加空间排他约束。
"""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def compile_role_keys(instruction, source_count):
    pattern = r"the same\s+(.+?)\s+(?:with|as(?:\s+in)?)\s+<?Product\s+(\d+)>?"
    matches = re.findall(pattern, instruction, flags=re.I)
    by_source = {int(index) - 1: role.strip().lower() for role, index in matches}
    assert len(matches) == source_count and set(by_source) == set(range(source_count)), "查询必须明示每个来源的角色；不能从候选标签补角色"
    roles = [by_source[i] for i in range(source_count)]
    assert len(set(roles)) == source_count, "当前 pilot 仅验证每来源一个不同角色的查询"
    return roles


@dataclass
class TraceScores:
    total: torch.Tensor
    utility: torch.Tensor
    affinity: torch.Tensor
    energies: torch.Tensor
    assignment_log_probs: torch.Tensor
    intended_support: torch.Tensor
    binding: torch.Tensor
    permutations: torch.Tensor


def assignment_energies(affinity: torch.Tensor, permutations: torch.Tensor):
    """[..., source, role] -> [..., assignment]；使用均值统一角色数量的尺度。"""
    roles = torch.arange(affinity.shape[-1], device=affinity.device)
    return affinity[..., permutations, roles].mean(-1)


class TraceBindHead(nn.Module):
    def __init__(self, hidden_dim: int, config: dict):
        super().__init__()
        dim = config["compact_dim"]
        self.probe_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim, bias=False))
        self.evidence_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim, bias=False))
        self.local_temperature = config["local_temperature"]
        self.assignment_temperature = config["assignment_temperature"]
        self.grid = config["evidence_grid"]
        self.log_support_weight = nn.Parameter(torch.tensor(math.log(math.expm1(config["support_weight"]))))
        self.log_binding_weight = nn.Parameter(torch.tensor(math.log(math.expm1(config["binding_weight"]))))

    def probes(self, hidden: torch.Tensor):
        return F.normalize(self.probe_projection(hidden.float()), dim=-1)

    def evidence(self, patches: torch.Tensor, height: int, width: int):
        # 输入为经过视觉 spatial merger 后、按行排列的语言层图像 token。
        assert patches.shape[0] == height * width
        patches = patches.float().reshape(height, width, -1).permute(2, 0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(patches, (self.grid, self.grid))
        pooled = pooled[0].permute(1, 2, 0).reshape(self.grid * self.grid, -1)
        return F.normalize(self.evidence_projection(pooled), dim=-1)

    def score(self, query_global, candidate_globals, probes, evidence,
              intended=None, arm="tracebind"):
        assert probes.ndim == 3 and evidence.ndim == 3
        sources, roles, _ = probes.shape
        assert sources == roles, "当前轨道要求每来源一个角色；不能静默用于非双射查询"
        permutations = torch.tensor(list(itertools.permutations(range(sources))),
                                    dtype=torch.long, device=probes.device)
        if intended is None:
            intended = torch.arange(roles, device=probes.device)
        else:
            intended = torch.as_tensor(intended, device=probes.device)
        matches = (permutations == intended).all(-1)
        assert matches.sum() == 1
        intended_index = matches.nonzero()[0, 0]
        local = torch.einsum("ijd,ckd->cijk", probes.float(), evidence.float())
        # 减 log K，消除证据槽数量引入的无条件加分。
        affinity = self.local_temperature * (
            torch.logsumexp(local / self.local_temperature, dim=-1) - math.log(evidence.shape[1]))
        energies = assignment_energies(affinity, permutations)
        log_probs = F.log_softmax(energies / self.assignment_temperature, dim=-1)
        support = energies[:, intended_index]
        binding = self.assignment_temperature * log_probs[:, intended_index]
        utility = candidate_globals.float() @ query_global.float()
        if arm == "ordinary" or arm == "permutation_augmentation":
            total = utility
        elif arm == "role_sum":
            total = utility + F.softplus(self.log_support_weight) * support
        else:
            assert arm == "tracebind"
            total = utility + F.softplus(self.log_support_weight) * support + F.softplus(self.log_binding_weight) * binding
        return TraceScores(total, utility, affinity, energies, log_probs, support, binding, permutations)


def assignment_set_nll(energies, valid, supervised, temperature):
    """未知交换不视为负例；多个有效绑定采用集合似然。"""
    valid = torch.as_tensor(valid, dtype=torch.bool, device=energies.device)
    supervised = torch.as_tensor(supervised, dtype=torch.bool, device=energies.device)
    assert valid.shape == supervised.shape == energies.shape
    assert valid.any(-1).all() and (valid & ~supervised).sum() == 0
    logits = energies / temperature
    return (torch.logsumexp(logits.masked_fill(~supervised, -torch.inf), -1) - torch.logsumexp(logits.masked_fill(~valid, -torch.inf), -1)).mean()


def binding_advantage(energies, valid, wrong, temperature):
    valid = torch.as_tensor(valid, dtype=torch.bool, device=energies.device)
    wrong = torch.as_tensor(wrong, dtype=torch.bool, device=energies.device)
    assert valid.any() and wrong.any() and not (valid & wrong).any()
    # 这里比较平均指数能量，避免不同集合大小产生人为优势。
    good = torch.logsumexp((energies / temperature).masked_fill(~valid, -torch.inf), -1) - valid.sum().log()
    bad = torch.logsumexp((energies / temperature).masked_fill(~wrong, -torch.inf), -1) - wrong.sum().log()
    return temperature * (good - bad)


def paired_dod_loss(positive_advantage, negative_advantage, margin, temperature):
    return temperature * F.softplus((margin - positive_advantage + negative_advantage) / temperature)


def teacher_ranking_loss(student, teacher, temperature):
    # teacher 必须来自微调前的原始模型；detach 防止蒸馏目标随学生更新。
    kl = F.kl_div(F.log_softmax(student / temperature, -1),
                  F.softmax(teacher.detach() / temperature, -1), reduction="none").sum(-1)
    # 排序损失同样乘 temperature，以匹配分数梯度尺度，避免额外乘一次温度削弱保护。
    return kl.mean() * temperature


def correspondence_loss(source_features, target_features, target_match_indices,
                        candidate_mask, temperature=0.07):
    """精细对应迁移接口：只用核验后的点；未匹配位置不能自动充当负例。"""
    source_features = F.normalize(source_features.float(), dim=-1)
    target_features = F.normalize(target_features.float(), dim=-1)
    logits = source_features @ target_features.T / temperature
    allowed = torch.as_tensor(candidate_mask, device=logits.device, dtype=torch.bool)
    matches = torch.as_tensor(target_match_indices, device=logits.device)
    if matches.ndim == 1:
        positives = torch.zeros_like(allowed)
        positives[torch.arange(len(logits), device=logits.device), matches.long()] = True
    else:
        positives = matches.bool()
    assert positives.shape == allowed.shape == logits.shape
    assert positives.any(-1).all() and not (positives & ~allowed).any()
    # 网格池化可使一个来源位置对应多个目标槽；不把其他正确位置当作负例。
    return (torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), -1)
            - torch.logsumexp(logits.masked_fill(~positives, -torch.inf), -1)).mean()


def tracebind_objective(scores, row, teacher_scores, config, arm="tracebind", geometry=None):
    """训练与运行审计共用真实标签；DoD 只施加于逐排列核验的偏序。"""
    tc, mc = config["training"], config["model"]
    zero = scores[0].total.new_zeros(())
    selected = scores if arm in {"permutation_augmentation", "tracebind"} else scores[:1]
    rank_terms = []
    for score in selected:
        assert score.total.numel() == 2, "本版本使用唯一正例及已核验的单角色困难负例"
        rank_terms.append(tc["temperature"] * F.softplus((tc["rank_margin"] - score.total[0] + score.total[1]) / tc["temperature"]))
    rank = torch.stack(rank_terms).mean()
    assignment, dod, consistency = zero, zero, zero
    if arm == "tracebind":
        assignment = assignment_set_nll(scores[0].energies[0], row["valid_assignments"], row["supervised_assignments"], mc["assignment_temperature"])
        assert row["verified_dod_assignments"], "不能用未知或无偏序依据的排列补 DoD"
        dod = torch.stack([paired_dod_loss(scores[0].energies[0, 0] - scores[0].energies[0, index],
                                         scores[0].energies[1, 0] - scores[0].energies[1, index],
                                         tc["dod_margin"], tc["temperature"])
                           for index in row["verified_dod_assignments"]]).mean()
        consistency = torch.stack([F.smooth_l1_loss(scores[0].total, score.total) for score in scores[1:]]).mean()
    if geometry is None:
        distillation = torch.stack([teacher_ranking_loss(score.utility, teacher_scores[index], tc["temperature"])
                                    for index, score in enumerate(scores)]).mean()
        anchors = zero
    else:
        q, g = geometry["student_queries"], geometry["student_candidates"]
        tq, tg = geometry["teacher_queries"].detach(), geometry["teacher_candidates"].detach()
        neighbours = geometry["teacher_neighbours"].detach()
        distillation = torch.stack([teacher_ranking_loss(value @ neighbours.T, teacher @ neighbours.T, tc["temperature"])
                                    for value, teacher in zip(q, tq)]).mean()
        anchors = (1 - (torch.stack(q) * tq).sum(-1)).mean() + (1 - (g * tg).sum(-1)).mean()
    loss = (rank + tc["assignment_weight"] * assignment + tc["dod_weight"] * dod
            + tc["consistency_weight"] * consistency + tc["distillation_weight"] * distillation
            + tc.get("geometry_anchor_weight", 0.0) * anchors)
    return loss, {"rank": rank, "assignment": assignment, "dod": dod,
                  "consistency": consistency, "distillation": distillation, "geometry_anchors": anchors}
