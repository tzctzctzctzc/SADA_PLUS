import torch
import torch.nn.functional as F


class ResidualAdapter(torch.nn.Module):
    """Zero-initialized residual adapter for one FPN level."""

    def __init__(self, channels, bottleneck=64):
        super().__init__()
        self.down = torch.nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.activation = torch.nn.ReLU()
        self.up = torch.nn.Conv1d(bottleneck, channels, kernel_size=1)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, features):
        return features + self.up(self.activation(self.down(features)))


def deterministic_subsample(values, max_points):
    """Keep a deterministic, approximately uniform subset of rows."""
    if max_points is None or values.shape[0] <= max_points:
        return values
    indices = torch.linspace(
        0,
        values.shape[0] - 1,
        max_points,
        device=values.device,
    ).round().long()
    return values.index_select(0, indices)


def action_label_distribution(action_labels=None, action_probs=None):
    """Turn action labels/probabilities into action-plus-background distributions."""
    if (action_labels is None) == (action_probs is None):
        raise ValueError("provide exactly one of action_labels/action_probs")
    if action_labels is not None:
        action = action_labels.float()
        action_mass = action.sum(dim=1, keepdim=True)
        normalized_action = action / action_mass.clamp_min(1.0)
        background = (action_mass == 0).float()
        return torch.cat([normalized_action, background], dim=1)

    action = action_probs.float().clamp(0.0, 1.0)
    background = (1.0 - action.max(dim=1, keepdim=True).values).clamp(0.0, 1.0)
    distribution = torch.cat([action, background], dim=1)
    return distribution / distribution.sum(dim=1, keepdim=True).clamp_min(1e-6)


def unbalanced_sinkhorn(cost, epsilon=0.1, marginal_relaxation=0.5, iterations=30):
    """Compute an entropic unbalanced transport plan for a small minibatch."""
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError("cost must be a non-empty matrix")
    if epsilon <= 0 or marginal_relaxation <= 0:
        raise ValueError("epsilon and marginal_relaxation must be positive")

    source_mass = cost.new_full((cost.shape[0],), 1.0 / cost.shape[0])
    target_mass = cost.new_full((cost.shape[1],), 1.0 / cost.shape[1])
    kernel = torch.exp(-cost / epsilon).clamp_min(1e-12)
    exponent = marginal_relaxation / (marginal_relaxation + epsilon)
    source_scaling = torch.ones_like(source_mass)
    target_scaling = torch.ones_like(target_mass)
    for _ in range(iterations):
        source_scaling = (
            source_mass
            / (kernel @ target_scaling).clamp_min(1e-12)
        ).pow(exponent)
        target_scaling = (
            target_mass
            / (kernel.transpose(0, 1) @ source_scaling).clamp_min(1e-12)
        ).pow(exponent)
    return source_scaling[:, None] * kernel * target_scaling[None, :]


def joint_unbalanced_ot_loss(
    source_features,
    target_features,
    source_labels,
    target_probs,
    feature_weight=1.0,
    label_weight=1.0,
    epsilon=0.1,
    marginal_relaxation=0.5,
    iterations=30,
    max_points=256,
):
    """Match source/target anchors with feature and soft-label costs."""
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("source and target feature dimensions must match")
    if source_features.shape[0] == 0 or target_features.shape[0] == 0:
        zero = source_features.sum() * 0.0 + target_features.sum() * 0.0
        return zero, {
            "transport_mass": zero.detach(),
            "rejected_mass": zero.detach(),
            "source_points": 0,
            "target_points": 0,
        }

    source_features = deterministic_subsample(source_features, max_points)
    target_features = deterministic_subsample(target_features, max_points)
    source_labels = deterministic_subsample(source_labels, max_points)
    target_probs = deterministic_subsample(target_probs, max_points)

    source_features = F.normalize(source_features.float(), dim=1)
    target_features = F.normalize(target_features.float(), dim=1)
    source_distribution = action_label_distribution(action_labels=source_labels)
    target_distribution = action_label_distribution(action_probs=target_probs)

    feature_cost = 1.0 - source_features @ target_features.transpose(0, 1)
    label_cost = 1.0 - source_distribution @ target_distribution.transpose(0, 1)
    cost = feature_weight * feature_cost + label_weight * label_cost
    plan = unbalanced_sinkhorn(
        cost,
        epsilon=epsilon,
        marginal_relaxation=marginal_relaxation,
        iterations=iterations,
    )
    detached_plan = plan.detach()
    transport_mass = detached_plan.sum().clamp_min(1e-8)
    loss = (detached_plan * cost).sum() / transport_mass
    rejected_mass = (1.0 - detached_plan.sum()).clamp_min(0.0)
    return loss, {
        "transport_mass": transport_mass.detach(),
        "rejected_mass": rejected_mass.detach(),
        "source_points": source_features.shape[0],
        "target_points": target_features.shape[0],
    }
