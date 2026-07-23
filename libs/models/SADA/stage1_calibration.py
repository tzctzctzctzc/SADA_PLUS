import hashlib
import json
from pathlib import Path

import torch

from .stage1_utils import weighted_quantile


def layer_balanced_weights(levels):
    weights = torch.zeros_like(levels, dtype=torch.float32)
    for level in torch.unique(levels):
        selected = levels == level
        weights[selected] = 1.0 / selected.sum()
    return weights


def calibrate_thresholds(records, quantile=0.90, epsilon_bg=0.01):
    """
    Calibrate beta/tau from detached source-validation anchor records.

    Required tensors: level, is_action, r, action_eligible, action_std,
    background_valid, background_std.
    """
    if not records:
        raise ValueError("empty Stage 1 source calibration records")
    data = {key: torch.cat([item[key].cpu() for item in records]) for key in records[0]}
    valid = data["background_valid"].bool()
    if not valid.any():
        raise ValueError("empty strict-valid source calibration set")
    beta_level = data["level"][valid]
    beta_r = data["r"][valid]
    beta_action = data["is_action"][valid]
    beta_weights = layer_balanced_weights(beta_level)
    candidates = torch.unique(beta_r).sort().values
    beta = None
    for candidate in candidates:
        selected = beta_r <= candidate
        denominator = beta_weights[selected].sum()
        if denominator <= 0:
            continue
        contamination = (
            beta_weights[selected] * beta_action[selected].float()
        ).sum() / denominator
        if contamination <= epsilon_bg:
            beta = candidate
    if beta is None:
        raise ValueError("no non-empty beta satisfies source contamination constraint")

    action = data["action_eligible"].bool()
    if not action.any():
        raise ValueError("empty eligible source action calibration set")
    action_weights = layer_balanced_weights(data["level"][action])
    tau_act = weighted_quantile(
        data["action_std"][action], action_weights, quantile
    )

    background = (
        data["background_valid"].bool()
        & (~data["is_action"].bool())
        & (data["r"] <= beta)
    )
    if not background.any():
        raise ValueError("empty eligible source background calibration set")
    background_weights = layer_balanced_weights(data["level"][background])
    tau_bg = weighted_quantile(
        data["background_std"][background], background_weights, quantile
    )
    selected = valid & (data["r"] <= beta)
    selected_count = int(selected.sum())
    pooled_contamination = float(
        data["is_action"][selected].float().mean()
    )
    selected_beta = beta_r <= beta
    balanced_contamination = float(
        (
            beta_weights[selected_beta]
            * beta_action[selected_beta].float()
        ).sum() / beta_weights[selected_beta].sum()
    )
    valid_action = valid & data["is_action"].bool()
    valid_background = valid & (~data["is_action"].bool())
    action_leakage = float(
        selected[valid_action].float().mean()
    ) if valid_action.any() else 0.0
    background_retention = float(
        selected[valid_background].float().mean()
    ) if valid_background.any() else 0.0
    return {
        "beta": float(beta),
        "tau_act": float(tau_act),
        "tau_bg": float(tau_bg),
        "quantile": quantile,
        "epsilon_bg": epsilon_bg,
        "selected_background": int(background.sum()),
        "selected_count": selected_count,
        "balanced_contamination": balanced_contamination,
        "pooled_contamination": pooled_contamination,
        "action_leakage": action_leakage,
        "background_retention": background_retention,
        "eligible_action": int(action.sum()),
    }


def save_calibration_artifact(path, thresholds, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    payload = dict(thresholds)
    payload["manifest"] = manifest
    payload["manifest_sha256"] = hashlib.sha256(
        manifest_json.encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_calibration_artifact(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
