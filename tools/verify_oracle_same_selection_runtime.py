"""Verify clean-pseudo and GT-boundary Oracle on the real model loss path."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA import meta_archs as meta_archs_module
from libs.utils.utils import _update_config, read_yaml


def move_video(video, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in video.items()
    }


def model_for_mode(cfg, checkpoint, state_key, device, mode):
    local_cfg = dict(cfg)
    local_cfg["diagnostic"] = {
        "oracle_routing": "off",
        "oracle_target_regression": mode,
        "oracle_target_regression_weight": 1.0,
        "target_regression_levels": [0, 1, 2],
    }
    local_cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    local_cfg["begtr"] = {"mode": "off"}
    model = make_meta_arch(
        local_cfg["model_name"], local_cfg, device=device
    ).to(device)
    state = {
        key: value for key, value in checkpoint[state_key].items()
        if not key.startswith("aux_boundary_head.")
    }
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-kind", choices=("ema", "student"), default="ema")
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(20260731)
    torch.cuda.manual_seed_all(20260731)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["begtr"] = {"mode": "off"}
    _update_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_key = "ema_state_dict" if args.model_kind == "ema" else "state_dict"
    source_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"]
    )
    target_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["tgt_train_split"], **cfg["dataset"]
    )
    source = move_video(source_dataset[args.source_index], args.device)
    target = move_video(target_dataset[args.target_index], args.device)

    captures = {}
    original = meta_archs_module.build_same_selection_supervision

    def capture(*values):
        mask, targets = original(*values)
        mode = values[-1]
        captures[mode] = {
            "mask": mask.detach().cpu(),
            "targets": targets[mask].detach().cpu(),
        }
        return mask, targets

    meta_archs_module.build_same_selection_supervision = capture
    losses = {}
    gradients = {}
    try:
        for mode in ("clean_pseudo", "same_selection"):
            model = model_for_mode(
                cfg, checkpoint, state_key, args.device, mode
            )
            output = model(
                video_list_src=[source], video_list_tgt=[target],
                mode="validation", domain="src", curr_epoch=0,
            )
            losses[mode] = output[0]
            model.zero_grad(set_to_none=True)
            losses[mode]["downstream_loss"].backward()
            gradients[mode] = {
                "parameters_with_gradients": sum(
                    parameter.grad is not None for parameter in model.parameters()
                ),
                "finite": all(
                    torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ),
            }
    finally:
        meta_archs_module.build_same_selection_supervision = original

    clean_mask = captures["clean_pseudo"]["mask"]
    oracle_mask = captures["same_selection"]["mask"]
    masks_equal = torch.equal(clean_mask, oracle_mask)
    if not masks_equal:
        raise AssertionError("same-selection modes used different runtime masks")
    common_names = sorted(
        set(losses["clean_pseudo"]) - {
            "oracle_target_reg_loss", "oracle_target_reg_positive",
            "downstream_loss",
        }
    )
    common_differences = {}
    for name in common_names:
        first = torch.as_tensor(losses["clean_pseudo"][name]).detach()
        second = torch.as_tensor(losses["same_selection"][name]).detach()
        difference = float((first - second).abs().max())
        common_differences[name] = difference
        if difference > 1e-6:
            raise AssertionError("common loss changed: {} {}".format(name, difference))
    clean_targets = captures["clean_pseudo"]["targets"]
    oracle_targets = captures["same_selection"]["targets"]
    target_difference = (
        float((clean_targets - oracle_targets).abs().mean())
        if clean_targets.numel() else 0.0
    )
    result = {
        "audit": "real_runtime_same_selection_oracle_equivalence",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "source_video": source["metadata"]["video_id"],
        "target_video": target["metadata"]["video_id"],
        "selected_anchors": int(clean_mask.sum()),
        "masks_exactly_equal": masks_equal,
        "common_loss_max_abs_differences": common_differences,
        "selected_target_mean_abs_difference": target_difference,
        "oracle_regression_loss": {
            mode: float(value["oracle_target_reg_loss"].detach())
            for mode, value in losses.items()
        },
        "backward": gradients,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
