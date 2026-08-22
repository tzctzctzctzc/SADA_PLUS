import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "libs"
    / "models"
    / "SADA"
    / "target_adapter.py"
)
SPEC = importlib.util.spec_from_file_location("target_adapter", MODULE_PATH)
target_adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(target_adapter)
build_oracle_target_distillation = (
    target_adapter.build_oracle_target_distillation
)


def test_selected_oracle_uses_only_high_confidence_enabled_anchors():
    result = build_oracle_target_distillation(
        scores=torch.tensor([[0.8, 0.4, 0.9, 0.7]]),
        gt_labels=torch.tensor([[
            [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]
        ]]),
        valid_mask=torch.tensor([[True, True, True, False]]),
        enabled_levels=torch.tensor([[True, True, False, True]]),
        alpha=0.6,
        mode="selected",
    )

    assert result["selected"].tolist() == [[True, False, False, False]]
    assert result["foreground"].tolist() == [[True, False, False, False]]
    assert not result["background"].any()


def test_full_oracle_includes_foreground_and_background_without_aliasing_gt():
    labels = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
    result = build_oracle_target_distillation(
        scores=torch.tensor([[0.1, 0.2, 0.3]]),
        gt_labels=labels,
        valid_mask=torch.tensor([[True, True, True]]),
        enabled_levels=torch.tensor([[True, True, False]]),
        alpha=0.6,
        mode="full",
    )

    assert result["selected"].tolist() == [[True, True, False]]
    assert result["foreground"].tolist() == [[True, False, False]]
    assert result["background"].tolist() == [[False, True, False]]
    assert result["targets"].data_ptr() != labels.data_ptr()


def test_oracle_rejects_bad_shapes_and_off_mode():
    values = torch.ones(1, 2)
    labels = torch.ones(1, 2, 1)
    with pytest.raises(ValueError, match="unknown"):
        build_oracle_target_distillation(
            values, labels, values.bool(), values.bool(), 0.6, "off"
        )
    with pytest.raises(ValueError, match="incompatible"):
        build_oracle_target_distillation(
            values, labels, torch.ones(1, 3).bool(), values.bool(), 0.6, "full"
        )
