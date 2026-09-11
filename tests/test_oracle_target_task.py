import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "libs"
    / "models"
    / "SADA"
    / "oracle_task.py"
)
SPEC = importlib.util.spec_from_file_location("oracle_task", MODULE_PATH)
oracle_task = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle_task)
compose_oracle_target_task_loss = oracle_task.compose_oracle_target_task_loss


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("off", 2.0),
        ("class", 5.0),
        ("boundary", 12.0),
        ("joint", 15.0),
    ],
)
def test_oracle_target_task_composes_only_selected_gt_component(mode, expected):
    source = torch.tensor(2.0)
    target_class = torch.tensor(3.0)
    target_boundary = torch.tensor(5.0)

    result = compose_oracle_target_task_loss(
        source,
        target_class,
        target_boundary,
        boundary_weight=2.0,
        mode=mode,
    )

    assert result.item() == expected


def test_oracle_target_task_rejects_unknown_mode():
    zero = torch.tensor(0.0)
    with pytest.raises(ValueError, match="unknown Oracle target task mode"):
        compose_oracle_target_task_loss(zero, zero, zero, 1.0, "invalid")


def test_category_only_oracle_has_no_boundary_gradient():
    source = torch.tensor(2.0, requires_grad=True)
    target_class = torch.tensor(3.0, requires_grad=True)
    target_boundary = torch.tensor(5.0, requires_grad=True)

    result = compose_oracle_target_task_loss(
        source,
        target_class,
        target_boundary,
        boundary_weight=2.0,
        mode="class",
    )
    result.backward()

    assert target_class.grad.item() == 1.0
    assert target_boundary.grad is None
