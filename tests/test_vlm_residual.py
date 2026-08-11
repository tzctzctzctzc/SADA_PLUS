import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "libs"
    / "models"
    / "SADA"
    / "vlm_residual.py"
)
SPEC = importlib.util.spec_from_file_location("vlm_residual", MODULE_PATH)
vlm_residual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vlm_residual)
bounded_residual_loss = vlm_residual.bounded_residual_loss
select_supported_residual = vlm_residual.select_supported_residual


@pytest.mark.parametrize(
    "source_samples,level,base,expected",
    [
        (2, 0, [True, False], [False, True]),
        (0, 0, [True, False], [False, False]),
        (2, 1, [True, False], [False, False]),
        (2, 0, [False, False], [False, False]),
    ],
)
def test_residual_requires_l0_source_and_base_target_support(
    source_samples, level, base, expected
):
    accepted = select_supported_residual(
        torch.tensor([False, True]),
        torch.tensor(base),
        source_samples,
        level,
        (0,),
    )

    assert accepted.tolist() == expected


def test_bounded_residual_cannot_exceed_five_percent_of_base():
    base = torch.tensor([2.0, 2.0], requires_grad=True)
    rescue = torch.tensor([100.0], requires_grad=True)

    total, residual, excess = bounded_residual_loss(base, rescue, 0.05)

    assert total.item() == pytest.approx(2.1)
    assert residual.item() == pytest.approx(0.1)
    assert excess.item() == pytest.approx(0.0)
    total.backward()
    assert rescue.grad.item() == pytest.approx(0.001)


def test_bounded_residual_rejects_rescue_without_base_target_support():
    base = torch.empty(0, requires_grad=True)
    rescue = torch.tensor([3.0], requires_grad=True)

    total, residual, excess = bounded_residual_loss(base, rescue, 0.05)

    assert total.item() == 0.0
    assert residual.item() == 0.0
    assert excess.item() == 0.0


def test_small_residual_is_not_scaled_when_already_under_budget():
    total, residual, excess = bounded_residual_loss(
        torch.tensor([2.0]), torch.tensor([0.01]), 0.05
    )

    assert total.item() == pytest.approx(2.01)
    assert residual.item() == pytest.approx(0.01)
    assert excess.item() == pytest.approx(0.0)
