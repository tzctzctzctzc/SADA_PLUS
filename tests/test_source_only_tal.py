from libs.trainer.training_regimes import (
    build_training_generators,
    consumes_target_train,
    epoch_length_from_config,
    training_epoch_length,
    uses_alignment_optimizer,
)
import torch
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest


def test_source_only_uses_only_source_epoch_length():
    assert training_epoch_length("source_only", 4, 7) == 4
    assert training_epoch_length("source_only", 11, 3) == 11
    assert not consumes_target_train("source_only")


def test_other_regimes_keep_existing_epoch_length_rules():
    assert training_epoch_length("target_only", 4, 7) == 7
    assert training_epoch_length("off", 4, 7) == 7
    assert training_epoch_length("source_target", 9, 7) == 9
    assert consumes_target_train("off")


def test_formal_so_and_uda_use_the_same_max_domain_budget():
    for source, target in ((25, 53), (62, 110), (26, 52), (61, 111), (172, 78), (78, 172)):
        assert training_epoch_length("source_only", source, target, "so") == max(source, target)
        assert training_epoch_length("off", source, target, "uda") == max(source, target)


def test_formal_so_uses_metadata_budget_not_its_source_alias():
    cfg = {
        "diagnostic": {"supervised_upper_bound": "source_only"},
        "rq2_formal": {"regime": "so", "epoch_budget": {
            "source_batches": 25, "target_batches": 53,
        }},
    }
    assert epoch_length_from_config(cfg, 25, 25) == 53
    assert not consumes_target_train("source_only")
    cfg["diagnostic"]["supervised_upper_bound"] = "off"
    cfg["rq2_formal"]["regime"] = "uda"
    assert epoch_length_from_config(cfg, 25, 53) == 53


def test_formal_source_optimizer_switch_matches_sada_warmup():
    for epoch in (1, 4, 5, 50):
        expected = epoch >= 5
        assert uses_alignment_optimizer("source_only", epoch, 5, "so") == expected
        assert uses_alignment_optimizer("off", epoch, 5, "uda") == expected
        assert not uses_alignment_optimizer("source_only", epoch, 5)


@pytest.mark.parametrize("downstream_weight", [2.0, 4.0])
def test_actual_trainer_source_update_matches_uda_with_alignment_removed(downstream_weight):
    # Execute the actual loss/optimizer/scheduler blocks, not a duplicate model
    # of their behavior. A scalar source loss suffices to expose branch drift.
    path = Path(__file__).parents[1] / "libs/trainer/da_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_train_epoch")
    loop = next(node for node in method.body if isinstance(node, ast.For))
    loss_branch = next(node for node in loop.body
                       if isinstance(node, ast.If) and isinstance(node.body[0], ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "task_loss"
                               for t in node.body[0].targets))
    optimizer_branch = next(node for node in loop.body
                            if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                            and isinstance(node.test.operand, ast.Name)
                            and node.test.operand.id == "use_alignment_optimizer")
    scheduler_branches = [node for node in loop.body if isinstance(node, ast.If)
                          and any(isinstance(part, ast.Attribute)
                                  and part.attr.startswith("lr_scheduler_")
                                  for part in ast.walk(node.test))]
    loss_code = compile(ast.Module(body=[loss_branch], type_ignores=[]), str(path), "exec")
    update_code = compile(ast.Module(body=[optimizer_branch, *scheduler_branches],
                                    type_ignores=[]), str(path), "exec")
    outcomes = []
    for mode, regime in (("source_only", "so"), ("off", "uda")):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        source = torch.optim.AdamW([parameter], lr=1e-3)
        alignment = torch.optim.AdamW([parameter], lr=1e-3)
        disc_steps = []
        cfg = {"model": {"loss_weights": {"weight_downstream": downstream_weight,
                                           "weight_adv": 1.0, "weight_semantic": 0.0}}}
        trainer = SimpleNamespace(
            cfg=cfg, model=SimpleNamespace(transport_alignment_mode="off"),
            optimizer_gen_source=source, optimizer_gen_source_alignment=alignment,
            optimizer_disc=SimpleNamespace(step=lambda: disc_steps.append(1)),
            lr_scheduler_gen_source=torch.optim.lr_scheduler.StepLR(source, step_size=10),
            lr_scheduler_gen_source_alignment=torch.optim.lr_scheduler.StepLR(alignment, step_size=10),
            lr_scheduler_disc=None,
        )
        for epoch in range(1, 7):
            source.zero_grad(set_to_none=True)
            alignment.zero_grad(set_to_none=True)
            loss = parameter.square()
            context = dict(self=trainer, supervised_mode=mode, curr_epoch=epoch,
                           warmup_adv=5, adv_method=[], adv_disc=False, iter_idx=0,
                           matched_source_only=regime == "so", alignment_components=[],
                           use_alignment_optimizer=uses_alignment_optimizer(mode, epoch, 5, regime),
                           losses={"downstream_loss": loss}, task_loss=loss * 0,
                           alignment_loss=loss * 0, final_loss=loss * 0)
            exec(loss_code, context)
            expected_weight = downstream_weight if epoch >= 5 else 1.0
            assert context["final_loss"].item() == pytest.approx(expected_weight * loss.item())
            context["final_loss"].backward()
            exec(update_code, context)
        assert source.state[parameter]["step"].item() == 4
        assert alignment.state[parameter]["step"].item() == 2
        assert trainer.lr_scheduler_gen_source_alignment.last_epoch == 6
        assert len(disc_steps) == (0 if regime == "so" else 2)
        outcomes.append(parameter.detach())
    torch.testing.assert_close(outcomes[0], outcomes[1], rtol=0, atol=0)


def test_target_iterator_does_not_change_source_order():
    source_a, target_a = build_training_generators(6)
    source_b, _ = build_training_generators(6)
    torch.randperm(100, generator=target_a)
    assert torch.equal(
        torch.randperm(100, generator=source_a),
        torch.randperm(100, generator=source_b),
    )
