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
import random
import numpy as np
from libs.datasets import make_data_loader


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


class RandomCropProbe(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return (index, random.random(), float(np.random.random()), float(torch.rand(())))


def test_epoch_reset_workers_preserve_order_and_crop_rng_after_resume():
    generator = torch.Generator().manual_seed(6)
    loader = make_data_loader(RandomCropProbe(), True, generator, 2, 2,
                              persistent_workers=False)
    list(loader)
    saved = generator.get_state()
    expected = list(loader)
    restored = torch.Generator()
    restored.set_state(saved)
    resumed = make_data_loader(RandomCropProbe(), True, restored, 2, 2,
                               persistent_workers=False)
    assert expected == list(resumed)


def test_actual_checkpoint_control_restores_normalizers_and_scheduler_phases():
    path = Path(__file__).parents[1] / "libs/trainer/base_trainer_da.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name in ("get_formal_training_state", "restore_formal_training_state")]
    namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)

    def trainer(normalizer):
        fields = dict(loss_normalizer=normalizer, loss_normalizer_pseudo_labels=127.5,
                      train_label_smoothing=0.1)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        return SimpleNamespace(
            model=SimpleNamespace(**fields), model_ema=SimpleNamespace(module=SimpleNamespace(**fields)),
            lr_scheduler_gen_source=torch.optim.lr_scheduler.StepLR(optimizer, 2),
            lr_scheduler_gen_source_alignment=torch.optim.lr_scheduler.StepLR(optimizer, 3),
            lr_scheduler_disc=torch.optim.lr_scheduler.StepLR(optimizer, 4),
        )

    continuous, resumed = trainer(318.8), trainer(250)
    # Deliberately distinct phases: SO must not reconstruct a progressing discriminator.
    continuous.lr_scheduler_gen_source.last_epoch = 4
    continuous.lr_scheduler_gen_source_alignment.last_epoch = 9
    continuous.lr_scheduler_disc.last_epoch = 0
    state = namespace["get_formal_training_state"](continuous)
    namespace["restore_formal_training_state"](resumed, state)
    assert namespace["get_formal_training_state"](resumed) == state
    assert 1600 / continuous.model.loss_normalizer == 1600 / resumed.model.loss_normalizer
    with pytest.raises(ValueError, match="complete model/scheduler"):
        namespace["restore_formal_training_state"](resumed, None)


def test_formal_latest_checkpoint_is_saved_after_rng_consuming_reporting():
    path = Path(__file__).parents[1] / "libs/trainer/base_trainer_da.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    train = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                 and node.name == "train")
    loop = next(node for node in train.body if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "epoch"
                and any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                        and item.func.attr == "_train_epoch" for item in ast.walk(node)))
    last_report = max(node.lineno for node in ast.walk(loop) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == "_test_epoch")
    formal_save = next(node for node in loop.body if isinstance(node, ast.If)
                       and isinstance(node.test, ast.BoolOp)
                       and isinstance(node.test.values[0], ast.Name)
                       and node.test.values[0].id == "rq2_formal")
    assert formal_save.lineno > last_report


def test_real_sada_next_loss_and_gradient_match_after_control_restore():
    pytest.importorskip("einops")  # Full-model regression runs in the remote SADA environment.
    from libs.models.SADA import make_meta_arch
    from libs.utils.utils import read_yaml, _update_config
    from libs.utils.parse_config import ConfigParser

    root = Path(__file__).parents[1]
    cfg = read_yaml(root / "configs/dark_vs_all_new_videos_full/version_final_ckpt.yaml")
    # Short CPU input; detector/loss code and all hidden dimensions are unchanged.
    cfg["dataset"]["max_seq_len"] = 256
    cfg["diagnostic"] = {"supervised_upper_bound": "source_only"}
    cfg = _update_config(cfg)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(6)
        model = make_meta_arch(cfg["model_name"], ConfigParser(cfg, save=False), device="cpu")
        model.train()
        video = {"feats": torch.randn(2304, 96), "segments": torch.tensor([[20., 40.]]),
                 "labels": torch.tensor([0]), "video_id": "cpu_probe"}
        model([video], [video], mode="train", curr_epoch=1)
        path = root / "libs/trainer/base_trainer_da.py"
        methods = [node for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                   if isinstance(node, ast.FunctionDef)
                   and node.name in ("get_formal_training_state", "restore_formal_training_state")]
        namespace = {}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
        trainer = SimpleNamespace(model=model, model_ema=SimpleNamespace(module=model),
                                  lr_scheduler_gen_source=None,
                                  lr_scheduler_gen_source_alignment=None, lr_scheduler_disc=None)
        state = namespace["get_formal_training_state"](trainer)
        rng = torch.get_rng_state()
        first = model([video], [video], mode="train", curr_epoch=1)[0]["downstream_loss"]
        first.backward()
        parameter = next(parameter for parameter in model.parameters() if parameter.grad is not None)
        gradient = parameter.grad.clone()
        model.zero_grad(set_to_none=True)
        model.loss_normalizer = model.loss_normalizer_pseudo_labels = 250
        namespace["restore_formal_training_state"](trainer, state)
        torch.set_rng_state(rng)
        restored = model([video], [video], mode="train", curr_epoch=1)[0]["downstream_loss"]
        restored.backward()
        torch.testing.assert_close(first, restored, rtol=0, atol=0)
        torch.testing.assert_close(gradient, parameter.grad, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
