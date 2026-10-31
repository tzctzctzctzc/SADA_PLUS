from pathlib import Path
from runpy import run_path
from copy import deepcopy
import torch
from tools import ttt_recovery as core
from tools.tta_core import unlabeled

helpers=run_path(str(Path(__file__).with_name('test_tta.py')))


def test_foreground_weight_is_detached_positive_bounded_and_prefers_action():
    p=torch.tensor([[.001,.01,.2,.8]],requires_grad=True);mask=torch.ones_like(p,dtype=torch.bool)
    w=core.foreground_weights(p,mask)
    assert not w.requires_grad and bool((w>0).all()) and float(w.max())<=5.1
    assert bool((w[:,1:]>w[:,:-1]).all())
    torch.testing.assert_close(core.foreground_weights(p*2,mask),w)


def test_weighted_recovery_keeps_reference_and_heads_frozen(tmp_path):
    model=helpers['small_model'](tmp_path);named,head=core.initialize(model);reference=deepcopy(model)
    for p in reference.parameters():p.requires_grad_(False)
    for p in head.parameters():p.requires_grad_(False)
    loss,proof=core.recovery_loss(model,reference,head,unlabeled(helpers['video'](64)),123)
    loss.backward()
    assert proof['foreground_weight_min']>0 and proof['foreground_probability_mean']>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in named.values())
    assert all(p.grad is None for p in reference.parameters())
    assert all(p.grad is None for p in head.parameters())


def test_revision_reuses_frozen_heads_and_only_runs_six_new_targets():
    from tools.baseline_protocol import CODE
    text=(CODE/'tools/run_ttt_recovery.py').read_text(encoding='utf-8')
    assert "prepared_path=args.previous_recovery/'prepare'/variant/s/'best.pt'" in text
    assert "prepared=[audit(args.previous_recovery/'prepare'/variant/s/'job.json') for s in SHIFTS]" in text
    assert "target_jobs=6,reused_original_prototype=6,revision=2" in text
