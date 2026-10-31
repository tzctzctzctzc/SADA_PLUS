from copy import deepcopy
from pathlib import Path
from runpy import run_path
import torch
import pytest
from tools import masked_uda as core
from tools.tta_core import dense,unlabeled

helpers=run_path(str(Path(__file__).with_name('test_tta.py')))


def test_masking_is_deterministic_and_input_is_not_mutated():
    video=unlabeled(helpers['video'](128));before=video['feats'].clone()
    a=core.masked_video(video,2);b=core.masked_video(video,2)
    assert torch.equal(a['feats'],b['feats']) and torch.equal(video['feats'],before)
    assert int((a['feats'].abs().sum(0)==0).sum())==64
    assert not torch.equal(a['feats'],core.masked_video(video,3)['feats'])
    with pytest.raises(ValueError):core.masked_video(helpers['video'](),1)


def test_unmasked_identical_teacher_kl_is_zero_and_masked_gradient_reaches_student(tmp_path):
    model=helpers['small_model'](tmp_path);core.trainable_parameters(model);teacher=deepcopy(model)
    for p in teacher.parameters():p.requires_grad_(False)
    video=unlabeled(helpers['video'](64))
    plain,_=core.consistency(model,teacher,[video],'ema_unmasked',1)
    assert abs(float(plain))<1e-6
    masked,proof=core.consistency(model,teacher,[video],'ema_masked',1)
    assert float(masked)>0 and proof['points']>0 and proof['pseudo_regression_loss']==0
    masked.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone_target.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    with pytest.raises(ValueError):core.consistency(model,teacher,[helpers['video']()],'ema_masked',1)


def test_teacher_ema_updates_after_student_and_remains_frozen(tmp_path):
    student=helpers['small_model'](tmp_path);teacher=deepcopy(student)
    for p in teacher.parameters():p.requires_grad_(False)
    before=next(teacher.parameters()).detach().clone()
    with torch.no_grad():next(student.parameters()).add_(1)
    core.update_teacher(teacher,student)
    torch.testing.assert_close(next(teacher.parameters()),before+.01,rtol=1e-5,atol=1e-7)
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())


def test_source_selection_first_boundary_and_ties():
    h=[dict(epoch=2,map=.8),dict(epoch=4,map=.8),dict(epoch=6,map=.7)]
    assert core.select(h)==(h[0],False)
    h.append(dict(epoch=8,map=.7));assert core.select(h)==(h[0],True)
    with pytest.raises(ValueError):core.select([dict(epoch=4,map=.2)])
    with pytest.raises(ValueError):core.select([dict(epoch=2,map=float('nan'))])


def test_consistency_ignores_student_regression_outputs_and_unselected_classes():
    logits=torch.full((1,4,10),-3.,requires_grad=True)
    teacher_logits=torch.full((1,4,10),-3.);teacher_logits[:,:,2]=2.
    student=dict(logits=[logits],offsets=[torch.randn(1,4,2,requires_grad=True)])
    teacher=dict(logits=[teacher_logits],masks=[torch.ones(1,4,dtype=torch.bool)])
    loss,proof=core.positive_kl(student,teacher);loss.backward()
    assert logits.grad[:,:,2].abs().sum()>0 and logits.grad[:,:,:2].abs().sum()==0
    assert student['offsets'][0].grad is None and not proof['target_gt_used']
