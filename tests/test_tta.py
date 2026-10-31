from copy import deepcopy
import hashlib
import json
from pathlib import Path
from runpy import run_path
import pytest
import torch
from tools import tta_core as core
from tools.baseline_protocol import CODE
from tools.run_tta import selection


def video(length=128):
    return dict(feats=torch.randn(8,length),segments=torch.tensor([[15.,35.]]),labels=torch.tensor([0]),
        metadata=dict(video_id='v',fps=30.,feat_stride=16,feat_num_frames=32,
            duration=length*16/30,video_length=length,use_temporal_grid_feats=True))


def small_model(tmp_path):
    from libs.models.SADA.meta_archs import SADA
    cfg=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']()
    cfg['model']['adv_method']=['None']
    path=tmp_path/'cache.json';path.write_text(json.dumps(dict(by_video={})))
    cfg['prototype_capacity']=dict(variant='control',prepared_cache=str(path),
        prepared_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    torch.manual_seed(123)
    return SADA(cfg,'cpu').eval()


def pseudo():
    return dict(segment=torch.tensor([15.,35.]),label=0,level=0,probability=.8,
        class_weight=.6,boundary_std=.03,location_weight=.9,support=2)


def test_gt_is_removed_and_rejected_at_model_boundary():
    original=video();clean=core.unlabeled(original)
    assert not clean['segments'].numel() and not clean['labels'].numel()
    assert original['segments'].numel() and original['labels'].numel()
    with pytest.raises(ValueError):core.assert_unlabeled(original)


def test_views_are_aligned_and_crop_metadata_does_not_mutate_original():
    clean=core.unlabeled(video(640));old=deepcopy(clean['metadata'])
    assert core.view_ranges(640,32)==[(0,640),(0,576),(64,640)]
    sub=core.crop(clean,64,640)
    assert sub['metadata']['video_length']==576 and clean['metadata']==old
    torch.testing.assert_close(sub['feats'],clean['feats'][:,64:])
    assert core.view_ranges(64,32)==[(0,64)]


def test_matching_is_class_aware_and_one_to_one():
    a=[dict(segment=torch.tensor([1.,5.]),label=0),dict(segment=torch.tensor([2.,6.]),label=0)]
    b=[dict(segment=torch.tensor([1.,5.]),label=0)]
    assert len(core.match(a,b))==1
    b[0]['label']=1;assert not core.match(a,b)


def test_custom_dense_inference_is_exact_original_target_path(tmp_path):
    model=small_model(tmp_path);clean=core.unlabeled(video())
    with torch.no_grad():
        original=model([clean],[clean],mode='test')[1][0]
        current=core.predict_full(model,clean)
    for key in ('segments','scores','labels'):assert torch.equal(original[key],current[key])


def test_update_reaches_shared_features_freezes_heads_and_resets_each_video(tmp_path,monkeypatch):
    model=small_model(tmp_path);teacher=deepcopy(model)
    for p in teacher.parameters():p.requires_grad_(False)
    named=core.adaptable_parameters(model);initial={n:p.detach().clone() for n,p in named.items()}
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if n not in named}
    monkeypatch.setattr(core,'consensus',lambda m,v:[pseudo()])
    clean=core.unlabeled(video(64))
    bank=dict(centers=torch.nn.functional.normalize(torch.randn(model.num_classes,128),dim=-1),
        ready=torch.ones(model.num_classes,dtype=torch.bool))
    one,a=core.adapt_video(model,teacher,initial,clean,'prototype',1e-4,1,bank)
    two,b=core.adapt_video(model,teacher,initial,clean,'prototype',1e-4,1,bank)
    assert a==b and a['changed'] and a['trace'][0]['gradient_norm']>0
    assert a['trace'][0]['prototype']>0 and a['trace'][0]['boundary_matches']>0
    for n,p in model.named_parameters():
        if n in frozen:assert torch.equal(p,frozen[n])
    for key in ('segments','scores','labels'):assert torch.equal(one[key],two[key])


def test_empty_pseudo_falls_back_exactly_without_parameter_update(tmp_path,monkeypatch):
    model=small_model(tmp_path);teacher=deepcopy(model)
    named=core.adaptable_parameters(model);initial={n:p.detach().clone() for n,p in named.items()}
    clean=core.unlabeled(video(64));before=core.predict_full(model,clean)
    monkeypatch.setattr(core,'consensus',lambda m,v:[])
    after,proof=core.adapt_video(model,teacher,initial,clean,'self_train',1e-4,3)
    assert not proof['changed'] and not any(x['updated'] for x in proof['trace'])
    for key in ('segments','scores','labels'):assert torch.equal(before[key],after[key])


def test_boundary_uncertainty_does_not_remove_reliable_class_supervision(tmp_path):
    model=small_model(tmp_path);core.adaptable_parameters(model)
    out=core.dense(model,core.unlabeled(video(64)));row=pseudo();row['boundary_std']=.5
    bank=dict(centers=torch.nn.functional.normalize(torch.randn(model.num_classes,128),dim=-1),
        ready=torch.ones(model.num_classes,dtype=torch.bool))
    loss,proof=core.adaptation_loss(model,out,[row],0,64,'uncertainty',bank)
    assert proof['positive_matches']==1 and proof['boundary_matches']==0 and proof['classification']>0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)


def test_source_selection_uses_six_direction_scores_and_favors_cheaper_ties():
    trials=[dict(lr=lr,steps=steps,scores=[.4]*6) for lr in (1e-5,1e-4) for steps in (1,3)]
    assert selection(trials)['lr']==1e-5 and selection(trials)['steps']==1
    trials[3]['scores']=[.5]*6;assert selection(trials)==trials[3]
    with pytest.raises(ValueError):selection(trials[:3])
    bad=deepcopy(trials);bad[0]['scores'][0]=float('nan')
    with pytest.raises(ValueError):selection(bad)
    bad=deepcopy(trials);bad[0]['lr']=.01
    with pytest.raises(ValueError):selection(bad)


def test_no_target_supervision_or_background_negatives_in_loss():
    import ast
    text=(CODE/'tools/tta_core.py').read_text(encoding='utf-8')
    node=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='adaptation_loss')
    source=ast.unparse(node)
    assert 'label_points' not in source and 'cross_entropy(' not in source
    assert 'binary_cross_entropy_with_logits' in source
    assert "row['label']" in source and "row['probability']" in source
