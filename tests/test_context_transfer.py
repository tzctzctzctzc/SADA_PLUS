import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest
import torch

# 纯坐标测试不要求Windows安装Linux NMS；实际SADA集成另在Linux执行。
CODE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('context_transfer_geometry',CODE/'libs/models/SADA/context_transfer.py')
geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)


def video(name,length=64,labelled=False):
    return dict(feats=torch.arange(8*length,dtype=torch.float32).reshape(8,length),
        segments=torch.tensor([[15.,35.]]) if labelled else torch.empty(0,2),
        labels=torch.tensor([0]) if labelled else torch.empty(0,dtype=torch.long),
        metadata=dict(video_id=name,fps=1.,feat_stride=1.,feat_num_frames=2,crop_start=0,crop_feat_len=length))


def context_config(tmp_path):
    path=tmp_path/'source.json'
    path.write_text(json.dumps(dict(database={
        's':dict(subset='source',annotations=[dict(segment=[16.,36.],label_id=0,label='TAKE')]),
        'held_out':dict(subset='validation',annotations=[dict(segment=[1.,2.],label_id=1,label='PUT')])})))
    return dict(context_transfer=dict(schema=geometry.SCHEMA,policy=copy.deepcopy(geometry.POLICY)),
        dataset=dict(json_file=str(path),action_subsampling='10_majority_classes'),src_train_split=['source'],
        baseline=dict(source_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def test_full_gt_uses_original_units_and_does_not_recover_clipped_boundaries():
    meta=dict(fps=32,feat_stride=16,feat_num_frames=32,crop_start=10)
    gt,labels=geometry.feature_ground_truth([dict(segment=[4.,8.],label_id=0),dict(segment=[0,9],label_id=20)],meta,10,'cpu')
    torch.testing.assert_close(gt,torch.tensor([[-3.,5.]]));assert labels.tolist()==[0]
    assert geometry.paste_pair(video('s',16,True),video('t',32),gt,labels,5,0) is None


def test_patch_exact_labels_translate_target_context_and_inputs_unchanged():
    src,tgt=video('s',64,True),video('t',64);tgt['feats'].fill_(-100)
    old_src,old_tgt=copy.deepcopy(src),copy.deepcopy(tgt)
    state=torch.get_rng_state().clone()
    mixed,payload=geometry.paste_pair(src,tgt,src['segments'],src['labels'],5,0)
    a,b=payload['original_source_patch'];c,d=payload['pasted']
    assert c>=1 and d<=63 and a==5 and b==45
    assert torch.equal(mixed['feats'][:,c:d],src['feats'][:,a:b])
    assert torch.equal(mixed['feats'][:,:c],tgt['feats'][:,:c]) and torch.equal(mixed['feats'][:,d:],tgt['feats'][:,d:])
    assert torch.equal(payload['segments'],src['segments']+c-a)
    assert mixed['segments'].numel()==mixed['labels'].numel()==0
    assert torch.equal(src['feats'],old_src['feats']) and torch.equal(tgt['feats'],old_tgt['feats'])
    assert torch.equal(torch.get_rng_state(),state)
    repeated=geometry.paste_pair(src,tgt,src['segments'],src['labels'],5,0)
    assert repeated[1]['pasted']==payload['pasted']


def test_target_gt_is_rejected_instead_of_used_to_choose_background():
    src,tgt=video('s',64,True),video('t',64,True)
    with pytest.raises(ValueError):geometry.paste_pair(src,tgt,src['segments'],src['labels'],5,0)


def test_no_fitting_patch_is_skipped_without_resampling_or_stretching():
    src,tgt=video('s',64,True),video('t',32)
    assert geometry.paste_pair(src,tgt,src['segments'],src['labels'],5,0) is None


def test_partial_instance_is_unknown_and_cannot_create_negative_points():
    src,tgt=video('s',40,True),video('t',20)
    gt=torch.tensor([[15.,19.],[0.,40.]])
    mixed,payload=geometry.paste_pair(src,tgt,gt,torch.tensor([0,1]),5,0)
    assert payload['labels'].tolist()==[0] and len(payload['partial'])==1
    points=[torch.stack([torch.arange(20),torch.zeros(20),torch.ones(20)*100,torch.ones(20)],-1)]
    known=geometry.known_points(points,[torch.ones(1,1,20,dtype=torch.bool)],[payload])
    assert not known.any()


def test_known_mask_uses_each_level_stride_padding_and_partial_region():
    grid=torch.tensor([[0.,0,100,1],[1,0,100,1],[2,0,100,1],[3,0,100,1],
                       [4,0,100,2],[5,0,100,2],[6,0,100,2],[7,0,100,2]])
    payload=dict(pasted=[0,8],partial=torch.tensor([[2.,3.]]))
    valid=torch.tensor([[[1,1,1,1,1,1,1,0]]],dtype=torch.bool)
    assert geometry.known_points([grid],[valid],[payload]).tolist()==[[False,True,False,False,True,True,True,False]]


def test_annotation_cache_only_keeps_training_split_and_rejects_sha_drift(tmp_path):
    cfg=context_config(tmp_path);state=torch.get_rng_state().clone()
    helper=geometry.SourceContextTransfer(cfg,10)
    assert set(helper.annotations)=={'s'} and torch.equal(torch.get_rng_state(),state)
    Path(cfg['dataset']['json_file']).write_text('{}')
    with pytest.raises(ValueError):geometry.SourceContextTransfer(cfg,10)


def test_selection_replays_with_iteration_and_valid_lengths_without_rng():
    src,tgt=video('s',64,True),video('t',128)
    state=torch.get_rng_state().clone()
    first=geometry.paste_pair(src,tgt,src['segments'],src['labels'],5,0,iteration=2)[1]
    replay=geometry.paste_pair(src,tgt,src['segments'],src['labels'],5,0,iteration=2)[1]
    assert first['pasted']==replay['pasted'] and torch.equal(torch.get_rng_state(),state)
    values={tuple(geometry.paste_pair(src,tgt,src['segments'],src['labels'],5,0,iteration=i)[1]['pasted']) for i in range(12)}
    assert len(values)>1


@pytest.mark.skipif(sys.platform=='win32',reason='真实损失依赖Linux SADA包')
def test_unknown_logits_and_offsets_have_zero_direct_supervision_gradient():
    from libs.models.SADA.context_transfer import masked_task_losses
    logits=torch.zeros(2,5,2,requires_grad=True);offsets=torch.ones(2,5,2,requires_grad=True)
    labels=torch.zeros(2,5,2);labels[0,2,0]=1
    truth=torch.ones(2,5,2)*2;known=torch.ones(2,5,dtype=torch.bool);known[0,4]=False
    cls,reg,count,known,positive=masked_task_losses(logits,offsets,labels,truth,known)
    assert count==1 and not known[1].any()
    cls.backward(retain_graph=True);reg.backward()
    assert logits.grad[~known].abs().sum()==0 and logits.grad[known].abs().sum()>0
    assert offsets.grad[~positive].abs().sum()==0 and offsets.grad[positive].abs().sum()>0


@pytest.mark.skipif(sys.platform=='win32',reason='真实SADA/NMS依赖在Linux独立环境验证')
def test_real_sada_initialization_warmup_loss_gradients_rng_and_restore(tmp_path,monkeypatch):
    from runpy import run_path
    from libs.models.SADA.meta_archs import SADA
    from libs.models.SADA.optimizer import make_optimizer
    cfg=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']()
    proto=run_path(str(CODE/'tests/test_prototype_capacity.py'))
    cfg['prototype_capacity']=proto['configuration'](tmp_path,'control')
    source_cfg=context_config(tmp_path);cfg['dataset'].update(source_cfg['dataset'])
    cfg['baseline']=source_cfg['baseline'];cfg['src_train_split']=source_cfg['src_train_split']
    cfg['model']['pl_threshold']=0
    native=torch.tensor
    def cpu_copy(*args,**kwargs):
        value=native(*args,**kwargs)
        return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    torch.manual_seed(123);base=SADA(copy.deepcopy(cfg),'cpu');initial_rng=torch.get_rng_state().clone()
    new_cfg=copy.deepcopy(cfg);new_cfg['context_transfer']=source_cfg['context_transfer']
    torch.manual_seed(123);model=SADA(new_cfg,'cpu');model.context_gradient_probe=True
    assert torch.equal(initial_rng,torch.get_rng_state())
    assert set(base.state_dict())==set(model.state_dict())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    source,target=proto['videos']()
    outputs=[];rng=[]
    for m in (base,model):
        torch.manual_seed(456)
        outputs.append(m(video_list_src=[source],video_list_tgt=[target],mode='train',curr_epoch=4)[0])
    for name in ('cls_loss','reg_loss','rq3_loss','downstream_loss'):assert torch.equal(outputs[0][name],outputs[1][name])
    outputs=[]
    for m in (base,model):
        torch.manual_seed(789)
        outputs.append(m(video_list_src=[source],video_list_tgt=[target],mode='train',curr_epoch=5)[0]);rng.append(torch.get_rng_state().clone())
    assert torch.equal(rng[0],rng[1])
    a,b=outputs
    for name in ('cls_loss','reg_loss','rq3_loss'):assert torch.equal(a[name],b[name])
    assert torch.equal(b['downstream_loss'],a['downstream_loss']+b['context_task_loss'])
    for key in ('positive_points','known_negative_points','feature_gradient_norm','cls_head_gradient_norm','reg_head_gradient_norm',
                'cls_backbone_gradient_norm','reg_backbone_gradient_norm','cls_fpn_gradient_norm','reg_fpn_gradient_norm'):
        assert b['context_'+key]>0
    for key in ('rng_restored','normalizers_unchanged','prototype_state_unchanged'):
        assert b['context_'+key]==1
    backbone=model.backbone_source.embd[0].conv.weight
    assert torch.autograd.grad(b['context_task_loss'],backbone,retain_graph=True)[0].abs().sum()>0
    before={name:p.detach().clone() for name,p in model.named_parameters() if name.startswith(('cls_head.','reg_head.','backbone_source.'))}
    optimizer=make_optimizer(model,cfg['opt'],.0001,exclude_params=['disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings'])
    optimizer.zero_grad();b['context_task_loss'].backward();optimizer.step()
    for prefix in ('cls_head.','reg_head.','backbone_source.'):
        assert any(not torch.equal(before[n],p) for n,p in model.named_parameters() if n in before and n.startswith(prefix))
    restored=SADA(new_cfg,'cpu');restored.load_state_dict(model.state_dict(),strict=True)


@pytest.mark.skipif(sys.platform=='win32',reason='协议依赖实际SADA包，在Linux完整运行')
def test_declared_pair_rejects_extra_seed_or_budget_changes():
    from tools.context_transfer_protocol import assert_pair
    base=dict(seed=123,loader={'batch_size':2},prototype_capacity={'variant':'control'},
              baseline=dict(code={'commit':'old'},tmpdir='old',target_gt_used=False))
    cfg=copy.deepcopy(base);cfg['baseline'].update(code={'commit':'new'},tmpdir='new')
    cfg['context_transfer']=dict(schema=geometry.SCHEMA,policy=copy.deepcopy(geometry.POLICY));assert_pair(cfg,base)
    bad=copy.deepcopy(cfg);bad['loader']['batch_size']=4
    with pytest.raises(ValueError):assert_pair(bad,base)
