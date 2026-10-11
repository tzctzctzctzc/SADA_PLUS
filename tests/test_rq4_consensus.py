import copy
import importlib
import json
import os
from pathlib import Path
import pytest
import torch
from test_rq4_boundary import PACKAGE,inputs,write
from test_rq4_relational import spec as relation_spec
from tools import rq4_consensus as p
from tools.rq4_protocol import digest,file_sha

M=importlib.import_module('rq4_cpu.rq4_consensus')
Q=importlib.import_module('rq4_cpu.rq4_quality')


def source_bundle(tmp,shift='S1'):
    database={'source_train':{'subset':'train'},'cal1':{'subset':'val'},'cal2':{'subset':'val'},
              'v':{'subset':'target'}}
    write(tmp/'annotations.json',dict(database=database))
    cfg=dict(seed=6,rq2_formal=dict(shift=shift),src_train_split=['train'],src_val_split=['val'],
        tgt_train_split=['target'])
    write(tmp/'baseline.json',cfg); write(tmp/'checkpoint.json',{})
    from tools.rq4_artifacts import reference
    table=dict(schema='rq4-quality-source-table-v1',target_gt_used=False,seed=6,shift=shift,
        train_video_ids=['source_train'],validation_video_ids=['cal1','cal2'],features=[],
        baseline_config=reference(tmp/'baseline.json'),annotations=reference(tmp/'annotations.json'),
        checkpoint=reference(tmp/'checkpoint.json'))
    table['train']=[dict(video_id='source_train',x=[0.]*134,y=[1.,0.,0.])]
    table['validation']=[dict(video_id=v,x=[0.]*134,y=[1.,0.,0.]) for v in ('cal1','cal2')]
    write(tmp/'source.json',table)
    with torch.random.fork_rng(devices=[]):
        model=Q.LocalizationQuality()
        for parameter in model.parameters(): parameter.data.zero_()
        model.net[-1].bias.data.copy_(torch.tensor([8.,-12.,-12.]))
    bundle=dict(schema='rq4-localization-quality-v1',shift=shift,seed=6,target_gt_used=False,
        fit_split='source_train',calibration_split='source_validation',source_table=reference(tmp/'source.json'),
        train_video_ids=['source_train'],validation_video_ids=['cal1','cal2'],
        code_sha256=file_sha(Q.__file__),state_dict={k:v.tolist() for k,v in model.state_dict().items()},
        calibration=dict(quantile=.9,quality_margin=0.,error_margin=[0.,0.],calibration_videos=2,
                         target_coverage_guaranteed=False))
    bundle['payload_sha256']=digest(bundle)
    write(tmp/'quality.json',bundle)
    return bundle


def spec(tmp,video='v'):
    base=relation_spec(tmp,video=video); source_bundle(tmp)
    base.update(variant=p.VARIANT,strategy='consensus',consensus_rule=copy.deepcopy(p.RULE),
        minimum_quality=.5,maximum_uncertainty=.5,anchor_fraction=.5,
        quality_path=str(tmp/'quality.json'),quality_file_sha256=file_sha(tmp/'quality.json'))
    base.pop('relation_rule')
    return base


def args_with_teacher():
    args=inputs(); args[0][0]['feats']=torch.randn(2304,10,requires_grad=True)
    centers=args[1][0][:,0]
    ema=torch.stack(((centers-1).clamp_min(.1),(9-centers).clamp_min(.1)),-1)[None].requires_grad_()
    teacher=dict(points=args[1],masks=args[2],offsets=[ema],video_keys=M.video_keys(args[0]))
    return args,teacher


# 第一轮：独立身份、数据隔离、失败关闭与旧协议兼容。
@pytest.mark.parametrize('field,value',[('weight',1.),('minimum_quality',0),('maximum_uncertainty',float('nan')),
    ('anchor_fraction',1.),('variant','rq4-soft-relational-boundary-v1'),('consensus_rule',{})])
def test_reject_protocol_drift(tmp_path,field,value):
    s=spec(tmp_path); s[field]=value
    with pytest.raises(ValueError): M.RQ4Consensus(s)


def test_source_membership_leakage(tmp_path):
    bundle=source_bundle(tmp_path); bundle['train_video_ids']=['v']
    bundle['payload_sha256']=digest({k:v for k,v in bundle.items() if k!='payload_sha256'})
    with pytest.raises(ValueError,match='leakage'):
        p.validate_quality(bundle,shift='S1',target_ids=['v'])


def test_source_calibration_cannot_claim_target_guarantee(tmp_path):
    b=source_bundle(tmp_path); b['calibration']['target_coverage_guaranteed']=True
    b['payload_sha256']=digest({k:v for k,v in b.items() if k!='payload_sha256'})
    with pytest.raises(ValueError,match='guarantee'): p.validate_quality(b,shift='S1',target_ids=['v'])


def test_changed_source_features_or_membership_rejected(tmp_path):
    b=source_bundle(tmp_path); table=p.read(tmp_path/'source.json')
    table['train'][0]['video_id']='cal1'
    with pytest.raises(ValueError,match='outside source'): p.audit_source_table(table)
    write(tmp_path/'source.json',table)
    with pytest.raises(ValueError,match='SHA'): p.validate_quality(b,shift='S1',target_ids=['v'])


def test_draft_cannot_start_by_boolean(tmp_path):
    from tools.rq4_protocol import assert_launch_ready
    s=spec(tmp_path)
    with pytest.raises(ValueError,match='freeze'): assert_launch_ready({'rq4_boundary':s})
    s['protocol_frozen']=True
    # 缺请求证据本身就失败；一个布尔值绝不是正式验收。
    with pytest.raises((KeyError,ValueError)): assert_launch_ready({'rq4_boundary':s})


# 第二轮：解析梯度、真实离散点、局部连通性、EMA/特征隔离。
def test_local_support_not_tiny_three_way_core():
    grid=torch.tensor([[i,0,100,1] for i in range(10)]).float()
    ids,support,edges=M.local_graph(grid,torch.ones(10,dtype=torch.bool),[1,9],[[1,5],[4,9]],4.,0)
    assert ids.tolist()==[3,4,5] and edges==[(0,1,.5)]
    assert not (4<3<5)  # 最小共同区间只有(4,5)，新方法仍有三个真实点。


def test_disconnected_contexts_or_mask_holes_never_bridge():
    grid=torch.tensor([[i,0,100,1] for i in range(10)]).float()
    valid=torch.ones(10,dtype=torch.bool)
    ids,_,edges=M.local_graph(grid,valid,[1,9],[[1,4],[5,9]],4.,0)
    for a,b,_ in edges: assert not (ids[a]<4 and ids[b]>5)
    valid[4]=False
    ids,_,edges=M.local_graph(grid,valid,[1,9],[[1,9],[1,9]],4.,0)
    for a,b,_ in edges: assert bool(valid[ids[a]:ids[b]+1].all())


def test_anchor_corrects_common_wrong_boundary():
    wrong=torch.tensor([[0.,12.],[0.,12.]],requires_grad=True)
    right=torch.tensor([1.,9.]); radius=torch.tensor([.1,.1])
    loss=M.bounded_anchor_loss(wrong,right,radius,8.).mean()
    assert loss>0; loss.backward()
    assert (wrong.grad[:,0]<0).all() and (wrong.grad[:,1]>0).all()


def test_uncertainty_dead_zone_does_not_pull_inside():
    boxes=torch.tensor([[1.1,8.9]],requires_grad=True)
    loss=M.bounded_anchor_loss(boxes,torch.tensor([1.,9.]),torch.tensor([.2,.2]),8.).sum()
    assert loss==0; loss.backward(); assert boxes.grad.abs().sum()==0


def test_weighted_median_resists_one_outlier_and_detaches():
    boxes=torch.tensor([[1.,9.],[1.2,9.2],[0.,20.]],requires_grad=True)
    ref,rad=M.consensus(boxes,torch.tensor([.8,.8,.1]),torch.zeros(3,2))
    assert torch.allclose(ref,torch.tensor([1.,9.2]))
    assert not ref.requires_grad and not rad.requires_grad


def test_forward_has_student_gradient_but_no_teacher_feature_quality_gradient(tmp_path):
    module=M.RQ4Consensus(spec(tmp_path)); args,teacher=args_with_teacher()
    result=module(*args,True,teacher=teacher)
    assert result['accepted']==2 and result['relation_pairs']>0 and result['anchor_points']>0
    assert result['gradient_norm']>0
    result['loss'].backward()
    assert args[-1][0].grad.abs().sum()>0
    assert teacher['offsets'][0].grad is None and args[0][0]['feats'].grad is None
    assert all(not p.requires_grad and p.grad is None for p in module.quality.parameters())


def test_single_real_point_has_anchor_not_fake_relation(tmp_path):
    module=M.RQ4Consensus(spec(tmp_path)); args,teacher=args_with_teacher()
    args[2][0].fill_(False); args[2][0][0,0,4]=True
    result=module(*args,True,teacher=teacher)
    assert result['anchor_points']==result['single_point_anchors']==2
    assert result['relation_pairs']==0 and result['gradient_norm']>0


def test_partial_crop_still_routes_visible_positions(tmp_path):
    module=M.RQ4Consensus(spec(tmp_path)); args,teacher=args_with_teacher()
    args[0][0]['metadata']['crop_start']=4
    teacher['video_keys']=M.video_keys(args[0])
    result=module(*args,True,teacher=teacher)
    assert result['routable']==2 and result['crop_rejected']==0


def test_target_annotations_ignored(tmp_path):
    module=M.RQ4Consensus(spec(tmp_path)); args,teacher=args_with_teacher()
    before=module(*args,True,teacher=teacher)
    args[0][0].update(labels=object(),segments=object())
    after=module(*args,True,teacher=teacher)
    assert torch.equal(before['loss'],after['loss'])


def test_model_construction_preserves_rng_and_strict_restore(tmp_path):
    s=spec(tmp_path); before=torch.random.get_rng_state().clone()
    one=M.RQ4Consensus(s); assert torch.equal(before,torch.random.get_rng_state())
    two=M.RQ4Consensus(s); two.load_state_dict(one.state_dict(),strict=True)
    assert torch.equal(before,torch.random.get_rng_state())


def test_warmup_is_noop_without_teacher(tmp_path):
    module=M.RQ4Consensus(spec(tmp_path)); args,_=args_with_teacher()
    result=module(*args,False); assert result['accepted']==0 and result['loss']==0
    result['loss'].backward(); assert args[-1][0].grad.abs().sum()==0


def test_ema_grid_mismatch_stops(tmp_path):
    module=M.RQ4Consensus(spec(tmp_path)); args,teacher=args_with_teacher()
    teacher['points']=[teacher['points'][0]+1]
    with pytest.raises(ValueError,match='geometry'): module(*args,True,teacher=teacher)


def test_stale_ema_crop_is_rejected(tmp_path):
    module=M.RQ4Consensus(spec(tmp_path)); args,teacher=args_with_teacher()
    args[0][0]['metadata']['crop_start']=1
    with pytest.raises(ValueError,match='crop/order'): module(*args,True,teacher=teacher)


def test_quality_abstention_does_not_invent_gradient(tmp_path):
    module=M.RQ4Consensus(spec(tmp_path)); args,teacher=args_with_teacher()
    module.calibration['quality_margin']=1.
    result=module(*args,True,teacher=teacher)
    assert result['routable']==result['quality_rejected']==2 and result['accepted']==0
    result['loss'].backward(); assert args[-1][0].grad.abs().sum()==0


def test_low_coverage_does_not_renormalize_remaining_candidate(tmp_path):
    module=M.RQ4Consensus(spec(tmp_path)); args,teacher=args_with_teacher()
    full=module(*args,True,teacher=teacher)['loss'].detach()
    record,region=module.records['v'][0]
    module.records['v'][0]=(record,dict(region,pair=[[20.,21.],[20.,21.]]))
    half=module(*args,True,teacher=teacher)
    assert half['routable']==2 and half['accepted']==1
    assert torch.allclose(half['loss']*2,full)


def test_source_crops_do_not_use_gt_to_select_locations():
    from tools.fit_rq4_localization_quality import source_crops
    video=dict(feats=torch.randn(2304,21),segments=torch.tensor([[2.,6.],[11.,19.]]),
               labels=torch.tensor([0,1]),metadata=dict(video_id='source'))
    crops=list(source_crops(video,10))
    assert [c['metadata']['crop_start'] for c in crops]==[0,5,11]
    video['segments']=torch.tensor([[0.,21.]])
    video['labels']=torch.tensor([1])
    altered=list(source_crops(video,10))
    assert [c['metadata']['crop_start'] for c in altered]==[0,5,11]
    assert altered[1]['segments'].tolist()==[[-5.,16.]]


def test_new_metrics_cannot_hide_target_leakage(tmp_path):
    from tools.rq4_consensus_audit import supervision
    module=M.RQ4Consensus(spec(tmp_path)); args,teacher=args_with_teacher()
    result=module(*args,True,teacher=teacher)
    result.update(M.gradient_diagnostics(args[-1][0].square().mean(),result['loss'],args[-1]))
    row=dict(epoch=1,**{'t_rq4_'+k:float(v) for k,v in result.items()})
    row.update({'t_target_gt_'+k+'_loss':0 for k in ('cls','reg','downstream')})
    write(tmp_path/'metrics.jsonl',row)
    assert supervision(tmp_path/'metrics.jsonl',1)['effective_gradient']
    row['t_target_gt_reg_loss']=1
    write(tmp_path/'metrics.jsonl',row)
    with pytest.raises(ValueError,match='leakage'): supervision(tmp_path/'metrics.jsonl',1)


@pytest.mark.skipif(not os.environ.get('RQ4_BASE_CONFIG'),reason='requires Linux package imports')
def test_preparation_rebuilds_real_protocol_evidence_and_source_policy(tmp_path):
    from test_rq4_artifacts import fixture
    from tools import rq4_artifacts as a
    args=fixture(tmp_path); a.build_artifacts(**args)
    # source_bundle只提供显式TEST合成质量头；检查工件链，不冒充真实源域拟合。
    quality_dir=tmp_path/'source_quality'; quality_dir.mkdir()
    source_bundle(quality_dir)
    result=p.prepare(Path(args['output'])/'direct.json',quality_dir/'quality.json',tmp_path/'new',
        minimum_quality=.5,maximum_uncertainty=.5,anchor_fraction=.5)
    cache=p.load_cache(result['cache_path'])
    p.audit_spec(result,cache)
    with pytest.raises(ValueError,match='acceptance'): p.launch_ready(result,cache)
    policy=p.read(result['source_policy_path']); policy['coverage']=.123
    write(Path(result['source_policy_path']),policy)
    result['source_policy_file_sha256']=file_sha(result['source_policy_path'])
    with pytest.raises(ValueError,match='policy evidence'): p.audit_spec(result,cache)


def test_quality_descriptor_is_finite_detached_and_dimensional():
    f=torch.randn(2304,10,requires_grad=True)
    boxes=torch.tensor([[-2.,6.],[11.,12.]])
    x=Q.descriptor(f,boxes,torch.tensor([3.,11.]),torch.ones(2))
    assert x.shape==(2,134) and not x.requires_grad and torch.isfinite(x).all()


def test_source_quality_fit_and_calibration_dont_touch_global_rng():
    torch.manual_seed(7); x=torch.randn(20,134); y=torch.cat((torch.ones(20,1)*.7,torch.ones(20,2)*.1),1)
    before=torch.random.get_rng_state().clone()
    model=Q.fit_quality(x,y,epochs=3,learning_rate=.001)
    assert torch.equal(before,torch.random.get_rng_state())
    cal=Q.calibrate(model(x),y,['v1']*10+['v2']*10,quantile=.9)
    assert cal['target_coverage_guaranteed'] is False and cal['calibration_videos']==2


def test_source_quality_targets_have_localization_meaning():
    boxes=torch.tensor([[1.,5.],[0.,8.],[10.,12.]])
    y=Q.source_targets(boxes,torch.tensor([[1.,5.]]))
    assert torch.allclose(y[:,0],torch.tensor([1.,.5,0.]))
    assert torch.allclose(y[0,1:],torch.zeros(2))


def test_shared_parameter_gradient_diagnostics():
    x=torch.tensor([1.,2.],requires_grad=True)
    result=M.gradient_diagnostics(x.square().sum(),-x.square().sum()*.1,[x])
    assert result['gradient_ratio']==pytest.approx(.1)
    assert result['gradient_cosine']==pytest.approx(-1)


# 第三轮：真实SADA类合成特征接线（明确不是正式视频验收）。
@pytest.mark.skipif(not os.environ.get('RQ4_BASE_CONFIG'),reason='requires SADA Linux CPU environment')
def test_actual_sada_consensus_ema_forward_and_strict_restore(tmp_path):
    from libs.models.SADA import make_meta_arch
    from tools.rq4_protocol import validate_config
    cfg=p.read(os.environ['RQ4_BASE_CONFIG'])
    cfg['dataset']['json_file']=str(Path(os.environ['RQ4_DATA_ROOT'])/cfg['dataset']['json_file'])
    db=p.read(cfg['dataset']['json_file'])['database']
    video=next(k for k,v in db.items() if v['subset'].lower() in {s.lower() for s in cfg['tgt_train_split']})
    cfg['rq4_boundary']=spec(tmp_path,video)
    validate_config(cfg,'uda')
    cfg['dataset']['max_seq_len']=cfg['model']['max_seq_len']=128
    torch.manual_seed(6); model=make_meta_arch(cfg['model_name'],cfg,device='cpu')
    source=dict(feats=torch.randn(2304,128),segments=torch.tensor([[2.,8.]]),labels=torch.tensor([1]),
        metadata=dict(video_id='source',fps=2,feat_stride=2,feat_num_frames=2,duration=128.,crop_start=0,crop_feat_len=128))
    target=copy.deepcopy(source); target['metadata']['video_id']=video
    target['segments']=torch.empty(0,2); target['labels']=torch.empty(0,dtype=torch.long)
    ema=copy.deepcopy(model).eval(); before=torch.random.get_rng_state().clone()
    teacher=ema.build_rq4_consensus_teacher([target,copy.deepcopy(target)])
    assert torch.equal(before,torch.random.get_rng_state())
    assert all(not x.requires_grad for x in teacher['offsets'])
    model.train()
    losses=model([source,copy.deepcopy(source)],[target,copy.deepcopy(target)],mode='train',
        curr_epoch=max(1,model.warmup_adv),rq4_consensus_teacher=teacher)[0]
    assert all(losses['target_gt_'+k+'_loss']==0 for k in ('cls','reg','downstream'))
    assert losses['rq4_accepted']>0 and losses['rq4_gradient_norm']>0
    losses['downstream_loss'].backward()
    assert model.reg_head.offset_head.conv.weight.grad.abs().sum()>0
    assert all(parameter.grad is None for parameter in model.rq4_boundary.quality.parameters())
    restored=make_meta_arch(cfg['model_name'],cfg,device='cpu')
    restored.load_state_dict(model.state_dict(),strict=True)
