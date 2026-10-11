import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
from tools import rq4_protocol as p, rq4_relational as r
from tools import rq4_artifacts as a
from test_rq4_boundary import PACKAGE, candidate, identity, inputs, write
from test_rq4_artifacts import fixture

SPEC = importlib.util.spec_from_file_location('rq4_cpu.rq4_relational',
                                             Path(PACKAGE.__path__[0])/'rq4_relational.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def spec(tmp, strategy='relational', video='v'):
    pool = [candidate(candidate_id=str(i),video_id=video,proposal=[2.,8.],window=[0.,10.]) for i in range(10)]
    budget = p.build_budget(pool,'S1')
    selected = p.select_candidates(pool,.2)
    teacher = 'sada_self' if strategy=='proposal_relation' else 'timelens2_8b'
    replies = [dict(candidate_id=x['candidate_id'],video_id=video,candidate_sha256=p.digest(x),
                    teacher=teacher,raw_text='[[2,8]]') for x in selected]
    cache = p.build_cache(selected,replies,teacher=teacher,shift='S1',identity=identity(),
                          min_iou=0.,ambiguity_margin=0.,budget=budget)
    regions = dict(schema=r.VARIANT,strategy=strategy,rule=r.RULE,builder_sha256=p.file_sha(r.__file__),
        cache_sha256=cache['payload_sha256'],rows=[dict(candidate_sha256=p.digest(x),
        pair=None if teacher=='sada_self' else [[2.,8.],[2.,8.]],core=[2.,8.]) for x in selected])
    regions['payload_sha256'] = p.digest(regions)
    write(tmp/'cache.json',cache); write(tmp/'regions.json',regions)
    result = dict(schema='rq4-boundary-training-v1',variant=r.VARIANT,strategy=strategy,
        relation_rule=copy.deepcopy(r.RULE),weight=.1,normalizer='common-routable-candidates',
        activation='sada-adaptation-phase',early_stop_patience=15,protocol_frozen=False)
    for name in ('cache','regions'):
        result[name+'_path'] = str(tmp/(name+'.json'))
        result[name+'_file_sha256'] = p.file_sha(tmp/(name+'.json'))
    return result


# 第一轮：协议、原响应重建、控制组隔离、工件身份与正式启动拒绝。
def test_rebuild_actual_paired_artifact_chain(tmp_path):
    args = fixture(tmp_path)
    a.build_artifacts(**args)
    s = r.prepare(tmp_path/'relation',original_spec=Path(args['output'])/'gated.json')
    cache = p.load_cache(s['cache_path'])
    regions = r.audit_spec(s,cache)
    assert len(regions['rows'])==cache['budget']['selected_count']==2
    assert all(x['core']==[2.,6.] for x in regions['rows'])
    s['protocol_frozen'] = True
    p.assert_launch_ready({'rq4_boundary':s})
    replies = a.read(args['responses']); replies[-1]['raw_text']='[[0,1]]'
    write(Path(args['responses']),replies)
    with pytest.raises(ValueError,match='SHA'): r.audit_spec(s,cache)


def test_proposal_control_never_uses_teacher_responses(tmp_path):
    args = fixture(tmp_path)
    # 即使所有教师输出损坏，对照准备也不能打开它们。
    Path(args['responses']).write_text('not-json')
    s = r.prepare(tmp_path/'control',requests=args['requests'])
    cache = p.load_cache(s['cache_path'])
    assert cache['teacher']=='sada_self'
    assert all(x['pair'] is None for x in r.audit_spec(s,cache)['rows'])
    s['protocol_frozen']=True
    p.assert_launch_ready({'rq4_boundary':s})


@pytest.mark.parametrize('key,value',[('weight',.2),('early_stop_patience',0),('variant','old'),
                                     ('relation_rule',{}),('gate_path','teacher-gate')])
def test_fixed_protocol_cannot_drift(tmp_path,key,value):
    s = spec(tmp_path); s[key]=value
    with pytest.raises(ValueError): r.validate_spec(s,p.load_cache(s['cache_path']))


def test_rehashed_regions_cannot_invent_a_core(tmp_path):
    s=spec(tmp_path); regions=a.read(s['regions_path'])
    regions['rows'][0]['core']=[3.,6.]
    regions['payload_sha256']=p.digest({k:v for k,v in regions.items() if k!='payload_sha256'})
    write(Path(s['regions_path']),regions); s['regions_file_sha256']=p.file_sha(s['regions_path'])
    with pytest.raises(ValueError,match='intersection'): M.RQ4Relational(s)


def test_timeple_3x_codec_failure_is_technical_not_zero_coverage(tmp_path):
    from tools.prepare_rq4_target_candidates import paired_candidates
    row=candidate(window=[0.,8.]); requests=[dict(candidate=x) for x in paired_candidates(row)]
    replies=[dict(candidate_id=x['candidate']['candidate_id'],video_id='v',teacher='timeple_8b',
        candidate_sha256=p.digest(x['candidate']),request_sha256=p.digest(x),
        raw_text='<|TIMESPAN|>',metadata={'codec_predictions':[[2.,4.]]}) for x in requests]
    cache=p.build_cache([row],[dict(replies[0],candidate_id=row['candidate_id'],candidate_sha256=p.digest(row))],
        teacher='timeple_8b',shift='S1',identity=identity(),min_iou=0.,ambiguity_margin=0.)
    replies[1]['metadata']={}
    with pytest.raises(RuntimeError,match='codec'):
        r.relation_rows(cache,dict(requests=requests),replies)


def test_legacy_acceptance_does_not_authorize_relational(tmp_path):
    from tools.run_rq4_entry import acceptance_ready
    s=spec(tmp_path)
    report=dict(status='passed',formal_result=False,real_video_features=True,batch_size=2,seed=6,
                spec=s,strategy='gated')
    write(tmp_path/'smoke.json',report)
    write(tmp_path/'acceptance.json',dict(status='passed',code_commit='head',reports=[a.reference(tmp_path/'smoke.json')]))
    with pytest.raises(ValueError,match='missing'):
        acceptance_ready(tmp_path/'acceptance.json','timelens2_8b','head','relational')


# 第二轮：解析值、坐标不变性、多点梯度和零干预。
def test_absolute_not_offset_consistency():
    centers=torch.tensor([2.,4.,7.])
    same_offsets=torch.ones(3,2,requires_grad=True)
    assert M.consistency_loss(same_offsets,centers,1.,6.).item()==pytest.approx(2.5/6)
    correct=torch.stack((centers-1.,9.-centers),dim=-1).requires_grad_()
    loss=M.consistency_loss(correct,centers,1.,6.)
    assert loss.item()==0
    loss.backward(); assert correct.grad.abs().sum()==0


def test_translation_and_unit_scale_invariant():
    x=torch.tensor([[1.,2.],[2.,4.],[5.,2.]])
    c=torch.tensor([2.,4.,7.])
    value=M.consistency_loss(x,c,2.,6.)
    assert torch.allclose(value,M.consistency_loss(x,c+50.,2.,6.))
    assert torch.allclose(value,M.consistency_loss(x,c*3.,6.,18.))


def test_only_three_real_points_receive_gradient(tmp_path):
    module=M.RQ4Relational(spec(tmp_path)); args=inputs()
    result=module(*args,active=True)
    assert (result['accepted'],result['relation_points'],result['relation_pairs'])==(2,6,4)
    assert result['raw_loss'].item()==pytest.approx(2.5/6)
    result['loss'].backward()
    grad=args[-1][0].grad
    assert grad[:,[2,7]].abs().sum()>0
    assert grad[:,[0,1,3,5,6,8,9]].abs().sum()==0
    assert list(module.parameters())==[]


def test_grid_shortage_does_not_interpolate_fake_points(tmp_path):
    module=M.RQ4Relational(spec(tmp_path)); args=inputs(anchor_stride=8)
    result=module(*args,active=True)
    assert result['routable']==result['multipoint_rejected']==2
    assert result['accepted']==0 and result['loss']==0
    result['loss'].backward(); assert args[-1][0].grad.abs().sum()==0


@pytest.mark.parametrize('condition',['warmup','crop','mask','missing','target_gt'])
def test_noop_and_gt_isolation(tmp_path,condition):
    module=M.RQ4Relational(spec(tmp_path)); args=inputs()
    original=module(*args,active=True)['loss'].detach()
    if condition=='crop': args[0][0]['metadata']['crop_start']=3
    if condition=='mask': args[2][0].fill_(False)
    if condition=='missing': args[0][0]['metadata']['video_id']='absent'
    if condition=='target_gt': args[0][0].update(labels=object(),segments=object(),logits=object())
    result=module(*args,active=condition!='warmup')
    assert torch.allclose(result['loss'],original) if condition=='target_gt' else result['loss']==0


def test_gate_rejection_keeps_common_denominator(tmp_path):
    s=spec(tmp_path); full=M.RQ4Relational(s)(*inputs(),active=True)
    regions=a.read(s['regions_path']); regions['rows'][0].update(pair=[[2.,8.],[7.,9.]],core=None)
    regions['payload_sha256']=p.digest({k:v for k,v in regions.items() if k!='payload_sha256'})
    write(Path(s['regions_path']),regions); s['regions_file_sha256']=p.file_sha(s['regions_path'])
    half=M.RQ4Relational(s)(*inputs(),active=True)
    assert half['accepted']==half['gate_rejected']==1 and half['routable']==2
    assert torch.allclose(half['loss']*2,full['loss'])


def test_points_do_not_depend_on_rng_and_masks_are_respected():
    grid=torch.tensor([[i,0,10,1] for i in range(10)]).float()
    valid=torch.ones(10,dtype=torch.bool); valid[4]=False
    before=torch.random.get_rng_state()
    assert M.select_points(grid,valid,[1.5,7.5],0).tolist()==[2,5,7]
    assert torch.equal(before,torch.random.get_rng_state())


def test_new_supervision_checks_multi_point_counts(tmp_path):
    from tools.run_rq4_entry import supervision
    result=M.RQ4Relational(spec(tmp_path))(*inputs(),active=True)
    row={'epoch':1,**{'t_rq4_'+k:float(v) for k,v in result.items()}}
    row.update({'t_target_gt_'+k+'_loss':0 for k in ('cls','reg','downstream')})
    path=tmp_path/'metrics.jsonl'; write(path,row)
    assert supervision(path,1,True)['effective_gradient']
    row['t_rq4_relation_pairs']=99; write(path,row)
    with pytest.raises(ValueError,match='multi-point'): supervision(path,1,True)


# 第三轮：真实SADA类接线/严格恢复；输入明确是合成特征，不冒充GPU验收。
@pytest.mark.skipif(not os.environ.get('RQ4_BASE_CONFIG'),reason='requires Linux SADA CPU environment')
@pytest.mark.parametrize('strategy',['relational','proposal_relation'])
def test_actual_sada_relational_forward_and_strict_restore(tmp_path,strategy):
    from libs.models.SADA import make_meta_arch
    cfg=a.read(os.environ['RQ4_BASE_CONFIG'])
    cfg['dataset']['json_file']=str(Path(os.environ['RQ4_DATA_ROOT'])/cfg['dataset']['json_file'])
    database=a.read(cfg['dataset']['json_file'])['database']
    splits={x.lower() for x in cfg['tgt_train_split']}
    video=next(k for k,v in database.items() if v['subset'].lower() in splits)
    cfg['rq4_boundary']=spec(tmp_path,strategy,video)
    p.validate_config(cfg,'uda')
    cfg['dataset']['max_seq_len']=cfg['model']['max_seq_len']=128
    torch.manual_seed(6)
    model=make_meta_arch(cfg['model_name'],cfg,device='cpu'); model.train()
    source=dict(feats=torch.randn(2304,128),segments=torch.tensor([[2.,8.]]),labels=torch.tensor([1]),
        metadata=dict(video_id='source',fps=2,feat_stride=2,feat_num_frames=2,duration=128.,
                      crop_start=0,crop_feat_len=128))
    target=copy.deepcopy(source); target['metadata']['video_id']=video
    target['segments']=torch.empty(0,2); target['labels']=torch.empty(0,dtype=torch.long)
    losses=model([source,copy.deepcopy(source)],[target,copy.deepcopy(target)],mode='train',
                  curr_epoch=max(1,model.warmup_adv))[0]
    assert all(losses['target_gt_'+k+'_loss']==0 for k in ('cls','reg','downstream'))
    assert losses['rq4_accepted']>0 and losses['rq4_gradient_norm']>0
    losses['downstream_loss'].backward()
    assert model.reg_head.offset_head.conv.weight.grad.abs().sum()>0
    restored=make_meta_arch(cfg['model_name'],cfg,device='cpu')
    restored.load_state_dict(model.state_dict(),strict=True)
