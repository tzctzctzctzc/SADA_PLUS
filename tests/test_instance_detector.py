from copy import deepcopy
import json
from runpy import run_path
import types
import pytest
import torch

from tools.baseline_protocol import CODE
from libs.models.SADA.instance_detector import InstanceTemporalDetector,specification,sample_1d,TemporalCrossAttention
from libs.models.SADA.instance_losses import source_losses,match,create_teacher,update_teacher,attach_vlm_labels,distill_instances
from tools.instance_engine import query_budget,validation_windows
from tools.run_instance_matrix import ordering,validate_results


def small():
    cfg=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']()
    cfg['model']['droppath']=0.;spec=specification(8)
    spec.update(dim=32,heads=4,decoder_layers=2,ff_dim=64,roi_bins=4,points_per_level=2)
    return cfg,spec


def videos():
    return [dict(feats=torch.randn(8,64),segments=torch.tensor([[9.,21.],[30.,43.]]),labels=torch.tensor([0,1]),
        metadata=dict(video_id='s',feat_stride=2,feat_num_frames=4,fps=2,duration=200))]


def test_real_structure_losses_all_main_heads_learn_and_strict_reload():
    torch.manual_seed(123);cfg,spec=small();model=InstanceTemporalDetector(cfg,spec);batch=videos()
    output=model(batch);loss=source_losses(output,batch);assert torch.isfinite(loss['loss']) and loss['source_instances']==2
    loss['loss'].backward()
    for module in (model.encoder,model.decoder,model.query_content,model.initial_reference,
                   model.class_heads,model.segment_heads,model.quality_head):
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters())
    assert not any(k in name for name in model.state_dict() for k in ('disc_','pt_uncertainty','rq3_capacity','backbone_target'))
    assert len(output['aux_outputs'])==1 and output['instance_embeddings'].shape==(1,8,32)
    other=InstanceTemporalDetector(cfg,spec);other.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())


def test_forward_does_not_read_ground_truth_and_is_stable_in_eval():
    cfg,spec=small();model=InstanceTemporalDetector(cfg,spec).eval();batch=videos()
    class Blind(dict):
        def __getitem__(self,key):
            if key in ('segments','labels'):raise AssertionError('前向访问了GT')
            return super().__getitem__(key)
    with torch.no_grad():
        a=model(batch);b=model([Blind(v) for v in batch])
    assert torch.equal(a['pred_logits'],b['pred_logits']) and torch.equal(a['pred_segments'],b['pred_segments'])


def test_one_dimensional_sampling_matches_linear_interpolation_and_has_coordinate_gradients():
    feature=torch.arange(4,dtype=torch.double).reshape(1,1,4,1).requires_grad_()
    position=torch.tensor([[[[.25,1.75]]]],dtype=torch.double,requires_grad=True);mask=torch.ones(1,4,dtype=torch.bool)
    actual,support=sample_1d(feature,position,mask)
    torch.testing.assert_close(actual.flatten(),torch.tensor([.25,1.75],dtype=torch.double))
    assert torch.autograd.gradcheck(lambda f,p:sample_1d(f,p,mask)[0],(feature,position))
    assert torch.equal(support,torch.ones_like(support))


def test_attention_never_reads_padding_and_uses_valid_feature_time_axis():
    attention=TemporalCrossAttention(4,1,1,1)
    with torch.no_grad():
        attention.offsets.bias.zero_();attention.value.weight.copy_(torch.eye(4));attention.value.bias.zero_()
        attention.output.weight.copy_(torch.eye(4));attention.output.bias.zero_()
    features=torch.tensor([[[1.,3.,999.,999.]]]).expand(1,4,4)
    result=attention(torch.zeros(1,1,4),torch.tensor([[[.5]]]),[features],[torch.tensor([[True,True,False,False]])],torch.tensor([2]),[1])
    torch.testing.assert_close(result,torch.full_like(result,3.))


def test_matching_is_one_to_one_and_not_query_index_order():
    output=dict(pred_logits=torch.tensor([[[-5.,5.],[5.,-5.]]]),pred_segments=torch.tensor([[[.7,.9],[.1,.3]]]))
    gt=[dict(labels=torch.tensor([0,1]),segments=torch.tensor([[.1,.3],[.7,.9]]))]
    a,b=match(output,gt)[0];assert list(zip(a.tolist(),b.tolist()))==[(0,1),(1,0)]
    gt[0]=dict(labels=torch.tensor([0,1,0]),segments=torch.tensor([[.1,.2],[.3,.4],[.5,.6]]))
    with pytest.raises(ValueError):match(output,gt)


def test_empty_source_and_extreme_class_logits_are_finite():
    cfg,spec=small();model=InstanceTemporalDetector(cfg,spec);batch=videos();batch[0].update(segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long))
    out=model(batch);loss=source_losses(out,batch)
    assert loss['localization_l1']==loss['localization_iou']==0 and torch.isfinite(loss['loss'])
    loss['loss'].backward()
    match(dict(pred_logits=torch.tensor([[[100.,-100.]]]),pred_segments=torch.tensor([[[.1,.3]]])),
        [dict(labels=torch.tensor([0]),segments=torch.tensor([[.1,.3]]))])


def test_teacher_copy_is_frozen_and_ema_update_is_correct():
    cfg,spec=small();model=InstanceTemporalDetector(cfg,spec);teacher=create_teacher(model)
    assert all(not p.requires_grad for p in teacher.parameters())
    name,param=next(iter(model.named_parameters()));old=dict(teacher.named_parameters())[name].clone()
    with torch.no_grad():param.add_(1.)
    update_teacher(teacher,model,.9)
    torch.testing.assert_close(dict(teacher.named_parameters())[name],old+.1)


def test_semantic_and_localization_interfaces_are_independent_and_teacher_has_no_grad():
    student=dict(pred_logits=torch.zeros(1,2,10,requires_grad=True),pred_segments=torch.tensor([[[.12,.32],[.65,.85]]],requires_grad=True))
    teacher=dict(pred_logits=torch.zeros(1,2,10,requires_grad=True),pred_segments=torch.tensor([[[.7,.9],[.1,.3]]],requires_grad=True))
    caches=[dict(origin='vlm-prediction',segments=torch.tensor([[.1,.3]]),labels=torch.tensor([4]))]
    labels=attach_vlm_labels(teacher,caches,.5);assert labels.tolist()==[[-1,4]]
    losses=distill_instances(student,teacher,labels>=0,torch.ones(1,2,dtype=torch.bool),labels)
    assert losses['semantic_instances']==1 and losses['localization_instances']==2
    (losses['semantic']+losses['localization']).backward()
    assert student['pred_segments'].grad.abs().sum()>0 and student['pred_logits'].grad[:,:,4].abs().sum()>0
    assert student['pred_logits'].grad[:,:,:4].abs().sum()==0 and student['pred_logits'].grad[:,:,5:].abs().sum()==0
    assert teacher['pred_segments'].grad is None and teacher['pred_logits'].grad is None
    caches[0]['origin']='GT-oracle'
    with pytest.raises(ValueError):attach_vlm_labels(teacher,caches,.5)


def test_query_budget_uses_source_training_only(tmp_path):
    db={'s':dict(subset='source',annotations=[dict(label_id=0,segment=[i,i+2]) for i in range(180)]),
        't':dict(subset='target',annotations=[dict(label_id=0,segment=[0,1])]*10000)}
    p=tmp_path/'source.json';p.write_text(json.dumps(dict(database=db)))
    cfg=dict(dataset=dict(json_file=str(p),max_seq_len=20,feat_stride=1,default_fps=1),src_train_split=['source'])
    first=query_budget(cfg);db['t']['annotations']=[];p.write_text(json.dumps(dict(database=db)));second=query_budget(cfg)
    assert first['num_queries']==second['num_queries'] and first['max_window_intersections']==second['max_window_intersections']
    assert first['num_queries']>=first['max_window_intersections']


def test_long_video_windows_preserve_absolute_time_and_do_not_use_gt():
    cfg,spec=small();cfg['test_cfg']['nms_method']='none';model=InstanceTemporalDetector(cfg,spec).eval()
    def fixed(self,videos):
        return dict(pred_logits=torch.zeros(1,1,10),pred_quality_logits=torch.zeros(1,1),pred_segments=torch.tensor([[[.25,.75]]]))
    model.forward=types.MethodType(fixed,model)
    v=videos()[0];v['feats']=torch.zeros(8,96);v['metadata']['crop_start']=10
    result=model.predict_video(v)
    assert result['segments'].shape==(2,2)
    torch.testing.assert_close(result['segments'],torch.tensor([[27.,59.],[59.,91.]]))
    windows=list(validation_windows(v,64));assert windows[1]['feats'].shape[-1]==32
    assert all((x['segments'][:,0]>=0).all() and (x['segments'][:,1]<=x['feats'].shape[-1]).all() for x in windows)


def test_six_direction_schedule_is_finite_and_longest_first():
    costs={'S1':1,'S2':2,'S3':3,'S4':4,'S5':5,'S6':6}
    assert ordering(costs)==['S6','S5','S4','S3','S2','S1']
    rows=[dict(status='passed',shift=s,map=.6) for s in costs]
    assert validate_results(rows)==pytest.approx(.6)
    with pytest.raises(ValueError):validate_results(rows[:-1])
    with pytest.raises(ValueError):validate_results(rows[:-1]+[rows[0]])
