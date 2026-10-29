from copy import deepcopy
import hashlib
import json
from pathlib import Path
from runpy import run_path

import pytest
import torch

from tools.prototype_oracle import (coverage,truth,build_caches,GROUPS,POLICY,
    config_for,assert_pair,ExposureTracker)
from tools.baseline_protocol import CODE
from libs.models.SADA.prototype_capacity import build_capacity


def fixture_pool():
    rows=[];database={}
    for i in range(200):
        video='v%03d'%i
        database[video]=dict(annotations=[dict(segment=[0.,10.],label_id=i%10)] if 20<=i<170 else [])
        rows.append(dict(video_id=video,candidate_id=video+'-p0000000',absolute_feature_index=0,
            proposal_start_sec=0.,proposal_end_sec=10.,source_fpn_levels=[0],class_id=0,
            semantic_label='TAKE',class_probabilities=[.9]+[.1/9]*9,participation_selected=i<40))
    return dict(shift='S1',by_video={r['video_id']:[r] for r in rows}),database


def test_truth_uses_interval_union_strict_half_and_refuses_multiple_or_unknown():
    anns=[dict(segment=[0.,4.],label_id=2),dict(segment=[2.,5.],label_id=2)]
    assert coverage(0.,10.,2,anns)==.5
    row=dict(proposal_start_sec=0.,proposal_end_sec=10.)
    assert truth(row,anns)['label'] is None
    anns.append(dict(segment=[5.,7.],label_id=2))
    assert truth(row,anns)['label']==2
    assert truth(row,anns+[dict(segment=[0.,8.],label_id=4)])['reason']=='multiple_known_majorities'
    assert truth(row,[dict(segment=[0.,10.],label_id=20)])['label'] is None


def test_four_sets_are_exact_nested_and_never_move_proposals_or_mutate_input():
    old,db=fixture_pool();before=deepcopy(old);caches=build_caches(old,db,set(db))
    assert old==before
    selected={g:{r['candidate_id'] for rr in c['by_video'].values() for r in rr if r['participation_selected']} for g,c in caches.items()}
    assert [len(selected[g]) for g in GROUPS]==[20,40,100,150]
    assert selected['clean20']<selected['clean50']<selected['clean100']
    assert selected['fixed40']<={r['candidate_id'] for rr in old['by_video'].values() for r in rr if r['participation_selected']}
    for group,c in caches.items():
        for video,rr in c['by_video'].items():
            row=rr[0];original=old['by_video'][video][0]
            for key in ['candidate_id','proposal_start_sec','proposal_end_sec','source_fpn_levels','absolute_feature_index']:
                assert row[key]==original[key]
            if row['participation_selected']:
                assert row['class_id']==int(video[1:])%10
                assert sum(row['class_probabilities'])==1 and row['class_probabilities'][row['class_id']]==1
            elif row['oracle_truth']['label'] is None:
                assert row['semantic_label']=='ABSTAIN' and not any(row['class_probabilities'])
    assert caches==build_caches(old,db,set(db))


def test_cache_builder_rejects_target_validation_members_and_inadequate_budget():
    old,db=fixture_pool()
    with pytest.raises(ValueError):build_caches(old,db,set(db)-{'v020'})
    for i in range(50,170):db['v%03d'%i]['annotations']=[]
    with pytest.raises(ValueError):build_caches(old,db,set(db))


def test_oracle_is_explicit_and_only_declared_changes_are_allowed(tmp_path):
    old,db=fixture_pool();payload=build_caches(old,db,set(db))['clean20']
    cache=tmp_path/'cache.json';cache.write_text(json.dumps(payload))
    base=dict(seed=123,loader=dict(batch_size=2),opt=dict(learning_rate=.0001),
        baseline=dict(regime='uda',code=dict(commit='old'),tmpdir='/old',target_gt_used=False))
    cfg=config_for(base,'clean20',cache,dict(commit='new'),tmp_path);assert_pair(cfg,base)
    assert cfg['baseline']['target_gt_used'] is True and cfg['oracle_auxiliary']['target_gt_for_auxiliary'] is True
    for name,value in [('seed',6),('opt',dict(learning_rate=.002))]:
        changed=deepcopy(cfg);changed[name]=value
        with pytest.raises(ValueError):assert_pair(changed,base)
    cap=cfg['prototype_capacity'];model=build_capacity(cap,8,10,[1],[[0,10000]],'cpu')
    assert model.branch.classifier.in_features==128
    cap=deepcopy(cap);cap.pop('oracle_auxiliary_ground_truth')
    with pytest.raises(ValueError):build_capacity(cap,8,10,[1],[[0,10000]],'cpu')


def test_true_sada_oracle_keeps_detector_rng_and_source_losses_and_gradients(tmp_path,monkeypatch):
    helper=run_path(str(CODE/'tests/test_prototype_capacity.py'))
    config=helper['configuration'](tmp_path,'control')
    payload=json.loads(Path(config['prepared_cache']).read_text());payload['schema']='prototype-gt-oracle-cache-v1'
    Path(config['prepared_cache']).write_text(json.dumps(payload))
    config['prepared_cache_sha256']=hashlib.sha256(Path(config['prepared_cache']).read_bytes()).hexdigest()
    config['oracle_auxiliary_ground_truth']=True
    small=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']()
    from libs.models.SADA.meta_archs import SADA
    from libs.models.SADA.optimizer import make_optimizer
    native=torch.tensor
    def copied(*args,**kwargs):
        x=native(*args,**kwargs)
        return x.clone() if x.requires_grad and x.device.type=='cpu' else x
    monkeypatch.setattr(torch,'tensor',copied)
    torch.manual_seed(123);base=SADA(deepcopy(small),'cpu')
    small['prototype_capacity']=config
    torch.manual_seed(123);model=SADA(small,'cpu')
    assert all(torch.equal(x,model.state_dict()[k]) for k,x in base.state_dict().items())
    opt=make_optimizer(model,small['opt'],.0001,exclude_params=['disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings'])
    ids={id(p) for g in opt.param_groups for p in g['params']}
    assert all(id(p) in ids for p in model.rq3_capacity.parameters())
    s,t=helper['videos']();assert t['labels'].numel()==t['segments'].numel()==0
    torch.manual_seed(777);a=base(video_list_src=[s],video_list_tgt=[t],mode='train',curr_epoch=5)[0]
    torch.manual_seed(777);b=model(video_list_src=[s],video_list_tgt=[t],mode='train',curr_epoch=5)[0]
    assert torch.equal(a['cls_loss'],b['cls_loss']) and torch.equal(a['reg_loss'],b['reg_loss'])
    assert b['rq3_target_gradient_norm']>0
    torch.testing.assert_close(b['downstream_loss'],a['downstream_loss']+b['rq3_loss'])
    restored=SADA(small,'cpu');restored.load_state_dict(model.state_dict(),strict=True)
    b['downstream_loss'].backward();assert model.rq3_capacity.branch.input_projection.weight.grad.norm()>0


def test_exposure_counts_real_crops_without_rng_or_loss_changes(tmp_path):
    old,db=fixture_pool();cache=build_caches(old,db,set(db))['clean100']['by_video']
    tracker=ExposureTracker(cache,[1],2,tmp_path/'exposure.json')
    video=dict(metadata=dict(video_id='v020',fps=1.,feat_stride=1.,feat_num_frames=0,crop_start=0,crop_feat_len=20),feats=torch.zeros(8,20))
    before=torch.get_rng_state().clone();tracker.observe([video],1,1)
    cropped=deepcopy(video);cropped['metadata']['crop_start']=5
    tracker.observe([cropped],1,0)
    assert torch.equal(before,torch.get_rng_state())
    out=json.loads((tmp_path/'exposure.json').read_text())
    assert out['epochs']['1']['exposures']==1 and len(out['unique_candidate_ids'])==1


def test_gpu_acceptance_requires_real_candidate_gradient_and_rng():
    from tools.run_prototype_oracle import acceptance
    report=dict(status='passed',rng_roundtrip=dict(status='passed'),initial_detector_sha='same',phases={
        '1':dict(target_gt_losses=[0.,0.,0.],branch_parameters_updated=True,auxiliary=dict(
            rq3_source_loss=1.,rq3_alignment_loss=.1,rq3_selected=2.,rq3_target_gradient_norm=.01))})
    acceptance(report,'same')
    for field in ['rq3_selected','rq3_target_gradient_norm']:
        bad=deepcopy(report);bad['phases']['1']['auxiliary'][field]=0.
        with pytest.raises(ValueError):acceptance(bad,'same')


def test_tracker_wrapper_preserves_real_auxiliary_values_and_rng(tmp_path):
    from types import SimpleNamespace
    from tools.run_prototype_oracle import install_tracker
    helpers=run_path(str(CODE/'tests/test_prototype_capacity.py'))
    cfg=helpers['configuration'](tmp_path,'control');path=Path(cfg['prepared_cache']);payload=json.loads(path.read_text())
    payload['by_video']['t'][0].update(video_id='t',absolute_feature_index=0)
    path.write_text(json.dumps(payload));cfg['prepared_cache_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    native=build_capacity(cfg,8,10,[1],[[0,10000]],'cpu');wrapped=deepcopy(native)
    engine=SimpleNamespace(build=lambda cfg:SimpleNamespace(model=SimpleNamespace(rq3_capacity=wrapped)))
    install_tracker(engine,tmp_path)
    wrapped=engine.build(dict(baseline=dict(epoch_budget=dict(iterations_per_epoch=1)))).model.rq3_capacity
    s,t=helpers['videos']();sf=[s['feats'][None].requires_grad_()];tf=[t['feats'][None].requires_grad_()]
    before=torch.get_rng_state().clone();a=native([s],[t],sf,tf,5);after=torch.get_rng_state().clone()
    torch.set_rng_state(before);b=wrapped([s],[t],sf,tf,5)
    assert torch.equal(after,torch.get_rng_state())
    for key in a:torch.testing.assert_close(torch.as_tensor(a[key]),torch.as_tensor(b[key]),rtol=0,atol=0)


def test_formal_gate_binds_code_inputs_reports_and_exact_cache_group(tmp_path):
    from tools.prototype_oracle import verify_acceptance,SCHEMA
    from tools.baseline_protocol import save,sha
    old,db=fixture_pool();cache=tmp_path/'cache.json';save(cache,build_caches(old,db,set(db))['clean20'])
    identity=dict(commit='new')
    base=dict(seed=123,loader=dict(batch_size=2),baseline=dict(regime='uda',code=dict(commit='old'),tmpdir='/old',target_gt_used=False,shift='S1'))
    cfg=config_for(base,'clean20',cache,identity,tmp_path)
    probe=tmp_path/'probe.json';save(probe,dict(status='passed'))
    save(tmp_path/'inputs.json',dict(prepared=dict(S1=dict(clean20=dict(path=str(cache),sha256=sha(cache))))))
    ready=dict(status='passed',schema=SCHEMA,code=identity,inputs_sha256=sha(tmp_path/'inputs.json'),report_sha256={str(probe):sha(probe)})
    save(tmp_path/'acceptance.json',ready);verify_acceptance(tmp_path,cfg)
    save(probe,dict(status='changed'))
    with pytest.raises(ValueError):verify_acceptance(tmp_path,cfg)
    save(probe,dict(status='passed'))
    changed=deepcopy(cfg);changed['oracle_auxiliary']['group']='fixed40'
    with pytest.raises(ValueError):verify_acceptance(tmp_path,changed)
