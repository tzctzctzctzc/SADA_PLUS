"""完整文本架构先源训练；完整源AP选模后再初始化UDA教师。"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
from tools import dyfadet_text_protocol as old
from tools.dyfadet_protocol import read, save, sha, CODE

SCHEMA = 'dyfadet-text-source-first-ap-v1'
SPEC = dict(old.SPEC, schema=SCHEMA,
    stages=['source_pretrain', 'uda', 'source_control'],
    source_pretrain_max_epochs=50, source_pretrain_visual_lr=1e-5,
    source_pretrain_semantic_lr=1e-4, source_pretrain_selection='full_source_mAP_0.1_to_0.5_student',
    source_pretrain_epoch0='diagnostic_only_random_new_modules_not_eligible',
    selection='full_source_mAP_0.1_to_0.5_with_original_foreground_NMS_and_voting',
    initialization='source_trained_AP_selected_student_cloned_into_student_and_EMA',
    source_control='same_source_pretrained_checkpoint_and_exact_UDA_source_updates',
    adaptation_selection='full_source_mAP_best_EMA_including_trained_epoch0',
    learning_budget='additional_source_pretraining_then_UDA_not_equal_to_original_SO',
    architecture_changes_from_previous=False, source_validation_cpu_workers=2,
    source_selection_target_GT=False)

def stage_config(cfg, mode):
    assert mode in SPEC['stages']
    c=deepcopy(cfg)
    c['opt'].update(epochs=50 if mode=='source_pretrain' else 30,
        warmup=False, warmup_epochs=0, learning_rate=1e-5, eta_min=1e-6)
    c['benchmark']['training_stage']=mode
    return c

def decision(history, mode):
    assert mode in SPEC['stages'] and history
    assert [x[0] for x in history]==list(range(0,history[-1][0]+1,5))
    assert all(math.isfinite(x[1]) and 0<=x[1]<=1 for x in history)
    eligible=history[1:] if mode=='source_pretrain' else history
    epoch=history[-1][0];maximum=50 if mode=='source_pretrain' else 30
    assert epoch<=maximum
    if not eligible:return dict(epoch=epoch,best_epoch=None,best_map=None,reason='running')
    best=max(eligible,key=lambda x:x[1])
    return dict(epoch=epoch,best_epoch=best[0],best_map=best[1],
        reason='source_AP_patience' if epoch-best[0]>=15 else 'max_epochs' if epoch==maximum else 'running')

def prepare(root,code):
    old.prepare(root,code);root=Path(root)
    for e in (root/'formal').glob('S*'):
        c=read(e/'config.json');c['benchmark'].update(schema=SCHEMA,spec=SPEC)
        save(e/'config.json',c)
        inputs=read(e/'inputs.json')
        inputs['manifests']={str(p):sha(p) for p in e.glob('*.json') if p.name!='inputs.json'}
        save(e/'inputs.json',inputs)
    save(root/'plan.json',dict(schema=SCHEMA,spec=SPEC,code=code,
        shifts=['S'+str(i) for i in range(1,7)],modes=SPEC['stages']))

def audit(entry):
    import torch
    e=Path(entry);c=read(e/'config.json');b=c['benchmark'];done=read(e/'complete.json')
    assert done['schema']==SCHEMA and b['spec']==SPEC
    for p,h in done['artifacts'].items():assert sha(p)==h,p
    for p,h in read(e/'inputs.json')['manifests'].items():assert sha(p)==h,p
    ref=Path(b['reference'])
    for n in ['config','complete','best']:
        assert sha(ref/(n+('.pt' if n=='best' else '.json')))==b['reference_'+n+'_sha256']
    for name in ['cache','provenance']:
        assert sha(b['text'][name])==b['text']['sha256' if name=='cache' else 'provenance_sha256']
    members=read(e/'inputs.json')['members']
    assert len(set().union(*(set(v) for v in members.values())))==sum(map(len,members.values()))
    assert all('annotations' not in x for x in read(b['target_train'])['database'].values())
    acceptance=read(e.parents[1]/'acceptance.json');assert acceptance['code']==b['code']
    for proof in acceptance['reports'].values():
        assert sha(proof['path'])==proof['sha256'];p=read(proof['path'])
        old.guard_audit(p['guard'],p['guard_sha256'],b['code']['commit'])
        assert p['full_state_restore'] and p['next_step_numerical'] and p['target_gradient_positive']
        assert p['staged_handoff']['teacher_equals_trained_student']
        assert p['staged_handoff']['target_updates_before_handoff']==0
        assert p['staged_handoff']['source_parameters_updated']
        assert p['staged_handoff']['foreground_affects_final_scores']
        assert p['staged_handoff']['source_AP_pipeline_passed']
    source_best=e/'source_pretrain/best.pt';source_sha=sha(source_best)
    source_state=torch.load(source_best,map_location='cpu')
    from tools.dyfadet_engine import digest
    source_digest=digest(source_state['model']);del source_state
    uda_stop=read(e/'uda/termination.json')['decision']['epoch']
    for mode in SPEC['stages']:
        d=e/mode;t=read(d/'termination.json');h=t['history'];rows=read(d/'metrics.json')
        expected=decision(h,mode)
        if mode=='source_control':
            assert t['decision']==dict(expected,reason='matched_UDA_source_update_budget')
            assert expected['epoch']==uda_stop
        else:
            assert expected==t['decision'] and expected['reason']!='running'
            assert all(decision(h[:i],mode)['reason']=='running' for i in range(1,len(h)))
        if mode=='source_pretrain':assert expected['best_epoch']>0
        stop=expected['epoch'];assert [x['epoch'] for x in rows]==list(range(1,stop+1))
        for epoch,value in h:
            p=d/'selection'/('epoch_%03d.json'%epoch);proof=read(p)
            assert proof['map_0.1_to_0.5']==value and not proof['target_GT_used']
            assert proof['annotation']==b['source_val'] and proof['annotation_sha256']==sha(b['source_val'])
            assert sha(proof['prediction'])==proof['prediction_sha256']
            assert proof['videos']==members['source_val']
        init=read(d/'initialization.json')
        if mode!='source_pretrain':
            assert init['source_checkpoint_sha256']==source_sha
            assert init['student_sha256']==init['teacher_sha256']==source_digest
            assert init['target_updates_before_handoff']==0
        for name in ['best','latest']:
            state=torch.load(d/(name+'.pt'),map_location='cpu')
            assert state['schema']==SCHEMA and state['mode']==mode and state['config']==stage_config(c,mode)
            assert state['epoch']==expected['best_epoch' if name=='best' else 'epoch']
            assert state['step']==state['epoch']*b['steps_per_epoch']
            assert all(k in state for k in ['optimizer','scheduler','rng','target_rng','augmentation_rng','normalizers'])
            assert state['rng']['cuda'][0].numel()>0
            assert state['history']==[x for x in h if x[0]<=state['epoch']]
            if name=='latest':assert state['metrics']==rows and state['ledger_digest']==t['ledger_digest']
            del state
        events=[json.loads(x) for x in (d/'events.jsonl').read_text().splitlines()]
        assert [x['step'] for x in events]==list(range(1,stop*b['steps_per_epoch']+1))
        chain=''
        for x in events:
            got=x.pop('sha256');assert x['previous']==chain
            chain=hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert got==chain
            assert set(x['source_videos'])<=set(members['source_train'])
            assert set(x['target_videos'])<=set(members['target_train'])
            if mode!='uda':assert not x['target_videos'] and x['target_proposals']==0
        assert chain==t['ledger_digest']
        if mode=='uda':
            assert sum(x['target_regression_points'] for x in events)>0
            assert all(all(math.isfinite(v) and v>0 for v in x['target_gradient_norms'].values()) for x in rows if x['target_gradient_norms'])
        old.guard_audit(t['guard'],t['guard_sha256'],b['code']['commit'])
        for role in ['source_val','target_val']:
            manifest=b['inference'][role];assert all('annotations' not in x for x in read(manifest)['database'].values())
            proof=read(d/(role+'_inference.json'));report=read(d/'evaluation'/(role+'.json'))
            assert proof['videos']==members[role] and not proof['target_GT_used']
            assert report['prediction_sha256']==sha(d/(role+'.npz')) and report['selected_epoch']==expected['best_epoch']
    g=read(e/'gpu_complete.json');old.guard_audit(g['guard'],g['guard_sha256'],b['code']['commit'])
    values={m:read(e/m/'evaluation/target_val.json')['map_0.1_to_0.5'] for m in SPEC['stages']}
    so=read(ref/'evaluation/target_val.json')['map_0.1_to_0.5']
    return dict(shift=b['shift'],map=values['uda'],reference_so_map=so,control_map=values['source_control'],
        source_pretrained_map=values['source_pretrain'],delta_pp=100*(values['uda']-so),
        delta_control_pp=100*(values['uda']-values['source_control']),delta_pretrained_pp=100*(values['uda']-values['source_pretrain']),
        complete=str(e/'complete.json'),complete_sha256=sha(e/'complete.json'))
