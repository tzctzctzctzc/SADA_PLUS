"""DyFADet SO：固定内部六方向，目标GT仅离线CPU评估。"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import math
import subprocess

CODE = Path(__file__).resolve().parents[1]
VENDOR = CODE / 'vendor/dyfadet'
REFERENCE = Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029')
SCHEMA = 'dyfadet-source-only-six-shift-v1'
SPEC = dict(seed=123, batch_size=2, max_epochs=50, valid_every=5, patience=15,
    selection='source_equal_video_cls_plus_reg_per_positive', ema_decay=.999,
    initialization='fresh_official_model_then_deepcopy_ema', optimizer='official_AdamW',
    learning_rate=.0001, warmup_epochs=10, cosine_total_epochs=50, eta_min=.00005,
    epoch_steps='same_as_existing_SO_reference', source_cycle='itertools_cycle_cached_batches',
    classes=list(range(10)), raw_features='unchanged_SlowFast', target_training=False,
    vlm=False, adversarial=False, precision='float32', gradient_clip=1.)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def save(p, x):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_name(p.name+'.tmp'); tmp.write_text(json.dumps(x, ensure_ascii=True, allow_nan=False, indent=2), encoding='utf8'); tmp.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def identity():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip(), 'code dirty'
    upstream=read(VENDOR/'upstream.json')
    for p,h in upstream['files'].items(): assert sha(VENDOR/p)==h, p
    return dict(commit=head,upstream=upstream['commit'],upstream_manifest_sha256=sha(VENDOR/'upstream.json'))

def decision(history):
    assert history and [x[0] for x in history]==list(range(5,history[-1][0]+1,5))
    assert history[-1][0]<=50 and all(math.isfinite(x[1]) for x in history)
    best=min(history,key=lambda x:x[1]); epoch=history[-1][0]
    return dict(epoch=epoch,best_epoch=best[0],best_loss=best[1],reason='source_patience' if epoch-best[0]>=15 else 'max_epochs' if epoch==50 else 'running')

def terminal(history):
    d=decision(history)
    assert d['reason']!='running'
    assert all(decision(history[:n])['reason']=='running' for n in range(1,len(history)))
    return d

def prepare(root, code):
    import sys
    sys.path.insert(0,str(VENDOR))
    from libs.core import load_config
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    for i in range(1,7):
        s='S'+str(i); entry=root/'formal'/s; entry.mkdir(parents=True)
        ref=REFERENCE/'formal'/s/'so'; old=read(ref/'config.json')
        source=read(old['dataset']['json_file'])['database']
        meta={}
        for role,field in [('source_train','src_train_split'),('source_val','src_val_split')]:
            db={k:dict(v,annotations=[a for a in v.get('annotations',[]) if int(a['label_id']) in SPEC['classes']]) for k,v in source.items() if v['subset'].lower() in old[field]}
            assert db
            save(entry/(role+'.json'),dict(database=db)); meta[role]=list(db)
        # This coordinator runs with CUDA hidden; only metadata leave the target annotation file.
        annotation=old['baseline']['annotation_for_offline_evaluation']
        full=read(annotation)['database']
        for role,field in [('source_val','src_val_split'),('target_val','tgt_val_split')]:
            db={k:{f:v[f] for f in ('subset','duration','fps') if f in v} for k,v in full.items() if v['subset'].lower() in old[field]}
            save(entry/(role+'_unlabeled.json'),dict(database=db)); meta[role]=list(db)
        assert not set(meta['source_train']) & set(meta['source_val'])
        assert not (set(meta['source_train'])|set(meta['source_val'])) & set(meta['target_val'])
        cfg=load_config(str(VENDOR/'configs/epic_slowfast_verb.yaml'))
        cfg['dataset'].update(num_classes=10,feat_folder=old['dataset']['feat_folder'],json_file=str(entry/'source_train.json'))
        cfg['model']['num_classes']=10
        cfg['init_rand_seed']=123; cfg['loader'].update(batch_size=2,num_workers=4)
        cfg['opt'].update(epochs=40,warmup_epochs=10)
        # The benchmark postprocessing is numerically identical to official EPIC settings.
        test=deepcopy(old['test_cfg']); test['ext_score_file']=None
        assert test==cfg['test_cfg'], 'unexpected postprocessing difference'
        cfg['model']['test_cfg']=test;cfg['model']['train_cfg']=cfg['train_cfg']
        cfg['benchmark']=dict(schema=SCHEMA,spec=SPEC,code=code,shift=s,reference=str(ref),
            reference_config_sha256=sha(ref/'config.json'),reference_complete_sha256=sha(ref/'complete.json'),
            steps_per_epoch=old['baseline']['epoch_budget']['iterations_per_epoch'],
            source_train=str(entry/'source_train.json'),source_val=str(entry/'source_val.json'),
            inference={role:str(entry/(role+'_unlabeled.json')) for role in ('source_val','target_val')})
        save(entry/'config.json',cfg)
        save(entry/'inputs.json',dict(members=meta,manifests={str(p):sha(p) for p in entry.glob('*.json')},source_GT_only=True))
        save(entry/'offline_evaluation.json',dict(annotation=annotation,annotation_sha256=sha(annotation),splits={r:old[f] for r,f in [('source_val','src_val_split'),('target_val','tgt_val_split')]}))
    save(root/'plan.json',dict(schema=SCHEMA,spec=SPEC,code=code,shifts=['S'+str(i) for i in range(1,7)],scope='new_SO_only_old_campaign_remains_stopped'))

def guard_audit(path,digest,commit):
    assert sha(path)==digest
    g=read(path); lease=read(g['lease'])
    assert sha(g['lease'])==g['lease_sha256'] and lease['commit']==commit
    assert g['owner_pid']==lease['owner_pid'] and g['observed']['uuid']==lease['observed']['uuid']
    assert g['observed']['free_mib']>=lease['minimum_free_mib']>=12288

def audit(entry):
    entry=Path(entry); c=read(entry/'complete.json'); cfg=read(entry/'config.json')
    assert c['schema']==SCHEMA and cfg['benchmark']['spec']==SPEC
    for p,h in c['artifacts'].items(): assert sha(p)==h,p
    for p,h in read(entry/'inputs.json')['manifests'].items(): assert sha(p)==h,p
    b=cfg['benchmark']; ref=Path(b['reference'])
    assert sha(ref/'config.json')==b['reference_config_sha256'] and sha(ref/'complete.json')==b['reference_complete_sha256']
    acceptance=read(entry.parents[1]/'acceptance.json');assert acceptance['status']=='passed' and acceptance['code']==b['code']
    for proof in acceptance['reports'].values():
        assert sha(proof['path'])==proof['sha256']
        report=read(proof['path']);guard_audit(report['guard'],report['guard_sha256'],b['code']['commit'])
        assert all(report['parameters_updated'].values()) and report['full_state_restore'] and report['next_step_exact']
    init=read(entry/'initialization.json');p=read(acceptance['reports'][b['shift']]['path'])
    assert init['model_sha256']==p['initial_model_sha256'] and not init['pretrained_checkpoint_loaded']
    t=read(entry/'termination.json'); assert terminal(t['history'])==t['decision']
    rows=read(entry/'metrics.json');assert [r['epoch'] for r in rows]==list(range(1,t['decision']['epoch']+1))
    assert [[r['epoch'],r['source_validation_loss']] for r in rows if 'source_validation_loss' in r]==t['history']
    assert all(r['steps']==b['steps_per_epoch'] and r['target_training_batches']==0 for r in rows)
    g=read(entry/'gpu_complete.json');guard_audit(g['guard'],g['guard_sha256'],b['code']['commit'])
    tr=read(entry/'train_gpu.json');guard_audit(tr['guard'],tr['guard_sha256'],b['code']['commit'])
    import torch
    for name in ('best','latest'):
        x=torch.load(entry/(name+'.pt'),map_location='cpu')
        assert x['epoch']==t['decision']['best_epoch' if name=='best' else 'epoch']
        assert x['config']==cfg and all(k in x for k in ('model','ema','optimizer','scheduler','rng','normalizers','history','metrics'))
        assert len(x['rng']['cuda'])==1 and x['rng']['cuda'][0].numel()>0
        if name=='latest': assert x['history']==t['history'] and x['metrics']==rows
        del x
    for role in ('source_val','target_val'):
        e=read(entry/'evaluation'/(role+'.json')); assert e['prediction_sha256']==sha(entry/(role+'.npz'))
        assert e['selected_epoch']==t['decision']['best_epoch'] and 0<=e['map_0.1_to_0.5']<=1
    result=read(entry/'evaluation/target_val.json')
    reference=read(ref/'evaluation/target_val.json')['map_0.1_to_0.5']
    return dict(shift=b['shift'],map=result['map_0.1_to_0.5'],reference_so_map=reference,
                delta_pp=100*(result['map_0.1_to_0.5']-reference),complete=str(entry/'complete.json'),complete_sha256=sha(entry/'complete.json'))
