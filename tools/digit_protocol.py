"""DiGIT SO：固定内部六方向，目标GT仅离线CPU评估。"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import math
import subprocess

CODE = Path(__file__).resolve().parents[1]
VENDOR = CODE / 'vendor/digit'
REFERENCE = Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029')
SCHEMA = 'digit-source-only-six-shift-v1'
SPEC = dict(seed=123,batch_size=2,max_epochs=50,valid_every=5,patience=15,
    selection='source_equal_video_main_decoder_weighted_cls_plus_iou_per_GT_instance',
    ema_decay=.999,initialization='fresh_official_model_then_deepcopy_ema',
    optimizer='official_AdamW_default_groups',learning_rate=.00005,weight_decay=0.,
    warmup_epochs=0,scheduler='StepLR_epoch50',gradient_clip=.1,
    architecture='official_DiGIT_MDGE_CAID_6encoder_6decoder_512dim',
    original_configuration='THUMOS14_InternVideo2_adapted_to_fixed_EPIC10_SlowFast',
    raw_features='unchanged_SlowFast',classes=list(range(10)),target_training=False,
    vlm=False,adversarial=False,precision='float32',max_queries=900,base_scale_frames=16,
    epoch_steps='same_as_existing_SO_reference',source_cycle='itertools_cycle_cached_batches',
    source_crop='same_benchmark_grid_crop_then_exact_DiGIT_seconds_conversion',
    postprocessing='official_one_class_per_query_then_fixed_benchmark_softNMS',
    scheduling='user_authorized_shared_GPU_no_public_exclusive_lock')


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
    import runpy
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
        author={k:v for k,v in runpy.run_path(str(VENDOR/'config/digit/_base_.py')).items() if not k.startswith('_')}
        author.update({k:v for k,v in runpy.run_path(str(VENDOR/'config/digit/internvideo2/thumos14.py')).items() if not k.startswith('_')})
        model=deepcopy(author)
        model.update(feature_dim=2304,num_classes=10,max_seq_len=2304,batch_size=2,stride=16,base_frame=32,
                     default_fps=30,epochs=50,lr_drop=50,device='cuda',amp=False,mem_cache=False)
        dataset=deepcopy(old['dataset'])
        dataset.update(num_classes=10,input_dim=2304,feat_stride=16,num_frames=32,default_fps=30,
                       downsample_rate=1,max_seq_len=2304,trunc_thresh=.3,crop_ratio=[.9,1.],file_ext='.npz',file_prefix=None)
        test=deepcopy(old['test_cfg']);test['ext_score_file']=None
        cfg=dict(model=model,dataset=dataset,loader=dict(batch_size=2,num_workers=4),test_cfg=test,
                 author_configuration=author,init_rand_seed=123)
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
    assert lease['policy']=='shared_without_public_exclusive_lock'

def audit(entry):
    entry=Path(entry); c=read(entry/'complete.json'); cfg=read(entry/'config.json')
    assert c['schema']==SCHEMA and cfg['benchmark']['spec']==SPEC
    for p,h in c['artifacts'].items(): assert sha(p)==h,p
    for p,h in read(entry/'inputs.json')['manifests'].items(): assert sha(p)==h,p
    b=cfg['benchmark']; ref=Path(b['reference'])
    assert sha(ref/'config.json')==b['reference_config_sha256'] and sha(ref/'complete.json')==b['reference_complete_sha256']
    acceptance=read(entry.parents[1]/'acceptance.json');assert acceptance['status']=='passed' and acceptance['code']==b['code']
    runtime=entry.parents[1]/'runtime.json';assert sha(runtime)==acceptance['runtime_sha256']
    for p,h in read(runtime)['extensions'].items():assert sha(p)==h
    members=read(entry/'inputs.json')['members']
    assert not set(members['source_train'])&set(members['source_val'])
    assert not (set(members['source_train'])|set(members['source_val']))&set(members['target_val'])
    for role,path in b['inference'].items():
        db=read(path)['database'];assert list(db)==members[role] and all('annotations' not in v for v in db.values())
    assert cfg['model']['feature_dim']==2304 and cfg['model']['num_classes']==10
    assert [cfg['model'][key] for key in ('hidden_dim','enc_layers','dec_layers')]==[512,6,6]
    for proof in acceptance['reports'].values():
        assert sha(proof['path'])==proof['sha256']
        report=read(proof['path']);guard_audit(report['guard'],report['guard_sha256'],b['code']['commit'])
        assert all(report['parameters_updated'].values()) and report['full_state_restore'] and report['next_step_exact']
        assert report['official_model_and_postprocessor'] and len(report['cuda_operator_reference'])==2
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
        assert x['config']==cfg and all(k in x for k in ('model','ema','optimizer','scheduler','rng','criterion','ema_updates','history','metrics'))
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
