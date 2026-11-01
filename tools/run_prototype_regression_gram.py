"""四卡联合回归几何×六方向；真实验收通过才fresh正式训练。"""
import argparse
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,POLICY
from tools.prototype_regression_gram import SPEC,SCHEMA,paired_config,assert_pair,audit
REFERENCE=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
SHIFTS=tuple('S%d'%i for i in range(1,7))


def gpu_stage(args):
    from tools.baseline_gpu import first_cuda_guard
    cfg=read(args.entry/'config.json');identity=code_identity();assert identity==cfg['baseline']['code']
    for path,digest in [(cfg['dataset']['json_file'],cfg['baseline']['source_manifest_sha256']),
                        (cfg['baseline']['target_manifest'],cfg['baseline']['target_manifest_sha256']),
                        (cfg['prototype_capacity']['prepared_cache'],cfg['prototype_capacity']['prepared_cache_sha256'])]:
        assert sha(path)==digest
    os.chdir(args.data_root);guard=first_cuda_guard()
    import torch
    from tools import baseline_engine as base
    e=args.entry
    if args.stage=='probe':
        ctx=base.build(cfg);reference=REFERENCE/args.shift
        assert ctx.initial_state_sha==read(reference/'termination.json')['initial_state_sha']
        assert ctx.initial_detector_sha==read(reference/'detector_initialization.json')['sha256']
        # 用已审成熟EMA仅核验新增loss/原优化器接线，formal另一个进程全新冷启动。
        saved=torch.load(reference/'best.pt',map_location='cpu');ctx.model.load_state_dict(saved['ema_state_dict'],strict=True)
        ctx.model_ema.module.load_state_dict(saved['ema_state_dict'],strict=True);del saved
        ctx.regression_gram_ledger=str(e/'probe_events.jsonl');phases={}
        for epoch,steps in [(1,2),(5,8)]:
            base.epoch_setup(ctx,epoch);loaders=ctx.train_loader_src,ctx.train_loader_tgt
            ctx.train_loader_src,ctx.train_loader_tgt=[base.StepLoader(z,steps) for z in loaders]
            before=base.tensor_digest(ctx.model.backbone_source.state_dict());reg_before=base.tensor_digest(ctx.model.reg_head.state_dict())
            torch.cuda.reset_peak_memory_stats();started=time.monotonic();ctx._train_epoch(epoch);torch.cuda.synchronize()
            metrics=ctx.train_metrics.result();g={k[2:]:float(v) for k,v in metrics.items() if k.startswith('t_gram_')}
            phases[str(epoch)]=dict(steps=steps,seconds=time.monotonic()-started,gram=g,
                backbone_updated=before!=base.tensor_digest(ctx.model.backbone_source.state_dict()),
                regression_updated=reg_before!=base.tensor_digest(ctx.model.reg_head.state_dict()),
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                source_gradient_norm=ctx.first_gradient_norm,target_gt_losses=[0.,0.,0.])
            ctx.train_loader_src,ctx.train_loader_tgt=loaders
        assert phases['1']['gram']['gram_events']==0
        for k in ['gram_loss','gram_events','gram_target_fpn_gradient_norm','gram_regression_gradient_norm']:
            assert phases['5']['gram'][k]>0 and math.isfinite(phases['5']['gram'][k]),k
        assert phases['5']['backbone_updated'] and phases['5']['regression_updated']
        ctx.valid_metrics_src.reset();validation=ctx._valid_epoch(5,base.StepLoader(ctx.val_loader_src,2),domain='src')
        assert math.isfinite(validation) and validation>0
        captured=base.capture(ctx,cfg,5,[[5,validation]]);stream=io.BytesIO();torch.save(captured,stream);stream.seek(0)
        captured=torch.load(stream,map_location='cpu');stream.close();base.restore(ctx,captured)
        actual=base.capture(ctx,cfg,5,[[5,validation]]);assert base.tensor_digest(actual)==base.tensor_digest(captured)
        save(e/'result.json',dict(status='passed',code=identity,phases=phases,initial_state_sha=ctx.initial_state_sha,
            initial_detector_sha=ctx.initial_detector_sha,source_validation_loss=validation,source_validation_scope='two videos only; formal uses complete split',
            full_state_roundtrip='passed',diagnostic_initialization='cold identity checked, then mature reference EMA loaded for mechanism only',
            checkpoint_written=False,formal_training=False,guard=guard,guard_sha256=sha(guard)))
    elif args.stage=='train':
        base.train(cfg,e)
    elif args.stage=='inference':
        import numpy as np
        from libs.models.SADA import make_meta_arch
        from libs.utils import fix_random_seed
        from libs.trainer.da_trainer import move_to_device
        saved=torch.load(e/'best.pt',map_location='cpu');fix_random_seed(cfg['seed'],include_cuda=True);torch.cuda.manual_seed(cfg['seed'])
        assert saved['config']==cfg and saved['epoch']==read(e/'termination.json')['decision']['best_epoch']
        model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda();model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval()
        for domain in ['source','target']:
            path=e/(domain+'_val_unlabeled.json');assert all('annotations' not in v for v in read(path)['database'].values())
            ds=base.dataset(cfg,path,cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'],False,True)
            columns={k:[] for k in ['video-id','t-start','t-end','label','score']}
            with torch.no_grad():
                for i in range(len(ds)):
                    video=move_to_device([ds[i]],'cuda:0');assert not video[0]['segments'].numel() and not video[0]['labels'].numel()
                    prediction=model(video_list_src=video,video_list_tgt=video,mode='test',domain=domain,curr_epoch=saved['epoch'])[0 if domain=='source' else 1][0]
                    columns['video-id'].extend([prediction['video_id']]*len(prediction['scores']))
                    for k,z in [('t-start',prediction['segments'][:,0]),('t-end',prediction['segments'][:,1]),('label',prediction['labels']),('score',prediction['scores'])]:columns[k].extend(z.cpu().tolist())
                    print('GRAM_OFFLINE_INFERENCE',domain,i,len(ds),flush=True)
            np.savez(e/(domain+'_val.npz'),**{k:np.asarray(v) for k,v in columns.items()})
    name='evaluate' if args.stage=='inference' else args.stage
    save(e/(name+'_gpu.json'),dict(guard=guard,guard_sha256=sha(guard),config_sha256=sha(e/'config.json'),target_gt_used=False))


def evaluate_cpu(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    from libs.evaluation import ANETdetection
    e=args.entry;cfg=read(e/'config.json');annotation=cfg['baseline']['annotation_for_offline_evaluation']
    assert sha(annotation)==cfg['baseline']['annotation_sha256'];results={}
    terminal=read(e/'termination.json')['decision'];checkpoint=sha(e/'best.pt')
    for domain in ['source','target']:
        path=e/(domain+'_val.npz')
        with np.load(path,allow_pickle=False) as f:prediction={k:f[k] for k in f.files}
        splits=cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split']
        evaluator=ANETdetection(annotation,splits,num_workers=2,action_subsampling='10_majority_classes')
        ap,mean,*_=evaluator.evaluate(prediction,verbose=True)
        result=dict(map_by_tiou=[float(x) for x in ap],
            videos=len(read(e/(domain+'_val_unlabeled.json'))['database']),prediction_count=len(prediction['score']),
            selected_epoch=terminal['best_epoch'],checkpoint_sha256=checkpoint,state='ema_state_dict',split=splits,prediction_sha256=sha(path))
        result['map_0.1_to_0.5']=float(mean)
        save(e/'evaluation'/(domain+'_val.json'),result);results[domain]=result
    save(e/'evaluation/manifest.json',dict(config_sha256=sha(e/'config.json'),checkpoint_sha256=checkpoint,
        annotation_sha256=sha(annotation),results=results,target_gt_scope='CPU offline task metric only'))


def complete(entry,root,identity):
    ref=REFERENCE/entry.parent.name
    paths=[p for p in entry.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.pt','.npz','.log'] and p.name!='complete.json']
    save(entry/'complete.json',dict(schema=SCHEMA,shift=entry.parent.name,regime='uda',code=identity,
        reference=str(ref),reference_complete_sha256=sha(ref/'complete.json'),target_gt_used=False,
        acceptance_sha256=sha(root/'acceptance.json'),artifacts={str(p):sha(p) for p in paths}))
    return audit(entry)


def main(args):
    from tools.run_prototype_target_cache import prepare,INVENTORY
    from tools.run_prototype_capacity import audit as audit_reference
    from tools.run_sada_baseline import acquire
    identity=code_identity();root=args.output;root.mkdir(parents=True,exist_ok=False)
    import shutil
    if shutil.disk_usage(root).free<15*2**30:raise ValueError('实验盘不足15GiB，不能启动六项checkpoint写入')
    tmp=Path('/vepfs/vlms/tmp')/('gram_'+identity['commit'][:7]);tmp.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(schema=SCHEMA,code=identity,specification=SPEC,shifts=SHIFTS,gpus=[0,1,2,3],
        seed=123,batch=2,policy=POLICY,reference_retrained=False,new_formal_count=6,initialization='fresh same cold initialization'))
    save(root/'state.json',dict(phase='input_audit',pid=os.getpid(),active={},completed=[]))
    prepare(root/'reference_snapshot',REFERENCE.parents[1],INVENTORY,identity)
    configs={};references={}
    for s in SHIFTS:
        ref=REFERENCE/s;references[s]=audit_reference(ref);old=read(ref/'config.json')
        cfg=paired_config(old,identity,tmp);configs[s]=cfg
        entry=root/'formal'/s/'gram';entry.mkdir(parents=True);save(entry/'config.json',cfg)
        for domain in ['source','target']:save(entry/(domain+'_val_unlabeled.json'),read(root/'reference_snapshot/formal'/s/(domain+'_val_unlabeled.json')),sort_keys=False)
        probe=root/'probes'/s;probe.mkdir(parents=True);save(probe/'config.json',cfg)
    save(root/'inputs.json',dict(code=identity,references=references,reference_snapshot_sha256=sha(root/'reference_snapshot/inputs.json')))
    active={};pending=list(SHIFTS);finished=[];reports={};infer=[];cpu=[];phase='real_data_acceptance';minimum=12288
    def state():save(root/'state.json',dict(phase=phase,pid=os.getpid(),pending=pending,completed=finished,
        active={k:dict(pid=t['p'].pid,shift=t['shift'],stage=t['stage'],gpu=t.get('gpu')) for k,t in active.items()}))
    def spawn(s,stage,card=None):
        e=root/('probes' if stage=='probe' else 'formal')/s
        if stage!='probe':e=e/'gram'
        cfg=configs[s];env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='123',
            CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        for name in ['RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE']:env.pop(name,None)
        if card:env.update(CUDA_VISIBLE_DEVICES=card['uuid'],SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
        with (e/(stage+'.log')).open('x') as log:p=subprocess.Popen([sys.executable,'-u','-B',__file__,'--output',str(root),'--entry',str(e),'--shift',s,'--stage',stage],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return dict(p=p,shift=s,stage=stage,gpu=card['index'] if card else None)
    try:
        while pending or active or infer or cpu:
            for key,t in list(active.items()):
                rc=t['p'].poll()
                if rc is None:continue
                if t.get('ctx'):t['ctx'].__exit__(None,None,None)
                del active[key]
                if rc:raise RuntimeError('阶段失败 '+key+' exit='+str(rc))
                s=t['shift']
                if t['stage']=='probe':
                    path=root/'probes'/s/'result.json';r=read(path);assert r['status']=='passed' and r['full_state_roundtrip']=='passed'
                    reports[s]=dict(path=str(path),sha256=sha(path),report=r)
                elif t['stage']=='train':infer.append(s)
                elif t['stage']=='inference':cpu.append(s)
                else:finished.append(complete(root/'formal'/s/'gram',root,identity))
            if phase=='real_data_acceptance' and len(reports)==6:
                minimum=max(12288,math.ceil(max(p['peak_reserved_mib'] for r in reports.values() for p in r['report']['phases'].values())+4096))
                save(root/'acceptance.json',dict(status='passed',batch=2,code=identity,reports=reports,minimum_free_mib=minimum,
                    scope='warm-reference mechanism diagnostics; formal initialization fresh and paired',inputs_sha256=sha(root/'inputs.json')))
                phase='formal';pending=list(SHIFTS)
            while cpu and sum(t['stage']=='evaluate_cpu' for t in active.values())<2:
                s=cpu.pop(0);active[s+'/evaluate_cpu']=spawn(s,'evaluate_cpu')
            queues=[('probe',pending)] if phase=='real_data_acceptance' else [('inference',infer),('train',pending)]
            for stage,queue in queues:
                for s in list(queue):
                    found=acquire([0,1,2,3],args.data_root,root,minimum,identity['commit'])
                    if not found:break
                    ctx,card=found
                    try:t=spawn(s,stage,card)
                    except BaseException:ctx.__exit__(None,None,None);raise
                    t['ctx']=ctx;active[s+'/'+stage]=t;queue.remove(s)
            state();time.sleep(3)
        assert phase=='formal' and len(finished)==6 and {r['shift'] for r in finished}==set(SHIFTS)
        save(root/'final_audit.json',dict(status='passed',count=6,results=finished,mean_map=sum(z['map'] for z in finished)/6,mean_delta_pp=sum(z['delta_pp'] for z in finished)/6))
        phase='complete';state()
    except BaseException as err:save(root/'error.json',dict(type=type(err).__name__,message=str(err)));phase='failed';state();raise
    finally:
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        for t in active.values():
            try:t['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['p'].pid,signal.SIGKILL);t['p'].wait()
            if t.get('ctx'):t['ctx'].__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--entry',type=Path)
    p.add_argument('--shift',choices=SHIFTS);p.add_argument('--stage',choices=['probe','train','inference','evaluate_cpu'])
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));a=p.parse_args()
    if a.stage=='evaluate_cpu':evaluate_cpu(a)
    elif a.stage:gpu_stage(a)
    else:
        def stop(sig,frame):raise SystemExit(128+sig)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main(a)
