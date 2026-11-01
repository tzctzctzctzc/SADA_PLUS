"""双卡源域定位诊断：不更新任何参数，不读取目标数据，不产出目标mAP。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_sada_baseline import acquire,prepare_file_limit
from tools.boundary_diagnostic import SPEC,statistics,decide

ACTIVE={}


def audit(path):
    j=read(path);d=Path(j['output']);a=read(d/'complete.json')
    assert a['status']=='passed' and a['job_sha256']==sha(path) and a['code']==j['code']
    for key in ('config','checkpoint','source_metadata','source_GT'):
        assert sha(j[key])==j[key+'_sha256']
    for name,h in a['artifacts'].items():assert sha(d/name)==h
    guard=read(a['guard']);lease=read(guard['lease'])
    assert sha(a['guard'])==a['guard_sha256'] and sha(guard['lease'])==guard['lease_sha256']
    assert lease['commit']==j['code']['commit'] and guard['observed']['uuid']==lease['observed']['uuid']
    assert guard['owner_pid']==lease['owner_pid'] and guard['observed']['free_mib']>=12288
    result=read(d/'result.json');assert result['model_unchanged'] and result['ordinary_prediction_exact']
    assert result['video_ids']==j['video_ids'] and not result['target_data_used']
    return dict(status='passed',shift=j['shift'],complete_sha256=sha(d/'complete.json'),
        statistics=result['statistics'],scored_rows=result['scored_rows'],videos=len(result['video_ids']))


def worker(args):
    import numpy as np
    import torch
    from tools.baseline_gpu import first_cuda_guard
    from tools.baseline_engine import dataset,tensor_digest
    from tools.run_tta import load_model
    from tools.tta_core import dense,unlabeled,crop,postprocess,raw_prediction
    from tools.boundary_diagnostic import ranges,candidates,attach_stability,match_source
    j=read(args.job);d=Path(j['output'])
    assert j['code']==code_identity() and j['specification']==SPEC
    for key in ('config','checkpoint','source_metadata','source_GT'):assert sha(j[key])==j[key+'_sha256']
    assert all('annotations' not in x for x in read(j['source_metadata'])['database'].values())
    guard=first_cuda_guard();os.chdir(args.data_root)
    cfg=read(j['config']);model=load_model(cfg,j['checkpoint']).eval()
    for p in model.parameters():p.requires_grad_(False)
    before=tensor_digest(model.state_dict());ds=dataset(cfg,j['source_metadata'],cfg['src_val_split'],False,True)
    indices={ds[i]['metadata']['video_id']:i for i in range(len(ds))}
    assert set(indices)==set(j['video_ids'])
    gt=read(j['source_GT'])['database'];matched=[];counts=[];exact=False
    with (d/'observations.jsonl').open('x') as stream,torch.no_grad():
        for n,vid in enumerate(j['video_ids']):
            video=unlabeled(ds[indices[vid]]);windows=ranges(video['feats'].shape[-1]);outputs=[]
            for begin,end in windows:
                raw=dense(model,crop(video,begin,end))
                if n==0 and begin==0 and end==video['feats'].shape[-1]:
                    ours=postprocess(model,video,raw_prediction(model,raw))
                    official=model([video],[video],mode='test')[0][0]
                    exact=all(torch.equal(ours[k],official[k]) for k in ('segments','scores','labels'))
                    assert exact,'新诊断必须保持原普通推理逐值相同'
                outputs.append({k:[x.detach().cpu() for x in raw[k]] for k in ('points','offsets','logits','masks')})
            rows=attach_stability(candidates(outputs[0],video['feats'].shape[-1]),outputs,windows)
            # 先落盘无GT观察，再读取本video源GT；GT不会决定候选、视图或稳定性。
            stream.write(json.dumps(dict(video_id=vid,windows=windows,candidates=rows))+'\n');stream.flush();os.fsync(stream.fileno())
            meta=video['metadata'];anns=gt[vid].get('annotations',[])
            segments=[[(z*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride'] for z in x['segment']] for x in anns]
            labels=[x['label_id'] for x in anns]
            scored=match_source(rows,segments,labels)
            matched.extend(dict(x,video_id=vid) for x in scored)
            counts.append(dict(video_id=vid,views=len(windows),candidates=len(rows),eligible=sum('relative_std' in x for x in rows),matched=sum(x['gt_supported'] for x in scored)))
            print('SOURCE_DIAGNOSTIC',n+1,len(ds),json.dumps(counts[-1]),flush=True)
    result=dict(status='passed',shift=j['shift'],video_ids=j['video_ids'],counts=counts,
        scored_rows=matched,statistics=statistics(matched),model_unchanged=tensor_digest(model.state_dict())==before,
        ordinary_prediction_exact=exact,target_data_used=False,candidate_selection_GT=False,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    assert result['model_unchanged'];save(d/'result.json',result)
    save(d/'complete.json',dict(status='passed',code=j['code'],job_sha256=sha(args.job),guard=guard,
        guard_sha256=sha(guard),artifacts={n:sha(d/n) for n in ('result.json','observations.jsonl')}))


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);identity=code_identity()
    save(root/'resources.json',prepare_file_limit());previous=read(args.previous/'monitor/final_independent_audit.json')
    assert previous['status']=='passed' and previous['count']==18
    save(root/'plan.json',dict(specification=SPEC,code=identity,previous_final_sha256=sha(args.previous/'monitor/final_independent_audit.json'),
        scope='fixed source-validation mechanism diagnostic only; no target training or target evaluation',count=6))
    pending=[];done=[]
    for i in range(1,7):
        s='S'+str(i);entry=args.prototype/'formal'/s/'prototype';cfg=read(entry/'config.json')
        source=read(cfg['dataset']['json_file'])['database']
        ids=[v for v,x in source.items() if x['subset'].lower() in cfg['src_val_split']]
        ids=sorted(ids,key=lambda v:hashlib.sha256(('123:'+v).encode()).hexdigest())[:8]
        inp=root/'inputs'/s;source_gt=inp/'source_GT.json';source_meta=inp/'source_metadata.json'
        save(source_gt,dict(database={v:source[v] for v in ids}))
        save(source_meta,dict(database={v:{k:x for k,x in source[v].items() if k!='annotations'} for v in ids}))
        d=root/'formal'/s;d.mkdir(parents=True);j=dict(shift=s,code=identity,specification=SPEC,output=str(d),video_ids=ids,
            config=str(entry/'config.json'),checkpoint=str(entry/'best.pt'),source_metadata=str(source_meta),source_GT=str(source_gt))
        for k in ('config','checkpoint','source_metadata','source_GT'):j[k+'_sha256']=sha(j[k])
        path=d/'job.json';save(path,j);pending.append(path)
    try:
        while pending or ACTIVE:
            for key,t in list(ACTIVE.items()):
                if t['process'].poll() is None:continue
                t['context'].__exit__(None,None,None);del ACTIVE[key]
                if t['process'].returncode:raise RuntimeError('源诊断失败:'+key)
                done.append(audit(t['job']))
            for path in list(pending):
                lease=acquire([0,1],args.data_root,root,12288,identity['commit'])
                if not lease:break
                context,slot=lease;j=read(path);env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',
                    PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=slot['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                    SADA_BASELINE_LEASE=slot['lease'],SADA_BASELINE_LEASE_SHA256=slot['lease_sha256'])
                with (Path(j['output'])/'run.log').open('x') as stream:
                    p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--job',str(path),'--data-root',str(args.data_root)],
                        cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                ACTIVE[j['shift']]=dict(process=p,context=context,job=path,gpu=slot['index']);pending.remove(path)
            save(root/'state.json',dict(phase='source_diagnostic',pid=os.getpid(),completed=len(done),total=6,
                active={k:dict(pid=t['process'].pid,gpu=t['gpu']) for k,t in ACTIVE.items()}))
            if pending or ACTIVE:time.sleep(3)
        assert len(done)==6 and {x['shift'] for x in done}=={'S1','S2','S3','S4','S5','S6'}
        final=decide(done);save(root/'final_audit.json',dict(final,count=6,results=done))
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),active={},count=6,mechanism_gate_passed=final['mechanism_gate_passed']))
    except BaseException as e:
        save(root/'error.json',dict(type=type(e).__name__,message=str(e)));save(root/'state.json',dict(phase='failed',pid=os.getpid(),active={}))
        raise
    finally:
        for t in ACTIVE.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in ACTIVE.values():
            try:t['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['process'].pid,signal.SIGKILL);t['process'].wait()
            t['context'].__exit__(None,None,None)


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);raise SystemExit(128+signum)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--prototype',type=Path);p.add_argument('--previous',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));args=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.job:worker(args)
    else:main(args)
