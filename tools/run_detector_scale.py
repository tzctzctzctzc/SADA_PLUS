"""原生检测器固定三尺度预测融合；全预测重新解码/NMS，非旧池删框。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_native_joint import DENSE,SOURCE
TASKS=tuple(s+'_'+role for s in ('S1','S2') for role in ('validation','target_val'))
MODES=('baseline','score_only','score_boundary')
SPEC=dict(schema='native-detector-fixed-scale-ensemble-v1',scales=[.75,1.,1.25],primary='score_boundary',
    resample='raw2304 temporal linear aligncorners, Tscaled=round((T-1)*scale)+1; effective=(Tscaled-1)/(T-1)',
    fuse='map perlevel probabilities and offsets to originalphysical grid; equal mean of validviews, offsets divided by effective ratio; originalNMS/voting config unchanged',
    score_only='same averaged classprobabilities but originalregression, diagnostic paired ablation not winnerselection',
    baseline='allvideos singleview predictions must match oldoriginal baseline arrays bitwise',
    source='S1/S2 entire sourcevalidation andtargetvalidation, noGT in GPUinputs, nofit',target_train_GT_read=False,target_val_used_for_selection=False,
    learning_steps=0,budget_seconds=1800,acceptance='primary two-direction target mAP@.1:.5 gain>=1pp and bothdirectionspositive; no scalesearch',
    scope='S1/S2 historical development, 3scale interpolation of cachedrawfeatures not newpixel/video backbone or paper reproduction',
    original_pool_constraint_relaxed='user authorized any metric-improving idea: new scores/endpoints/pool allowed; baseline detectorfiles unchanged')


def prepare(root):
    import torch
    from tools import baseline_engine as base
    folder=root/'inputs';folder.mkdir();report={}
    allowed={'video','segments','scores','labels','hard','grid_boxes','video_metadata','raw_path','raw_size','raw_mtime_ns','raw_sha256','downsample'}
    for s in ('S1','S2'):
        path=DENSE/'inputs'/s/'data.pt';assert sha(path)==read(DENSE/'prepare_complete.json')['results'][s]['data_sha256'];data=torch.load(path,map_location='cpu')
        for role in ('validation','target_val'):
            rows=data['roles'][role];assert all(set(r)<=allowed for r in rows);p=folder/(s+'_'+role+'.pt');base.checkpoint(p,rows);report[s+'_'+role]=dict(sha256=sha(p),videos=[r['video'] for r in rows],source_sha256=sha(path))
    save(root/'prepare.json',dict(passed=True,tasks=report,GPU_inputs_no_GT=True))


def infer(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.run_foreground_calibration_oracle import load_detector
    from tools.run_dense_revisit import raw_load,seed
    from tools.detector_scale_math import rescale_raw,fuse_heads
    guard=first_cuda_guard(entry);s,role=entry.name.split('_',1);path=entry.parent/'inputs'/(entry.name+'.pt');assert sha(path)==read(entry.parent/'prepare.json')['tasks'][entry.name]['sha256'];rows=torch.load(path,map_location='cpu');seed();model=load_detector(SOURCE/s);assert not model.use_trident_head
    model.requires_grad_(False);model.eval();digest=base.tensor_digest(model.state_dict());cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0);started=time.monotonic();outputs={m:[] for m in MODES};cases=[];(entry/'videos').mkdir()
    def heads(raw,metadata):
        video=[dict(feats=raw.T,segments=raw.new_empty(0,2),labels=torch.empty(0,dtype=torch.long,device='cuda'),metadata=metadata)]
        x,mask=model.preprocessing(video,mode='test');f,m,_=model.backbone_source(model.dropout(x),mask);f,m=model.neck_source(f,m);cls,_=model.cls_head(f,m);reg,_=model.reg_head(f,m)
        return video,f,m,[v[0] for v in cls],[v[0] for v in reg]
    with torch.no_grad():
        for index,r in enumerate(rows):
            tick=time.monotonic();raw=raw_load(r).cuda();video,f,m,cls,reg=heads(raw,r['video_metadata']);points=model.point_generator(f)
            def decode(c,z):return model.inference(video,points,[v[:,0] for v in m],[v.T[None] for v in c],[v.T[None] for v in z],None,None)[0]
            single=decode(cls,reg)
            for k in ('segments','scores','labels'):assert torch.equal(single[k].cpu(),r[k])
            views=[];proof=[]
            for scale in (.75,1.25):
                rx,ratio=rescale_raw(raw,scale);_,_,vm,vc,vr=heads(rx,r['video_metadata']);views.append((vc,vr,[x[0,0].bool() for x in vm],ratio));proof.append(dict(scale=scale,ratio=ratio,length=len(rx),cls=[x.cpu() for x in vc],reg=[x.cpu() for x in vr],valid=[x[0,0].cpu().bool() for x in vm]))
            fused_cls=[];fused_reg=[]
            for l in range(len(cls)):
                a,b=fuse_heads(cls[l],reg[l],[(c[l],z[l],v[l],q) for c,z,v,q in views]);fused_cls.append(a);fused_reg.append(b)
            result=dict(baseline=single,score_only=decode(fused_cls,reg),score_boundary=decode(fused_cls,fused_reg))
            case=dict(video=r['video'],raw_sha256=r['raw_sha256'],base_length=len(raw),base_cls=[x.cpu() for x in cls],base_reg=[x.cpu() for x in reg],base_valid=[x[0,0].cpu().bool() for x in m],views=proof,fused_cls=[x.cpu() for x in fused_cls],fused_reg=[x.cpu() for x in fused_reg])
            # 保存的CPU头张量再搬GPU，重放全部融合张量，排除缓存接线差异。
            for l in range(len(cls)):
                a,b=fuse_heads(case['base_cls'][l].cuda(),case['base_reg'][l].cuda(),[(v['cls'][l].cuda(),v['reg'][l].cuda(),v['valid'][l].cuda(),v['ratio']) for v in proof]);assert torch.equal(a,fused_cls[l]) and torch.equal(b,fused_reg[l])
            p=entry/'videos'/(r['video']+'.pt');base.checkpoint(p,case);cases.append(dict(video=r['video'],path=str(p),sha256=sha(p),seconds=time.monotonic()-tick))
            for mode,out in result.items():outputs[mode].append(dict(video=r['video'],segments=out['segments'].cpu(),scores=out['scores'].cpu(),labels=out['labels'].cpu()))
            print('SCALE_VIDEO',entry.name,index+1,len(rows),cases[-1]['seconds'],flush=True)
            if index==2:
                estimate=(time.monotonic()-started)*len(rows)/3;save(entry/'throughput.json',dict(first3_seconds=time.monotonic()-started,projected_seconds=estimate,limit=1800));assert estimate<1800,'read-only time budget exceeded'
    assert digest==base.tensor_digest(model.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0));base.checkpoint(entry/'predictions.pt',outputs)
    save(entry/'gpu_complete.json',dict(passed=True,guard=guard,guard_sha256=sha(guard),input_sha256=sha(path),predictions_sha256=sha(entry/'predictions.pt'),cases=cases,model_digest_unchanged=digest,all_baseline_arrays_bitwise_equal=True,all_cached_fusion_GPU_bitwise_equal=True,CPUCUDA_RNG_unchanged=True,optimizer_steps=0,seconds=time.monotonic()-started))


def evaluate(root,s):
    import torch,numpy as np,pandas as pd
    from tools.audit_background_ap_cost import IndependentAP,official
    from tools.run_foreground_calibration_oracle import load_truth
    cfg=read(SOURCE/s/'config.json');results={}
    for role in ('validation','target_val'):
        gt=load_truth(cfg['dataset']['json_file'] if role=='validation' else cfg['baseline']['annotation_for_offline_evaluation'],cfg['src_val_split'] if role=='validation' else cfg['tgt_val_split'])
        path=root/(s+'_'+role)/'predictions.pt';out=torch.load(path,map_location='cpu');methods={}
        for mode,rows in out.items():
            frame=pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c),float(v)) for r in rows for b,c,v in zip(r['segments'],r['labels'],r['scores'])],columns=['video-id','t-start','t-end','label','score']);curves=[]
            for c in range(10):
                a=gt[gt.label==c].reset_index(drop=True);b=frame[frame.label==c].reset_index(drop=True);engine=IndependentAP(a,b)
                # 官方完整曲线用于最终指标；独立AP核均值，固定10类无GT类按0。
                ap=official(a,b) if len(a) and len(b) else np.zeros(5)
                ap=np.asarray(ap);assert ap.shape==(5,);assert np.allclose(ap,engine.ap(np.ones(len(b),bool)),atol=1e-7,rtol=0);curves.append(ap)
            table=np.asarray(curves);methods[mode]=dict(map=float(table.mean()),per_tiou_AP=table.mean(0).tolist(),per_class_AP=table.mean(1).tolist(),per_class_tiou_AP=table.tolist(),predictions=len(frame))
        for mode in MODES:methods[mode]['delta_pp']=100*(methods[mode]['map']-methods['baseline']['map'])
        if role=='target_val':assert abs(methods['baseline']['map']-read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/s/'evaluation/target_val.json')['map_0.1_to_0.5'])<1e-7
        results[role]=methods
    save(root/(s+'_evaluation.json'),dict(passed=True,results=results,official_full_AP_independent_means_equal=True,target_val_used_for_selection=False))


def run(root):
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity()));prepare(root);active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    try:
        for stage,tasks in [('infer',TASKS),('evaluate',('S1','S2'))]:
            pending=list(tasks)
            while pending or active:
                for task,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[task]
                    if t['p'].returncode:raise RuntimeError(task+'/'+stage+' failed')
                    done.append(task+'/'+stage)
                devices=[None]*max(0,2-len(active)) if stage=='evaluate' else [g for g in range(4) if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    task=pending.pop(0);e=root/task;e.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='123')
                    if g is not None:
                        v=snapshot(g);p=e/'lease_infer.json';save(p,dict(owner_pid=os.getpid(),index=g,uuid=v['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']));env.update(CUDA_VISIBLE_DEVICES=v['uuid'],PROPOSAL_FOREGROUND_LEASE=str(p),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(p))
                    log=(e/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(e),'--worker',stage],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()},pending=pending,completed=done));time.sleep(3)
        save(root/'final.json',dict(passed=True,allGPU_sha256={t:sha(root/t/'gpu_complete.json') for t in TASKS},evaluation_sha256={s:sha(root/(s+'_evaluation.json')) for s in ('S1','S2')}));save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex),automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['infer','evaluate']);a=p.parse_args()
    if a.worker=='infer':infer(a.entry)
    elif a.worker=='evaluate':evaluate(a.entry.parent,a.entry.name)
    else:run(a.root)
