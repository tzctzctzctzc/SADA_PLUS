"""四卡：源语义对照、目标重加权、条件CORAL、EMA一致性；目标GT不入训练。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_layer_signals import seed
from tools.run_highscore_pairs import DIAG,OLD,SOURCE,concatenate
from tools.prepare_uda_background import prepare_source,extract,density,target_rows
from tools.uda_background_math import UDAHead,MODES,source_objective,conditional_coral,consistency_loss,ema_update
TASKS=tuple(s+'_'+m for s in ('S1','S2') for m in MODES)
EXTRACT=tuple(s+'_'+r for s in ('S1','S2') for r in ('target_train','target_val'))
SPEC=dict(schema='highscore-ontology-UDA-short-v1',source_only=False,target_train_unlabeled=True,target_train_GT_read=False,
    shifts=['S1','S2'],modes=MODES,tasks=TASKS,epochs=20,checkpoints=[1,5,10,20],batch=128,pair_batch=64,target_batch=128,
    seed=311104,order_seed=123,pair_seed=311105,target_seed=311108,augmentation_seed=311109,
    lr=3e-4,weight_decay=.05,clip=1.,rates=[.005,.01,.02,.05],budget_seconds=900,
    shared_source_loss='balancedBG_BCE+.25quality+.25 originalAPcriticalPairs+.1source97verbCE; task foreground priority, ignore ambiguous semantic label',
    density='100steps linear domain discriminator on sourceclass-demeaned inner512, equal domains/video-balanced sampling, ratio pT/(1-pT) clipped.25..4; importance within sourceBG/nonBG groups and APpairs',
    coral='same predicted-class covariance alignment of learned inner64, unbiasedcov/(4*d*d), coefficient10, no targetBG pseudo labels',
    teacher='EMA teacher weak original sequence; student additive Gaussian .05 sourcefeatureSTD; independent CUDA augmentationgenerator; sigmoidMSE of BG/quality, coefficient1; no hard pseudoBG',
    ramp='CORAL/consistency0 for5epochs then linear(epoch-5)/15; no epoch selection',ema='all4methods same min(.99,1-1/(globalstep+1)); all4useEMA at fixed20',
    precision='originalfloat32/TF32 inherited, noAMP/quantization',
    evaluation='fullsourcefit/headheldout/sourceval/targetval after8heads frozen, fixed4rates, allscores/classes/endpoints and low-score replacements unchanged',
    caveat='S1/S2 only, head-only source holdout; targetval has historical development exposure, frozen-round evaluation not untouched independent test; no fullsixdirection claim',
    literature=dict(coral='https://arxiv.org/html/1607.01719v1',teacher='https://arxiv.org/abs/1703.01780',teacher_code='https://github.com/CuriousAI/mean-teacher',importance='https://jmlr.org/papers/v8/sugiyama07a.html'),
    adaptation='97verb auxiliary and APcritical background objective are task-specific additions; not full original paper benchmark reproductions')


class TargetStream:
    def __init__(self,n):
        import torch
        self.n=n;self.generator=torch.Generator().manual_seed(311108);self.order=torch.randperm(n,generator=self.generator);self.cursor=0;self.seen=torch.zeros(n,dtype=torch.long)
    def draw(self,count=128):
        import torch
        parts=[];remaining=count
        while remaining:
            take=min(remaining,self.n-self.cursor);parts.append(self.order[self.cursor:self.cursor+take]);self.cursor+=take;remaining-=take
            if self.cursor==self.n:self.order=torch.randperm(self.n,generator=self.generator);self.cursor=0
        ids=torch.cat(parts);self.seen.index_add_(0,ids,torch.ones(len(ids),dtype=torch.long));return ids
    def state(self):return dict(generator=self.generator.get_state(),order=self.order,cursor=self.cursor,seen=self.seen)
    def restore(self,s):self.generator.set_state(s['generator']);self.order=s['order'].clone();self.cursor=s['cursor'];self.seen=s['seen'].clone()


def step(model,teacher,opt,data,target,ids,pi,table,pw,importance,mode,ramp,ti,noise,global_step):
    import torch
    opt.zero_grad(set_to_none=True);loss,parts,source_inside=source_objective(model,data,ids,table[pi],pw[pi],importance if mode=='density' else None)
    tx=target['seq'][ti];tv=target['valid'][ti];tm=target['metadata'][ti]
    target_logits,_,target_inside=model.all_outputs(tx,tv,tm)
    coral=conditional_coral(source_inside,target_inside,data['metadata'][ids,:10].argmax(1),tm[:,:10].argmax(1))
    with torch.no_grad():teacher_logits=teacher(tx,tv,tm)
    student_logits=model(tx+noise,tv,tm);cons=consistency_loss(student_logits,teacher_logits)
    if mode=='coral' and ramp:loss=loss+10*ramp*coral
    if mode=='teacher' and ramp:loss=loss+ramp*cons
    assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();ema_update(model,teacher,global_step)
    return dict(loss=float(loss.detach()),gradient=float(norm),**{k:float(v.detach()) for k,v in parts.items()},coral=float(coral.detach()),consistency=float(cons.detach()))


def capture(model,teacher,opt,order,pr,stream,aug,history,global_step,root,shift):
    import torch
    return dict(model=model.state_dict(),ema=teacher.state_dict(),optimizer=opt.state_dict(),order_rng=order.get_state(),pair_rng=pr.get_state(),target_stream=stream.state(),augmentation_rng=aug.get_state(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),history=history,global_step=global_step,spec=SPEC,
        source_sha256=read(root/'source_prepare.json')['results'][shift]['source_sha256'],target_train_complete_sha256=sha(root/(shift+'_target_train')/'complete.json'),density_complete_sha256=sha(root/(shift+'_density')/'complete.json'))


def kernel(entry,data,target,table,pw,importance,scale):
    import torch
    from tools import baseline_engine as base
    from tools.highscore_pair_math import make_model,objective,build_batch
    from tools.run_highscore_pairs import update
    shift,mode=entry.name.split('_');seed();old=make_model().cuda();seed();new=UDAHead().cuda();batch=build_batch(data,torch.arange(32,device='cuda'),table[:8]);ids=torch.arange(32,device='cuda');pi=torch.arange(8,device='cuda')
    la,_,_=source_objective(new,data,ids,table[pi],pw[pi],None,semantic_weight=0);lb,_=objective(old,batch,len(ids),pw[pi],True);assert torch.equal(la,lb)
    la.backward();lb.backward()
    for name,p in old.named_parameters():assert torch.equal(p.grad,dict(new.named_parameters())[name].grad)
    oa=torch.optim.AdamW(old.parameters(),lr=3e-4,weight_decay=.05);ob=torch.optim.AdamW(new.parameters(),lr=3e-4,weight_decay=.05);oa.step();ob.step()
    for name,p in old.named_parameters():assert torch.equal(p,dict(new.named_parameters())[name])
    seed();model=UDAHead().cuda();teacher=deepcopy(model).eval();teacher.requires_grad_(False);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05);order=torch.Generator().manual_seed(123);pr=torch.Generator().manual_seed(311105);stream=TargetStream(len(target['seq']));aug=torch.Generator(device='cuda').manual_seed(311109)
    ti=stream.draw(32).cuda();noise=torch.randn(target['seq'][ti].shape,device='cuda',generator=aug)*scale*.05
    before=base.tensor_digest(model.state_dict());terms=step(model,teacher,opt,data,target,ids,pi,table,pw,importance,mode,1.,ti,noise,0);assert before!=base.tensor_digest(model.state_dict())
    base.checkpoint(entry/'kernel.pt',capture(model,teacher,opt,order,pr,stream,aug,[],1,entry.parent,shift));saved=torch.load(entry/'kernel.pt',map_location='cpu');fresh=UDAHead().cuda();fresh.load_state_dict(saved['model']);ft=deepcopy(fresh).eval();ft.load_state_dict(saved['ema']);ft.requires_grad_(False);fo=torch.optim.AdamW(fresh.parameters());fo.load_state_dict(saved['optimizer'])
    def restore():
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],0);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);order.set_state(saved['order_rng']);pr.set_state(saved['pair_rng']);stream.restore(saved['target_stream']);aug.set_state(saved['augmentation_rng'])
    def draw():
        t=stream.draw(32).cuda();n=torch.randn(target['seq'][t].shape,device='cuda',generator=aug)*scale*.05
        return t,n,torch.randperm(13,generator=order),torch.randperm(17,generator=pr),torch.rand(3,device='cuda'),torch.rand(3)
    restore();a=draw();restore();b=draw();assert all(torch.equal(x,y) for x,y in zip(a,b))
    restore();ti,noise,*_=draw();step(model,teacher,opt,data,target,ids,pi,table,pw,importance,mode,1.,ti,noise,1)
    restore();ti,noise,*_=draw();step(fresh,ft,fo,data,target,ids,pi,table,pw,importance,mode,1.,ti,noise,1)
    assert base.tensor_digest(model.state_dict())==base.tensor_digest(fresh.state_dict()) and base.tensor_digest(teacher.state_dict())==base.tensor_digest(ft.state_dict()) and base.tensor_digest(opt.state_dict())==base.tensor_digest(fo.state_dict())
    assert all(p.grad is None for p in teacher.parameters())
    save(entry/'kernel_complete.json',dict(passed=True,original_core_loss_grad_update_exact=True,real_update=True,model_EMA_optimizer_RNG_nextstep_exact=True,draws_and_target_cursor_exact=True,
        teacher_detached=True,terms=terms,state_sha256=sha(entry/'kernel.pt')))


def train(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);shift,mode=entry.name.split('_');began=time.monotonic();sourcepath=entry.parent/'inputs'/shift/'source.pt';assert sha(sourcepath)==read(entry.parent/'source_prepare.json')['results'][shift]['source_sha256']
    roles=torch.load(sourcepath,map_location='cpu');targettrain=target_rows(entry.parent,shift,'target_train');data=concatenate(roles['fit'],('seq','valid','metadata','bg','quality','semantic'),'cuda');target=concatenate(targettrain,('seq','valid','metadata'),'cuda')
    density_done=read(entry.parent/(shift+'_density')/'complete.json');dp=entry.parent/(shift+'_density')/'density.pt';assert sha(dp)==density_done['artifact_sha256'];importance=torch.load(dp,map_location='cpu')['weights'].cuda()
    cost=read(DIAG/shift/'fit_costs_and_pairs.json');pairs=cost['pairing']['pairs'];table=torch.tensor([[p['BG'],p['foreground']] for p in pairs],device='cuda');pw=torch.tensor([p['BG_benefit']+p['FG_harm'] for p in pairs],device='cuda')
    feature_scale=data['seq'][:,4:20].reshape(-1,512).std(0,unbiased=False).clamp_min(1e-5);kernel(entry,data,target,table,pw,importance,feature_scale)
    seed();model=UDAHead().cuda();teacher=deepcopy(model).eval();teacher.requires_grad_(False);initial=base.tensor_digest(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
    order=torch.Generator().manual_seed(123);pr=torch.Generator().manual_seed(311105);stream=TargetStream(len(target['seq']));aug=torch.Generator(device='cuda').manual_seed(311109);assert aug.initial_seed()==311109;history=[];global_step=0
    for epoch in range(1,21):
        tick=time.monotonic();records=[];permutation=torch.randperm(len(data['bg']),generator=order);ramp=max(0,(epoch-5)/15)
        for ids in permutation.split(128):
            pi=torch.randperm(len(table),generator=pr)[:64].cuda();ti=stream.draw().cuda();noise=torch.randn(target['seq'][ti].shape,device='cuda',generator=aug)*feature_scale*.05
            records.append(step(model,teacher,opt,data,target,ids.cuda(),pi,table,pw,importance,mode,ramp,ti,noise,global_step));global_step+=1
        torch.cuda.synchronize();seconds=time.monotonic()-tick;history.append(dict(epoch=epoch,steps=len(records),seconds=seconds,ramp=ramp,**{k:float(np.mean([r[k] for r in records])) for k in records[0]}))
        base.checkpoint(entry/'latest.pt',capture(model,teacher,opt,order,pr,stream,aug,history,global_step,entry.parent,shift))
        if epoch in (1,5,10,20):os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
        save(entry/'history.json',history);print('UDA_EPOCH',entry.name,epoch,seconds,history[-1]['loss'],flush=True)
        if epoch==1:
            save(entry/'throughput.json',dict(first_epoch_seconds=seconds,projected_training_seconds=20*seconds,setup_seconds=time.monotonic()-began-seconds,limit=900))
            if seconds*20>900:save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False,checkpoint_sha256=sha(entry/'latest.pt')));return
    roles['target_val']=target_rows(entry.parent,shift,'target_val');teacher.eval();digest=base.tensor_digest(teacher.state_dict());cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    def readout(m):
        out={}
        with torch.no_grad():
            for role,rows in roles.items():
                out[role]=[]
                for r in rows:
                    probs=[]
                    for start in range(0,len(r['seq']),128):
                        sl=slice(start,start+128);probs.append(m(r['seq'][sl].cuda(),r['valid'][sl].cuda(),r['metadata'][sl].cuda())[:,0].sigmoid().cpu())
                    p=torch.zeros(len(r['scores']));p[r['hard']]=torch.cat(probs) if probs else torch.empty(0);out[role].append({k:r[k] for k in ('video','segments','scores','labels')}|dict(pbg=p))
        return out
    out=readout(teacher);base.checkpoint(entry/'predictions.pt',out)
    assert digest==base.tensor_digest(teacher.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    restored=UDAHead().cuda();restored.load_state_dict(torch.load(entry/'epoch020.pt',map_location='cpu')['ema']);restored.eval();again=readout(restored)
    for role in out:
        for a,b in zip(out[role],again[role]):assert torch.equal(a['pbg'],b['pbg'])
    save(entry/'gpu_complete.json',dict(passed=True,guard=guard,guard_sha256=sha(guard),initial_digest=initial,final_student_digest=base.tensor_digest(model.state_dict()),final_ema_digest=digest,
        kernel_sha256=sha(entry/'kernel_complete.json'),predictions_sha256=sha(entry/'predictions.pt'),checkpoint_sha256=sha(entry/'epoch020.pt'),parameters=sum(p.numel() for p in model.parameters()),
        all_saved_EMA_predictions_disk_GPU_bitwise_equal=True,readout_parameters_RNG_unchanged=True,target_GT_read=False,target_train_candidates=len(target['seq']),target_train_candidates_seen=int((stream.seen>0).sum()),seconds=time.monotonic()-began))


def evaluate(entry):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import ClassAP,prediction_frame,background_mask
    from tools.audit_background_ap_cost import IndependentAP,selected_indices
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    root=entry.parent;assert all((root/t/'gpu_complete.json').exists() for t in TASKS);shift,mode=entry.name.split('_');roles=torch.load(root/'inputs'/shift/'source.pt',map_location='cpu');roles['target_val']=target_rows(root,shift,'target_val')
    gp=read(entry/'gpu_complete.json');assert sha(entry/'predictions.pt')==gp['predictions_sha256'];outputs=torch.load(entry/'predictions.pt',map_location='cpu');gt=read(SOURCE/shift/'source_truth.json');cfg=read(SOURCE/shift/'config.json')
    truth={r:pd.DataFrame([(x['video'],float(b[0]),float(b[1]),int(c)) for x in roles[r] for b,c in zip(gt[x['video']]['segments'],gt[x['video']]['labels'])],columns=['video-id','t-start','t-end','label']) for r in ('fit','heldout')}
    truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split'])
    # 只有全部8头冻结后此CPU函数才读取targetval标注；targettrain标注从未打开。
    truth['target_val']=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split']);report={}
    for role,rows in outputs.items():
        for r,o in zip(rows,roles[role]):
            assert r['video']==o['video']
            for k in ('segments','scores','labels'):assert torch.equal(r[k],o[k])
            assert (r['pbg'][~o['hard']]==0).all()
        frame=prediction_frame(rows);bg=background_mask(truth[role],frame);positions={};engines={};refs={};baseline=[]
        for c in range(10):
            ids=np.flatnonzero(frame.label.to_numpy()==c);positions[c]=ids;engines[c]=ClassAP(truth[role][truth[role].label==c],frame.iloc[ids]);refs[c]=IndependentAP(truth[role][truth[role].label==c],frame.iloc[ids].reset_index(drop=True))
            ap=engines[c].evaluate(np.ones(len(ids),bool))['ap'];assert np.allclose(ap,refs[c].ap(np.ones(len(ids),bool)),atol=1e-7,rtol=0);baseline.append(ap)
        report[role]={}
        for method in (mode,'confidence'):
            auc=[];budgets=[];value=1-frame.score.to_numpy() if method=='confidence' else frame.pbg.to_numpy()
            for _,ids in frame.groupby('video-id',sort=False).indices.items():
                ids=np.asarray(ids);ids=ids[frame.iloc[ids].score.to_numpy()>=.3];a=independent_auc(bg[ids],value[ids])
                if a is not None:auc.append(a)
            for rate in SPEC['rates']:
                selected=selected_indices(frame,rate,method=='confidence');keep=np.ones(len(frame),bool);keep[selected]=False;ap=[engines[c].evaluate(keep[positions[c]])['ap'] for c in range(10)];ref=[refs[c].ap(keep[positions[c]]).tolist() for c in range(10)]
                assert np.allclose(ap,ref,atol=1e-7,rtol=0);budgets.append(dict(rate=rate,selected=selected.tolist(),delta_pp=float(100*(np.mean(ap)-np.mean(baseline))),per_class_AP=ap,BG_removed=int(bg[selected].sum()),nonBG_removed=int((~bg[selected]).sum())))
            report[role][method]=dict(baseline=float(np.mean(baseline)),original_per_class_AP=baseline,video_macro_AUC=float(np.mean(auc)) if auc else None,AUC_valid_videos=len(auc),budgets=budgets)
        print('EVALUATED',entry.name,role,flush=True)
    original=read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/shift/'evaluation'/'target_val.json')['map_0.1_to_0.5'];assert abs(report['target_val'][mode]['baseline']-original)<1e-7
    save(entry/'complete.json',dict(passed=True,UDA_training=True,target_train_GT_read=False,target_val_used_for_selection=False,gpu_complete_sha256=sha(entry/'gpu_complete.json'),results=report))


def run(root):
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity()));prepare_source(root);active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,stop)
    try:
        for stage,tasks in [('extract',EXTRACT),('density',('S1_density','S2_density')),('train',TASKS),('evaluate',TASKS)]:
            pending=list(tasks)
            while pending or active:
                for task,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[task]
                    if t['p'].returncode:raise RuntimeError(task+'/'+stage+' failed')
                    if (root/task/'budget_pause.json').exists():raise RuntimeError(task+' scheduling pause not scientific failure')
                    done.append(task+'/'+stage)
                devices=[None]*max(0,2-len(active)) if stage=='evaluate' else [g for g in (0,1,2,3) if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    task=pending.pop(0);entry=root/task;entry.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='123')
                    if g is not None:
                        observed=snapshot(g);lease=entry/('lease_'+stage+'.json');save(lease,dict(owner_pid=os.getpid(),index=g,uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']));env.update(CUDA_VISIBLE_DEVICES=observed['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(entry/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(entry),'--worker',stage],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),completed=done,pending=pending,active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()}));time.sleep(3)
        save(root/'final.json',dict(passed=True,spec=SPEC,complete_sha256={t:sha(root/t/'complete.json') for t in TASKS}));save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['extract','density','train','evaluate']);a=p.parse_args()
    if a.worker:globals()[a.worker](a.entry)
    else:run(a.root)
