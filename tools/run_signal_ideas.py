"""四卡四个有依据的表示idea；两个方向、固定预算、复用已审配对基线。"""
import argparse
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
from tools.signal_ideas_math import MODES,make_model
from tools.prepare_signal_ideas import extract,load_roles,LAYER
from tools.run_layer_signals import seed,EXTRACT
from tools.run_highscore_pairs import DIAG,OLD,SOURCE,concatenate,update,evaluate
BASE=Path('/vepfs/vlms/highscore_matched_pairs_7df125c_1103')
TASKS=tuple(s+'_'+m for s in ('S1','S2') for m in MODES)
SPEC=dict(schema='highscore-four-signal-ideas-v1',source_only=True,target_read=False,shifts=['S1','S2'],modes=MODES,tasks=TASKS,
    ideas=dict(early='embedding24points replaces FPN0',centered='FPN0 minus unlabeled valid-video mean',dual='32embedding +32FPN projected channels with same total parameters',centerdual='dual with each stream subtracting its own video mean'),
    reference='reuse independently audited originalFPN0 AP-pairs head; verify input,initial weights,loss,allgrad,AdamWupdate equivalence; no duplicate reference training',
    pairs='unchanged frozen7011 sourceFIT AP-critical cross-video condition-matched pairs; original metadata shared within pairs',
    loss='balancedBG_BCE+.25quality+.25 APweighted pair softplus(.5-BGlogit+FGlogit)',
    epochs=20,checkpoints=[1,5,10,20],seed=311104,order_seed=123,pair_seed=311105,batch=128,pair_batch=64,
    lr=3e-4,weight_decay=.05,clip=1.,rates=[.005,.01,.02,.05],budget_seconds=900,
    precision='inherit nativefloat32/TF32; no AMP/precision change',
    caveat='sourcehead-only split, detector saw source_train; no target/fullsixdirection claims; centering also removes action-distribution information')


def state(model,opt,order,pair_rng,history,root,shift):
    import torch
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),order_rng=order.get_state(),pair_rng=pair_rng.get_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),history=history,spec=SPEC,input_sha256=read(DIAG/shift/'fit_costs_and_pairs.json')['inputs_sha256'],pairs_sha256=sha(DIAG/shift/'fit_costs_and_pairs.json'),
        sequence_complete_sha256={d:sha(root/(shift+'_'+d)/'complete.json') for d in ('source','validation')})


def kernel(entry,data,table,weights,reference):
    import torch
    from tools import baseline_engine as base
    from tools.highscore_pair_math import make_model as old_model
    shift,mode=entry.name.split('_');seed();a=old_model().cuda();seed();b=make_model('reference').cuda()
    expected=read(BASE/(shift+'_pairs')/'gpu_complete.json')['initial_digest'];assert base.tensor_digest(a.state_dict())==base.tensor_digest(b.state_dict())==expected
    # 真GPU原目标/全部梯度/clip/AdamW更新对照，不仅是CPU接线。
    oa=torch.optim.AdamW(a.parameters(),lr=3e-4,weight_decay=.05);ob=torch.optim.AdamW(b.parameters(),lr=3e-4,weight_decay=.05)
    ids=torch.arange(min(64,len(data['bg'])),device='cuda');pi=torch.arange(min(16,len(table)),device='cuda')
    la=update(a,oa,reference,ids,pi,table,weights,True);lb=update(b,ob,reference,ids,pi,table,weights,True)
    assert la==lb and all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters())) and base.tensor_digest(a.state_dict())==base.tensor_digest(b.state_dict()) and base.tensor_digest(oa.state_dict())==base.tensor_digest(ob.state_dict())
    seed();model=make_model(mode).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05);order=torch.Generator().manual_seed(123);pr=torch.Generator().manual_seed(311105)
    initial=base.tensor_digest(model.state_dict());loss,norm,_=update(model,opt,data,ids,pi,table,weights,True);assert norm>0 and base.tensor_digest(model.state_dict())!=initial
    base.checkpoint(entry/'kernel_state.pt',state(model,opt,order,pr,[],entry.parent,shift));saved=torch.load(entry/'kernel_state.pt',map_location='cpu');fresh=make_model(mode).cuda();fresh.load_state_dict(saved['model']);fo=torch.optim.AdamW(fresh.parameters());fo.load_state_dict(saved['optimizer'])
    def restore():
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],0);order.set_state(saved['order_rng']);pr.set_state(saved['pair_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    restore();draw=(torch.rand(3),torch.rand(3,device='cuda'),torch.randperm(13,generator=order),torch.randperm(17,generator=pr),random.random(),np.random.rand())
    restore();again=(torch.rand(3),torch.rand(3,device='cuda'),torch.randperm(13,generator=order),torch.randperm(17,generator=pr),random.random(),np.random.rand());assert all(torch.equal(draw[i],again[i]) for i in range(4)) and draw[4:]==again[4:]
    restore();update(model,opt,data,ids,pi,table,weights,True);restore();update(fresh,fo,data,ids,pi,table,weights,True)
    assert base.tensor_digest(model.state_dict())==base.tensor_digest(fresh.state_dict()) and base.tensor_digest(opt.state_dict())==base.tensor_digest(fo.state_dict())
    save(entry/'kernel_complete.json',dict(passed=True,original_reference_loss_grad_update_exact=True,reference_initial_digest=expected,real_update=True,full_state_nextstep_and_draws_exact=True,explicit_cuda_seed=311104,state_sha256=sha(entry/'kernel_state.pt')))


def train(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);shift,mode=entry.name.split('_');began=time.monotonic();roles=load_roles(entry.parent,shift,mode)
    data=concatenate(roles['fit'],('seq','valid','metadata','bg','quality'),'cuda');refroles=torch.load(DIAG/shift/'inputs.pt',map_location='cpu');reference=concatenate(refroles['fit'],('seq','valid','metadata','bg','quality'),'cuda')
    cost=read(DIAG/shift/'fit_costs_and_pairs.json');assert np.array_equal(data['bg'].cpu().numpy().astype(bool),cost['BG']);pairs=cost['pairing']['pairs']
    table=torch.tensor([[p['BG'],p['foreground']] for p in pairs],device='cuda');weights=torch.tensor([p['BG_benefit']+p['FG_harm'] for p in pairs],device='cuda');kernel(entry,data,table,weights,reference);del reference,refroles
    seed();model=make_model(mode).cuda();initial=base.tensor_digest(model.state_dict());assert initial==read(BASE/(shift+'_pairs')/'gpu_complete.json')['initial_digest']
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05);order=torch.Generator().manual_seed(123);pr=torch.Generator().manual_seed(311105);history=[]
    for epoch in range(1,21):
        tick=time.monotonic();losses=[];ordinary=[];rank=[];permutation=torch.randperm(len(data['bg']),generator=order)
        for ids in permutation.split(128):
            pi=torch.randperm(len(table),generator=pr)[:64].cuda();loss,norm,terms=update(model,opt,data,ids.cuda(),pi,table,weights,True);losses.append(loss);ordinary.append(terms['ordinary']);rank.append(terms['pair'])
        torch.cuda.synchronize();seconds=time.monotonic()-tick;history.append(dict(epoch=epoch,loss=float(np.mean(losses)),ordinary=float(np.mean(ordinary)),pair=float(np.mean(rank)),seconds=seconds,steps=len(losses)))
        base.checkpoint(entry/'latest.pt',state(model,opt,order,pr,history,entry.parent,shift))
        if epoch in (1,5,10,20):os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
        save(entry/'history.json',history);print('EPOCH',entry.name,epoch,seconds,flush=True)
        if epoch==1:
            save(entry/'throughput.json',dict(first_epoch_seconds=seconds,projected_training_seconds=20*seconds,setup_seconds=time.monotonic()-began-seconds,limit=900))
            if seconds*20>900:save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False,checkpoint_sha256=sha(entry/'latest.pt')));return
    def readout(m):
        result={}
        with torch.no_grad():
            for role,rows in roles.items():
                result[role]=[]
                for row in rows:
                    parts=[]
                    for start in range(0,len(row['seq']),128):
                        sl=slice(start,start+128);parts.append(m(row['seq'][sl].cuda(),row['valid'][sl].cuda(),row['metadata'][sl].cuda())[:,0].sigmoid().cpu())
                    p=torch.zeros(len(row['scores']));p[row['hard']]=torch.cat(parts) if parts else torch.empty(0);result[role].append({k:row[k] for k in ('video','segments','scores','labels')}|dict(pbg=p))
        return result
    model.eval();digest=base.tensor_digest(model.state_dict());cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0);out=readout(model)
    assert digest==base.tensor_digest(model.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0));base.checkpoint(entry/'predictions.pt',out)
    replay=make_model(mode).cuda();replay.load_state_dict(torch.load(entry/'epoch020.pt',map_location='cpu')['model']);replay.eval();again=readout(replay)
    for role in out:
        for a,b in zip(out[role],again[role]):assert torch.equal(a['pbg'],b['pbg'])
    save(entry/'gpu_complete.json',dict(passed=True,shift=shift,mode=mode,spec=SPEC,guard=guard,guard_sha256=sha(guard),kernel_sha256=sha(entry/'kernel_complete.json'),initial_digest=initial,final_digest=digest,
        parameters=sum(p.numel() for p in model.parameters()),input_sha256=cost['inputs_sha256'],pairs_sha256=sha(DIAG/shift/'fit_costs_and_pairs.json'),history_sha256=sha(entry/'history.json'),
        predictions_sha256=sha(entry/'predictions.pt'),checkpoint_sha256=sha(entry/'epoch020.pt'),all_saved_predictions_disk_GPU_bitwise_equal=True,readout_parameter_and_RNG_unchanged=True,seconds=time.monotonic()-began))


def run(root):
    from tools.baseline_gpu import snapshot
    assert sha(BASE/'independent_final_audit.json')=='b52cd9a57e2b74015277fff1a9cf1c5569dc9fae97315fdb979225b114ae2ed6'
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity(),reference_audit_sha256=sha(BASE/'independent_final_audit.json'),layer_audit_sha256=sha(LAYER/'independent_final_audit.json')));active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,stop)
    try:
        for stage,tasks in [('extract',EXTRACT),('train',TASKS),('evaluate',TASKS)]:
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
                    log=(entry/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(entry),'--worker',stage],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),completed=done,pending=pending,active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()}));time.sleep(3)
        save(root/'final.json',dict(passed=True,source_only=True,spec=SPEC,complete_sha256={t:sha(root/t/'complete.json') for t in TASKS}));save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['extract','train','evaluate']);a=p.parse_args()
    if a.worker:globals()[a.worker](a.entry)
    else:run(a.root)
