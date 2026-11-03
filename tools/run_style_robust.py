"""四卡短预检：早层风格混合与参与者组稳健学习，不改原检测器或原候选。"""
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
from tools.style_robust_math import MODES
from tools.run_dense_revisit import SOURCE,seed,raw_load
DENSE=Path('/vepfs/vlms/highscore_dense_revisit_5ef4cca_1103')
TASKS=tuple(s+'_'+m for s in ('S1','S2') for m in MODES)
SPEC=dict(schema='task10-temporal-style-group-v1',shifts=['S1','S2'],modes=MODES,tasks=TASKS,
    source='same139FIT/full15100sourceannotations, same97tensor but onlytask0..9 occupancy objective; exact oldsource split and frozenrawinput',
    model='same DenseEvidence180771 nominalparams; styles only after raw2304->64 GELU and before temporalblock; same inference graph',
    sourcemix='p=.5,Beta(.1,.1),unbiasedtimevariance+1e-6; detach both style stats; partneruniform among sourceFIT videos of a different participant',
    targetmix='sameMixStyle but partnercycling sanitized targetTRAIN, no targetlabels or pseudoBG; targetprojection no_grad style stats only',
    groupdro='sourceparticipant from video prefix; perstep Nvideo/ngroupvideo*q_g*denseTaskLoss; adversarial q uniforminitial; exponential eta.01 update onceperepoch using ALLparticipantmean rawdense losses',
    targetdro='targetmix and groupdro combined, same primaryorder and augmentationdrawstream as othermodes',
    adaptation='MixStyle temporal lower-layer feature adaptation, GroupDRO epoch-block adversary not originalperminibatch; no wholepaper reproduction claim',
    objective='unchanged10task per-class balancedsoftoccupancy BCE; no newBGhead,qualityaux,pairweight or97classaux',
    epochs=20,checkpoints=[1,5,10,20],seed=311104,order_seed=123,target_seed=311108,style_seed=311110,
    lr=3e-4,weight_decay=.05,clip=1.,rates=[.005,.01,.02,.05],budget_seconds=900,
    control='reuse audited dense10 ERM, same initialization/order/inference, no duplicatereference training; realGPU disableloss/grad/AdamW exact',
    candidate_pool='originalpostNMSscore>=.3, originalscore/class/endpoints andlow-score replacements unchanged',
    inference='oldtaskreadout1-maxprob over16insidepoints/all10tasks; no augmentation,qweight or targetdata at inference',
    target_train_GT_read=False,target_val_used_for_selection=False,
    acceptance='primary5percent: mean targetAP gain>=1pp andbothS1/S2 positive, above samecountconfidence inboth; all4budgets/modes reported; no threshold/epochsearch; passonly permits budgetreview before newdirections',
    caveat='S1/S2 historicdevelopment, not untouchedtest; temporalmean/std also encode actiondistribution, style/GT invariance is hypothesis not proven',
    literature=['https://arxiv.org/abs/2104.02008','https://raw.githubusercontent.com/KaiyangZhou/mixstyle-release/master/reid/models/mixstyle.py','https://arxiv.org/abs/1911.08731','https://raw.githubusercontent.com/kohpangwei/group_DRO/master/loss.py'])


def groups_for(fit):
    groups=sorted({r['video'].split('_')[0] for r in fit});index={p:i for i,p in enumerate(groups)}
    ids=np.array([index[r['video'].split('_')[0]] for r in fit]);counts=np.bincount(ids,minlength=len(groups));return groups,ids,counts


def train(entry):
    import torch
    from tools import baseline_engine as base
    from tools.dense_revisit_math import DenseEvidence,dense_loss,sample_sequence
    from tools.style_robust_math import StyledEvidence,StyleDraws,group_update
    from tools.run_uda_background import TargetStream
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);seed();s,mode=entry.name.split('_');began=time.monotonic();source=DENSE/'inputs'/s/'data.pt'
    assert sha(DENSE/'independent_final_audit.json')=='3cf9fcc7eb3f06e2bc557af745816847bbde67f2725ab5e2820a86a0664f0141'
    assert sha(source)==read(DENSE/'prepare_complete.json')['results'][s]['data_sha256'];data=torch.load(source,map_location='cpu');roles=data['roles'];fit=roles['fit'];target=roles['target_train'];mean=data['mean'];std=data['std']
    groups,groupids,counts=groups_for(fit);G=len(groups);robust=mode in ('groupdro','targetdro');uses_target=mode in ('targetmix','targetdro');cache={}
    for r in fit+(target if uses_target else []):cache[r['video']]=((raw_load(r)-mean)/std).pin_memory()
    needed=sum(x.numel()*x.element_size() for x in cache.values());resident=needed+4096*2**20<torch.cuda.mem_get_info()[0]
    if resident:cache={k:v.cuda() for k,v in cache.items()}
    def get(r):return cache[r['video']].to('cuda',non_blocking=True)
    def fresh():
        m=StyledEvidence().cuda();o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.05);return m,o
    def capture(m,o,order,stream,draws,q,seen,history,step):
        return dict(model=deepcopy(m.state_dict()),optimizer=deepcopy(o.state_dict()),order_rng=order.get_state(),target_order=stream.state(),style_rng=draws.state(),adv_probs=q.cpu().clone(),effective_target_seen=seen.clone(),
            groups=groups,group_video_counts=counts.tolist(),group_ids=groupids.tolist(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),
            history=deepcopy(history),step=step,spec=SPEC,inputs_sha256=sha(source),old_dense_GPU_sha256=sha(DENSE/(s+'_dense10')/'gpu_complete.json'))
    def restore(st,m,o,order,stream,draws):
        m.load_state_dict(st['model']);o.load_state_dict(st['optimizer']);order.set_state(st['order_rng']);stream.restore(st['target_order']);draws.restore(st['style_rng']);torch.set_rng_state(st['cpu_rng']);torch.cuda.set_rng_state(st['cuda_rng'],0);random.setstate(st['python_rng']);np.random.set_state(st['numpy_rng']);return st['adv_probs'].cuda().clone(),st['effective_target_seen'].clone()
    def event(i,stream,draws):
        e=draws.draw();ti=int(stream.draw(1)[0]);candidates=np.flatnonzero(groupids!=groupids[i]);si=int(candidates[min(len(candidates)-1,int(e['source_fraction']*len(candidates)))]);return e,si,ti
    def update(m,o,i,q,e,si,ti,seen,disabled=False,cpu_transfer=False):
        r=fit[i];x=((raw_load(r)-mean)/std).cuda() if cpu_transfer else get(r);partner=None
        if not disabled and mode!='groupdro' and e['apply']:
            partner=get(target[ti] if uses_target else fit[si])
            if uses_target:seen[ti]+=1
        o.zero_grad(set_to_none=True);logits,_=m(x,partner,e['weight']);ordinary=dense_loss(logits[:,:10],r['dense'][:,:10].cuda());weight=q[int(groupids[i])]*len(fit)/int(counts[groupids[i]]) if robust and not disabled else x.new_tensor(1.)
        loss=ordinary*weight;assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);o.step()
        return dict(loss=float(loss.detach()),ordinary=float(ordinary.detach()),weight=float(weight),grad=float(norm),mixed=int(partner is not None),participant=int(groupids[i]))
    # 真实禁用风格和DRO后，与旧dense10完整loss/全部grad/AdamW逐tensor等价。
    seed();old=DenseEvidence().cuda();seed();m,o=fresh();oo=torch.optim.AdamW(old.parameters(),lr=3e-4,weight_decay=.05);x=get(fit[0]);oldloss=dense_loss(old(x)[0][:,:10],fit[0]['dense'][:,:10].cuda());oldloss.backward();torch.nn.utils.clip_grad_norm_(old.parameters(),1.);oo.step()
    order=torch.Generator().manual_seed(123);stream=TargetStream(len(target));draws=StyleDraws();q=torch.ones(G,device='cuda')/G;seen=torch.zeros(len(target),dtype=torch.long);forced=dict(apply=True,weight=.3,source_fraction=.2);si=int(np.flatnonzero(groupids!=groupids[0])[0]);update(m,o,0,q,forced,si,0,seen,disabled=True)
    assert base.tensor_digest(old.state_dict())==base.tensor_digest(m.state_dict()) and base.tensor_digest(oo.state_dict())==base.tensor_digest(o.state_dict())
    for (n,a),(nn,b) in zip(old.named_parameters(),m.named_parameters()):
        assert n==nn and ((a.grad is None and b.grad is None) or torch.equal(a.grad,b.grad))
    # 各干预真实更新与完整状态磁盘重建；风格/目标游标/组对手都在协议内。
    update(m,o,0,q,forced,si,0,seen);q=group_update(q,torch.arange(G,device='cuda').float()/10) if robust else q
    base.checkpoint(entry/'kernel.pt',capture(m,o,order,stream,draws,q,seen,[],2));st=torch.load(entry/'kernel.pt',map_location='cpu');other,otheropt=fresh()
    def draw_all():return torch.randperm(len(fit),generator=order),event(0,stream,draws),torch.rand(4),torch.rand(4,device='cuda')
    q,seen=restore(st,m,o,order,stream,draws);a=draw_all();q2,seen2=restore(st,other,otheropt,order,stream,draws);b=draw_all();assert torch.equal(a[0],b[0]) and a[1]==b[1] and torch.equal(a[2],b[2]) and torch.equal(a[3],b[3])
    q,seen=restore(st,m,o,order,stream,draws);e,si,ti=event(0,stream,draws);record=update(m,o,0,q,e,si,ti,seen);digest=base.tensor_digest(m.state_dict());od=base.tensor_digest(o.state_dict())
    q2,seen2=restore(st,other,otheropt,order,stream,draws);e2,s2,t2=event(0,stream,draws);record2=update(other,otheropt,0,q2,e2,s2,t2,seen2)
    assert digest==base.tensor_digest(other.state_dict()) and od==base.tensor_digest(otheropt.state_dict()) and record==record2 and torch.equal(seen,seen2)
    q,seen=restore(st,m,o,order,stream,draws);update(m,o,0,q,forced,si,0,seen);d=base.tensor_digest(m.state_dict());q2,seen2=restore(st,other,otheropt,order,stream,draws);update(other,otheropt,0,q2,forced,si,0,seen2,cpu_transfer=True);assert d==base.tensor_digest(other.state_dict())
    save(entry/'kernel_complete.json',dict(passed=True,disabled_old_loss_allgrad_AdamW_exact=True,real_style_or_DRO_update=True,complete_model_optimizer_style_RNG_adversary_nextstep_exact=True,target_cursor_draws_exact=True,CPUtransfer_vs_resident_update_exact=True,kernel_sha256=sha(entry/'kernel.pt'),guard_sha256=sha(guard)))
    del old,oo,m,o,other,otheropt,st,x;torch.cuda.empty_cache();seed();m,o=fresh();order=torch.Generator().manual_seed(123);stream=TargetStream(len(target));draws=StyleDraws();q=torch.ones(G,device='cuda')/G;seen=torch.zeros(len(target),dtype=torch.long);history=[];step=0
    initial=base.tensor_digest(m.state_dict());assert initial==read(DENSE/(s+'_dense10')/'gpu_complete.json')['initial_digest']
    for epoch in range(1,21):
        tick=time.monotonic();records=[];before=q.cpu().tolist()
        for i in torch.randperm(len(fit),generator=order).tolist():
            e,si,ti=event(i,stream,draws);records.append(update(m,o,i,q,e,si,ti,seen));step+=1
        losses=[float(np.mean([a['ordinary'] for a in records if a['participant']==g])) for g in range(G)]
        if robust:q=group_update(q,torch.tensor(losses,device='cuda'))
        torch.cuda.synchronize();seconds=time.monotonic()-tick;history.append(dict(epoch=epoch,steps=len(records),seconds=seconds,q_before=before,q_after=q.cpu().tolist(),group_losses=losses,mixed_videos=sum(a['mixed'] for a in records),
            loss=float(np.mean([a['loss'] for a in records])),ordinary=float(np.mean([a['ordinary'] for a in records]))))
        base.checkpoint(entry/'latest.pt',capture(m,o,order,stream,draws,q,seen,history,step))
        if epoch in (1,5,10,20):os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
        save(entry/'history.json',history);print('STYLE_EPOCH',entry.name,epoch,seconds,history[-1]['loss'],flush=True)
        if epoch==1:
            save(entry/'throughput.json',dict(first_epoch_seconds=seconds,projected_training_seconds=seconds*20,setup_seconds=time.monotonic()-began-seconds,normalized_input_bytes=needed,GPU_resident=resident,limit=900))
            if seconds*20>900:save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False,checkpoint_sha256=sha(entry/'latest.pt')));return
    m.eval();digest=base.tensor_digest(m.state_dict());cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    def readout(model):
        out={}
        with torch.no_grad():
            for role in ('fit','heldout','validation','target_val'):
                out[role]=[]
                for r in roles[role]:
                    raw=get(r) if r['video'] in cache else ((raw_load(r)-mean)/std).cuda();logits,z=model(raw);h=r['hard'];boxes=r['grid_boxes'][h].cuda();bg,_,_=model.candidate(logits,z,boxes,r['labels'][h].cuda(),r['scores'][h].cuda());p=torch.zeros(len(h));p[h]=bg.cpu();_,valid=sample_sequence(z,boxes);eligible=torch.zeros(len(h),dtype=torch.bool);eligible[h]=(valid.sum(1)>=4).cpu()
                    out[role].append({k:r[k] for k in ('video','segments','scores','labels')}|dict(pbg=p,eligible=eligible))
        return out
    outputs=readout(m);base.checkpoint(entry/'predictions.pt',outputs);assert digest==base.tensor_digest(m.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    restored=StyledEvidence().cuda();restored.load_state_dict(torch.load(entry/'epoch020.pt',map_location='cpu')['model']);restored.eval();replayed=readout(restored)
    for role in outputs:
        for a,b in zip(outputs[role],replayed[role]):assert torch.equal(a['pbg'],b['pbg']) and torch.equal(a['eligible'],b['eligible'])
    save(entry/'gpu_complete.json',dict(passed=True,guard=guard,guard_sha256=sha(guard),kernel_sha256=sha(entry/'kernel_complete.json'),checkpoint_sha256=sha(entry/'epoch020.pt'),predictions_sha256=sha(entry/'predictions.pt'),initial_digest=initial,final_digest=digest,
        inference_readonly_RNG_unchanged=True,saved_model_GPU_readout_bitwise_equal=True,parameters=sum(p.numel() for p in m.parameters()),effective_target_videos=int((seen>0).sum()) if uses_target else 0,target_train_GT_read=False,groups=groups,seconds=time.monotonic()-began))


def evaluate(root,s):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import prediction_frame,background_mask,ClassAP
    from tools.audit_background_ap_cost import IndependentAP
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    from tools.run_dense_revisit import choose
    assert all((root/t/'gpu_complete.json').exists() for t in TASKS);roles=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu')['roles'];outputs={m:torch.load(root/(s+'_'+m)/'predictions.pt',map_location='cpu') for m in MODES};outputs['erm']=torch.load(DENSE/(s+'_dense10')/'predictions.pt',map_location='cpu')
    cfg=read(SOURCE/s/'config.json');gt=read(SOURCE/s/'source_truth.json');truth={role:pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c)) for r in roles[role] for b,c in zip(gt[r['video']]['segments'],gt[r['video']]['labels'])],columns=['video-id','t-start','t-end','label']) for role in ('fit','heldout')}
    truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);truth['target_val']=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split']);result={}
    for role in ('fit','heldout','validation','target_val'):
        frames={};eligible=None
        for m,output in outputs.items():
            for a,b in zip(output[role],roles[role]):
                assert a['video']==b['video']
                for k in ('segments','scores','labels'):assert torch.equal(a[k],b[k])
                assert (a['pbg'][~b['hard']]==0).all()
            frames[m]=prediction_frame(output[role]);e=torch.cat([r['eligible'] for r in output[role]]).numpy()
            if eligible is None:eligible=e
            else:assert np.array_equal(e,eligible)
        frame=frames['erm'];bg=background_mask(truth[role],frame);pos={};fast={};refs={};baseline=[]
        for c in range(10):
            ids=np.flatnonzero(frame.label.to_numpy()==c);pos[c]=ids;fast[c]=ClassAP(truth[role][truth[role].label==c],frame.iloc[ids]);refs[c]=IndependentAP(truth[role][truth[role].label==c],frame.iloc[ids].reset_index(drop=True));baseline.append(refs[c].ap(np.ones(len(ids),bool)))
        def effect(chosen):
            keep=np.ones(len(frame),bool);keep[chosen]=False;a=[refs[c].ap(keep[pos[c]]) for c in range(10)];b=[fast[c].evaluate(keep[pos[c]])['ap'] for c in range(10)];assert np.allclose(a,b,atol=1e-7,rtol=0);return float(100*(np.mean(a)-np.mean(baseline))),np.asarray(a).tolist()
        result[role]={}
        for m in (*outputs,'confidence'):
            f=frame if m=='confidence' else frames[m];score=1-f.score.to_numpy() if m=='confidence' else f.pbg.to_numpy();aucs=[];budgets=[]
            for _,ids in f.groupby('video-id',sort=False).indices.items():
                ids=np.asarray(ids);ids=ids[f.iloc[ids].score.to_numpy()>=.3];a=independent_auc(bg[ids],score[ids])
                if a is not None:aucs.append(a)
            for rate in SPEC['rates']:
                chosen=choose(f,eligible,rate,m=='confidence');delta,ap=effect(chosen);gain,_=effect(chosen[bg[chosen]]);harm,_=effect(chosen[~bg[chosen]])
                budgets.append(dict(rate=rate,selected=chosen.tolist(),delta_pp=delta,per_class_AP=ap,BG_only_delta_pp=gain,nonBG_only_delta_pp=harm,BG_removed=int(bg[chosen].sum()),nonBG_removed=int((~bg[chosen]).sum())))
            result[role][m]=dict(baseline=float(np.mean(baseline)),original_per_class_AP=np.asarray(baseline).tolist(),video_macro_AUC=float(np.mean(aucs)),AUC_valid_videos=len(aucs),budgets=budgets)
        old=read(DENSE/(s+'_evaluation.json'))['results'][role]['dense10'];assert abs(result[role]['erm']['baseline']-old['baseline'])<1e-7
        for a,b in zip(result[role]['erm']['budgets'],old['budgets']):assert abs(a['delta_pp']-b['delta_pp'])<1e-7 and a['selected']==b['selected']
        print('STYLE_EVALUATED',s,role,flush=True)
    save(root/(s+'_evaluation.json'),dict(passed=True,results=result,all_original_arrays_exact=True,all_AP_independent_numpy_equal=True,ERM_reproduced=True,target_train_GT_read=False,target_val_used_for_selection=False))


def run(root):
    from tools.baseline_gpu import snapshot
    from tools.style_shift_diagnosis import run as diagnose
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity(),old_dense_audit_sha256=sha(DENSE/'independent_final_audit.json')));diagnose(root);active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        for stage,tasks in [('train',TASKS),('evaluate',('S1','S2'))]:
            pending=list(tasks)
            while pending or active:
                for task,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[task]
                    if t['p'].returncode:raise RuntimeError(task+'/'+stage+' failed')
                    if (root/task/'budget_pause.json').exists():raise RuntimeError(task+' schedulingpause')
                    done.append(task+'/'+stage)
                devices=[None]*max(0,2-len(active)) if stage=='evaluate' else [g for g in range(4) if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    task=pending.pop(0);entry=root/task;entry.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='123')
                    if g is not None:
                        v=snapshot(g);lease=entry/'lease.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=v['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']));env.update(CUDA_VISIBLE_DEVICES=v['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(entry/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(entry),'--worker',stage],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()},pending=pending,completed=done));time.sleep(3)
        save(root/'final.json',dict(passed=True,allGPU_sha256={t:sha(root/t/'gpu_complete.json') for t in TASKS},evaluation_sha256={s:sha(root/(s+'_evaluation.json')) for s in ('S1','S2')}));save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['train','evaluate']);a=p.parse_args()
    if a.worker=='train':train(a.entry)
    elif a.worker=='evaluate':evaluate(a.entry.parent,a.entry.name)
    else:run(a.root)
