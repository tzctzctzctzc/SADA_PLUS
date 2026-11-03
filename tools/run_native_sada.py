"""从原Prototype基座重训：source-only、SADA动作+背景、SADA仅动作。"""
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
from tools import run_native_joint as native
ORIGIN=Path('/vepfs/vlms/highscore_native_full20_706d076_1103')
MODES=('control','sadafull','sadafg');TASKS=tuple(s+'_'+m for s in ('S1','S2') for m in MODES)
SPEC=dict(schema='native-SADA-background-fresh-v1',shifts=['S1','S2'],modes=MODES,epochs=20,checkpoints=[1,5,10,20],
    fresh='ALL6groups retrained from originalPrototype bestEMA nativebody and originalSADAbranch; NEW BGhead/AdamW/order/targetstream/RNG; no backgroundexperiment weights/optimizer/history loaded',
    source_objective='samejointcost .5balancedBG+.5sourceAPcostweightedBG+.25quality plus sourcecriticalFG originalinsidefeature MSE1',
    representation='actual originalpretrained SGP/neck ALL6FPN shared by source and target; single nativefeatures pass per source reused by BG and SADA',
    alignment='originalSADA perclass conditional discriminators, learnableclassembeddings, exact originalGRL and maskedclassBCE/normalization/level/domain/backgroundweights',
    modes_description=dict(control='noDA fresh20 sourcejointcost',sadafull='10task actions plus originalpseudo-no-class alignment',sadafg='sameSADA but no backgroundgroup in alignment'),
    target='all106/221 sanitized targetTRAIN rawvideos, cyclicvideo stream, one per sourceFITstep fromepoch6; no targetGT/pseudoBG from new head',
    no_class_scope='SADA no-class group is original detector pseudo labels below configured threshold; NOT groundtruth pureBG(anytaskIoU<.01)',
    source_assignment='originalSADA point_generator and label_points including center/range; sourceGT only0..9; target class via currentSGP and frozen originalclshead',
    inherited='perS cfg beta, plthreshold, weight_source/target/no_class/level, discriminator architecture/dropout/lr/clip inherited exactly; weight_adv/weight_downstream ratio multiplies alignment',
    schedule='E1 BGhead warmup; E2..20 nativebody+head; first5 noDA, E6..20 SADA active; no newcosine/EMA or epochselection',
    body_lr=3e-5,head_lr=3e-4,weight_decay=.05,clip=1.,seed=311104,order_seed=123,target_seed=311108,
    rates=[.005,.01,.02,.05],budget_seconds=1800,target_train_GT_read=False,target_val_used_for_selection=False,
    immutable='originalscores/classes/endpoints/NMS pool/low-score replacements untouched; newlearnedbranch only filters originalscore>=.3 candidates',
    acceptance='fixed5percent targetmean>=1pp, bothshiftspositive and eachabove samecountconfidence; allmodes/allbudgets report',
    caveat='S1/S2 historical development, not full6directions/untouchedtest/full50epochSADA reproduction; sourceheadholdout still seen by originaldetector',
    literature=['https://arxiv.org/html/2312.13377v3','https://github.com/davidpujol/SADA'])
SPEC['efficiency']='batch only perlevel maskcount CPU synchronization; sameclass order, discriminator Dropout calls, loss reductions, precision and RNG; realGPU fullstep legacy/native equivalence and measured kernel speed required before fresh formaltraining'


def train(entry):
    import torch
    from tools import baseline_engine as base
    from tools.native_joint_math import NativeBackground,cost_proposal_loss,critical_anchor
    from tools.native_sada_math import NativeSADA,sample_pyramid,conditional_loss,official_conditional_reference
    from tools.native_sada_legacy import conditional_loss_legacy
    from tools.run_foreground_calibration_oracle import load_detector
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.run_uda_background import TargetStream
    guard=first_cuda_guard(entry);s,mode=entry.name.split('_');uses_da=mode!='control';with_bg=mode!='sadafg';began=time.monotonic()
    datafile=native.DENSE/'inputs'/s/'data.pt';data=torch.load(datafile,map_location='cpu');roles=data['roles'];fit=roles['fit'];target=roles['target_train']
    assert sha(datafile)==read(native.DENSE/'prepare_complete.json')['results'][s]['data_sha256']
    allowed={'video','segments','scores','labels','hard','grid_boxes','video_metadata','raw_path','raw_size','raw_mtime_ns','raw_sha256','downsample'}
    assert all(set(r)<=allowed for r in target);assert not {r['video'] for r in target}&{r['video'] for r in roles['target_val']}
    cfg=read(native.SOURCE/s/'config.json');truth=read(native.SOURCE/s/'source_truth.json');teachers=torch.load(ORIGIN/s/'teacher.pt',map_location='cpu')['rows'];assert [r['video'] for r in teachers]==[r['video'] for r in fit]
    teacher=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in r.items()} for r in teachers]
    raw={r['video']:native.raw_load(r).pin_memory() for r in fit+(target if uses_da else [])}
    def get(r):return raw[r['video']].to('cuda',non_blocking=True)
    def build():
        detector=load_detector(native.SOURCE/s);detector.requires_grad_(False);native.seed();m=NativeSADA(detector).cuda();m.eval();m.disc_dom_conditional_adv.requires_grad_(uses_da);m.learnable_embeddings.requires_grad_(uses_da)
        o=torch.optim.AdamW([dict(params=list(m.backbone.parameters())+list(m.neck.parameters()),lr=3e-5),dict(params=m.head.parameters(),lr=3e-4)],weight_decay=.05)
        d=torch.optim.AdamW(m.da_parameters(),lr=cfg['opt']['disc_learning_rate'],weight_decay=.05)
        return m,o,d
    labels_cache={};assignment_records={}
    def update(m,o,d,i,epoch,ti=None,official=False,legacy=False):
        joint=epoch>1;active=uses_da and epoch>5;m.body_mode(joint);m.head.train();m.disc_dom_conditional_adv.train(active);m.cls_head.eval();o.zero_grad(set_to_none=True);d.zero_grad(set_to_none=True)
        r=fit[i];t=teacher[i]
        if joint:
            features,masks=m.features(get(r),r['video_metadata']);seq,valid=sample_pyramid(features,masks,m.strides,r['grid_boxes'][r['hard']].cuda())
        else:seq,valid=t['seq'],t['valid']
        assert torch.equal(valid,t['valid']);z=m.logits(seq,valid,t['metadata']);ordinary=cost_proposal_loss(z,t['bg'],t['quality'],t['delta']);anchor=critical_anchor(seq,t['seq'],valid,t['bg'],t['delta']);loss=ordinary+anchor;adv=loss.new_zeros(());details={}
        if active:
            tr=target[ti];tf,tm=m.features(get(tr),tr['video_metadata']);tl=m.pseudo_logits(tf,tm)
            if i not in labels_cache:
                gt=truth[r['video']];labels_cache[i]=m.source_point_labels(features,torch.tensor(gt['segments'],device='cuda',dtype=torch.float32).reshape(-1,2),torch.tensor(gt['labels'],device='cuda',dtype=torch.long))
                assignment_records[r['video']]=dict(points=[p.detach().cpu() for p in m.point_generator([f[None] for f in features])],masks=[v.cpu() for v in masks],labels=[v.cpu() for v in labels_cache[i]],center_sample=m.train_center_sample,radius=m.train_center_sample_radius,reg_distance=m.train_reg_distance)
            if official:adv=official_conditional_reference(m,features,masks,tf,tm,labels_cache[i],tl,with_bg)
            else:adv,details=(conditional_loss_legacy if legacy else conditional_loss)(m,features,masks,tf,tm,labels_cache[i],tl,with_bg)
            loss=loss+cfg['model']['loss_weights']['weight_adv']/cfg['model']['loss_weights']['weight_downstream']*adv
        assert torch.isfinite(loss);loss.backward()
        # 保留现有SADA先裁判别器权值、再全局梯度裁剪/两优化器step的次序。
        if active and cfg['model']['weight_clip_value']>0:
            for p in m.disc_dom_conditional_adv.parameters():p.data.clamp_(-cfg['model']['weight_clip_value'],cfg['model']['weight_clip_value'])
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm)
        if active:d.step()
        o.step()
        return dict(loss=float(loss.detach()),ordinary=float(ordinary.detach()),anchor=float(anchor.detach()),adv=float(adv.detach()),grad=float(norm),**details)
    def capture(m,o,d,order,stream,history,epoch,step):
        return dict(model=deepcopy(m.state_dict()),optimizer=deepcopy(o.state_dict()),disc_optimizer=deepcopy(d.state_dict()),order_rng=order.get_state(),target_order=stream.state(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),history=deepcopy(history),epoch=epoch,step=step,spec=SPEC,
            input_sha256=sha(datafile),source_teacher_sha256=sha(ORIGIN/s/'teacher.pt'),original_native_checkpoint_sha256=read(ORIGIN/s/'complete.json')['native_checkpoint_sha256'])
    def restore(st,m,o,d,order,stream):
        m.load_state_dict(st['model']);o.load_state_dict(st['optimizer']);d.load_state_dict(st['disc_optimizer']);order.set_state(st['order_rng']);stream.restore(st['target_order']);random.setstate(st['python_rng']);np.random.set_state(st['numpy_rng']);torch.set_rng_state(st['cpu_rng']);torch.cuda.set_rng_state(st['cuda_rng'],0)
    m,o,d=build();initial=deepcopy(m.state_dict());initial_core=base.tensor_digest(m.core_state());order=torch.Generator().manual_seed(123);stream=TargetStream(len(target))
    # 关闭DA后，实际单次共享FPN的源loss/allgrad/AdamW必须等价原nativejointcost。
    detector=load_detector(native.SOURCE/s);detector.requires_grad_(False);native.seed();old=NativeBackground(detector).cuda();old.load_state_dict(m.core_state())
    oo=torch.optim.AdamW([dict(params=list(old.backbone.parameters())+list(old.neck.parameters()),lr=3e-5),dict(params=old.head.parameters(),lr=3e-4)],weight_decay=.05)
    old.body_mode(True);old.head.train();oo.zero_grad(set_to_none=True);r=fit[0];t=teacher[0];seq,v,_=old.sample(get(r),r['video_metadata'],r['grid_boxes'][r['hard']].cuda());zz=old.logits(seq,v,t['metadata']);oldloss=cost_proposal_loss(zz,t['bg'],t['quality'],t['delta'])+critical_anchor(seq,t['seq'],v,t['bg'],t['delta']);oldloss.backward();torch.nn.utils.clip_grad_norm_(old.parameters(),1.);oo.step();a=update(m,o,d,0,2)
    assert oldloss.item()==a['loss'] and base.tensor_digest(old.state_dict())==base.tensor_digest(m.core_state()) and base.tensor_digest(oo.state_dict())==base.tensor_digest(o.state_dict())
    for name,p in old.named_parameters():
        q=dict(m.named_parameters())[name];assert (p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad)
    del old,oo,detector,seq,zz,oldloss;torch.cuda.empty_cache()
    m.load_state_dict(initial);o=torch.optim.AdamW([dict(params=list(m.backbone.parameters())+list(m.neck.parameters()),lr=3e-5),dict(params=m.head.parameters(),lr=3e-4)],weight_decay=.05);d=torch.optim.AdamW(m.da_parameters(),lr=cfg['opt']['disc_learning_rate'],weight_decay=.05)
    if uses_da:
        # GPU参照保留原SADA条件分支AST；相同CUDA dropout随机流比较全loss、全部梯度和双AdamW。
        state=capture(m,o,d,order,stream,[],0,0);a=update(m,o,d,0,6,0);digest=base.tensor_digest(m.state_dict());od=base.tensor_digest(o.state_dict());dd=base.tensor_digest(d.state_dict());grads={n:p.grad.clone() if p.grad is not None else None for n,p in m.named_parameters()};after_cpu=torch.get_rng_state();after_cuda=torch.cuda.get_rng_state(0)
        ref,ro,rd=build();restore(state,ref,ro,rd,order,stream);b=update(ref,ro,rd,0,6,0,True)
        assert a['loss']==b['loss'] and digest==base.tensor_digest(ref.state_dict()) and od==base.tensor_digest(ro.state_dict()) and dd==base.tensor_digest(rd.state_dict())
        for n,p in ref.named_parameters():assert (p.grad is None and grads[n] is None) or torch.equal(p.grad,grads[n])
        restore(state,ref,ro,rd,order,stream);c=update(ref,ro,rd,0,6,0,legacy=True)
        assert a==c and digest==base.tensor_digest(ref.state_dict()) and od==base.tensor_digest(ro.state_dict()) and dd==base.tensor_digest(rd.state_dict()) and torch.equal(after_cpu,torch.get_rng_state()) and torch.equal(after_cuda,torch.cuda.get_rng_state(0))
        for n,p in ref.named_parameters():assert (p.grad is None and grads[n] is None) or torch.equal(p.grad,grads[n])
        assert base.tensor_digest(m.backbone.state_dict())!=base.tensor_digest({k[len('backbone.'):]:v for k,v in initial.items() if k.startswith('backbone.')})
        del ref,ro,rd,state,grads;torch.cuda.empty_cache()
        with torch.no_grad():
            sf,sm=m.features(get(fit[0]),fit[0]['video_metadata']);tf,tm=m.features(get(target[0]),target[0]['video_metadata']);tl=m.pseudo_logits(tf,tm)
        sf=[x.detach().requires_grad_() for x in sf];tf=[x.detach().requires_grad_() for x in tf];old_times=[];new_times=[]
        for _ in range(10):
            cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(0)
            for fn,times in ((conditional_loss_legacy,old_times),(conditional_loss,new_times)):
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng,0);m.zero_grad(set_to_none=True)
                for x in sf+tf:x.grad=None
                torch.cuda.synchronize();tick=time.monotonic();value,_=fn(m,sf,sm,tf,tm,labels_cache[0],tl,with_bg);value.backward();torch.cuda.synchronize();times.append(time.monotonic()-tick)
        save(entry/'efficiency_benchmark.json',dict(passed_numeric=True,full_model_dualopt_grad_and_RNG_exact=True,legacy_seconds=old_times,optimized_seconds=new_times,speedup=sum(old_times)/sum(new_times),scope='conditional loss forward-backward on real fullFPN inputs; not endtoendtraining speedup'))
        assert sum(old_times)>sum(new_times),'no measured synchronization optimization gain'
        del sf,sm,tf,tm,tl,value;torch.cuda.empty_cache()
    ti=int(stream.draw(1)[0]) if uses_da else None;update(m,o,d,0,6 if uses_da else 2,ti);base.checkpoint(entry/'kernel.pt',capture(m,o,d,order,stream,[],0,1));st=torch.load(entry/'kernel.pt',map_location='cpu')
    def draws():return torch.randperm(len(fit),generator=order),stream.draw(1),torch.rand(3),torch.rand(3,device='cuda')
    restore(st,m,o,d,order,stream);first=draws();restore(st,m,o,d,order,stream);again=draws();assert all(torch.equal(a,b) for a,b in zip(first,again))
    restore(st,m,o,d,order,stream);ti=int(stream.draw(1)[0]) if uses_da else None;a=update(m,o,d,0,6 if uses_da else 2,ti);md=base.tensor_digest(m.state_dict());od=base.tensor_digest(o.state_dict());dd=base.tensor_digest(d.state_dict())
    fresh,fo,fd=build();restore(st,fresh,fo,fd,order,stream);ti=int(stream.draw(1)[0]) if uses_da else None;b=update(fresh,fo,fd,0,6 if uses_da else 2,ti)
    assert a==b and md==base.tensor_digest(fresh.state_dict()) and od==base.tensor_digest(fo.state_dict()) and dd==base.tensor_digest(fd.state_dict())
    save(entry/'kernel_complete.json',dict(passed=True,disabled_DA_native_loss_allgrad_AdamW_exact=True,original_SADA_conditional_loss_allgrad_dualAdamW_exact=uses_da,real_SGP_head_discriminator_updates=uses_da,complete_model_twoopts_order_target_PythonNumPyCPUCUDA_RNG_reconstruction_nextstep_exact=True,kernel_sha256=sha(entry/'kernel.pt'),guard_sha256=sha(guard)))
    del fresh,fo,fd,m,o,d,st,initial;torch.cuda.empty_cache()
    # 明确全新初始化正式实验，kernel状态、旧背景训练状态均不转移。
    labels_cache.clear();assignment_records.clear();m,o,d=build();assert base.tensor_digest(m.core_state())==initial_core;order=torch.Generator().manual_seed(123);stream=TargetStream(len(target));history=[];step=0;formal=time.monotonic();da_initial=base.tensor_digest({k:v for k,v in m.state_dict().items() if k.startswith('disc_dom_conditional_adv.') or k=='learnable_embeddings'})
    for epoch in range(1,21):
        tick=time.monotonic();records=[]
        for i in torch.randperm(len(fit),generator=order).tolist():
            ti=int(stream.draw(1)[0]) if uses_da and epoch>5 else None;records.append(update(m,o,d,i,epoch,ti));step+=1
        torch.cuda.synchronize();seconds=time.monotonic()-tick;row=dict(epoch=epoch,steps=len(records),seconds=seconds,joint_backbone=epoch>1,alignment_active=uses_da and epoch>5,**{k:float(np.mean([a[k] for a in records])) for k in ('loss','ordinary','anchor','adv','grad')})
        if row['alignment_active']:row.update(class_counts=np.sum([a['class_counts'] for a in records],axis=0).tolist(),domain_group_accuracy=float(np.mean([a['domain_group_accuracy'] for a in records])))
        history.append(row);base.checkpoint(entry/'latest.pt',capture(m,o,d,order,stream,history,epoch,step))
        if epoch in (1,5,10,20):os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
        if epoch==5:
            prior=torch.load(ORIGIN/(s+'_jointcost')/'epoch005.pt',map_location='cpu');assert base.tensor_digest(m.core_state())==base.tensor_digest(prior['model']) and base.tensor_digest(o.state_dict())==base.tensor_digest(prior['optimizer'])
            save(entry/'fresh_epoch5_equivalence.json',dict(passed=True,fresh_not_resumed=True,all_source_core_parameters_and_optimizer_bitwise_equal=True,prior_sha256=sha(ORIGIN/(s+'_jointcost')/'epoch005.pt')))
        save(entry/'history.json',history);print('NATIVE_SADA_EPOCH',entry.name,epoch,row,flush=True)
        if epoch in (1,2,6):
            estimate=seconds*(20-epoch);save(entry/'throughput.json',dict(epoch=epoch,epoch_seconds=seconds,elapsed_formal=time.monotonic()-formal,setup_seconds=formal-began,projected_remaining_seconds=estimate,limit=1800))
            if time.monotonic()-began+estimate>1800:save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False,checkpoint_sha256=sha(entry/'latest.pt')));return
    base.checkpoint(entry/'source_point_labels.pt',assignment_records)
    if mode=='control':
        prior=torch.load(ORIGIN/(s+'_jointcost')/'epoch020.pt',map_location='cpu');assert base.tensor_digest(m.core_state())==base.tensor_digest(prior['model']) and base.tensor_digest(o.state_dict())==base.tensor_digest(prior['optimizer'])
    m.eval();m.body_mode(False);digest=base.tensor_digest(m.state_dict());cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    def readout():
        out={}
        with torch.no_grad():
            for role in ('fit','heldout','validation','target_val'):
                out[role]=[]
                for r in roles[role]:
                    h=r['hard'];seq,v,length=m.sample(native.raw_load(r).cuda(),r['video_metadata'],r['grid_boxes'][h].cuda());meta=torch.cat((torch.nn.functional.one_hot(r['labels'][h].long(),10).float(),torch.logit(r['scores'][h].clamp(1e-5,1-1e-5))[:,None]/5,torch.log1p((r['grid_boxes'][h,1]-r['grid_boxes'][h,0])/max(1,length))[:,None]),1)
                    z=m.logits(seq,v,meta.cuda());p=torch.zeros(len(h));p[h]=z[:,:,0].sigmoid().mean(0).cpu();eligible=torch.zeros(len(h),dtype=torch.bool);eligible[h]=(v[:,:,4:20].sum(-1).min(0).values>=4).cpu();out[role].append({k:r[k] for k in ('video','segments','scores','labels')}|dict(pbg=p,eligible=eligible))
        return out
    outputs=readout();base.checkpoint(entry/'predictions.pt',outputs);assert digest==base.tensor_digest(m.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0));m.load_state_dict(torch.load(entry/'epoch020.pt',map_location='cpu')['model']);again=readout()
    if mode=='control':old=torch.load(ORIGIN/(s+'_jointcost')/'predictions.pt',map_location='cpu')
    for role in outputs:
        for i,(a,b) in enumerate(zip(outputs[role],again[role])):
            assert torch.equal(a['pbg'],b['pbg']) and torch.equal(a['eligible'],b['eligible'])
            if mode=='control':assert torch.equal(a['pbg'],old[role][i]['pbg']) and torch.equal(a['eligible'],old[role][i]['eligible'])
    save(entry/'gpu_complete.json',dict(passed=True,guard=guard,guard_sha256=sha(guard),kernel_sha256=sha(entry/'kernel_complete.json'),checkpoint_sha256=sha(entry/'epoch020.pt'),predictions_sha256=sha(entry/'predictions.pt'),fresh_epoch5_equivalence_sha256=sha(entry/'fresh_epoch5_equivalence.json'),
        initial_core_digest=initial_core,final_digest=digest,final_core_digest=base.tensor_digest(m.core_state()),source_point_labels_sha256=sha(entry/'source_point_labels.pt'),discriminator_or_embeddings_changed=da_initial!=base.tensor_digest({k:v for k,v in m.state_dict().items() if k.startswith('disc_dom_conditional_adv.') or k=='learnable_embeddings'}),target_train_videos=len(target),target_seen=int((stream.seen>0).sum()),target_draws=int(stream.seen.sum()),
        saved_model_GPU_predictions_bitwise_equal=True,readout_parameters_RNG_unchanged=True,control_fresh20_exact_old20=(mode=='control'),no_background_checkpoint_or_history_loaded_for_training=True,target_train_GT_read=False,seconds=time.monotonic()-began,formal_seconds=time.monotonic()-formal))


def run(root):
    from tools.baseline_gpu import snapshot
    assert read(ORIGIN/'independent_final_audit.json')['passed'];assert not Path('/proc',str(read(ORIGIN/'state.json')['owner_pid'])).exists();root.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(spec=SPEC,code=code_identity(),origin_audit_sha256=sha(ORIGIN/'independent_final_audit.json'),dense_input_audit_sha256=sha(native.DENSE/'independent_input_audit.json'),source_model_core_unchanged=True))
    active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    try:
        order=tuple(s+'_'+m for m in ('sadafull','sadafg','control') for s in ('S1','S2'))
        for stage,tasks in [('train',order),('evaluate',('S1','S2'))]:
            pending=list(tasks)
            while pending or active:
                for task,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[task]
                    if t['p'].returncode:raise RuntimeError(task+'/'+stage+' failed')
                    if (root/task/'budget_pause.json').exists():raise RuntimeError(task+' budgetpause')
                    done.append(task+'/'+stage)
                free=[snapshot(g) for g in range(4) if g not in [t['gpu'] for t in active.values()]] if stage=='train' else []
                devices=[None]*max(0,2-len(active)) if stage=='evaluate' else [v['index'] for v in sorted(free,key=lambda v:(v['utilization'],v['index'])) if v['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    task=pending.pop(0);e=root/task;e.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='123')
                    if g is not None:
                        v=snapshot(g);lease=e/'lease_train.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=v['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']));env.update(CUDA_VISIBLE_DEVICES=v['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(e/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(e),'--worker',stage],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()},pending=pending,completed=done));time.sleep(3)
        save(root/'final.json',dict(passed=True,allGPU_sha256={t:sha(root/t/'gpu_complete.json') for t in TASKS},evaluation_sha256={s:sha(root/(s+'_evaluation.json')) for s in ('S1','S2')}));save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex),automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['train','evaluate']);a=p.parse_args()
    if a.worker=='train':train(a.entry)
    elif a.worker=='evaluate':native.evaluate(a.entry.parent,a.entry.name,MODES=MODES,TASKS=TASKS,SPEC=SPEC)
    else:run(a.root)
