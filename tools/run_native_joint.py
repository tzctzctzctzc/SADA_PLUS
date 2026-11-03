"""配对短实验：冻结原生金字塔BG读出 versus 预训练SGP/neck联合更新+关键FG锚定。"""
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
from tools.run_dense_revisit import SOURCE,DIAG,seed,raw_load
DENSE=Path('/vepfs/vlms/highscore_dense_revisit_5ef4cca_1103')
MODES=('frozen','joint');TASKS=tuple(s+'_'+m for s in ('S1','S2') for m in MODES)
SPEC=dict(schema='native-pretrained-SGP-background-joint-v1',shifts=['S1','S2'],modes=MODES,tasks=TASKS,
    source='unchanged head-holdout sourceFIT40/99, cached original highscore proposals/BG/quality and fullFIT APcost; heldout/sourceval not used in optimizer',
    representation='originalPrototype pretrained sourceSGP and neck, ALL6nativeFPNlevels; same4/16/4 interpolation at boxes/stride, no newimagefeatures',
    head='same64dim ProposalEncoder, shared across6levels; BG/quality losses mean overlevels; candidate pBG mean(sigmoid BGlogits) acrosslevels',
    frozen='native teacherpyramid samples read-only cached once, only head updates; numeric loss/allgrad/update equivalent to directnative features',
    joint='epoch1headwarmup; epochs2..5 sourceSGP/neck and head jointly learn BG/quality; sourcecriticalFG originalinsidefeature MSE anchor coefficient1, harmfulnonBG costs normalized withinvideo',
    anchor='sourceFIT only negative actual fullFIT delete-one APmarginals on nonBG; no anchoring originalBG hallucinations; marginal weights not additiveAP',
    source_body_lr=3e-5,head_lr=3e-4,weight_decay=.05,clip=1.,epochs=5,warmup_epochs=1,checkpoints=[1,3,5],seed=311104,order_seed=123,
    training_mode='native pretrained backbone/neck/dropout remain eval-mode while gradients allowed; no stochasticdropout/EMA; head train-mode',
    precision='nativefloat32 inheritedTF32, noAMP/quantization or inputnormalization changes',
    rates=[.005,.01,.02,.05],budget_seconds=900,target_train_GT_read=False,target_train_used=False,target_val_used_for_selection=False,
    immutable='old detector files, originalscore/class/endpoints/all low-score replacements remain unchanged; learned model only filters oldpool',
    acceptance='primary5percent target two-shiftmean>=1pp andbothshiftspositive, eachabove samecountconfidence; reportallmodesandrates no epoch/thresholdsearch',
    caveat='paired short representation-learning precheck, not retrained originalA or fullsixdirections; sourceheadholdout and historical targetvaldevelopment')


def raw_video(raw,r):
    import torch
    return [dict(feats=raw.T,segments=raw.new_empty(0,2),labels=torch.empty(0,dtype=torch.long,device=raw.device),metadata=r['video_metadata'])]


def extract(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.run_foreground_calibration_oracle import load_detector,native_features,REFERENCE
    from tools.native_joint_math import NativeBackground
    guard=first_cuda_guard(entry);seed();s=entry.name;began=time.monotonic();data=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu');old=torch.load(DIAG/s/'inputs.pt',map_location='cpu');lookup={r['video']:r for r in old['fit']}
    expected=read(SOURCE/'plan.json')['jobs'][s]['checkpoint_sha256'];cp=REFERENCE/s/'best.pt';assert sha(cp)==expected
    detector=load_detector(SOURCE/s);initial=base.tensor_digest(detector.state_dict());assert initial==read(SOURCE/s/'cache_index.json')['detector_digest'];detector.requires_grad_(False);seed();model=NativeBackground(detector).cuda();rows=[];cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    with torch.no_grad():
        for i,r in enumerate(data['roles']['fit']):
            x=raw_load(r).cuda();h=r['hard'];assert h.any();boxes=r['grid_boxes'][h].cuda();seq,valid,_=model.sample(x,r['video_metadata'],boxes);prior=lookup[r['video']]
            assert torch.equal(seq[0].cpu(),prior['seq']) and torch.equal(valid[0].cpu(),prior['valid'])
            if i==0:
                f,v,result=native_features(detector,raw_video(x,r),True);assert torch.equal(result['scores'].cpu(),r['scores']) and torch.equal(result['labels'].cpu(),r['labels'])
                from tools.proposal_foreground_math import seconds_to_grid
                assert torch.equal(seconds_to_grid(result['segments'],r['video_metadata']).cpu(),r['grid_boxes'])
            rows.append(dict(video=r['video'],seq=seq.cpu(),valid=valid.cpu(),metadata=prior['metadata'],bg=prior['bg'],quality=prior['quality'],delta=r['delta']))
            print('NATIVE_TEACHER',s,i+1,len(data['roles']['fit']),flush=True)
            if i==2:
                projected=(time.monotonic()-began)*len(data['roles']['fit'])/3;save(entry/'throughput.json',dict(projected_extract_seconds=projected,limit=900))
                if projected>900:save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False));return
    assert initial==base.tensor_digest(detector.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    base.checkpoint(entry/'teacher.pt',dict(rows=rows,input_sha256=sha(DENSE/'inputs'/s/'data.pt'),native_checkpoint_sha256=expected))
    save(entry/'complete.json',dict(passed=True,guard=guard,guard_sha256=sha(guard),teacher_sha256=sha(entry/'teacher.pt'),native_checkpoint_sha256=expected,original_detector_digest=initial,
        all_original_FPN0_samples_bitwise_equal=True,first_original_prediction_arrays_bitwise_equal=True,parameters_RNG_unchanged=True,seconds=time.monotonic()-began))


def train(entry):
    import torch
    from tools import baseline_engine as base
    from tools.native_joint_math import NativeBackground,proposal_loss,critical_anchor
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.run_foreground_calibration_oracle import load_detector
    guard=first_cuda_guard(entry);seed();s,mode=entry.name.split('_');began=time.monotonic();data=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu');roles=data['roles'];fit=roles['fit'];done=read(entry.parent/s/'complete.json');assert sha(entry.parent/s/'teacher.pt')==done['teacher_sha256'];teacher=torch.load(entry.parent/s/'teacher.pt',map_location='cpu')['rows'];assert [r['video'] for r in fit]==[r['video'] for r in teacher]
    detector=load_detector(SOURCE/s);detector.requires_grad_(False);assert base.tensor_digest(detector.state_dict())==done['original_detector_digest'];seed();model=NativeBackground(detector).cuda();initial_body=deepcopy({k:v.cpu().clone() for k,v in model.state_dict().items() if not k.startswith('head.')});initial_head=deepcopy({k:v.cpu().clone() for k,v in model.state_dict().items() if k.startswith('head.')})
    raw=[raw_load(r).pin_memory() for r in fit];teacher_gpu=[{k:(v.cuda() if torch.is_tensor(v) else v) for k,v in r.items()} for r in teacher]
    def optimizer(m):return torch.optim.AdamW([dict(params=list(m.backbone.parameters())+list(m.neck.parameters()),lr=3e-5),dict(params=m.head.parameters(),lr=3e-4)],weight_decay=.05)
    def capture(m,o,order,history,epoch,step):
        return dict(model=deepcopy(m.state_dict()),optimizer=deepcopy(o.state_dict()),order_rng=order.get_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),
            history=deepcopy(history),epoch=epoch,step=step,spec=SPEC,teacher_sha256=done['teacher_sha256'],input_sha256=sha(DENSE/'inputs'/s/'data.pt'),native_checkpoint_sha256=done['native_checkpoint_sha256'])
    def restore(st,m,o,order):
        m.load_state_dict(st['model']);o.load_state_dict(st['optimizer']);order.set_state(st['order_rng']);random.setstate(st['python_rng']);np.random.set_state(st['numpy_rng']);torch.set_rng_state(st['cpu_rng']);torch.cuda.set_rng_state(st['cuda_rng'],0)
    def update(m,o,i,joint,direct_frozen=False):
        r=fit[i];t=teacher_gpu[i];m.body_mode(joint);m.head.train();o.zero_grad(set_to_none=True)
        if joint or direct_frozen:seq,valid,_=m.sample(raw[i].cuda(non_blocking=True),r['video_metadata'],r['grid_boxes'][r['hard']].cuda())
        else:seq,valid=t['seq'],t['valid']
        assert torch.equal(valid,t['valid']);logits=m.logits(seq,valid,t['metadata']);ordinary=proposal_loss(logits,t['bg'],t['quality']);anchor=critical_anchor(seq,t['seq'],valid,t['bg'],t['delta']);loss=ordinary+anchor
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);o.step()
        return dict(loss=float(loss.detach()),ordinary=float(ordinary.detach()),anchor=float(anchor.detach()),grad=float(norm))
    # 缓存和原生全金字塔的真实loss/allgrad/AdamW更新等价，不仅核接口。
    order=torch.Generator().manual_seed(123);o=optimizer(model);initial=deepcopy(model.state_dict());update(model,o,0,False);a=base.tensor_digest(model.head.state_dict());oa=base.tensor_digest(o.state_dict());ga={n:p.grad.clone() if p.grad is not None else None for n,p in model.named_parameters()};model.load_state_dict(initial);o=optimizer(model);update(model,o,0,False,True)
    assert a==base.tensor_digest(model.head.state_dict()) and oa==base.tensor_digest(o.state_dict())
    for n,p in model.named_parameters():assert (ga[n] is None and p.grad is None) or torch.equal(ga[n],p.grad)
    model.load_state_dict(initial);o=optimizer(model);body_before=base.tensor_digest(model.backbone.state_dict());update(model,o,0,mode=='joint');assert mode=='frozen' or body_before!=base.tensor_digest(model.backbone.state_dict())
    base.checkpoint(entry/'kernel.pt',capture(model,o,order,[],0,1));st=torch.load(entry/'kernel.pt',map_location='cpu');restore(st,model,o,order);rngdraw=(torch.randperm(len(fit),generator=order),torch.rand(3),torch.rand(3,device='cuda'));restore(st,model,o,order);again=(torch.randperm(len(fit),generator=order),torch.rand(3),torch.rand(3,device='cuda'));assert all(torch.equal(a,b) for a,b in zip(rngdraw,again))
    restore(st,model,o,order);record=update(model,o,0,mode=='joint');final=base.tensor_digest(model.state_dict());osd=base.tensor_digest(o.state_dict())
    rebuilt_detector=load_detector(SOURCE/s);rebuilt_detector.requires_grad_(False);seed();rebuilt=NativeBackground(rebuilt_detector).cuda();ro=optimizer(rebuilt);restore(st,rebuilt,ro,order);record2=update(rebuilt,ro,0,mode=='joint')
    assert record==record2 and final==base.tensor_digest(rebuilt.state_dict()) and osd==base.tensor_digest(ro.state_dict());del rebuilt,rebuilt_detector,ro;torch.cuda.empty_cache()
    save(entry/'kernel_complete.json',dict(passed=True,cached_vs_native_loss_allgrad_update_exact=True,real_joint_backbone_update=mode=='joint',complete_model_optimizer_RNG_nextstep_exact=True,guard_sha256=sha(guard),kernel_sha256=sha(entry/'kernel.pt')))
    # 接线状态不进入正式训练，原预训练主干和新头均精确回到同一初始状态。
    model.load_state_dict(initial);seed();o=optimizer(model);order=torch.Generator().manual_seed(123);history=[];step=0
    for epoch in range(1,6):
        tick=time.monotonic();joint=mode=='joint' and epoch>1;records=[]
        for i in torch.randperm(len(fit),generator=order).tolist():records.append(update(model,o,i,joint));step+=1
        torch.cuda.synchronize();seconds=time.monotonic()-tick;history.append(dict(epoch=epoch,joint_backbone=joint,steps=len(records),seconds=seconds,**{k:float(np.mean([a[k] for a in records])) for k in records[0]}));base.checkpoint(entry/'latest.pt',capture(model,o,order,history,epoch,step))
        if epoch in (1,3,5):os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
        save(entry/'history.json',history);print('NATIVE_EPOCH',entry.name,epoch,seconds,history[-1],flush=True)
        # joint的warmup耗时不代表主干反传；epoch2再用真实速度检查剩余预算。
        if epoch in (1,2):
            projected=seconds*(5-epoch);save(entry/'throughput.json',dict(epoch=epoch,seconds=seconds,projected_remaining_seconds=projected,elapsed=time.monotonic()-began,limit=900))
            if time.monotonic()-began+projected>900:save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False,checkpoint_sha256=sha(entry/'latest.pt')));return
    final_body={k:v.cpu() for k,v in model.state_dict().items() if not k.startswith('head.')};changed=base.tensor_digest(final_body)!=base.tensor_digest(initial_body);assert changed==(mode=='joint')
    model.eval();model.body_mode(False);digest=base.tensor_digest(model.state_dict());cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    def readout():
        output={}
        with torch.no_grad():
            for role in ('fit','heldout','validation','target_val'):
                output[role]=[]
                for r in roles[role]:
                    x=raw_load(r).cuda();h=r['hard'];seq,valid,length=model.sample(x,r['video_metadata'],r['grid_boxes'][h].cuda());meta=torch.cat((torch.nn.functional.one_hot(r['labels'][h].long(),10).float(),torch.logit(r['scores'][h].clamp(1e-5,1-1e-5))[:,None]/5,torch.log1p((r['grid_boxes'][h,1]-r['grid_boxes'][h,0])/max(1,length))[:,None]),1)
                    logits=model.logits(seq,valid,meta.cuda());p=torch.zeros(len(h));p[h]=logits[:,:,0].sigmoid().mean(0).cpu();eligible=torch.zeros(len(h),dtype=torch.bool);eligible[h]=(valid[:,:,4:20].sum(-1).min(0).values>=4).cpu()
                    output[role].append({k:r[k] for k in ('video','segments','scores','labels')}|dict(pbg=p,eligible=eligible))
        return output
    outputs=readout();base.checkpoint(entry/'predictions.pt',outputs);assert digest==base.tensor_digest(model.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0));model.load_state_dict(torch.load(entry/'epoch005.pt',map_location='cpu')['model']);again=readout()
    for role in outputs:
        for a,b in zip(outputs[role],again[role]):assert torch.equal(a['pbg'],b['pbg']) and torch.equal(a['eligible'],b['eligible'])
    save(entry/'gpu_complete.json',dict(passed=True,guard=guard,guard_sha256=sha(guard),kernel_sha256=sha(entry/'kernel_complete.json'),checkpoint_sha256=sha(entry/'epoch005.pt'),predictions_sha256=sha(entry/'predictions.pt'),teacher_sha256=done['teacher_sha256'],backbone_neck_changed=changed,
        parameters=sum(p.numel() for p in model.parameters()),readout_parameters_RNG_unchanged=True,saved_model_GPU_predictions_bitwise_equal=True,initial_digest=base.tensor_digest(initial),final_digest=digest,seconds=time.monotonic()-began,target_train_GT_read=False))


def evaluate(root,s):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import prediction_frame,background_mask,ClassAP
    from tools.audit_background_ap_cost import IndependentAP
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    from tools.run_dense_revisit import choose
    assert all((root/t/'gpu_complete.json').exists() for t in TASKS);roles=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu')['roles'];outputs={m:torch.load(root/(s+'_'+m)/'predictions.pt',map_location='cpu') for m in MODES}
    cfg=read(SOURCE/s/'config.json');gt=read(SOURCE/s/'source_truth.json');truth={role:pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c)) for r in roles[role] for b,c in zip(gt[r['video']]['segments'],gt[r['video']]['labels'])],columns=['video-id','t-start','t-end','label']) for role in ('fit','heldout')};truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);truth['target_val']=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split']);results={}
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
        f=frames['frozen'];bg=background_mask(truth[role],f);positions={};fast={};refs={};baseline=[]
        for c in range(10):
            ids=np.flatnonzero(f.label.to_numpy()==c);positions[c]=ids;fast[c]=ClassAP(truth[role][truth[role].label==c],f.iloc[ids]);refs[c]=IndependentAP(truth[role][truth[role].label==c],f.iloc[ids].reset_index(drop=True));baseline.append(refs[c].ap(np.ones(len(ids),bool)))
        def effect(chosen):
            keep=np.ones(len(f),bool);keep[chosen]=False;a=[refs[c].ap(keep[positions[c]]) for c in range(10)];b=[fast[c].evaluate(keep[positions[c]])['ap'] for c in range(10)];assert np.allclose(a,b,atol=1e-7,rtol=0);return float(100*(np.mean(a)-np.mean(baseline))),np.asarray(a).tolist()
        results[role]={}
        for m in (*MODES,'confidence'):
            frame=f if m=='confidence' else frames[m];value=1-frame.score.to_numpy() if m=='confidence' else frame.pbg.to_numpy();auc=[];budgets=[]
            for _,ids in frame.groupby('video-id',sort=False).indices.items():
                ids=np.asarray(ids);ids=ids[frame.iloc[ids].score.to_numpy()>=.3];a=independent_auc(bg[ids],value[ids])
                if a is not None:auc.append(a)
            for rate in SPEC['rates']:
                chosen=choose(frame,eligible,rate,m=='confidence');delta,ap=effect(chosen);gain,_=effect(chosen[bg[chosen]]);harm,_=effect(chosen[~bg[chosen]])
                budgets.append(dict(rate=rate,selected=chosen.tolist(),delta_pp=delta,per_class_AP=ap,BG_only_delta_pp=gain,nonBG_only_delta_pp=harm,BG_removed=int(bg[chosen].sum()),nonBG_removed=int((~bg[chosen]).sum())))
            results[role][m]=dict(baseline=float(np.mean(baseline)),original_per_class_AP=np.asarray(baseline).tolist(),video_macro_AUC=float(np.mean(auc)),AUC_valid_videos=len(auc),budgets=budgets)
        if role=='target_val':assert abs(np.mean(baseline)-read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/s/'evaluation'/'target_val.json')['map_0.1_to_0.5'])<1e-7
        print('NATIVE_EVALUATED',s,role,flush=True)
    save(root/(s+'_evaluation.json'),dict(passed=True,results=results,all_original_arrays_exact=True,all_AP_independent_NumPy_equal=True,target_train_GT_read=False,target_val_used_for_selection=False))


def run(root):
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity(),dense_input_audit_sha256=sha(DENSE/'independent_input_audit.json')));active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        for stage,tasks in [('extract',('S1','S2')),('train',TASKS),('evaluate',('S1','S2'))]:
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
                        v=snapshot(g);lease=entry/('lease_'+stage+'.json');save(lease,dict(owner_pid=os.getpid(),index=g,uuid=v['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']));env.update(CUDA_VISIBLE_DEVICES=v['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(entry/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(entry),'--worker',stage],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()},pending=pending,completed=done));time.sleep(3)
        save(root/'final.json',dict(passed=True,allGPU_sha256={t:sha(root/t/'gpu_complete.json') for t in TASKS},teacher_sha256={s:sha(root/s/'complete.json') for s in ('S1','S2')},evaluation_sha256={s:sha(root/(s+'_evaluation.json')) for s in ('S1','S2')}));save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['extract','train','evaluate']);a=p.parse_args()
    if a.worker=='extract':extract(a.entry)
    elif a.worker=='train':train(a.entry)
    elif a.worker=='evaluate':evaluate(a.entry.parent,a.entry.name)
    else:run(a.root)
