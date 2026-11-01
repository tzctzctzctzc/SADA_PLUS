"""六方向真实回归特征/梯度核验；无优化步骤、无mAP、无正式训练。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.regression_gram import SPEC


def probe(args):
    from tools.baseline_gpu import first_cuda_guard
    guard=first_cuda_guard()
    import torch
    import numpy as np
    from libs.models.SADA import make_meta_arch
    from libs.trainer.da_trainer import move_to_device
    from libs.utils import fix_random_seed
    from tools import baseline_engine as base
    from tools.prototype_target_cache import detector_features
    from tools.regression_gram import regression_inputs,gram_alignment
    e=args.entry;cfg=read(e/'config.json');inputs=read(e/'inputs.json');identity=code_identity()
    assert identity==read(args.output/'plan.json')['code']
    os.chdir(args.data_root);fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    ref=Path(inputs['reference']);assert sha(ref/'best.pt')==inputs['checkpoint_sha256']
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda()
    ck=torch.load(ref/'best.pt',map_location='cpu');model.load_state_dict(ck['ema_state_dict'],strict=True);del ck
    model.eval();initial=base.tensor_digest(model.state_dict());rows={};nodes={};observations={}
    target_db=read(e/'target_train_unlabeled.json')['database']
    assert all('annotations' not in v for v in target_db.values())
    assert set(read(e/'source_train_gt.json')['database']).isdisjoint(target_db)
    for domain in ['source','target']:
        source=domain=='source';path=e/('source_train_gt.json' if source else 'target_train_unlabeled.json')
        ds=base.dataset(cfg,path,cfg['src_train_split'] if source else cfg['tgt_train_split'],True,not source)
        values=[];all_nodes=[];obs=[]
        for i in range(min(SPEC['videos_per_domain'],len(ds))):
            v=move_to_device([ds[i]],'cuda:0')[0]
            if not source:assert not v['segments'].numel() and not v['labels'].numel()
            f,m=detector_features(model,v);points=model.point_generator(f);x=regression_inputs(model.reg_head,f,m)[0]
            valid=torch.cat([z[0].reshape(-1) for z in m]).bool()
            logits,_=model.cls_head(f,m);scores=torch.cat([z[0].T for z in logits]).sigmoid().amax(1).detach()
            if source:
                classes,_=model.label_points(points,[v['segments']],[v['labels']]);eligible=valid&(classes[0].sum(-1)>0)
            else:eligible=valid&(scores>=SPEC['target_confidence'])
            ids=eligible.nonzero(as_tuple=True)[0]
            # 等距取样不消费训练随机流，也不按未知目标定位质量筛选。
            if len(ids)>SPEC['max_points_per_video']:
                ids=ids[torch.linspace(0,len(ids)-1,SPEC['max_points_per_video'],device=ids.device).long()]
            values.append(x[ids]);all_nodes.extend(f)
            obs.append(dict(video=v['metadata']['video_id'],valid_points=int(valid.sum()),eligible_points=int(eligible.sum()),
                retained=len(ids),retained_foreground_score_mean=float(scores[ids].mean()) if len(ids) else None))
        rows[domain]=torch.cat(values);nodes[domain]=all_nodes;observations[domain]=obs
    info=dict(schema=SPEC['schema'],specification=SPEC,code=identity,guard=guard,guard_sha256=sha(guard),
        observations=observations,target_gt_used=False,optimization_steps=0,checkpoint_written=False,
        detector_mode='eval to isolate the alignment path; not a training efficacy test')
    if min(len(rows['source']),len(rows['target']))<4:
        info.update(status='mechanism_unready',reason='insufficient source-positive or target-confident regression points')
    else:
        n=min(len(rows['source']),len(rows['target']));source=rows['source'][:n];target=rows['target'][:n]
        loss,details=gram_alignment(source,target)
        identity_loss,identity_details=gram_alignment(source.detach(),source.detach())
        gradients=torch.autograd.grad(loss,nodes['target'],retain_graph=True,allow_unused=True)
        target_gradient=sum(float(g.square().sum()) for g in gradients if g is not None)**.5
        assert torch.isfinite(loss) and target_gradient>0 and all(torch.isfinite(g).all() for g in gradients if g is not None)
        model.zero_grad();loss.backward()
        norm=lambda params:sum(float(p.grad.square().sum()) for p in params if p.grad is not None)**.5
        backbone_gradient=norm(model.backbone_source.parameters());regression_gradient=norm(model.reg_head.head.parameters())
        assert backbone_gradient>0 and regression_gradient>0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        info.update(status='passed' if float(identity_loss)<1e-7 else 'mechanism_unready',loss=float(loss),alignment=details,
            identical_domains_loss=float(identity_loss),identical_domain_alignment=identity_details,
            target_fpn_gradient_norm=target_gradient,backbone_gradient_norm=backbone_gradient,
            regression_tower_gradient_norm=regression_gradient,numerical_reference='thin-SVD truncated Gram inverse; paper rank-count rule')
    assert initial==base.tensor_digest(model.state_dict());info['detector_parameters_unchanged']=True
    save(e/'probe_complete.json',info);print('REGRESSION_GRAM_READINESS',info['status'],flush=True)


def main(args):
    from tools.run_prototype_target_cache import prepare,REFERENCE,INVENTORY,SHIFTS
    from tools.run_sada_baseline import acquire
    identity=code_identity();args.output.mkdir(exist_ok=False)
    save(args.output/'plan.json',dict(schema=SPEC['schema'],specification=SPEC,code=identity,
        scope='six-shift source/target regression-feature gradient readiness only',formal_training=False,
        gpus=[0,1,2,3],minimum_free_mib=12288,reference=str(REFERENCE)))
    prepare(args.output,REFERENCE,INVENTORY,identity)
    for s in SHIFTS:
        e=args.output/'formal'/s;cfg=read(e/'config.json');db=read(cfg['baseline']['annotation_for_offline_evaluation'])['database']
        wanted={z.lower() for z in cfg['src_train_split']}
        save(e/'source_train_gt.json',dict(database={k:v for k,v in db.items() if v['subset'].lower() in wanted}),sort_keys=False)
    pending=list(SHIFTS);active={};completed=[]
    try:
        while pending or active:
            for s,t in list(active.items()):
                code=t['process'].poll()
                if code is None:continue
                t['ctx'].__exit__(None,None,None);del active[s]
                if code:raise RuntimeError(s+' readiness process failed: '+str(code))
                p=args.output/'formal'/s/'probe_complete.json';completed.append(dict(shift=s,status=read(p)['status'],path=str(p),sha256=sha(p)))
            for s in list(pending):
                acquired=acquire([0,1,2,3],args.data_root,args.output,12288,identity['commit'])
                if not acquired:break
                ctx,card=acquired;e=args.output/'formal'/s
                env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=card['uuid'],
                    OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
                try:
                    with (e/'probe.log').open('x') as log:p=subprocess.Popen([sys.executable,'-u','-B',__file__,'--output',str(args.output),'--entry',str(e)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:ctx.__exit__(None,None,None);raise
                active[s]=dict(process=p,ctx=ctx,gpu=card['index']);pending.remove(s)
            save(args.output/'state.json',dict(phase='mechanism_probe',pid=os.getpid(),pending=pending,completed=completed,
                active={s:dict(pid=t['process'].pid,gpu=t['gpu']) for s,t in active.items()}));time.sleep(3)
        assert len(completed)==6
        save(args.output/'final_probe.json',dict(count=6,entries=completed,formal_training=False,task_metric_evaluated=False))
        save(args.output/'state.json',dict(phase='probe_complete',active={},completed=completed))
    except BaseException as err:save(args.output/'error.json',dict(error=str(err)));raise
    finally:
        for t in active.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in active.values():
            try:t['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['process'].pid,signal.SIGKILL);t['process'].wait()
            t['ctx'].__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--entry',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));a=p.parse_args()
    if a.entry:probe(a)
    else:
        def stop(sig,frame):raise SystemExit(128+sig)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main(a)
