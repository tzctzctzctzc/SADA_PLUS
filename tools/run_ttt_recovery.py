"""双卡时序恢复TTT：源域准备/源校准/冻结目标评估；不接续或覆盖旧根。"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_sada_baseline import acquire,prepare_file_limit
from tools.ttt_recovery import SPEC
from tools.run_tta import load_model,evaluate_predictions

SHIFTS=tuple('S%d'%i for i in range(1,7))
ACTIVE={}


def state_cpu(module):
    return {n:p.detach().cpu().clone() for n,p in module.items()}


def predict_dataset(model,ds):
    from tools.tta_core import unlabeled,predict_full
    videos=[]
    for i in range(len(ds)):
        prediction=predict_full(model,unlabeled(ds[i]))
        videos.append(serializable(prediction))
    return videos


def serializable(prediction):
    return dict(video_id=prediction['video_id'],segments=prediction['segments'].tolist(),
        labels=prediction['labels'].tolist(),scores=prediction['scores'].tolist())


def prepare(model,reference,head,named,cfg,job):
    import random
    import numpy as np
    import torch
    from tools.baseline_engine import dataset
    from tools.tta_core import unlabeled,parameter_digest
    from tools.ttt_recovery import source_detection_loss,recovery_loss,seed_for
    output=Path(job['output']);kind=job['variant'];history=[];best=-1
    # 三种方法一模一样的数据顺序、crop随机种子、检测损失及共享参数优化器。
    optimizer=torch.optim.AdamW([dict(params=list(named.values()),lr=SPEC['preparation_lr']),
        dict(params=list(head.parameters()),lr=SPEC['decoder_lr'])],weight_decay=.01)
    train=dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],True)
    val=dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False)
    initial=parameter_digest(named)
    for epoch in range(1,SPEC['preparation_epochs']+1):
        generator=torch.Generator().manual_seed(123+epoch)
        order=torch.randperm(len(train),generator=generator).tolist();rows=[]
        for step,index in enumerate(order):
            seed=seed_for(str(index),'source%d'%epoch)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
            video=train[index]
            detection=source_detection_loss(model,video)
            loss=detection;metrics={}
            if kind!='source_only':
                rec,metrics=recovery_loss(model,reference,head,unlabeled(video),seed,kind=='temporal_change')
                loss=loss+SPEC['reconstruction_weight']*rec
            if not torch.isfinite(loss):raise ValueError('源准备loss非有限')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(list(named.values())+list(head.parameters()),1.)
            if not torch.isfinite(norm):raise ValueError('源准备梯度非有限')
            optimizer.step()
            rows.append(dict(source_detection=float(detection.detach()),total=float(loss.detach()),
                             gradient_norm=float(norm),**metrics))
            if step%20==0:print('PREP',kind,epoch,step+1,len(order),rows[-1],flush=True)
        predictions=predict_dataset(model,val)
        score=evaluate_predictions(predictions,cfg['dataset']['json_file'],cfg['src_val_split'])
        history.append(dict(epoch=epoch,source_map=score['map'],steps=len(order),
            mean_loss=sum(x['total'] for x in rows)/len(rows),
            mean_detection=sum(x['source_detection'] for x in rows)/len(rows),
            mean_reconstruction=sum(x.get('reconstruction',0) for x in rows)/len(rows),
            mean_change=sum(x.get('change',0) for x in rows)/len(rows)))
        payload=dict(schema=SPEC['schema'],variant=kind,epoch=epoch,config_sha256=job['config_sha256'],
            parent_checkpoint_sha256=job['checkpoint_sha256'],code=job['code'],specification=SPEC,
            adaptable=state_cpu(named),head=state_cpu(head.state_dict()),history=history,
            optimizer=optimizer.state_dict(),rng=dict(python=random.getstate(),numpy=np.random.get_state(),
            cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0)))
        torch.save(payload,output/'latest.pt')
        if score['map']>best:
            best=score['map'];torch.save(payload,output/'best.pt')
        save(output/'history.json',history)
        print('PREP_EPOCH',json.dumps(history[-1]),flush=True)
    if initial==parameter_digest(named):raise ValueError('源准备没有更新共享检测参数')
    selected=torch.load(output/'best.pt',map_location='cpu')
    return dict(status='passed',source_best_map=best,best_epoch=selected['epoch'],epochs=SPEC['preparation_epochs'],
        source_train_videos=len(train),source_val_videos=len(val),target_data_seen=False,
        initial_parameter_sha256=initial,final_parameter_sha256=parameter_digest(named))


def diagnostic(model,reference,head,named,cfg,job):
    import torch
    from tools.baseline_engine import dataset
    from tools.tta_core import unlabeled,predict_full,parameter_digest
    from tools.ttt_recovery import source_detection_loss,recovery_loss,adapt,restore
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],True)
    video=ds[0];clean=unlabeled(video);initial=state_cpu(named)
    original=model([clean],[clean],mode='test')[1][0];current=predict_full(model,clean)
    if any(not torch.equal(original[k],current[k]) for k in ('segments','scores','labels')):
        raise ValueError('无适应推理与正式原路径不相同')
    official=model([video],[clean],mode='val',domain='src')[0]['downstream_loss']
    ours=source_detection_loss(model,video)
    torch.testing.assert_close(ours,official,rtol=1e-5,atol=1e-6)
    frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    reference_digest=parameter_digest(dict(reference.named_parameters()))
    optimizer=torch.optim.AdamW([dict(params=list(named.values()),lr=SPEC['preparation_lr']),
        dict(params=list(head.parameters()),lr=SPEC['decoder_lr'])])
    head_before=parameter_digest(dict(head.named_parameters()));encoder_before=parameter_digest(named)
    rec,metrics=recovery_loss(model,reference,head,clean,123,job['variant']=='temporal_change')
    rec.backward()
    grads=[p.grad for p in named.values() if p.grad is not None]
    norm=torch.stack([g.norm() for g in grads]).norm()
    if not torch.isfinite(norm) or float(norm)<=0:raise ValueError('真实恢复梯度无效')
    optimizer.step()
    if head_before==parameter_digest(dict(head.named_parameters())) or encoder_before==parameter_digest(named):
        raise ValueError('真实源准备分支未更新')
    if reference_digest!=parameter_digest(dict(reference.named_parameters())):raise ValueError('参考被更新')
    initial=state_cpu(named);head_before=parameter_digest(dict(head.named_parameters()))
    one,a=adapt(model,reference,head,initial,clean,1e-4,1,job['variant']=='temporal_change')
    two,b=adapt(model,reference,head,initial,clean,1e-4,1,job['variant']=='temporal_change')
    if a!=b or not a['changed'] or any(not torch.equal(one[k],two[k]) for k in ('segments','scores','labels')):
        raise ValueError('真实TTT逐视频重置不精确')
    if head_before!=parameter_digest(dict(head.named_parameters())):raise ValueError('测试恢复头未冻结')
    if any(not torch.equal(p.detach().cpu(),frozen[n]) for n,p in model.named_parameters() if n in frozen):
        raise ValueError('测试头或其他冻结参数改变')
    return dict(status='passed',source_detection_exact=True,inference_exact=True,reference_frozen=True,
        decoder_frozen_at_test=True,episodic_reset=True,real_encoder_gradient=float(norm),
        target_gt_seen=False,source_recovery=metrics,adaptation=a)


def worker(args):
    import torch
    from tools.baseline_gpu import first_cuda_guard
    from tools.baseline_engine import dataset
    from tools.tta_core import unlabeled,predict_full
    from tools.ttt_recovery import initialize,apply_prepared,adapt,restore
    job=read(args.job);output=Path(job['output']);cfg=read(job['config'])
    if code_identity()!=job['code'] or SPEC!=job['specification']:raise ValueError('执行身份改变')
    for path,field in [('config','config_sha256'),('checkpoint','checkpoint_sha256'),
                       ('metadata_manifest','metadata_sha256')]:
        if sha(job[path])!=job[field]:raise ValueError('输入SHA改变: '+path)
    guard=first_cuda_guard();os.chdir(args.data_root)
    model=load_model(cfg,job['checkpoint']);named,head=initialize(model)
    reference=deepcopy(model).eval()
    for p in reference.parameters():p.requires_grad_(False)
    if job['kind']=='diagnostic':result=diagnostic(model,reference,head,named,cfg,job)
    elif job['kind']=='prepare':result=prepare(model,reference,head,named,cfg,job)
    else:
        if sha(job['prepared'])!=job['prepared_sha256']:raise ValueError('准备checkpoint改变')
        prepared=torch.load(job['prepared'],map_location='cpu')
        if (prepared['parent_checkpoint_sha256']!=job['checkpoint_sha256'] or
                prepared['variant']!=job['variant'] or prepared['config_sha256']!=job['config_sha256']):
            raise ValueError('准备checkpoint与方向不匹配')
        apply_prepared(model,head,prepared);initial=state_cpu(named)
        splits=cfg['src_val_split'] if job['kind']=='calibration' else cfg['tgt_val_split']
        if job['kind']=='target' and job['adapt']:
            chosen=read(job['selection_file'])
            if (sha(job['selection_file'])!=job['selection_sha256'] or chosen['target_results_seen'] or
                job['lr']!=chosen['chosen']['lr'] or job['steps']!=chosen['chosen']['steps']):
                raise ValueError('正式TTT必须使用冻结源选择')
        ds=dataset(cfg,job['metadata_manifest'],splits,False,True)
        predictions=[];proofs=[]
        for i in range(len(ds)):
            video=unlabeled(ds[i])
            if job['adapt']:
                prediction,proof=adapt(model,reference,head,initial,video,job['lr'],job['steps'],job['variant']=='temporal_change')
            else:
                restore(named,initial);prediction=predict_full(model,video)
                proof=dict(initial_parameter_sha256=__import__('tools.tta_core',fromlist=['parameter_digest']).parameter_digest(named),
                    changed=False,trace=[],target_gt_seen=False)
            predictions.append(serializable(prediction));proofs.append(proof)
            print('VIDEO',i+1,len(ds),video['metadata']['video_id'],'updated',proof['changed'],flush=True)
        save(output/'predictions.json',predictions);save(output/'adaptation_proofs.json',proofs)
        result=evaluate_predictions(predictions,job['evaluation_annotation'],splits)
        result.update(status='passed',videos=len(ds),updated_videos=sum(p['changed'] for p in proofs),
                      target_gt_seen_during_adaptation=False)
    result['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
    artifacts={p.name:sha(p) for p in output.iterdir() if p.name in
        ('best.pt','latest.pt','history.json','predictions.json','adaptation_proofs.json')}
    save(output/'complete.json',dict(status='passed',job_sha256=sha(args.job),code=job['code'],guard=guard,
        guard_sha256=sha(guard),artifacts=artifacts,result=result))


def audit(path):
    job=read(path);out=Path(job['output']);record=read(out/'complete.json')
    if record['status']!='passed' or record['job_sha256']!=sha(path) or record['code']!=job['code']:
        raise ValueError('TTT完整身份改变')
    for p,field in [('config','config_sha256'),('checkpoint','checkpoint_sha256'),('metadata_manifest','metadata_sha256'),
                     ('evaluation_annotation','evaluation_annotation_sha256'),
                     ('prepared','prepared_sha256'),('selection_file','selection_sha256')]:
        if p in job and sha(job[p])!=job[field]:raise ValueError('TTT输入改变: '+p)
    if sha(record['guard'])!=record['guard_sha256']:raise ValueError('首CUDA凭证改变')
    guard=read(record['guard']);lease=read(guard['lease'])
    if (sha(guard['lease'])!=guard['lease_sha256'] or lease['commit']!=job['code']['commit'] or
        guard['observed']['uuid']!=lease['observed']['uuid'] or guard['observed']['free_mib']<lease['minimum_free_mib']):
        raise ValueError('GPU共享/锁/UUID凭证不匹配')
    for p,h in record['artifacts'].items():
        if sha(out/p)!=h:raise ValueError('TTT工件改变: '+p)
    if (out/'history.json').exists():
        history=read(out/'history.json')
        if [x['epoch'] for x in history]!=list(range(1,SPEC['preparation_epochs']+1)):
            raise ValueError('准备历史不完整')
        chosen=max(history,key=lambda x:x['source_map'])
        if chosen['epoch']!=record['result']['best_epoch']:raise ValueError('源选best错误')
    if (out/'adaptation_proofs.json').exists():
        proofs=read(out/'adaptation_proofs.json')
        if len({p['initial_parameter_sha256'] for p in proofs})>1 or any(p['target_gt_seen'] for p in proofs):
            raise ValueError('视频重置/GT隔离失败')
    return dict(job=str(path),shift=job['shift'],variant=job['variant'],adapt=job.get('adapt'),
                complete_sha256=sha(out/'complete.json'),**record['result'])


def run_jobs(root,jobs,data,identity,phase):
    queue=list(jobs);done=[]
    while queue or ACTIVE:
        for key,t in list(ACTIVE.items()):
            if t['process'].poll() is None:continue
            t['context'].__exit__(None,None,None);del ACTIVE[key]
            if t['process'].returncode:raise RuntimeError('TTT任务失败，保留工件: '+key)
            done.append(audit(t['job']))
        for path in list(queue):
            job=read(path);out=Path(job['output'])
            if (out/'complete.json').exists():done.append(audit(path));queue.remove(path);continue
            found=acquire([0,1],data,root,12288,identity['commit'])
            if not found:break
            context,lease=found
            env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
                TMPDIR=str(root/'tmp'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
            try:
                with (out/'run.log').open('x') as stream:
                    process=subprocess.Popen([sys.executable,'-B',str(CODE/'tools/run_ttt_recovery.py'),
                        '--job',str(path),'--data-root',str(data)],cwd=CODE,env=env,
                        stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            except BaseException:context.__exit__(None,None,None);raise
            key=str(out.relative_to(root));ACTIVE[key]=dict(process=process,context=context,job=path,gpu=lease['index'])
            queue.remove(path)
        save(root/'state.json',dict(phase=phase,pid=os.getpid(),phase_completed=len(done),phase_total=len(jobs),
            active={k:dict(pid=t['process'].pid,gpu=t['gpu']) for k,t in ACTIVE.items()}))
        if queue or ACTIVE:time.sleep(3)
    return done


def job_file(root,relative,kind,s,variant,prototype,identity,manifest,annotation,**extra):
    entry=prototype/'formal'/s/'prototype';out=root/relative;out.mkdir(parents=True,exist_ok=False)
    save(out/'job.json',dict(kind=kind,shift=s,variant=variant,config=str(entry/'config.json'),
        config_sha256=sha(entry/'config.json'),checkpoint=str(entry/'best.pt'),checkpoint_sha256=sha(entry/'best.pt'),
        prototype_complete_sha256=sha(entry/'complete.json'),metadata_manifest=str(manifest),metadata_sha256=sha(manifest),
        evaluation_annotation=str(annotation),evaluation_annotation_sha256=sha(annotation),
        output=str(out),code=identity,specification=SPEC,**extra))
    return out/'job.json'


def source_selection(candidates):
    expected={(lr,n) for lr in SPEC['learning_rates'] for n in SPEC['steps']}
    if (len(candidates)!=4 or {(x['lr'],x['steps']) for x in candidates}!=expected or
            any(len(x['results'])!=6 or {r['shift'] for r in x['results']}!=set(SHIFTS) or
                any(not math.isfinite(r['map']) or not 0<=r['map']<=1 for r in x['results']) for x in candidates)):
        raise ValueError('源校准必须四设置乘六方向')
    return max(candidates,key=lambda x:(sum(r['map'] for r in x['results'])/6,-x['steps'],-x['lr']))


def main(args):
    root=args.output.resolve();prototype=args.prototype.resolve();prior=args.tta.resolve();data=args.data_root.resolve()
    identity=code_identity();root.mkdir(parents=True,exist_ok=False);(root/'tmp').mkdir()
    save(root/'runtime_resources.json',prepare_file_limit())
    save(root/'plan.json',dict(specification=SPEC,code=identity,prototype=str(prototype),prior_tta=str(prior),
        target_jobs=30,route_revisions_remaining=dict(route1=2,route2=2),
        development='exploratory-all-positive-negative-results-retained',
        references=['https://arxiv.org/abs/2411.17869','https://jmlr.org/papers/v26/24-0439.html']))
    try:
        proto_final=read(prototype/'final_audit.json');tta_final=read(prior/'final_audit.json')
        if (proto_final['count']!=6 or tta_final['count']!=24 or
                proto_final['status']!='passed' or tta_final['status']!='passed'):
            raise ValueError('前序精确六方向/24项未完成')
        # 已终审前序只复核完整凭证的小文件SHA；不再次循环读取24份相同大权重。
        for row in proto_final['results']:
            if sha(prototype/'formal'/row['shift']/'prototype/complete.json')!=row['complete_sha256']:
                raise ValueError('前序Prototype完整凭证改变')
        for rows in tta_final['results'].values():
            for row in rows:
                if sha(Path(row['job']).parent/'complete.json')!=row['complete_sha256']:
                    raise ValueError('前序TTA完整凭证改变')
        save(root/'predecessor.json',dict(prototype_final_sha256=sha(prototype/'final_audit.json'),
            tta_final_sha256=sha(prior/'final_audit.json')))
        manifests={}
        for s in SHIFTS:
            # 直接复用已经冻结的同八源验证视频及无GT目标元数据；不按目标结果重新挑选。
            cfg=read(prototype/'formal'/s/'prototype/config.json')
            manifests[s]=dict(source=prior/'inputs'/s/'source_unlabeled.json',source_gt=prior/'inputs'/s/'source_evaluation.json',
                target=prior/'inputs'/s/'target_unlabeled.json',target_gt=cfg['baseline']['annotation_for_offline_evaluation'])
        diagnostics=[job_file(root,'diagnostic/'+v,'diagnostic','S1',v,prototype,identity,
            manifests['S1']['source'],manifests['S1']['source_gt']) for v in ('reconstruction','temporal_change')]
        acceptance=run_jobs(root,diagnostics,data,identity,'real_gpu_diagnostic')
        save(root/'acceptance.json',dict(status='passed',code=identity,results=acceptance))
        all_results={}
        for variant in SPEC['preparation_variants']:
            jobs=[job_file(root,'prepare/'+variant+'/'+s,'prepare',s,variant,prototype,identity,
                manifests[s]['source'],manifests[s]['source_gt']) for s in SHIFTS]
            prepared=run_jobs(root,jobs,data,identity,'source_prepare_'+variant)
            save(root/'prepare'/variant/'final_audit.json',dict(status='passed',count=6,results=prepared))
            selection_file=None
            if variant!='source_only':
                trials=[]
                for lr in SPEC['learning_rates']:
                    for steps in SPEC['steps']:
                        jobs=[]
                        for s in SHIFTS:
                            prepared_path=root/'prepare'/variant/s/'best.pt'
                            jobs.append(job_file(root,'calibration/%s/lr%s_steps%d/%s'%(variant,lr,steps,s),
                                'calibration',s,variant,prototype,identity,manifests[s]['source'],manifests[s]['source_gt'],
                                prepared=str(prepared_path),prepared_sha256=sha(prepared_path),adapt=True,lr=lr,steps=steps))
                        results=run_jobs(root,jobs,data,identity,'source_calibration_'+variant)
                        trials.append(dict(lr=lr,steps=steps,results=results))
                chosen=source_selection(trials);selection_file=root/'selection'/variant/'frozen.json'
                save(selection_file,dict(status='passed',target_results_seen=False,chosen=chosen,candidates=trials))
            for use_adaptation in ([False] if variant=='source_only' else [False,True]):
                label=variant+('_ttt' if use_adaptation else '_no_ttt');jobs=[]
                for s in SHIFTS:
                    prepared_path=root/'prepare'/variant/s/'best.pt'
                    options=dict(prepared=str(prepared_path),prepared_sha256=sha(prepared_path),adapt=use_adaptation)
                    if use_adaptation:options.update(lr=chosen['lr'],steps=chosen['steps'],
                        selection_file=str(selection_file),selection_sha256=sha(selection_file))
                    jobs.append(job_file(root,'formal/'+label+'/'+s,'target',s,variant,prototype,identity,
                        manifests[s]['target'],manifests[s]['target_gt'],**options))
                results=run_jobs(root,jobs,data,identity,'target_'+label)
                all_results[label]=results
                save(root/'formal'/label/'final_audit.json',dict(status='passed',count=6,results=results,
                    mean_map=sum(x['map'] for x in results)/6))
        save(root/'final_audit.json',dict(status='passed',count=30,results=all_results))
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),active={},count=30))
    except BaseException as error:
        save(root/'error.json',dict(type=type(error).__name__,message=str(error)))
        save(root/'state.json',dict(phase='failed',pid=os.getpid(),active={}))
        raise
    finally:
        for t in ACTIVE.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in ACTIVE.values():
            try:t['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['process'].pid,signal.SIGKILL);t['process'].wait()
            t['context'].__exit__(None,None,None)


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    raise SystemExit(128+signum)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--prototype',type=Path)
    p.add_argument('--tta',type=Path);p.add_argument('--job',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));args=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.job:worker(args)
    else:main(args)
