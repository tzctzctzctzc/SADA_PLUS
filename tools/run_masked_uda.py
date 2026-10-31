"""三组匹配续训：Prototype控制/EMA无mask/EMA有mask；训练target与评估target分离。"""
import argparse
from copy import deepcopy
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,assert_no_target_gt
from tools.run_sada_baseline import acquire,prepare_file_limit
from tools.masked_uda import SPEC
from tools.run_tta import load_model,evaluate_predictions

SHIFTS=tuple('S%d'%i for i in range(1,7));ACTIVE={}


def prediction_item(p):
    return dict(video_id=p['video_id'],segments=p['segments'].tolist(),scores=p['scores'].tolist(),labels=p['labels'].tolist())


def validation(model,cfg):
    import torch
    from tools.baseline_engine import dataset
    from tools.tta_core import unlabeled
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False)
    predictions=[];model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            clean=unlabeled(ds[i]);predictions.append(prediction_item(model([clean],[clean],mode='test')[0][0]))
    return evaluate_predictions(predictions,cfg['dataset']['json_file'],cfg['src_val_split'])['map']


def training(job,probe=False):
    import numpy as np
    import torch
    from tools.baseline_engine import dataset,loader,move_to_device,tensor_digest,controls,checkpoint
    from tools.tta_core import unlabeled
    from tools.masked_uda import trainable_parameters,consistency,update_teacher,select
    output=Path(job['output']);cfg=read(job['config']);variant=job['variant']
    model=load_model(cfg,job['checkpoint']);initial=tensor_digest(model.state_dict())
    trainable=trainable_parameters(model);teacher=deepcopy(model).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    teacher_initial=tensor_digest(teacher.state_dict())
    if teacher_initial!=initial:raise ValueError('EMA教师必须由相同完整原EMA权重复制')
    optimizer=torch.optim.AdamW(list(trainable.values()),lr=SPEC['learning_rate'],weight_decay=SPEC['weight_decay'])
    generators={name:torch.Generator().manual_seed(123+index*1000003) for index,name in enumerate(('source','target'))}
    src=dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],True)
    tgt=dataset(cfg,cfg['baseline']['target_manifest'],cfg['tgt_train_split'],True,True)
    source=loader(src,True,generators['source'],SPEC['batch'],0)
    target=loader(tgt,True,generators['target'],SPEC['batch'],0)
    budget=cfg['baseline']['epoch_budget']['iterations_per_epoch']
    if budget<max(len(source),len(target)):raise ValueError('续训预算小于完整原epoch')
    history=[];all_epochs=[];best=-1;target_gradient_seen=False;proto_gradient_seen=False
    current_epoch=0
    for epoch in range(1,(1 if probe else SPEC['max_epochs'])+1):
        source_iter=itertools.cycle(source);target_iter=itertools.cycle(target);model.train()
        model.train_label_smoothing=0.;reports=[]
        for step in range(budget):
            global_step=(epoch-1)*budget+step
            random.seed(123+global_step);np.random.seed(123+global_step)
            source_videos=move_to_device(next(source_iter),'cuda:0')
            target_videos=move_to_device(next(target_iter),'cuda:0')
            assert_no_target_gt(target_videos)
            optimizer.zero_grad(set_to_none=True)
            returned=model(video_list_src=source_videos,video_list_tgt=target_videos,
                mode='train',domain='src',curr_epoch=cfg['train_cfg']['warmup_adv']+epoch)
            losses=returned[0]
            if returned[2].count_nonzero():raise ValueError('target输入生成了非零GT类别')
            native=losses['downstream_loss']*cfg['model']['loss_weights']['weight_downstream']
            # 额外增强/teacher调用不改变下轮原训练随机流；loader/crop流各组相同。
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(123+global_step);torch.cuda.manual_seed(123+global_step)
                auxiliary,metrics=consistency(model,teacher,[unlabeled(v) for v in target_videos],variant,global_step)
            loss=native+SPEC['consistency_weight']*auxiliary
            if not torch.isfinite(loss):raise ValueError('续训loss非有限')
            target_norm=0.
            if probe and variant!='continuation':
                parameters=list(model.backbone_target.stem[-1].parameters())
                grads=torch.autograd.grad(auxiliary,parameters,retain_graph=True,allow_unused=True)
                valid=[g.norm() for g in grads if g is not None]
                target_norm=float(torch.stack(valid).norm()) if valid else 0.
                target_gradient_seen|=target_norm>1e-8
            proto_gradient_seen|=float(losses.get('rq3_target_gradient_norm',0))>0
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),SPEC['clip_grad'])
            if not torch.isfinite(norm) or float(norm)<=0:raise ValueError('续训共享/检测梯度无效')
            optimizer.step();update_teacher(teacher,model)
            report=dict(native=float(native.detach()),source_cls=float(losses['cls_loss'].detach()),
                source_reg=float(losses['reg_loss'].detach()),consistency=float(auxiliary.detach()),gradient_norm=float(norm),
                prototype_target_gradient=float(losses.get('rq3_target_gradient_norm',0)),
                target_consistency_gradient=target_norm,target_annotations=0,**metrics)
            reports.append(report)
            if step%20==0:print('TRAIN',variant,epoch,step+1,budget,json.dumps(report),flush=True)
        current_epoch=epoch
        aggregate=dict(epoch=epoch,steps=budget,native=sum(x['native'] for x in reports)/budget,
            consistency=sum(x['consistency'] for x in reports)/budget,
            mean_points=sum(x['points'] for x in reports)/budget,
            prototype_target_gradient_seen=any(x['prototype_target_gradient']>0 for x in reports),
            target_annotations=0,target_GT_used=False)
        all_epochs.append(aggregate);save(output/'epochs.json',all_epochs)
        if probe:
            if variant=='ema_masked' and not target_gradient_seen:raise ValueError('遮挡一致性没有有效目标共享梯度')
            if not proto_gradient_seen:raise ValueError('完整源epoch没有覆盖原VLM Prototype目标辅助梯度')
            source_map=validation(teacher,cfg)
            return dict(status='passed',variant=variant,steps=budget,batch=2,initial_state_sha256=initial,
                teacher_initial_state_sha256=teacher_initial,student_updated=tensor_digest(model.state_dict())!=initial,
                teacher_updated=tensor_digest(teacher.state_dict())!=initial,target_GT_used=False,
                target_consistency_gradient_seen=target_gradient_seen,prototype_target_gradient_seen=proto_gradient_seen,
                source_map=source_map,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        if epoch%SPEC['validate_every']==0:
            score=validation(teacher,cfg)
            if not math.isfinite(score) or not 0<=score<=1:raise ValueError('源mAP非法')
            history.append(dict(epoch=epoch,map=score));save(output/'source_history.json',history)
            if score>best:
                best=score
                checkpoint(output/'best.pt',dict(config=cfg,ema_state_dict=teacher.state_dict(),epoch=epoch,
                    training_protocol=SPEC,variant=variant,parent_checkpoint_sha256=job['checkpoint_sha256']))
            print('SOURCE_VALIDATION',epoch,score,flush=True)
        latest=dict(config=cfg,training_protocol=SPEC,variant=variant,epoch=epoch,history=history,
            state_dict=model.state_dict(),ema_state_dict=teacher.state_dict(),optimizer=optimizer.state_dict(),
            model_controls=controls(model),ema_controls=controls(teacher),
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(0),loaders={k:g.get_state() for k,g in generators.items()}))
        checkpoint(output/'latest.pt',latest)
        if history and history[-1]['epoch']==epoch and select(history)[1]:break
    chosen,stopped=select(history)
    if not stopped:raise ValueError('未到合法终止点')
    termination=dict(status='passed',stop_epoch=current_epoch,best_epoch=chosen['epoch'],best_source_map=chosen['map'],
        initial_state_sha256=initial,teacher_initial_state_sha256=teacher_initial,
        source_train_videos=len(src),target_train_videos=len(tgt),iterations_per_epoch=budget,
        target_GT_used=False,epochs=all_epochs,source_history=history,training_protocol=SPEC)
    save(output/'termination.json',termination)
    return dict(status='passed',variant=variant,stop_epoch=current_epoch,best_epoch=chosen['epoch'],
        best_source_map=chosen['map'],initial_state_sha256=initial,teacher_initial_state_sha256=teacher_initial,
        target_GT_used=False,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)


def worker(args):
    from tools.baseline_gpu import first_cuda_guard
    from tools.baseline_engine import dataset
    from tools.tta_core import unlabeled,predict_full
    job=read(args.job);output=Path(job['output']);cfg=read(job['config'])
    if code_identity()!=job['code'] or job['specification']!=SPEC:raise ValueError('续训执行身份改变')
    for field,hash_field in [('config','config_sha256'),('checkpoint','checkpoint_sha256'),
        ('source_manifest','source_manifest_sha256'),('target_manifest','target_manifest_sha256'),
        ('evidence','evidence_sha256')]:
        if sha(job[field])!=job[hash_field]:raise ValueError('续训输入改变: '+field)
    evidence=read(job['evidence'])
    if evidence['target_gt_used'] or evidence['selected_count']!=40 or evidence['candidate_count']!=200:
        raise ValueError('必须复用原无GT 40/200教师证据')
    if any('annotations' in row for row in read(job['target_manifest'])['database'].values()):
        raise ValueError('目标训练manifest必须仅含无标注元数据')
    if job['kind']=='evaluate':
        if sha(job['evaluation_metadata'])!=job['evaluation_metadata_sha256'] or sha(job['evaluation_annotation'])!=job['evaluation_annotation_sha256']:
            raise ValueError('离线评估成员/GT改变')
    guard=first_cuda_guard();os.chdir(args.data_root)
    if job['kind'] in ('probe','train'):result=training(job,job['kind']=='probe')
    else:
        training_record=read(output/'train_complete.json')
        if sha(output/'best.pt')!=training_record['artifacts']['best.pt']:raise ValueError('续训best改变')
        model=load_model(cfg,str(output/'best.pt'))
        ds=dataset(cfg,job['evaluation_metadata'],cfg['tgt_val_split'],False,True);predictions=[]
        for i in range(len(ds)):
            p=predict_full(model,unlabeled(ds[i]));predictions.append(prediction_item(p));print('EVAL',i+1,len(ds),flush=True)
        save(output/'predictions.json',predictions)
        result=evaluate_predictions(predictions,job['evaluation_annotation'],cfg['tgt_val_split'])
        result.update(status='passed',videos=len(ds),target_GT_during_training=False)
    names=('epochs.json','source_history.json','best.pt','latest.pt','termination.json','predictions.json')
    artifacts={name:sha(output/name) for name in names if (output/name).exists()}
    target='train_complete.json' if job['kind']=='train' else 'complete.json'
    save(output/target,dict(status='passed',kind=job['kind'],job_sha256=sha(args.job),code=job['code'],
        guard=guard,guard_sha256=sha(guard),result=result,artifacts=artifacts))


def summary_record(job,path,record,complete_path):
    # worker结果本身已带variant；先合并映射，再绑定job身份，避免重复keyword报错。
    return dict(record['result'],shift=job['shift'],variant=job['variant'],job=str(path),
        complete_sha256=sha(complete_path))


def audit(path):
    job=read(path);out=Path(job['output']);record=read(out/('train_complete.json' if job['kind']=='train' else 'complete.json'))
    if record['job_sha256']!=sha(path) or record['code']!=job['code'] or record['status']!='passed':raise ValueError('续训身份改变')
    for field in ('config','checkpoint','source_manifest','target_manifest','evidence'):
        if sha(job[field])!=job[field+'_sha256']:raise ValueError('续训输入SHA改变')
    if job['kind']=='evaluate':
        for field in ('evaluation_metadata','evaluation_annotation'):
            if sha(job[field])!=job[field+'_sha256']:raise ValueError('离线评估输入改变')
    if sha(record['guard'])!=record['guard_sha256']:raise ValueError('GPU凭证改变')
    guard=read(record['guard']);lease=read(guard['lease'])
    if sha(guard['lease'])!=guard['lease_sha256'] or lease['commit']!=job['code']['commit']:
        raise ValueError('GPU lease身份改变')
    if guard['observed']['uuid']!=lease['observed']['uuid'] or guard['observed']['free_mib']<lease['minimum_free_mib']:
        raise ValueError('GPU UUID/显存不符')
    for name,h in record['artifacts'].items():
        if sha(out/name)!=h:raise ValueError('续训工件改变:'+name)
    if job['kind']!='probe':
        from tools.masked_uda import select
        term=read(out/'termination.json');chosen,stop=select(term['source_history'])
        if not stop or chosen['epoch']!=term['best_epoch'] or [x['epoch'] for x in term['epochs']]!=list(range(1,term['stop_epoch']+1)):
            raise ValueError('源选模/终止/训练历史不符')
        if any(select(term['source_history'][:i])[1] for i in range(1,len(term['source_history']))):
            raise ValueError('错过首次合法停止点')
        if any(x['target_GT_used'] or x['target_annotations'] for x in term['epochs']):raise ValueError('目标GT泄漏')
    return summary_record(job,path,record,out/('train_complete.json' if job['kind']=='train' else 'complete.json'))


def run_jobs(root,jobs,data,identity,phase):
    pending=list(jobs);done=[]
    if pending and read(pending[0])['kind']=='train':
        # 先派长任务，空出的卡接短任务；只影响调度，不改变每项独立随机流。
        pending.sort(key=lambda p:read(read(p)['config'])['baseline']['epoch_budget']['iterations_per_epoch'],reverse=True)
    while pending or ACTIVE:
        for key,t in list(ACTIVE.items()):
            if t['process'].poll() is None:continue
            t['context'].__exit__(None,None,None);del ACTIVE[key]
            if t['process'].returncode:raise RuntimeError('新UDA任务失败: '+key)
            done.append(audit(t['job']))
        for path in list(pending):
            job=read(path);out=Path(job['output']);found=acquire([0,1],data,root,12288,identity['commit'])
            if not found:break
            context,lease=found
            env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
                TMPDIR=str(root/'tmp'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
            try:
                with (out/(job['kind']+'.log')).open('x') as stream:
                    p=subprocess.Popen([sys.executable,'-B',str(CODE/'tools/run_masked_uda.py'),'--job',str(path),
                        '--data-root',str(data)],cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            except BaseException:context.__exit__(None,None,None);raise
            ACTIVE[str(out.relative_to(root))]=dict(process=p,context=context,job=path,gpu=lease['index']);pending.remove(path)
        save(root/'state.json',dict(phase=phase,pid=os.getpid(),phase_completed=len(done),phase_total=len(jobs),
            active={k:dict(pid=t['process'].pid,gpu=t['gpu']) for k,t in ACTIVE.items()}))
        if pending or ACTIVE:time.sleep(3)
    return done


def make_job(root,relative,kind,s,variant,prototype,identity):
    entry=prototype/'formal'/s/'prototype';cfg=read(entry/'config.json');out=root/relative;out.mkdir(parents=True,exist_ok=False)
    evidence=cfg['prototype_capacity']['prepared_cache']
    payload=dict(kind=kind,shift=s,variant=variant,config=str(entry/'config.json'),config_sha256=sha(entry/'config.json'),
        checkpoint=str(entry/'best.pt'),checkpoint_sha256=sha(entry/'best.pt'),output=str(out),code=identity,specification=SPEC,
        source_manifest=cfg['dataset']['json_file'],target_manifest=cfg['baseline']['target_manifest'],evidence=evidence)
    for field in ('source_manifest','target_manifest','evidence'):payload[field+'_sha256']=sha(payload[field])
    if kind=='evaluate':
        target=read(cfg['baseline']['annotation_for_offline_evaluation'])['database']
        meta=root/'inputs'/s/'evaluation_unlabeled.json'
        save(meta,dict(database={v:{k:x for k,x in row.items() if k!='annotations'} for v,row in target.items()
            if row['subset'].lower() in cfg['tgt_val_split']}))
        payload.update(evaluation_metadata=str(meta),evaluation_metadata_sha256=sha(meta),
            evaluation_annotation=cfg['baseline']['annotation_for_offline_evaluation'],
            evaluation_annotation_sha256=sha(cfg['baseline']['annotation_for_offline_evaluation']))
    path=out/(kind+'_job.json');save(path,payload);return path


def main(args):
    root=args.output.resolve();base=args.prototype.resolve();data=args.data_root.resolve();identity=code_identity()
    root.mkdir(parents=True,exist_ok=False);(root/'tmp').mkdir();save(root/'runtime_resources.json',prepare_file_limit())
    save(root/'plan.json',dict(specification=SPEC,code=identity,prototype=str(base),formal_count=18,
        method='training-time masked prediction consistency, not test-time feature reconstruction',
        controls='same initialization/optimizer/batch/source-target order/budget/EMA/selection across all three arms'))
    try:
        predecessor=read(args.previous_ttt/'monitor/final_independent_audit.json')
        if predecessor['status']!='passed' or predecessor['count']!=12:raise ValueError('最后TTT未独立全审')
        final=read(base/'final_audit.json')
        if final['status']!='passed' or final['count']!=6:raise ValueError('原Prototype未完整')
        for row in final['results']:
            if sha(base/'formal'/row['shift']/'prototype/complete.json')!=row['complete_sha256']:
                raise ValueError('原完整项身份改变')
        save(root/'predecessor.json',dict(prototype_final_sha256=sha(base/'final_audit.json'),
            previous_ttt_final_sha256=sha(args.previous_ttt/'monitor/final_independent_audit.json')))
        if args.previous_root:
            # 仅接续此次验收汇总失败：旧根尚无正式目录，两份完整GPU验收直接按旧身份复用。
            if (args.previous_root/'formal').exists():raise ValueError('该恢复入口拒绝已有正式/部分训练根')
            reports=[audit(args.previous_root/'probe'/v/'probe_job.json') for v in ('continuation','ema_masked')]
            save(root/'reused_acceptance.json',dict(status='passed',previous_root=str(args.previous_root),
                reports=reports,reason='audit summary duplicate-key repair; training/core algorithms unchanged'))
        else:
            probes=[make_job(root,'probe/'+v,'probe','S1',v,base,identity) for v in ('continuation','ema_masked')]
            reports=run_jobs(root,probes,data,identity,'real_gpu_acceptance')
        if len({x['initial_state_sha256'] for x in reports})!=1:raise ValueError('对照/新方法起点不同')
        if not all(x['student_updated'] and x['teacher_updated'] for x in reports):raise ValueError('模型或教师未更新')
        save(root/'acceptance.json',dict(status='passed',code=identity,results=reports))
        all_results={}
        for variant in SPEC['variants']:
            jobs=[make_job(root,'formal/'+variant+'/'+s,'train',s,variant,base,identity) for s in SHIFTS]
            trained=run_jobs(root,jobs,data,identity,'training_'+variant)
            # 输出目录已有训练工件，仅新增独立evaluation job，不覆盖训练。
            evaluations=[]
            for s in SHIFTS:
                train_job=read(root/'formal'/variant/s/'train_job.json');out=Path(train_job['output'])
                evaluate=deepcopy(train_job);evaluate['kind']='evaluate'
                cfg=read(train_job['config']);raw=read(cfg['baseline']['annotation_for_offline_evaluation'])['database']
                meta=root/'inputs'/s/'evaluation_unlabeled.json'
                save(meta,dict(database={v:{k:x for k,x in row.items() if k!='annotations'} for v,row in raw.items()
                    if row['subset'].lower() in cfg['tgt_val_split']}))
                evaluate.update(evaluation_metadata=str(meta),evaluation_metadata_sha256=sha(meta),
                    evaluation_annotation=cfg['baseline']['annotation_for_offline_evaluation'],
                    evaluation_annotation_sha256=sha(cfg['baseline']['annotation_for_offline_evaluation']))
                path=out/'evaluate_job.json';save(path,evaluate);evaluations.append(path)
            result=run_jobs(root,evaluations,data,identity,'evaluation_'+variant);all_results[variant]=result
            save(root/'formal'/variant/'final_audit.json',dict(status='passed',count=6,results=result,training=trained,
                mean_map=sum(x['map'] for x in result)/6))
        for s in SHIFTS:
            hashes=[read(root/'formal'/v/s/'termination.json')['initial_state_sha256'] for v in SPEC['variants']]
            if len(set(hashes))!=1:raise ValueError('六方向三组初始化配对失败')
        save(root/'final_audit.json',dict(status='passed',count=18,results=all_results))
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),active={},count=18))
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
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN);raise SystemExit(128+signum)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prototype',type=Path);p.add_argument('--previous-ttt',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--job',type=Path)
    p.add_argument('--previous-root',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));args=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.job:worker(args)
    else:main(args)
