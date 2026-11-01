"""两种完整实例联合学习，双卡锁准入、完整源mAP选模、落盘后目标评估。"""
import argparse
from copy import deepcopy
import itertools,json,math,os,random,signal,subprocess,sys,time,shutil
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,assert_no_target_gt
from tools.run_sada_baseline import acquire,prepare_file_limit
from tools.joint_instance import SPEC,JointHead,select
from tools.run_tta import load_model,evaluate_predictions
ACTIVE={};SHIFTS=tuple('S%d'%i for i in range(1,7))

def prediction_item(p):
    return dict(video_id=p['video_id'],segments=p['segments'].tolist(),scores=p['scores'].tolist(),labels=p['labels'].tolist())

def validate(model,head,cfg,epoch):
    import torch
    from tools.baseline_engine import dataset
    from tools.tta_core import unlabeled
    from tools.joint_instance import predict
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False)
    model.eval();head.eval();items=[]
    with torch.no_grad():
        for i in range(len(ds)):items.append(prediction_item(predict(model,head,unlabeled(ds[i]),epoch)))
    return evaluate_predictions(items,cfg['dataset']['json_file'],cfg['src_val_split'])['map']

def train(job,probe=False):
    import numpy as np
    import torch
    from tools.baseline_engine import dataset,loader,move_to_device,tensor_digest,checkpoint,controls
    from tools.masked_uda import trainable_parameters,update_teacher
    from tools.tta_core import dense,unlabeled,predict_full
    from tools.ttt_recovery import temporal_mask,seed_for
    from tools.joint_instance import source_proposals,source_objective,pseudo_instances,target_objective,predict
    cfg=read(job['config']);out=Path(job['output']);variant=job['variant']
    model=load_model(cfg,job['checkpoint']);initial=tensor_digest(model.state_dict())
    named=trainable_parameters(model)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(123);torch.cuda.manual_seed(123);head=JointHead(cfg['model']['fpn_dim'],len(model.fpn_strides)).cuda()
    head_initial=tensor_digest(head.state_dict());teacher=deepcopy(model).eval();ema_head=deepcopy(head).eval()
    teacher.requires_grad_(False);ema_head.requires_grad_(False)
    optimizer=torch.optim.AdamW([dict(params=list(named.values()),lr=1e-5),dict(params=head.parameters(),lr=1e-4)],weight_decay=.01)
    generators={k:torch.Generator().manual_seed(123+i*1000003) for i,k in enumerate(('source','target'))}
    src=dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],True)
    tgt=dataset(cfg,job['target_manifest'],cfg['tgt_train_split'],True,True)
    sl=loader(src,True,generators['source'],2,0);tl=loader(tgt,True,generators['target'],2,0)
    budget=cfg['baseline']['epoch_budget']['iterations_per_epoch'];assert budget>=max(len(sl),len(tl))
    history=[dict(epoch=0,map=validate(teacher,ema_head,cfg,0))];epochs=[]
    assert abs(history[0]['map']-read(Path(job['checkpoint']).parent/'termination.json')['decision']['best_map'])<1e-6
    signals=dict(source_backbone=False,source_region_encoder=False,target_backbone=False,target_classifier=False,target_regressor=False,target_region_encoder=False)
    def norm(loss,parameters):
        gs=torch.autograd.grad(loss,list(parameters),allow_unused=True,retain_graph=True)
        return sum(float(g.detach().square().sum()) for g in gs if g is not None)**.5
    if probe:
        probe_video=unlabeled(src[0]);a=predict_full(model,probe_video);b=predict(model,head,probe_video,0)
        assert all(torch.equal(a[k],b[k]) for k in ('segments','scores','labels'))
        head_reset={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    def best_payload(epoch):
        return dict(config=cfg,ema_state_dict=teacher.state_dict(),head_state_dict=ema_head.state_dict(),epoch=epoch,
            specification=SPEC,variant=variant,parent_checkpoint_sha256=job['checkpoint_sha256'],initial_state_sha256=initial,initial_head_sha256=head_initial)
    if not probe:checkpoint(out/'best.pt',best_payload(0))
    max_epoch=3 if probe else 30
    for epoch in range(1,max_epoch+1):
        if shutil.disk_usage(out).free<2*2**30:raise RuntimeError('剩余磁盘低于2GiB，保留工件停止，不删除旧结果')
        # 真实重新遍历loader；不使用cycle缓存整轮crop。
        si=iter(sl);ti=iter(tl);reports=[]
        for group,index in zip(optimizer.param_groups,range(2)):group['lr']=(1e-5 if index==0 else 1e-4)*(.1 if epoch>15 else 1.)
        model.train();head.train();model.train_label_smoothing=0.
        for step in range(budget):
            global_step=(epoch-1)*budget+step;random.seed(123+global_step);np.random.seed(123+global_step)
            try:sv=next(si)
            except StopIteration:si=iter(sl);sv=next(si)
            try:tv=next(ti)
            except StopIteration:ti=iter(tl);tv=next(ti)
            sv=move_to_device(sv,'cuda:0');tv=move_to_device(tv,'cuda:0');assert_no_target_gt(tv)
            optimizer.zero_grad(set_to_none=True)
            ret=model(sv,tv,mode='train',domain='src',curr_epoch=cfg['train_cfg']['warmup_adv']+epoch)
            assert not ret[2].count_nonzero()
            losses=ret[0];native=losses['downstream_loss']*cfg['model']['loss_weights']['weight_downstream']
            if not torch.isfinite(native):raise ValueError('原检测/Prototype损失非有限')
            native.backward();native_value=float(native.detach());del native,ret
            row=dict(native=native_value,source_region=0.,source_regions=0,source_region_positive=0,source_region_background=0,pseudo_instances=0,pseudo_native_points=0,target=0.,target_annotations=0,target_GT_used=False)
            # 单视频反传释放图；仍只在整个batch所有损失之后做一次optimizer step。
            model.eval()
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(123+global_step);torch.cuda.manual_seed(123+global_step)
                for v in sv:
                    packet=dense(model,unlabeled(v));boxes,base=source_proposals(packet,v,global_step)
                    outputs=head(packet,boxes,base)
                    loss,metrics=source_objective(outputs,v['segments'],v['labels'])
                    if not torch.isfinite(loss):raise ValueError('完整源实例损失非有限')
                    if probe and step in (1,10,budget-1):
                        signals['source_backbone'] |= norm(loss,model.backbone_source.stem[-1].parameters())>1e-8
                        signals['source_region_encoder'] |= norm(loss,head.stages[0].encoder.parameters())>1e-8
                    (loss/len(sv)).backward();row['source_region']+=float(loss.detach())/len(sv)
                    for k,vv in metrics.items():row[k]+=vv
                    del loss,outputs,packet
                active=variant=='ema_joint' and epoch>=3
                if active:
                    for v in tv:
                        clean=unlabeled(v);pseudo=pseudo_instances(teacher,ema_head,clean,epoch)
                        strong=unlabeled(clean);mask=temporal_mask(strong['feats'].shape[-1],seed_for(v['metadata']['video_id'],global_step),.2,8).to(strong['feats'].device)
                        strong['feats']=strong['feats'].masked_fill(mask[None],0.)
                        loss,metrics=target_objective(model,head,strong,pseudo)
                        if not torch.isfinite(loss):raise ValueError('完整目标实例损失非有限')
                        if probe and len(pseudo['boxes']) and step in (1,10,budget-1):
                            for key,params in [('target_backbone',model.backbone_target.stem[-1].parameters()),('target_classifier',model.cls_head.parameters()),('target_regressor',model.reg_head.parameters()),('target_region_encoder',head.stages[0].encoder.parameters())]:signals[key] |= norm(loss,params)>1e-8
                        weight=min((epoch-2)/5.,1.)
                        (weight*loss/len(tv)).backward();row['target']+=float(loss.detach())/len(tv)
                        for k,vv in metrics.items():row[k]+=vv
                        del loss,pseudo
            parameters=list(named.values())+list(head.parameters());grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(grad) or grad<=0:raise ValueError('联合实际梯度无效')
            optimizer.step();update_teacher(teacher,model)
            assert all(p.grad is None for p in teacher.parameters()) and all(p.grad is None for p in ema_head.parameters())
            with torch.no_grad():
                for a,b in zip(ema_head.parameters(),head.parameters()):a.mul_(.99).add_(b,alpha=.01)
            model.train();row['gradient_norm']=float(grad);reports.append(row)
            if step%20==0:print('TRAIN',variant,epoch,step+1,budget,json.dumps(row),flush=True)
        aggregate={k:sum(r[k] for r in reports)/budget for k in reports[0] if isinstance(reports[0][k],(int,float))}
        aggregate.update(epoch=epoch,steps=budget,target_annotations=0,target_GT_used=False);epochs.append(aggregate);save(out/'epochs.json',epochs)
        if probe and epoch<3:continue
        if probe:
            assert signals['source_backbone'] and signals['source_region_encoder'],signals
            if variant=='ema_joint':assert all(signals.values()),signals
            assert tensor_digest(model.state_dict())!=initial and tensor_digest(head.state_dict())!=head_initial
            assert tensor_digest(teacher.state_dict())!=initial and tensor_digest(ema_head.state_dict())!=head_initial
            # 精确恢复所有参数和控制之后，普通原模型推理必须逐值相同。
            parent=torch.load(job['checkpoint'],map_location='cpu');model.load_state_dict(parent['ema_state_dict'],strict=True);model.eval();head.load_state_dict(head_reset,strict=True);head.eval()
            assert tensor_digest(model.state_dict())==initial and tensor_digest(head.state_dict())==head_initial
            z=predict(model,head,probe_video,0);assert all(torch.equal(z[k],a[k]) for k in ('segments','scores','labels'))
            return dict(status='passed',signals=signals,ordinary_epoch0_inference_exact=True,updated_student_and_head=True,updated_independent_teacher_and_head=True,restore_digest_and_original_prediction_exact=True,initial_state_sha256=initial,initial_head_sha256=head_initial,source_map0=history[0]['map'],steps=budget,source_regions=sum(r['source_regions'] for r in reports),pseudo_instances=sum(r['pseudo_instances'] for r in reports),target_GT_used=False,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        if epoch%2==0:
            score=validate(teacher,ema_head,cfg,epoch);previous=select(history)[0]['map'];history.append(dict(epoch=epoch,map=score));save(out/'source_history.json',history)
            if score>previous:checkpoint(out/'best.pt',best_payload(epoch))
            print('SOURCE_VALIDATION',epoch,score,flush=True)
        checkpoint(out/'latest.pt',dict(config=cfg,specification=SPEC,variant=variant,epoch=epoch,history=history,epochs=epochs,
            state_dict=model.state_dict(),ema_state_dict=teacher.state_dict(),head_state_dict=head.state_dict(),ema_head_state_dict=ema_head.state_dict(),optimizer=optimizer.state_dict(),model_controls=controls(model),ema_controls=controls(teacher),
            parent_checkpoint_sha256=job['checkpoint_sha256'],initial_state_sha256=initial,initial_head_sha256=head_initial,
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0),loaders={k:g.get_state() for k,g in generators.items()})))
        if epoch%2==0 and select(history)[1]:break
    chosen,stopped=select(history);assert stopped
    save(out/'termination.json',dict(status='passed',best_epoch=chosen['epoch'],stop_epoch=epoch,source_history=history,epochs=epochs,initial_state_sha256=initial,initial_head_sha256=head_initial,target_GT_used=False,specification=SPEC))
    return dict(status='passed',best_epoch=chosen['epoch'],stop_epoch=epoch,initial_state_sha256=initial,initial_head_sha256=head_initial,target_GT_used=False,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)

def worker(args):
    import torch
    from tools.baseline_gpu import first_cuda_guard
    from tools.baseline_engine import dataset
    from tools.tta_core import unlabeled
    from tools.joint_instance import predict
    job=read(args.job);out=Path(job['output']);cfg=read(job['config'])
    assert code_identity()==job['code'] and job['specification']==SPEC
    for field in ('config','checkpoint','source_manifest','target_manifest','evidence'):assert sha(job[field])==job[field+'_sha256']
    assert not any('annotations' in x for x in read(job['target_manifest'])['database'].values())
    guard=first_cuda_guard();os.chdir(args.data_root)
    if job['kind'] in ('train','probe'):result=train(job,job['kind']=='probe')
    else:
        for k in ('evaluation_metadata','evaluation_annotation'):assert sha(job[k])==job[k+'_sha256']
        assert sha(out/'train_complete.json')==job['train_complete_sha256'] and sha(out/'best.pt')==read(out/'train_complete.json')['artifacts']['best.pt']
        saved=torch.load(out/'best.pt',map_location='cpu');assert saved['specification']==SPEC and saved['parent_checkpoint_sha256']==job['checkpoint_sha256']
        model=load_model(cfg,job['checkpoint']);model.load_state_dict(saved['ema_state_dict'],strict=True)
        head=JointHead(cfg['model']['fpn_dim'],len(model.fpn_strides)).cuda().eval();head.load_state_dict(saved['head_state_dict'],strict=True)
        ds=dataset(cfg,job['evaluation_metadata'],cfg['tgt_val_split'],False,True);items=[]
        for i in range(len(ds)):
            items.append(prediction_item(predict(model,head,unlabeled(ds[i]),saved['epoch'])));print('EVAL',i+1,len(ds),flush=True)
        save(out/'predictions.json',items);result=evaluate_predictions(items,job['evaluation_annotation'],cfg['tgt_val_split']);result.update(status='passed',videos=len(ds),target_GT_used=False)
    names=('epochs.json','source_history.json','best.pt','latest.pt','termination.json','predictions.json')
    save(out/('train_complete.json' if job['kind']=='train' else 'complete.json'),dict(status='passed',result=result,job_sha256=sha(args.job),code=job['code'],guard=guard,guard_sha256=sha(guard),artifacts={n:sha(out/n) for n in names if (out/n).exists()}))

def audit(path):
    job=read(path);out=Path(job['output']);cp=out/('train_complete.json' if job['kind']=='train' else 'complete.json');r=read(cp)
    assert r['status']=='passed' and r['job_sha256']==sha(path) and r['code']==job['code'] and job['specification']==SPEC
    for f in ('config','checkpoint','source_manifest','target_manifest','evidence','evaluation_metadata','evaluation_annotation'):
        if f in job:assert sha(job[f])==job[f+'_sha256']
    for n,h in r['artifacts'].items():assert sha(out/n)==h
    assert sha(r['guard'])==r['guard_sha256'];g=read(r['guard']);lease=read(g['lease']);assert sha(g['lease'])==g['lease_sha256'] and lease['commit']==job['code']['commit']
    assert g['observed']['uuid']==lease['observed']['uuid'] and g['observed']['free_mib']>=12288
    if job['kind']!='probe':
        t=read(out/'termination.json');best,stop=select(t['source_history']);assert stop and best['epoch']==t['best_epoch']
        assert [x['epoch'] for x in t['epochs']]==list(range(1,t['stop_epoch']+1))
        assert not any(select(t['source_history'][:i])[1] for i in range(1,len(t['source_history'])))
        assert all(x['target_annotations']==0 and not x['target_GT_used'] for x in t['epochs'])
        if job['kind']=='evaluate':assert sha(out/'train_complete.json')==job['train_complete_sha256']
    return dict(r['result'],kind=job['kind'],shift=job['shift'],variant=job['variant'],complete_sha256=sha(cp),job=str(path))

def run_jobs(root,jobs,data,identity,phase):
    pending=sorted(jobs,key=lambda p:read(read(p)['config'])['baseline']['epoch_budget']['iterations_per_epoch'],reverse=True);done=[]
    while pending or ACTIVE:
        for key,t in list(ACTIVE.items()):
            if t['process'].poll() is None:continue
            t['context'].__exit__(None,None,None);del ACTIVE[key]
            if t['process'].returncode:raise RuntimeError('实例联合任务失败:'+key)
            done.append(audit(t['job']))
        for path in list(pending):
            found=acquire([0,1],data,root,12288,identity['commit'])
            if not found:break
            context,lease=found;j=read(path);out=Path(j['output'])
            env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],TMPDIR=str(root/'tmp'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (out/(j['kind']+'.log')).open('x') as stream:p=subprocess.Popen([sys.executable,'-B',str(CODE/'tools/run_joint_instance.py'),'--job',str(path),'--data-root',str(data)],cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            ACTIVE[str(out.relative_to(root))]=dict(process=p,context=context,job=path,gpu=lease['index']);pending.remove(path)
        save(root/'state.json',dict(phase=phase,pid=os.getpid(),completed=len(done),total=len(jobs),active={k:dict(pid=t['process'].pid,gpu=t['gpu']) for k,t in ACTIVE.items()}))
        if pending or ACTIVE:time.sleep(3)
    return done

def make_job(root,relative,kind,s,variant,base,identity):
    entry=base/'formal'/s/'prototype';cfg=read(entry/'config.json');out=root/relative;out.mkdir(parents=True,exist_ok=False)
    j=dict(kind=kind,shift=s,variant=variant,output=str(out),code=identity,specification=SPEC,config=str(entry/'config.json'),checkpoint=str(entry/'best.pt'),source_manifest=cfg['dataset']['json_file'],target_manifest=cfg['baseline']['target_manifest'],evidence=cfg['prototype_capacity']['prepared_cache'])
    for k in ('config','checkpoint','source_manifest','target_manifest','evidence'):j[k+'_sha256']=sha(j[k])
    p=out/(kind+'_job.json');save(p,j);return p

def main(args):
    root=args.output.resolve();base=args.prototype.resolve();identity=code_identity();data=args.data_root.resolve()
    root.mkdir(parents=True,exist_ok=False);(root/'tmp').mkdir();save(root/'runtime_resources.json',prepare_file_limit())
    save(root/'plan.json',dict(specification=SPEC,code=identity,formal_count=12,prototype=str(base),prototype_final_sha256=sha(base/'final_audit.json'),exploratory=True,controls='identical architecture/base/head initialization/native losses/source-target flow/budget; target pseudo task only differs',checkpoint_storage='full fp32 best and latest only; preserve prior artifacts',resume='refuse existing root; no blind restart of partial matrix'))
    try:
        probes=[make_job(root,'probe/'+v,'probe','S1',v,base,identity) for v in SPEC['variants']]
        reports=run_jobs(root,probes,data,identity,'real_gpu_acceptance')
        assert len({x['initial_state_sha256'] for x in reports})==len({x['initial_head_sha256'] for x in reports})==1
        save(root/'acceptance.json',dict(status='passed',results=reports,code=identity))
        # 完整真实GPU验收后，父进程CPU独立再次审计才进入正式矩阵。
        accepted=[audit(p) for p in probes];save(root/'monitor/acceptance_independent_audit.json',dict(status='passed',count=2,results=accepted))
        groups={}
        for v in SPEC['variants']:
            jobs=[make_job(root,'formal/'+v+'/'+s,'train',s,v,base,identity) for s in SHIFTS]
            trained=run_jobs(root,jobs,data,identity,'training_'+v);evals=[]
            for p in jobs:
                j=read(p);out=Path(j['output']);cfg=read(j['config']);meta=root/'inputs'/j['shift']/'evaluation_unlabeled.json'
                prior=Path('/root/autodl-tmp/vlm/prototype_masked_uda18_cd9402f_1101/inputs')/j['shift']/'evaluation_unlabeled.json'
                save(meta,read(prior));j.update(kind='evaluate',train_complete_sha256=sha(out/'train_complete.json'),evaluation_metadata=str(meta),evaluation_metadata_sha256=sha(meta),evaluation_annotation=cfg['baseline']['annotation_for_offline_evaluation'],evaluation_annotation_sha256=sha(cfg['baseline']['annotation_for_offline_evaluation']))
                ep=out/'evaluate_job.json';save(ep,j);evals.append(ep)
            result=run_jobs(root,evals,data,identity,'evaluation_'+v);groups[v]=result
            save(root/'formal'/v/'final_audit.json',dict(status='passed',count=6,training=trained,results=result,mean_map=sum(x['map'] for x in result)/6))
        for s in SHIFTS:
            terms=[read(root/'formal'/v/s/'termination.json') for v in SPEC['variants']]
            assert terms[0]['initial_state_sha256']==terms[1]['initial_state_sha256'] and terms[0]['initial_head_sha256']==terms[1]['initial_head_sha256']
        save(root/'final_audit.json',dict(status='passed',count=12,results=groups));save(root/'state.json',dict(phase='complete',active={},pid=os.getpid(),count=12))
    except BaseException as e:save(root/'error.json',dict(type=type(e).__name__,message=str(e)));raise
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
    p=argparse.ArgumentParser();p.add_argument('--prototype',type=Path);p.add_argument('--output',type=Path);p.add_argument('--job',type=Path);p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));args=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.job:worker(args)
    else:main(args)
