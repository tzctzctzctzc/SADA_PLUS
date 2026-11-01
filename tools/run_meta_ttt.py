"""元TTT显式阶段入口；默认不运行GPU，无自动恢复/探索或隐式训练队列。"""
import argparse
from collections import OrderedDict
from copy import deepcopy
import json,os,random,signal,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.meta_ttt import SPEC


def binding(job):
    return {key:sha(job[key]) for key in ('config','checkpoint','source_manifest','cache')}


def verify(job):
    if job['specification']!=SPEC or job['code']!=code_identity(): raise ValueError('代码/协议变动')
    if binding(job)!=job['input_sha256']: raise ValueError('冻结输入SHA改变')
    if job.get('prepared') and sha(job['prepared'])!=job['prepared_sha256']:
        raise ValueError('准备权重SHA改变')
    for key in ['resume_checkpoint','resume_best']:
        if job.get(key) and sha(job[key])!=job[key+'_sha256']:
            raise ValueError('恢复工件SHA改变: '+key)
    cfg=read(job['config']);allowed=set(cfg['src_train_split']+cfg['src_val_split'])
    if any(v['subset'].lower() not in allowed for v in read(job['source_manifest'])['database'].values()):
        raise ValueError('源准备manifest混入目标视频')


def load(job,device):
    import torch
    from libs.models.SADA import make_meta_arch
    from tools.baseline_engine import tensor_digest
    from tools.meta_ttt import configure,Recovery
    cfg=read(job['config']);base=torch.load(job['checkpoint'],map_location='cpu')
    if base['config']!=cfg: raise ValueError('checkpoint不是原完整配置')
    runtime=deepcopy(cfg)
    runtime['prototype_capacity']['prepared_cache']=job['cache']
    if sha(job['cache'])!=cfg['prototype_capacity']['prepared_cache_sha256']: raise ValueError('原Proto缓存不一致')
    torch.manual_seed(123)
    if device.startswith('cuda'):torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',runtime,device=device).to(device)
    model.load_state_dict(base['ema_state_dict'],strict=True);model.eval()
    initial=tensor_digest(model.state_dict());reference=deepcopy(model).eval().requires_grad_(False)
    encoder=configure(model)
    if job.get('execution_backend'):
        from tools.meta_ttt_speed import BACKEND,install
        if job['execution_backend']!=BACKEND:raise ValueError('未知执行后端')
        install(encoder)
    with torch.random.fork_rng(devices=[0] if device.startswith('cuda') else []):
        torch.manual_seed(123)
        decoder=Recovery(cfg['model']['fpn_dim'],len(model.fpn_strides),SPEC['decoder_hidden']).to(device)
    decoder_initial=tensor_digest(decoder.state_dict())
    if job.get('prepared'):
        prepared=torch.load(job['prepared'],map_location='cpu')
        if (prepared['specification']!=SPEC or prepared['parent_checkpoint_sha256']!=job['input_sha256']['checkpoint']
            or prepared['config']!=cfg or prepared['arm']!=job['arm']):raise ValueError('准备权重身份不匹配')
        model.load_state_dict(prepared['state_dict'],strict=True)
        decoder.load_state_dict(prepared['decoder'],strict=True)
    return model,reference,encoder,decoder,cfg,initial,decoder_initial


def data(job,cfg,split,training,manifest=None):
    from tools.baseline_engine import dataset
    runtime=deepcopy(cfg)
    runtime['dataset']['feat_folder']=job['features']
    return dataset(runtime,manifest or job['source_manifest'],split,training,False)


def item(prediction):
    return {k:prediction[k].tolist() if k!='video_id' else prediction[k]
            for k in ('video_id','segments','scores','labels')}


def source_map(job,model,reference,encoder,decoder,cfg,steps):
    from tools.meta_ttt import predict,clean_video
    from tools.run_tta import evaluate_predictions
    ds=data(job,cfg,cfg['src_val_split'],False)
    predictions=[item(predict(model,reference,encoder,decoder,clean_video(ds[i]),steps)) for i in range(len(ds))]
    return evaluate_predictions(predictions,job['source_manifest'],cfg['src_val_split'])['map']


def train(job, acceptance=False):
    import numpy as np
    import torch
    from tools.baseline_engine import checkpoint,loader,tensor_digest
    from tools.meta_ttt import episode,choose_checkpoint
    model,ref,enc,dec,cfg,initial,head_initial=load(job,'cuda:0')
    # 原Prototype等非任务分支冻结，分类/回归在源准备参与外层；测试时从未加入fast weights。
    model.cls_head_target.requires_grad_(True);model.reg_head_target.requires_grad_(True)
    base_params=list(enc.parameters())+list(model.cls_head_target.parameters())+list(model.reg_head_target.parameters())
    optimizer=torch.optim.AdamW([dict(params=base_params,lr=SPEC['outer_lr']),
        dict(params=dec.parameters(),lr=SPEC['decoder_lr'])],weight_decay=SPEC['weight_decay'])
    generator=torch.Generator().manual_seed(123)
    ds=data(job,cfg,cfg['src_train_split'],True)
    dl=loader(ds,True,generator,SPEC['source_batch'],0)
    budget=cfg['baseline']['epoch_budget']['iterations_per_epoch']
    if not len(dl):raise ValueError('源训练集为空')
    out=Path(job['output']);start_epoch=0
    if job.get('resume_checkpoint'):
        from tools.meta_ttt_speed import check_resume,restore_training
        saved=torch.load(job['resume_checkpoint'],map_location='cpu')
        start_epoch=check_resume(saved,job,cfg,initial,head_initial)
        history=saved['history'];epochs=saved['epochs']
        best_saved=torch.load(job['resume_best'],map_location='cpu')
        if (best_saved['epoch']!=choose_checkpoint(history)[0]['epoch']
                or best_saved['config']!=cfg or best_saved['parent_checkpoint_sha256']!=job['input_sha256']['checkpoint']):
            raise ValueError('旧best不对应源选择')
        checkpoint(out/'best.pt',best_saved)
        save(out/'epochs.json',epochs)
        if len(history)>1:save(out/'source_history.json',history)
        restore_training(saved,model,dec,optimizer,generator)
        print('RESUME_EPOCH_BOUNDARY',start_epoch,'NEXT',start_epoch+1,flush=True)
        del saved,best_saved
    else:
        history=[dict(epoch=0,map=source_map(job,model,ref,enc,dec,cfg,0))];epochs=[]
    if acceptance:
        frozen_initial=tensor_digest({n:p for n,p in model.named_parameters() if not p.requires_grad})
        reference_initial=tensor_digest(ref.state_dict())
    task_head_gradient=dict(classification=False,regression=False)
    def payload(epoch,latest=False):
        result=dict(config=cfg,specification=SPEC,arm=job['arm'],epoch=epoch,
            state_dict=model.state_dict(),decoder=dec.state_dict(),
            parent_checkpoint_sha256=job['input_sha256']['checkpoint'],
            initial_model_sha256=initial,initial_decoder_sha256=head_initial,
            adaptation_enabled=epoch>0,history=history,epochs=epochs)
        if latest:result.update(optimizer=optimizer.state_dict(),rng=dict(
            python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state(0),source_loader=generator.get_state()))
        return result
    if not start_epoch:checkpoint(out/'best.pt',payload(0))
    for epoch in range(start_epoch+1,(1 if acceptance else SPEC['max_epochs'])+1):
        it=iter(dl);losses=[];gradients=[];hypergradients=[]
        for step in range(budget):
            try: batch=next(it)
            except StopIteration:it=iter(dl);batch=next(it)
            optimizer.zero_grad(set_to_none=True)
            for source in batch:
                loss=episode(model,ref,enc,dec,source,job['arm'],((epoch-1)*budget+step)*SPEC['inner_steps'])/len(batch)
                if not torch.isfinite(loss):raise ValueError('元目标非有限')
                loss.backward();losses.append(float(loss.detach()))
            # 保留每张量原float32平方求和与Python double累计的数值，合并为一次GPU同步。
            squares=torch.stack([p.grad.square().sum() for p in dec.parameters() if p.grad is not None])
            hg=sum(squares.detach().cpu().tolist())**.5
            for key,module in [('classification',model.cls_head_target),('regression',model.reg_head_target)]:
                task_head_gradient[key] |= any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters())
            norm=torch.nn.utils.clip_grad_norm_(base_params+list(dec.parameters()),SPEC['clip'])
            if not torch.isfinite(norm) or norm<=0 or not math_finite(hg) or hg<=0:raise ValueError('模型/恢复头梯度无效')
            optimizer.step();gradients.append(float(norm));hypergradients.append(hg)
            if step%10==0:print('TRAIN',epoch,step+1,budget,losses[-1],float(norm),hg,flush=True)
        epochs.append(dict(epoch=epoch,steps=budget,loss=sum(losses)/budget,
            gradient_norm=sum(gradients)/budget,decoder_gradient=sum(hypergradients)/budget,
            source_videos_seen=budget*SPEC['source_batch'],target_videos_seen=0,target_GT_used=False))
        save(out/'epochs.json',epochs)
        if epoch%SPEC['validate_every']==0:
            previous=choose_checkpoint(history)[0]['map']
            score=source_map(job,model,ref,enc,dec,cfg,SPEC['inner_steps'])
            history.append(dict(epoch=epoch,map=score));save(out/'source_history.json',history)
            if score>previous:checkpoint(out/'best.pt',payload(epoch))
            print('SOURCE_POST_ADAPT_MAP',epoch,score,flush=True)
        checkpoint(out/'latest.pt',payload(epoch,True))
        if acceptance:
            from tools.meta_ttt import predict,clean_video
            sample=clean_video(ds[0]);before=predict(model,ref,enc,dec,sample,SPEC['inner_steps'])
            saved=torch.load(out/'latest.pt',map_location='cpu')
            model.load_state_dict(saved['state_dict'],strict=True);dec.load_state_dict(saved['decoder'],strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy'])
            torch.set_rng_state(saved['rng']['torch_cpu']);torch.cuda.set_rng_state(saved['rng']['torch_cuda'],0)
            generator.set_state(saved['rng']['source_loader'])
            after=predict(model,ref,enc,dec,sample,SPEC['inner_steps'])
            assert all(torch.equal(before[k],after[k]) for k in ('segments','scores','labels'))
            assert all(task_head_gradient.values())
            assert frozen_initial==tensor_digest({n:p for n,p in model.named_parameters() if not p.requires_grad})
            assert reference_initial==tensor_digest(ref.state_dict())
            post=source_map(job,model,ref,enc,dec,cfg,SPEC['inner_steps'])
            return dict(status='passed',arm=job['arm'],epochs=1,steps=budget,
                source_map0=history[0]['map'],diagnostic_source_post_adapt_map=post,
                source_task_heads_gradient=task_head_gradient,decoder_gradient_seen=min(hypergradients)>0,
                exact_checkpoint_prediction_roundtrip=True,optimizer_and_all_rng_restored=True,
                frozen_parameters_and_reference_unchanged=True,target_GT_used=False,target_videos_seen=0,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        if epoch%SPEC['validate_every']==0 and choose_checkpoint(history)[1]:break
    best,stop=choose_checkpoint(history)
    if not stop:raise ValueError('未达到合法停止')
    result=dict(status='passed',best_epoch=best['epoch'],stop_epoch=epoch,history=history,
        initial_model_sha256=initial,initial_decoder_sha256=head_initial,target_GT_used=False)
    save(out/'termination.json',result)
    return result


def math_finite(value):
    import math
    return math.isfinite(value)


def probe(job,device):
    import torch
    from tools.baseline_engine import tensor_digest
    from tools.meta_ttt import (episode,clean_video,detection_packet,detection_loss,
        reference_targets,recovery_loss,predict)
    from tools.tta_core import predict_full
    from tools.ttt_recovery import source_detection_loss
    model,ref,enc,dec,cfg,initial,head_initial=load(job,device)
    ds=data(job,cfg,cfg['src_train_split'],True)
    # 使用真实训练crop，长度仍是原2304上限，不偷换短合成序列代替全图验收。
    source=ds[0];clean=clean_video(source)
    ordinary=predict_full(model,clean);zero=predict(model,ref,enc,dec,clean,0)
    assert all(torch.equal(ordinary[k],zero[k]) for k in ('segments','scores','labels'))
    old=source_detection_loss(model,source)
    new=detection_loss(model,detection_packet(model,enc,clean),source)
    torch.testing.assert_close(old,new,rtol=0,atol=0)
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    loss=episode(model,ref,enc,dec,source,'meta')
    parameters=list(enc.parameters())+list(dec.parameters())
    grads=torch.autograd.grad(loss,parameters)
    assert all(torch.isfinite(g).all() for g in grads)
    nenc=len(list(enc.parameters()))
    enc_norm=sum(float(g.square().sum()) for g in grads[:nenc])**.5
    hyper_norm=sum(float(g.square().sum()) for g in grads[nenc:])**.5
    assert enc_norm>0 and hyper_norm>0
    per_decoder=[];at=nenc
    for module in dec.decoders:
        n=len(list(module.parameters()));value=sum(float(g.square().sum()) for g in grads[at:at+n])**.5
        assert value>0;per_decoder.append(value);at+=n
    assert tensor_digest(model.state_dict())==initial and tensor_digest(ref.state_dict())==initial
    # 真正执行一次外层优化，不能把仅autograd计算当作参数实际更新。
    optimizer=torch.optim.AdamW(parameters,lr=SPEC['outer_lr'])
    for parameter,gradient in zip(parameters,grads):parameter.grad=gradient
    optimizer.step();del grads,loss
    assert tensor_digest(model.state_dict())!=initial and tensor_digest(dec.state_dict())!=head_initial
    assert tensor_digest(ref.state_dict())==initial
    frozen_before={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    a=predict(model,ref,enc,dec,clean,SPEC['inner_steps'])
    b=predict(model,ref,enc,dec,clean,SPEC['inner_steps'])
    assert all(torch.equal(a[k],b[k]) for k in ('segments','scores','labels'))
    assert all(torch.equal(p,frozen_before[n]) for n,p in model.named_parameters() if n in frozen_before)
    base=torch.load(job['checkpoint'],map_location='cpu')
    model.load_state_dict(base['ema_state_dict'],strict=True)
    reset=predict_full(model,clean)
    assert all(torch.equal(ordinary[k],reset[k]) for k in ('segments','scores','labels'))
    result=dict(status='passed',input_tokens=source['feats'].shape[-1],source_annotations=len(source['labels']),
        encoder_parameters=sum(p.numel() for p in enc.parameters()),
        sgp_parameters=sum(p.numel() for p in enc.backbone.parameters()),
        decoder_parameters=sum(p.numel() for p in dec.parameters()),encoder_gradient=enc_norm,
        decoder_meta_gradient=hyper_norm,per_scale_decoder_meta_gradient=per_decoder,
        old_source_loss_exact=True,original_inference_exact=True,second_order=True,
        real_outer_update=True,reference_unchanged=True,episodic_repeat_exact=True,
        frozen_parameters_unchanged=True,full_reset_inference_exact=True,
        target_videos_seen=0,target_GT_used=False,initial_model_sha256=initial,initial_decoder_sha256=head_initial,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device.startswith('cuda') else 0.)
    return result


def calibrate(job):
    model,ref,enc,dec,cfg,_,_=load(job,'cuda:0')
    import torch
    saved=torch.load(job['prepared'],map_location='cpu')
    return dict(status='passed',source_only=True,arm=job['arm'],shift=job['shift'],
        source_maps={str(s):source_map(job,model,ref,enc,dec,cfg,s if saved['adaptation_enabled'] else 0)
            for s in SPEC['calibration_steps']},prepared_sha256=job['prepared_sha256'],target_GT_used=False)


def selection(calibrations,arm):
    from tools.meta_ttt import choose_steps
    if len(calibrations)!=6 or {read(p)['result']['shift'] for p in calibrations}!=set('S%d'%i for i in range(1,7)):
        raise ValueError('需要精确六份完整源校准')
    candidates=[dict(steps=s,source_maps={}) for s in SPEC['calibration_steps']]
    for path in calibrations:
        proof=read(path);r=proof['result']
        audit(Path(proof['job_path']))
        if proof['stage']!='calibrate':raise ValueError('必须源校准阶段凭证')
        if r['arm']!=arm or not r['source_only'] or r['target_GT_used']:raise ValueError('校准身份非法')
        for row in candidates:row['source_maps'][r['shift']]=r['source_maps'][str(row['steps'])]
    chosen=choose_steps(candidates)
    return dict(schema=SPEC['schema'],arm=arm,steps=chosen['steps'],candidates=candidates,
        calibration_sha256={str(p):sha(p) for p in calibrations},target_results_seen=False)


def evaluate(job):
    import torch
    from tools.meta_ttt import predict,clean_video
    from tools.run_tta import evaluate_predictions
    if sha(job['selection'])!=job['selection_sha256']:raise ValueError('源校准凭证改变')
    chosen=read(job['selection'])
    if chosen['target_results_seen'] or chosen['arm']!=job['arm']:raise ValueError('选择包含目标结果')
    if sha(job['evaluation_manifest'])!=job['evaluation_manifest_sha256']:raise ValueError('目标无标签manifest改变')
    if any(v.get('annotations') for v in read(job['evaluation_manifest'])['database'].values()):
        raise ValueError('目标评估manifest包含GT')
    model,ref,enc,dec,cfg,_,_=load(job,'cuda:0');saved=torch.load(job['prepared'],map_location='cpu')
    ds=data(job,cfg,cfg['tgt_val_split'],False,job['evaluation_manifest'])
    steps=chosen['steps'] if job['mode']=='adapt' and saved['adaptation_enabled'] else 0
    rows=[]
    for i in range(len(ds)):rows.append(item(predict(model,ref,enc,dec,clean_video(ds[i]),steps)))
    path=Path(job['output'])/'predictions.json'
    # 真正fsync落盘后，才打开GT路径进行离线评估。
    with path.open('x') as f:
        json.dump(rows,f,allow_nan=False);f.flush();os.fsync(f.fileno())
    prediction_sha=sha(path)
    scores=evaluate_predictions(read(path),job['annotation'],cfg['tgt_val_split'])
    return dict(status='passed',steps=steps,predictions_sha256=prediction_sha,metrics=scores,
        videos=len(rows),target_GT_for_adaptation=False,predictions_saved_before_GT_evaluation=True)


def worker(job_path,stage,device):
    import torch
    job=read(job_path);verify(job)
    if stage!=job['stage']:raise ValueError('阶段与job身份不符')
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=False)
    guard=None
    if device.startswith('cuda'):
        from tools.baseline_gpu import first_cuda_guard
        guard=first_cuda_guard()
    os.chdir(job['data_root']);torch.set_num_threads(8)
    random.seed(123)
    import numpy as np
    np.random.seed(123)
    started=time.time()
    if stage=='probe':result=probe(job,device)
    elif stage in ('train','training_probe'):result=train(job,acceptance=stage=='training_probe')
    elif stage=='calibrate':result=calibrate(job)
    elif stage=='evaluate':result=evaluate(job)
    else:raise ValueError(stage)
    files={str(p.name):sha(p) for p in out.iterdir() if p.is_file()}
    save(out/'complete.json',dict(stage=stage,result=result,code=job['code'],specification=SPEC,
        job_path=str(job_path),job_sha256=sha(job_path),input_sha256=binding(job),artifacts=files,guard=guard,
        elapsed_seconds=time.time()-started))


def audit(job_path):
    job=read(job_path);verify(job);out=Path(job['output']);receipt=read(out/'complete.json')
    if receipt['job_sha256']!=sha(job_path) or receipt['input_sha256']!=binding(job):raise ValueError('工件身份变化')
    if receipt['code']!=job['code'] or receipt['specification']!=SPEC:raise ValueError('凭证代码/协议变化')
    if receipt['result']['status']!='passed':raise ValueError('阶段未通过')
    if any(sha(out/name)!=value for name,value in receipt['artifacts'].items()):raise ValueError('产物SHA改变')
    if receipt['stage']=='train':
        from tools.meta_ttt import choose_checkpoint
        epochs=read(out/'epochs.json');termination=read(out/'termination.json');history=termination['history']
        if [r['epoch'] for r in epochs]!=list(range(1,termination['stop_epoch']+1)):raise ValueError('训练历史不连续')
        if any(choose_checkpoint(history[:i])[1] for i in range(1,len(history))):raise ValueError('非首次合法停止')
        if choose_checkpoint(history)[0]['epoch']!=termination['best_epoch']:raise ValueError('源选模不一致')
    return dict(status='passed',stage=receipt['stage'],job=str(job_path),complete_sha256=sha(out/'complete.json'))


def dispatch(job_path,stage,gpu):
    from tools.baseline_gpu import slot
    from tools.run_sada_baseline import prepare_file_limit
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('调度父必须CUDA空')
    job=read(job_path);verify(job);prepare_file_limit()
    # 独占公共物理锁期间仅启动本worker，外部共享进程不检查命令、不发送信号。
    with slot(gpu,job['data_root'],job['admission_root'],24576,job['code']['commit']) as lease:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['uuid'],PYTHONDONTWRITEBYTECODE='1',
            SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONHASHSEED='123',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        log=Path(job['output']).with_suffix('.'+stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as stream:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker',
                '--job',str(job_path),'--stage',stage],env=env,cwd=CODE,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
            def stop(signum,frame):
                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                raise SystemExit(128+signum)
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            save(log.with_suffix('.launch.json'),dict(parent_pid=os.getpid(),worker_pid=process.pid,
                command=process.args,lease=lease,job_sha256=sha(job_path),stage=stage))
            code=process.wait()
            if code:raise RuntimeError('阶段失败，现场保留: '+str(log))
    return audit(job_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['worker','dispatch','audit','select','make-job'])
    parser.add_argument('--job',type=Path);parser.add_argument('--stage',choices=['probe','train','training_probe','calibrate','evaluate'],default='probe')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--gpu',type=int,choices=range(4))
    parser.add_argument('--calibrations',nargs='+',type=Path);parser.add_argument('--arm',choices=SPEC['arms']);parser.add_argument('--output',type=Path)
    parser.add_argument('--inputs',type=Path,help='显式机器路径JSON，不自动猜测/换数据')
    parser.add_argument('--shift',choices=['S%d'%i for i in range(1,7)])
    parser.add_argument('--prepared',type=Path);parser.add_argument('--selection',type=Path)
    parser.add_argument('--mode',choices=SPEC['target_modes'],default='adapt')
    args=parser.parse_args()
    if args.command=='make-job':
        if args.job.exists() or args.output.exists():raise FileExistsError('新job/输出必须是全新路径')
        paths=read(args.inputs)
        job=dict(paths,stage=args.stage,shift=args.shift,arm=args.arm or 'meta',mode=args.mode,
            code=code_identity(),specification=SPEC,output=str(args.output.resolve()))
        job['input_sha256']=binding(job)
        if args.prepared:
            job.update(prepared=str(args.prepared.resolve()),prepared_sha256=sha(args.prepared))
        if args.stage in ('calibrate','evaluate') and not args.prepared:raise ValueError('需要源选准备权重')
        if args.stage=='evaluate':
            job.update(selection=str(args.selection.resolve()),selection_sha256=sha(args.selection),
                evaluation_manifest_sha256=sha(job['evaluation_manifest']))
        verify(job);save(args.job,job);return
    if args.command=='select':
        if args.output.exists():raise FileExistsError(args.output)
        save(args.output,selection(args.calibrations,args.arm));return
    if args.command=='worker':worker(args.job.resolve(),args.stage,args.device)
    elif args.command=='dispatch':print(json.dumps(dispatch(args.job.resolve(),args.stage,args.gpu)))
    else:print(json.dumps(audit(args.job.resolve())))


if __name__=='__main__': main()
