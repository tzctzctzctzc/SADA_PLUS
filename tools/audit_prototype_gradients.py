"""已有源选EMA的梯度诊断；无优化器、无参数更新，目标GT只在第二条离线评价路径。"""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,assert_no_target_gt
from tools.prototype_refresh import CONTROL,SHIFTS


def cosine(a,b):
    denom=float(a.norm()*b.norm())
    return float((a@b)/denom) if denom>0 else None


def restore_controls(model,controls):
    import torch
    with torch.no_grad():
        for name,value in controls.items():
            current=getattr(model,name)
            if torch.is_tensor(current):current.copy_(value.to(current.device))
            elif isinstance(current,list):
                for a,b in zip(current,value):a.copy_(b.to(a.device))
            else:setattr(model,name,value)


def parameter_groups(model):
    modules=dict(shared_fpn=[model.backbone_source,model.neck_source],
        detection_classification=[model.cls_head,model.cls_head_target],
        detection_regression=[model.reg_head,model.reg_head_target],auxiliary=[model.rq3_capacity])
    params=[];positions={};groups={}
    for name,mods in modules.items():
        indices=set()
        for module in mods:
            for p in module.parameters():
                if not p.requires_grad:continue
                if id(p) not in positions:positions[id(p)]=len(params);params.append(p)
                indices.add(positions[id(p)])
        groups[name]=sorted(indices)
    return params,groups


def gradients(loss,params,groups,retain):
    import torch
    values=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain)
    norms={}
    for name,indices in groups.items():
        norms[name]=sum(float(values[i].detach().square().sum()) for i in indices if values[i] is not None)**.5
    chunks=[values[i].detach().flatten().cpu() if values[i] is not None else torch.zeros(params[i].numel())
            for i in groups['shared_fpn']]
    return dict(norms=norms,shared_vector=torch.cat(chunks))


def offline_detection_loss(model,videos,epoch):
    # 官方验证入口按单视频(src/tgt为同一视频)填充；不传入异长的验证batch。
    losses=[model(video_list_src=[v],video_list_tgt=[v],mode='validation',domain='tgt',
                  curr_epoch=epoch)[0]['downstream_loss'] for v in videos]
    return sum(losses)/len(losses)


def select_samples(ds,count,selected_cache=None):
    """按视频ID固定排序；目标选择只看缓存/特征长度，不看GT正确与否。"""
    from libs.models.SADA.prototype_capacity import fully_visible_cache
    indices=sorted(range(len(ds)),key=lambda i:ds.data_list[i]['id']);result=[]
    for i in indices:
        if selected_cache is not None:
            v=ds.data_list[i]['id']
            if not any(x['participation_selected'] for x in selected_cache.get(v,[])):continue
        sample=ds[i]
        if sample['feats'].shape[-1]>2304:continue
        if selected_cache is None:
            if not sample['labels'].numel():continue
        else:
            assert_no_target_gt([sample])
            if not fully_visible_cache(selected_cache,[sample],0.)[0][0]:continue
        result.append(sample)
        if len(result)==count:break
    return result


def worker(root,data):
    import torch
    from copy import deepcopy
    from libs.models.SADA import make_meta_arch
    from libs.trainer.da_trainer import move_to_device
    from tools.baseline_engine import dataset,tensor_digest
    from tools.baseline_gpu import first_cuda_guard
    os.chdir(data);proof=first_cuda_guard();results={}
    for s in SHIFTS:
        entry=CONTROL/'formal/control'/s;cfg=read(entry/'config.json');checkpoint=entry/'best.pt'
        done=read(entry/'complete.json');expected=done['artifacts'][str(checkpoint)]
        if sha(checkpoint)!=expected:raise ValueError('诊断checkpoint身份不符')
        saved=torch.load(checkpoint,map_location='cpu')
        if saved['epoch']!=read(entry/'termination.json')['decision']['best_epoch']:raise ValueError('不是源选EMA')
        torch.manual_seed(123);torch.cuda.manual_seed(123)
        model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').to('cuda:0')
        model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval()
        restore_controls(model,saved['ema_controls'])
        source=dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],False)
        target=dataset(cfg,cfg['baseline']['target_manifest'],cfg['tgt_train_split'],False,True)
        sources=select_samples(source,6);targets=select_samples(target,6,model.rq3_capacity.cache)
        # 离线GT仅准备独立评价副本；主辅助路径始终使用target_unlabeled样本。
        offline=dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_train_split'],False)
        gt_indices={x['id']:i for i,x in enumerate(offline.data_list)}
        params,groups=parameter_groups(model);captured={}
        original_source=model.rq3_capacity.branch.source_loss
        original_align=model.rq3_capacity.branch.prototype_alignment
        original_aux=model.rq3_capacity.forward
        def source_loss(*a,**kw):
            value=original_source(*a,**kw);captured['source_ce']=value;return value
        def align(*a,**kw):
            value=original_align(*a,**kw);captured['alignment']=value[0];return value
        def auxiliary(*a,**kw):
            value=original_aux(*a,**kw);captured['auxiliary']=value['loss'];return value
        model.rq3_capacity.branch.source_loss=source_loss
        model.rq3_capacity.branch.prototype_alignment=align;model.rq3_capacity.forward=auxiliary
        parameter_before=tensor_digest(dict(model.named_parameters()));batches=[]
        n=min(len(sources)//2,len(targets)//2,3)
        if n==0:raise ValueError('无符合长度和固定缓存的真实诊断样本')
        for batch in range(n):
            model.load_state_dict(saved['ema_state_dict'],strict=True);restore_controls(model,saved['ema_controls'])
            torch.manual_seed(123+batch);torch.cuda.manual_seed(123+batch)
            src=move_to_device(deepcopy(sources[2*batch:2*batch+2]),'cuda:0')
            tgt=move_to_device(deepcopy(targets[2*batch:2*batch+2]),'cuda:0');assert_no_target_gt(tgt)
            captured.clear();out=model(video_list_src=src,video_list_tgt=tgt,mode='train',domain='src',curr_epoch=saved['epoch'])[0]
            scale=1. if saved['epoch']<cfg['train_cfg']['warmup_adv'] else cfg['model']['loss_weights']['weight_downstream']
            task=scale*(out['downstream_loss']-captured['auxiliary'])
            if saved['epoch']>=cfg['train_cfg']['warmup_adv']:
                task=task+cfg['model']['loss_weights']['weight_adv']*out.get('loss_conditional_adv_rev',0.)
            stats=dict(batch=batch,source_videos=[v['metadata']['video_id'] for v in src],
                target_videos=[v['metadata']['video_id'] for v in tgt],selected=int(out['rq3_selected']),
                source_aux_ce=float(captured['source_ce']),target_alignment=float(captured['alignment']),
                effective_target_coefficient=scale*.1,auxiliary_path_target_gt_empty=True)
            g_base=gradients(task,params,groups,True)
            g_source=gradients(scale*captured['source_ce'],params,groups,True)
            g_target=gradients(scale*.1*captured['alignment'],params,groups,False)
            if g_target['norms']['detection_classification'] or g_target['norms']['detection_regression']:
                raise ValueError('辅助目标出现预期外的检测头直接梯度')
            captured.clear();del out,task;gc.collect()
            # 这次GT forward不计算辅助分支、没有优化器，也不保存任何模型。
            gt_cpu=[offline[gt_indices[v['metadata']['video_id']]] for v in tgt]
            for a,b in zip(gt_cpu,targets[2*batch:2*batch+2]):
                if not torch.equal(a['feats'],b['feats']):raise ValueError('GT评价与无标签路径特征不一致')
            gt=move_to_device(gt_cpu,'cuda:0')
            gt_loss=offline_detection_loss(model,gt,saved['epoch'])
            g_gt=gradients(gt_loss,params,groups,False)
            stats.update(norms={k:g['norms'] for k,g in [('SADA',g_base),('source_aux_ce',g_source),('target_alignment',g_target),('offline_target_GT_detection',g_gt)]},
                cosine_with_target_GT={k:cosine(g['shared_vector'],g_gt['shared_vector']) for k,g in
                    [('SADA',g_base),('source_aux_ce',g_source),('target_alignment',g_target)]},
                target_alignment_cosine_with_SADA=cosine(g_target['shared_vector'],g_base['shared_vector']),
                offline_target_gt_detection_loss=float(gt_loss),offline_GT_aggregation='mean_official_single_video_validation')
            batches.append(stats);del gt_loss,g_base,g_source,g_target,g_gt,src,tgt,gt;gc.collect()
            print('GRADIENT_BATCH',s,batch,json.dumps(stats['cosine_with_target_GT']),flush=True)
        model.load_state_dict(saved['ema_state_dict'],strict=True);restore_controls(model,saved['ema_controls'])
        if tensor_digest(dict(model.named_parameters()))!=parameter_before:raise ValueError('诊断改变了模型参数')
        if sha(checkpoint)!=expected:raise ValueError('诊断写回了原始模型')
        save(root/(s+'.json'),dict(shift=s,checkpoint=str(checkpoint),checkpoint_sha256=expected,
            source_best_epoch=saved['epoch'],model_eval_mode=True,batches=batches,parameters_unchanged=True,
            optimizer_constructed=False,target_gt_only_for_separate_validation_gradient=True,
            sampling_not_based_on_GT=True,scope='three fixed short-video batches; local gradient evidence, not mAP causality'))
        results[s]=dict(path=str(root/(s+'.json')),sha256=sha(root/(s+'.json')))
        # 绑定方法的闭包持有model；显式恢复，避免跨shift残留GPU图。
        model.rq3_capacity.branch.source_loss=original_source;model.rq3_capacity.branch.prototype_alignment=original_align
        model.rq3_capacity.forward=original_aux
        del model,saved,params,groups,original_source,original_align,original_aux;gc.collect();torch.cuda.empty_cache()
    save(root/'audit.json',dict(status='passed',shifts=results,no_training=True,guard_path=proof,guard_sha256=sha(proof)))


def main(args):
    from tools.run_sada_baseline import acquire
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);identity=code_identity()
    save(root/'plan.json',dict(code=identity,no_training=True,no_optimizer=True,checkpoints='existing_source_selected_EMA',
        target_gt='offline diagnostic only',batch_size=2,max_batches_per_shift=3,gpus_allowed=[0,1,2,3]))
    while not (found:=acquire([3,2,0,1],args.data_root,root,16384,identity['commit'])):
        save(root/'state.json',dict(phase='waiting_shared_gpu',pid=os.getpid()));time.sleep(15)
    context,card=found;p=None
    try:
        env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=card['uuid'],
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
        with (root/'worker.log').open('x') as f:
            p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--worker',
                '--output',str(root),'--data-root',str(args.data_root)],cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'state.json',dict(phase='gradient_diagnostic',pid=os.getpid(),child_pid=p.pid,gpu=card['index']))
        if p.wait():raise RuntimeError('梯度诊断失败，保留全部工件')
        save(root/'state.json',dict(phase='complete',pid=os.getpid()))
    finally:
        if p is not None and p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        context.__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    def stop(sig,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN);raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:worker(args.output,args.data_root) if args.worker else main(args)
    except BaseException as e:
        if args.output.exists():save(args.output/'error.json',dict(type=type(e).__name__,message=str(e)))
        raise
