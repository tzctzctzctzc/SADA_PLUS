"""禁CUDA的因果诊断：原梯度裁剪、源GT方向校验、源内步幅敏感性；不修改正式模型。

只读取冻结目标预测/无标签轨迹。GT相关机制检查限定固定hash的源验证视频；
1/10/100倍单步扰动只是源内敏感性诊断，不选新超参、不启动目标试验。
"""
import argparse,hashlib,json,math,os,sys,time
from copy import deepcopy
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity


def norm(gradients):
    return sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)**.5


def cosine(a,b):
    product=sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
    return product/max(norm(a)*norm(b),1e-30)


def save_before_GT(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())


def source_truth(original,clean):
    import torch
    meta=clean['metadata'];entry=original['database'][meta['video_id']]
    selected=[a for a in entry.get('annotations',[]) if a['label_id']<10]
    result=deepcopy(clean)
    result['labels']=torch.tensor([a['label_id'] for a in selected],dtype=torch.long)
    seconds=torch.tensor([a['segment'] for a in selected],dtype=torch.float32).reshape(-1,2)
    result['segments']=(seconds*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']
    return result


def quality(rows,truth):
    import torch
    from tools.tta_core import tiou
    bounds=truth['segments'];labels=truth['labels'];values=[];covered=set()
    for r in rows:
        ids=(labels==r['label']).nonzero().flatten()
        overlap=tiou(r['segment'][None,:],bounds[ids]) if len(ids) else torch.empty(0)
        best=float(overlap.max()) if len(ids) else 0.
        values.append(best)
        if len(ids):covered.update(ids[overlap>=.5].tolist())
    return dict(pseudo_count=len(rows),source_GT_count=len(labels),same_class_IoU=values,
        pseudo_IoU50_count=sum(x>=.5 for x in values),pseudo_wrong_or_IoU_below10_count=sum(x<.1 for x in values),
        GT_covered_IoU50=len(covered))


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('本诊断严格禁CUDA')
    import torch,numpy as np
    from libs.models.SADA import make_meta_arch
    from tools.baseline_engine import dataset,tensor_digest
    from tools.direct_detection_ttt import (SPEC,configure,reset,teacher_rows,clean_video,window,context,marks,seed,objective,preservation)
    from tools.tta_core import dense
    from tools.meta_ttt import detection_loss
    torch.set_num_threads(args.threads);torch.manual_seed(123);np.random.seed(123)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    identity=code_identity();root=args.root.resolve();plan=read(root/'plan.json');paths=plan['paths'][args.shift]
    os.chdir(paths['data_root'])
    # 诊断来自独立分支，但正式数学函数逐字节不变。
    for name in ['tools/direct_detection_ttt.py','tools/tta_core.py','tools/meta_ttt.py']:
        import subprocess
        original=subprocess.check_output(['git','show',plan['code']['commit']+':'+name],cwd=CODE)
        if hashlib.sha256(original).hexdigest()!=sha(CODE/name):raise ValueError('历史数学函数改变')
    cfg=read(paths['config']);runtime=deepcopy(cfg);runtime['prototype_capacity']['prepared_cache']=paths['cache']
    saved=torch.load(paths['checkpoint'],map_location='cpu')
    if saved['config']!=cfg:raise ValueError('不是原完整配置')
    model=make_meta_arch('sada',runtime,device='cpu').to('cpu');model.load_state_dict(saved['ema_state_dict'],strict=True);del saved
    teacher=deepcopy(model).eval().requires_grad_(False);named=configure(model);parameters=list(named.values())
    initial={n:p.detach().clone() for n,p in named.items()};cold=tensor_digest(model.state_dict());teacher_cold=tensor_digest(teacher.state_dict())
    parameter_norm=norm(parameters)
    runtime['dataset']['feat_folder']=paths['features']
    ds=dataset(runtime,paths['source_manifest'],cfg['src_val_split'],False,True)
    indices=sorted(range(len(ds)),key=lambda i:hashlib.sha256(('123:'+ds.data_list[i]['id']).encode()).hexdigest())[:args.samples]
    observations=[];all_rows={};all_videos={}
    for i in indices:
        v=clean_video(ds[i]);check_id=v['metadata']['video_id'];all_videos[check_id]=v
        rows,coverage=teacher_rows(teacher,v);all_rows[check_id]=rows
        observations.append(dict(video_id=check_id,coverage=coverage,rows=[dict(r,segment=r['segment'].tolist()) for r in rows]))
    save_before_GT(out/'unlabeled_observations.json',observations)
    # 原源GT只在所有无GT候选先落盘后读取，不接触任何目标GT文件。
    original=read(paths['source_annotation'])
    if any(v['subset'].lower() not in cfg['src_train_split']+cfg['src_val_split'] for v in original['database'].values()):raise ValueError('GT文件非纯源')
    rows_out=[];started=time.time()
    for observation in observations:
        vid=observation['video_id'];clean=all_videos[vid];pseudo=all_rows[vid];truth=source_truth(original,clean)
        record=dict(video_id=vid,input_tokens=clean['feats'].shape[-1],quality=quality(pseudo,truth))
        if not pseudo:
            record['skip']='no-reliable-pseudo';rows_out.append(record);continue
        ordered=sorted(pseudo,key=lambda r:r['anchor'])
        a,b=window(ordered[0],clean['feats'].shape[-1],model.max_seq_len,max(model.fpn_strides))
        with torch.no_grad():ref=dense(teacher,context(clean,a,b))
        reference=dict(logits=ref['logits'],masks=ref['masks']);del ref
        packet=dense(model,context(clean,a,b))
        unmasked_loss,unmasked_metrics,_=objective(packet,reference,ordered,a,b)
        unmasked_grad=torch.autograd.grad(unmasked_loss,parameters,allow_unused=True)
        unmasked_norm=norm(unmasked_grad);del unmasked_grad,packet,unmasked_loss
        mask=marks(b-a,SPEC['student_mask_fraction'],seed(vid,1))
        packet=dense(model,context(clean,a,b,mask))
        loss,stats,parts=objective(packet,reference,ordered,a,b)
        class_grad=torch.autograd.grad(parts['classification'],parameters,retain_graph=True,allow_unused=True)
        reg_grad=torch.autograd.grad(parts['regression'],parameters,retain_graph=True,allow_unused=True)
        preserve_grad=torch.autograd.grad(SPEC['preservation_weight']*preservation(packet,reference),parameters,retain_graph=True,allow_unused=True)
        adaptation_grad=torch.autograd.grad(loss,parameters,allow_unused=True)
        gradient_norm=norm(adaptation_grad)
        full=dense(model,clean);supervised=detection_loss(model,full,truth)
        supervised_grad=torch.autograd.grad(supervised,parameters,allow_unused=True)
        supervised_before=float(supervised.detach())
        record.update(window=[a,b],TTT=stats,gradient_norm=gradient_norm,unmasked_gradient_norm=unmasked_norm,
            classification_gradient_norm=norm(class_grad),regression_gradient_norm=norm(reg_grad),
            weighted_preservation_gradient_norm=norm(preserve_grad),class_reg_gradient_cosine=cosine(class_grad,reg_grad),
            TTT_GT_gradient_cosine=cosine(adaptation_grad,supervised_grad),class_GT_gradient_cosine=cosine(class_grad,supervised_grad),
            reg_GT_gradient_cosine=cosine(reg_grad,supervised_grad),source_clean_detection_loss=supervised_before)
        baseline_logits=torch.cat([x.detach().sigmoid() for x in full['logits']],1)
        baseline_offsets=torch.cat([x.detach() for x in full['offsets']],1)
        del full,supervised,supervised_grad,class_grad,reg_grad,preserve_grad,loss,packet,parts
        factor=SPEC['learning_rate']*min(1.,SPEC['gradient_clip']/(gradient_norm+1e-6))
        record['effective_lr']=factor;record['sensitivity']=[]
        for scale in [1,10,100]:
            with torch.no_grad():
                for (name,p),g in zip(named.items(),adaptation_grad):
                    if g is not None:p.copy_(initial[name]-scale*factor*g)
                test=dense(model,clean);value=float(detection_loss(model,test,truth))
                logits=torch.cat([x.sigmoid() for x in test['logits']],1)
                offsets=torch.cat(test['offsets'],1)
                delta=norm([p.detach()-initial[n] for n,p in named.items()])
                record['sensitivity'].append(dict(source_only=True,scale=scale,parameter_delta_l2=delta,
                    parameter_relative_l2=delta/parameter_norm,source_clean_detection_loss=value,
                    source_detection_loss_change_percent=(value/supervised_before-1)*100,
                    clean_probability_mean_abs_change=float((logits-baseline_logits).abs().mean()),
                    clean_offset_mean_abs_change=float((offsets-baseline_offsets).abs().mean())))
            reset(named,initial)
        del adaptation_grad,baseline_logits,baseline_offsets,reference
        rows_out.append(record);save(out/'progress.json',dict(completed=len(rows_out),total=len(observations),rows=rows_out))
        print('SOURCE_DIAG',args.shift,vid,'cos',record['TTT_GT_gradient_cosine'],'seconds',round(time.time()-started,1),flush=True)
    if tensor_digest(model.state_dict())!=cold or tensor_digest(teacher.state_dict())!=teacher_cold:raise ValueError('诊断污染了原权重')
    save(out/'complete.json',dict(status='passed-source-only-diagnosis',shift=args.shift,code=identity,
        historical_experiment_code=plan['code'],root=str(root),checkpoint_sha256=paths['checkpoint_sha256'],
        parameter_norm=parameter_norm,parameters=sum(p.numel() for p in parameters),videos=len(rows_out),rows=rows_out,
        model_unchanged=True,teacher_unchanged=True,target_GT_read=False,source_only_sensitivity_not_hyperparameter_selection=True,
        observations_sha256=sha(out/'unlabeled_observations.json'),elapsed_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shift',choices=['S%d'%i for i in range(1,7)],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=8);p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();run(args)
