"""冻结成熟学生/EMA，只做源GT误差头可学性诊断；不冒充在线EMA风险方案或正式验收。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import random
import sys

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,termination

SCHEMA='source-frozen-pair-endpoint-risk-learnability-v1'
SPEC=dict(steps=500,batch_videos=2,seed=123,learning_rate=.001,clip_norm=1.,
    head='original-detached-1x1-Gaussian-NLL',teacher_and_student_heads='separately source supervised',
    detector_update=False,target_data_used=False,margin=0.,parameter_search=False,
    sampling='uniform source-video with replacement; one fixed source-training crop per video',
    checkpoint_choice='original source-best; not selected by this diagnostic',
    random_reference='same-video-level-class-duration-bin-side expected gain at same coverage')


def dense(model,video,mode):
    x,mask=model.preprocessing([video],mode=mode)
    features,masks,_=model.backbone_source(model.dropout(x),mask)
    fpn,masks=model.neck_source(features,masks)
    offsets,_,regression=model.reg_head(fpn,masks,return_features=True)
    logits,_=model.cls_head(fpn,masks)
    return dict(points=model.point_generator(fpn),offsets=[x.permute(0,2,1) for x in offsets],
        masks=masks,regression=regression),logits


def positive_record(observation,classes,targets):
    """1x1头逐点独立：只缓存源正点，与完整前向后mask的NLL含义相同。"""
    import torch
    features=[];means=[];truth=[]
    for level,(feature,offset,mask) in enumerate(zip(observation['regression'],observation['offsets'],observation['masks'])):
        positive=mask[0].reshape(-1).bool() & (classes[level].sum(-1)>0)
        features.append(feature[:,:,positive].detach());means.append(offset[:,positive].detach())
        truth.append(targets[level][positive][None].detach())
    return dict(features=torch.cat(features,-1),mean=torch.cat(means,1),target=torch.cat(truth,1))


def fit_heads(cache,head_options,steps=SPEC['steps'],seed=SPEC['seed']):
    import torch
    from libs.models.SADA.endpoint_risk import EndpointRisk,risk_nll
    channels=cache[0]['student']['features'].shape[1]
    heads={role:EndpointRisk(channels,head_options) for role in ('student','teacher')}
    optimizers={role:torch.optim.AdamW(head.parameters(),lr=SPEC['learning_rate'],weight_decay=0.) for role,head in heads.items()}
    generator=torch.Generator().manual_seed(seed);history=[]
    for step in range(1,steps+1):
        ids=torch.randint(len(cache),(SPEC['batch_videos'],),generator=generator).tolist();metrics={}
        for role,head in heads.items():
            records=[cache[i][role] for i in ids]
            feature=torch.cat([r['features'] for r in records],-1)
            mean=torch.cat([r['mean'] for r in records],1);target=torch.cat([r['target'] for r in records],1)
            if not feature.shape[-1]:raise ValueError('源拟合批没有有效正点')
            opt=optimizers[role];opt.zero_grad(set_to_none=True)
            variance=head([feature])[0]
            loss=risk_nll(variance,mean,target,torch.ones(mean.shape[:-1],dtype=torch.bool))
            if not torch.isfinite(loss):raise ValueError('源误差头拟合非有限')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),SPEC['clip_norm'])
            if not torch.isfinite(norm) or float(norm)<=0:raise ValueError('误差头缺少源梯度')
            opt.step();metrics[role]=dict(nll=float(loss.detach()),gradient_norm=float(norm))
        history.append(dict(step=step,source_indices=ids,metrics=metrics))
    for head in heads.values():head.eval()
    return heads,optimizers,history


def collect(args):
    import numpy as np
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.endpoint_risk import SCHEMA as ONLINE_SCHEMA,POLICY
    from libs.models.SADA.prototype_capacity import feature_interval
    from tools.baseline_engine import dataset,tensor_digest
    from tools.validate_source_windows import source_inputs
    from tools.calibrate_prototype_geometry import prepare_source_video
    from tools.audit_mature_endpoint_teacher import point_record
    from tools.endpoint_risk_protocol import calibrate
    os.chdir(args.data_root);args.output.mkdir(parents=True,exist_ok=False)
    identity=code_identity();entry=args.control_root/'formal/control'/args.shift
    cfg,candidates,requests=source_inputs(args.control_root,args.source_pool,args.shift)
    holdout=read(args.holdout);pool=read(args.source_pool)
    if sha(requests)!=holdout['source_requests_sha256']:raise ValueError('固定源候选改变')
    fit,audit=holdout['fit_videos'],holdout['audit_videos'];training=pool['membership'][args.shift]['train_videos']
    if set(fit)&set(audit) or set(fit+audit)&set(training):raise ValueError('物理视频成员交叉')
    terminal=termination(read(entry/'termination.json')['history']);complete=read(entry/'complete.json')
    checkpoint=entry/'best.pt'
    if sha(checkpoint)!=complete['artifacts'][str(checkpoint)] or sha(entry/'config.json')!=complete['artifacts'][str(entry/'config.json')]:
        raise ValueError('原source-best工件改变')
    saved=torch.load(checkpoint,map_location='cpu')
    if saved['config']!=cfg or saved['epoch']!=terminal['best_epoch']:raise ValueError('不是源验证选定保存点')
    models={}
    for role,key in [('student','state_dict'),('teacher','ema_state_dict')]:
        model=make_meta_arch('sada',deepcopy(cfg),device='cpu').to('cpu');model.load_state_dict(saved[key],strict=True)
        model.eval()
        for param in model.parameters():param.requires_grad_(False)
        models[role]=model
    del saved
    before=tensor_digest({k:m.state_dict() for k,m in models.items()})
    plan=dict(schema=SCHEMA,spec=SPEC,code=identity,shift=args.shift,source_checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),epoch=terminal['best_epoch'],holdout=holdout,training_videos=training,
        source_pool_sha256=sha(args.source_pool),online_method_validated=False,formal_acceptance_eligible=False)
    save(args.output/'plan.json',plan)
    random.seed(123);np.random.seed(123);torch.manual_seed(123)
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],True);cache=[];crops=[];features={}
    if {x['id'] for x in ds.data_list}!=set(training):raise ValueError('实际源训练成员不同')
    with torch.no_grad():
        for i in range(len(ds)):
            original=ds[i];clean,_=prepare_source_video(original,'cpu');pair={}
            observations={role:dense(model,clean,'train')[0] for role,model in models.items()}
            classes,targets=models['student'].label_points(observations['student']['points'],[original['segments']],[original['labels']])
            lengths=[len(x) for x in observations['student']['points']];classes=classes[0].split(lengths);targets=targets[0].split(lengths)
            for role,obs in observations.items():pair[role]=positive_record(obs,classes,targets)
            cache.append(pair);vid=original['metadata']['video_id']
            crops.append(dict(video_id=vid,crop_sha256=tensor_digest(original),positive_points=pair['student']['features'].shape[-1]))
            feature=Path(cfg['dataset']['feat_folder'])/((cfg['dataset']['file_prefix'] or '')+vid+cfg['dataset']['file_ext']);features[str(feature)]=sha(feature)
            print('SOURCE_TRAIN_FEATURE',i+1,len(ds),vid,flush=True)
    save(args.output/'source_training_crops.json',crops)
    opts=dict(schema=ONLINE_SCHEMA,policy=deepcopy(POLICY),mode='head_only',head_learning_rate=.001,head_clip=1.,target_weight=.1,warmup_epochs=5)
    rng=torch.get_rng_state().clone();heads,optimizers,history=fit_heads(cache,opts)
    if not torch.equal(rng,torch.get_rng_state()):raise ValueError('拟合使用了全局随机流')
    save(args.output/'head_fit_history.json',history)
    torch.save(dict(schema=SCHEMA,plan=plan,heads={k:h.state_dict() for k,h in heads.items()},
        optimizers={k:o.state_dict() for k,o in optimizers.items()}),args.output/'risk_heads.pt')
    del cache
    grouped={}
    for c in candidates:grouped.setdefault(c['video_id'],[]).append(c)
    val=dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False);rows=[];excluded=[]
    with torch.no_grad():
        for i,item in enumerate(val.data_list):
            if item['id'] not in grouped:continue
            original=val[i];clean,_=prepare_source_video(original,'cpu');observations={};teacher_logits=None
            for role,model in models.items():
                obs,logits=dense(model,clean,'test');obs['variance']=heads[role](obs['regression']);observations[role]=obs
                if role=='teacher':teacher_logits=logits
            so,to=observations['student'],observations['teacher']
            classes,targets=models['student'].label_points(so['points'],[original['segments']],[original['labels']])
            lengths=[len(x) for x in so['points']];classes=classes[0].split(lengths);targets=targets[0].split(lengths)
            for c in grouped[item['id']]:
                segment=feature_interval(dict(proposal_start_sec=c['proposal'][0],proposal_end_sec=c['proposal'][1]),original['metadata'])
                row,reason=point_record(so,to,teacher_logits,classes,targets,segment,int(c['fpn_level']),original['feats'].shape[-1])
                common=dict(candidate_id=c['candidate_id'],video_id=item['id'])
                if row is None:excluded.append(dict(common,reason=reason));continue
                level=row['level'];index=int((so['points'][level][:,0]-row['absolute_point']).abs().argmin());stride=so['points'][level][index,3]
                for role,obs in observations.items():row[role+'_sigma']=(obs['variance'][level][0,index].sqrt()*stride).cpu().tolist()
                rows.append(dict(common,**row))
            feature=Path(cfg['dataset']['feat_folder'])/((cfg['dataset']['file_prefix'] or '')+item['id']+cfg['dataset']['file_ext']);features[str(feature)]=sha(feature)
            print('SOURCE_VALIDATION_FEATURE',item['id'],len(rows),flush=True)
    cert=calibrate(rows,fit,audit,training,SPEC['margin'])
    if before!=tensor_digest({k:m.state_dict() for k,m in models.items()}):raise ValueError('冻结检测器被修改')
    if any(p.grad is not None for m in models.values() for p in m.parameters()):raise ValueError('检测器意外收到梯度')
    if torch.cuda.is_initialized():raise ValueError('CPU机制诊断意外触发CUDA')
    save(args.output/'source_rows.json',rows);save(args.output/'excluded.json',excluded)
    result=dict(schema=SCHEMA,status=cert['status'],source_calibration=cert,code=identity,spec=SPEC,
        detector_weights_unchanged=True,detector_gradients_none=True,risk_heads_actually_fitted=True,
        source_checkpoint_sha256=sha(checkpoint),source_rows_sha256=sha(args.output/'source_rows.json'),
        risk_heads_sha256=sha(args.output/'risk_heads.pt'),feature_inputs=features,target_data_used=False,
        target_map_used=False,online_method_validated=False,formal_acceptance_eligible=False,cuda_initialized=False)
    save(args.output/'result.json',result);print('FROZEN_RISK_DONE',cert['status'],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-root',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    p.add_argument('--source-pool',type=Path,default=Path('/vepfs/vlms/qwen8b32b_source_0adb413_1030/frozen_candidates.json'))
    p.add_argument('--holdout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shift',default='S1');p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    args=p.parse_args()
    for key in ('control_root','source_pool','holdout','output','data_root'):setattr(args,key,getattr(args,key).resolve())
    collect(args)
