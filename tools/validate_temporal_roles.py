"""仅S1源域机制实验：同一冻结UDA特征、同一batch序列，三种汇聚独立比较。"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity


def encode(branch,rows,valid,variant,disable_roles=False):
    import torch
    from torch.nn import functional as F
    if variant=='temporal_roles':
        return branch.encode_tokens(rows,valid,disable_roles)
    tokens=branch.input_projection(rows)+branch.position_embedding[None]
    tokens=tokens.masked_fill(~valid[...,None],0.)
    tokens=branch.encoder(tokens,src_key_padding_mask=~valid)
    if variant=='masked_mean':
        value=(tokens*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp(min=1)
    else:
        inside,_=branch.pool_group(tokens[:,1:6],valid[:,1:6],branch.inside_attention)
        outside,_=branch.pool_group(tokens[:,[0,6]],valid[:,[0,6]],branch.outside_attention)
        value=branch.pool_fusion(torch.cat([inside,outside],1))
    value=branch.output_norm(value)
    return dict(embedding=F.normalize(value,dim=-1),logits=branch.classifier(value))


def metrics(logits,labels,classes):
    import torch
    from torch.nn import functional as F
    prediction=logits.argmax(1)
    counts=torch.bincount(labels,minlength=classes)
    correct=torch.bincount(labels[prediction==labels],minlength=classes)
    perclass=[float(correct[c]/counts[c]) if counts[c] else None for c in range(classes)]
    cm=torch.bincount(labels*classes+prediction,minlength=classes*classes).reshape(classes,classes)
    return dict(count=len(labels),accuracy=float((prediction==labels).float().mean()),
        macro_accuracy=sum(x for x in perclass if x is not None)/sum(x is not None for x in perclass),
        cross_entropy=float(F.cross_entropy(logits,labels)),class_counts=counts.tolist(),
        class_accuracy=perclass,confusion_matrix=cm.tolist())


def extract(cfg,checkpoint,device,output):
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.prototype_capacity import source_gt_proposals
    from libs.models.SADA.prototype_temporal_roles import sample_masked_tokens,TemporalRoleBranch
    from tools.baseline_engine import dataset
    model=make_meta_arch('sada',deepcopy(cfg),device=device).to(device)
    state=torch.load(checkpoint,map_location='cpu');model.load_state_dict(state['ema_state_dict'],strict=True)
    selected_epoch=state['epoch'];del state
    model.eval().requires_grad_(False)
    payload={};scope={}
    for domain,split in [('train',cfg['src_train_split']),('validation',cfg['src_val_split'])]:
        ds=dataset(cfg,cfg['dataset']['json_file'],split,False)
        rows,masks,labels,ids=[],[],[],[];excluded=0;total=0
        for vi in range(len(ds)):
            video=ds[vi];length=video['feats'].shape[-1];maximum=cfg['dataset']['max_seq_len']
            used=set();total+=len(video['labels'])
            for start in range(0,length,max(1,maximum//2)):
                end=min(length,start+maximum)
                kept=[i for i,seg in enumerate(video['segments'])
                      if i not in used and float(seg[0])>=start and float(seg[1])<=end]
                if not kept:
                    if end==length:break
                    continue
                used.update(kept)
                part=dict(video,metadata=dict(video['metadata']))
                part.update(feats=video['feats'][:,start:end],segments=video['segments'][kept]-start,labels=video['labels'][kept])
                part['metadata'].update(crop_start=start,crop_feat_len=end-start)
                part={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in part.items()}
                with torch.no_grad():
                    inputs,valid=model.preprocessing([part],mode='train')
                    feats,valid,_=model.backbone_source(inputs,valid);feats,valid=model.neck_source(feats,valid)
                    prop=source_gt_proposals([part],model.num_classes,len(model.fpn_strides),model.reg_range)
                    token,mask=sample_masked_tokens(feats,prop,model.fpn_strides,valid,TemporalRoleBranch.sample_ratios)
                    active=mask.any(1)
                rows.append(token[active].cpu());masks.append(mask[active].cpu());labels.append(part['labels'][active].cpu())
                ids.extend([video['metadata']['video_id']+':'+str(i) for i,ok in zip(kept,active.tolist()) if ok])
                if end==length:break
            excluded+=len(video['labels'])-len(used)
            if vi%10==0:print('EXTRACT',domain,vi,len(ds),flush=True)
        payload[domain]=dict(rows=torch.cat(rows),valid=torch.cat(masks),labels=torch.cat(labels),ids=ids)
        scope[domain]=dict(videos=[x['id'] for x in ds.data_list],total_source_segments=total,
            longer_than_chunk_or_outside_feature_range=excluded,retained=len(ids))
    assert not set(scope['train']['videos'])&set(scope['validation']['videos'])
    torch.save(payload,output/'frozen_source_tokens.pt')
    save(output/'feature_provenance.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        selected_ema_epoch=selected_epoch,source_json_sha256=sha(cfg['dataset']['json_file']),
        tokens_sha256=sha(output/'frozen_source_tokens.pt'),scope=scope,target_data_loaded=False,
        source_sampling='all frozen split videos; 50pct overlapping max_seq_len chunks; each GT at most once',
        holdout_limit='frozen UDA EMA was selected on this source validation split; head fitting never uses validation labels'))
    del model;gc.collect();torch.cuda.empty_cache()
    return payload


def run(args):
    import torch
    from torch.nn import functional as F
    from libs.models.SADA.prototype_capacity import VLMProposalCoAdaptationBranch
    from libs.models.SADA.prototype_methods import TemporalPoolingBranch
    from libs.models.SADA.prototype_temporal_roles import TemporalRoleBranch
    os.chdir(args.data_root)
    args.output.mkdir(parents=True,exist_ok=False)
    identity=code_identity()
    cfg=read(args.baseline/'config.json')
    if cfg['seed']!=123 or cfg['loader']['batch_size']!=2:raise ValueError('机制实验需绑定当前seed123/batch2基线')
    protocol=dict(status='running',kind='source_only_frozen_feature_mechanism',shift='S1',code=identity,
        seed=123,steps=500,batch=64,learning_rate=.001,weight_decay=.05,selection='fixed_last_step_no_validation_selection',
        variants=['masked_mean','temporal_pool','temporal_roles'],target_data_loaded=False,
        downstream_detection_claim=False,positive_rule='role macro accuracy exceeds both comparators; report all outcomes')
    save(args.output/'protocol.json',protocol)
    torch.set_num_threads(4)
    from tools.baseline_gpu import slot,snapshot
    while True:
        try:
            context=slot(args.gpu,args.data_root,args.output,12288,identity['commit']);lease=context.__enter__();break
        except BlockingIOError:print('WAIT_GPU',args.gpu,flush=True);time.sleep(15)
    try:
        now=snapshot(args.gpu)
        if now['free_mib']<12288 or now['uuid']!=lease['uuid']:raise ValueError('首CUDA前显存/UUID改变')
        os.environ['CUDA_VISIBLE_DEVICES']=lease['uuid']
        save(args.output/'first_cuda.json',dict(lease=lease,observed=now,owner_pid=os.getpid()))
        torch.manual_seed(123);torch.cuda.manual_seed(123);device='cuda:0'
        payload=extract(cfg,args.baseline/'best.pt',device,args.output)
        train=payload['train'];val=payload['validation'];n=len(train['labels']);d=train['rows'].shape[-1]
        generator=torch.Generator().manual_seed(123)
        batches=torch.randint(n,(500,64),generator=generator)
        save(args.output/'batch_order.json',dict(sha256=__import__('hashlib').sha256(batches.numpy().tobytes()).hexdigest(),samples=n))
        results={}
        for variant,cls in [('masked_mean',VLMProposalCoAdaptationBranch),('temporal_pool',TemporalPoolingBranch),('temporal_roles',TemporalRoleBranch)]:
            torch.manual_seed(0);branch=cls(d,10,confidence_threshold=0.).to(device)
            opt=torch.optim.AdamW(branch.parameters(),lr=.001,weight_decay=.05)
            branch.train();losses=[];role_first_gradient=None
            for step,indices in enumerate(batches):
                out=encode(branch,train['rows'][indices].to(device),train['valid'][indices].to(device),variant)
                loss=F.cross_entropy(out['logits'],train['labels'][indices].to(device))/math.log(10)
                opt.zero_grad();loss.backward()
                if variant=='temporal_roles' and step==0:role_first_gradient=float(branch.role_residual.weight.grad.norm())
                opt.step();losses.append(float(loss.detach()))
                if (step+1)%100==0:print('HEAD_TRAIN',variant,step+1,losses[-1],flush=True)
            branch.eval();logits=[];ablated=[];similarity=[]
            with torch.no_grad():
                for start in range(0,len(val['labels']),128):
                    rows=val['rows'][start:start+128].to(device);valid=val['valid'][start:start+128].to(device)
                    out=encode(branch,rows,valid,variant);logits.append(out['logits'].cpu())
                    if variant=='temporal_roles':
                        off=encode(branch,rows,valid,variant,True);ablated.append(off['logits'].cpu())
                        similarity.extend((out['embedding']*off['embedding']).sum(1).cpu().tolist())
            result=metrics(torch.cat(logits),val['labels'],10)
            result.update(train_loss_first=losses[0],train_loss_last=losses[-1],parameters=sum(p.numel() for p in branch.parameters()))
            if variant=='temporal_roles':
                result.update(role_first_gradient=role_first_gradient,role_weight_norm=float(branch.role_residual.weight.norm()),
                    roles_disabled=metrics(torch.cat(ablated),val['labels'],10),
                    learned_vs_roles_disabled_cosine=sum(similarity)/len(similarity))
                if not role_first_gradient>0 or not math.isfinite(role_first_gradient):raise ValueError('真实角色路径没有梯度')
            torch.save(dict(kind='mechanism_head_only_NOT_formal_detection',state_dict=branch.state_dict(),protocol=protocol),args.output/(variant+'.pt'))
            results[variant]=result;save(args.output/'results.json',dict(status='running',results=results))
            del branch,opt;gc.collect();torch.cuda.empty_cache()
        role=results['temporal_roles']['macro_accuracy']
        save(args.output/'results.json',dict(status='passed',execution_passed=True,
            evidence_supports_role_pooling=all(role>results[v]['macro_accuracy'] for v in ['masked_mean','temporal_pool']),
            scope='single source direction, one seed, frozen backbone head comparison; NOT downstream mAP',results=results))
        print('SOURCE_MECHANISM_COMPLETE',flush=True)
    finally:context.__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--gpu',type=int,default=0)
    run(p.parse_args())
