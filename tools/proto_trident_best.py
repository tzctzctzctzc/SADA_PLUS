"""冻结最强单模型发布：原Trident权重、边界uncertainty、原型/CE读出。

python -m tools.proto_trident_best --reference-root ROOT --output-root OUT --shift S1
追加 --cached-root 已有读出缓存根目录 可零GPU重放；--shared 使用共用读出参数。
模型权重和数据不入git，运行时校验发布manifest中的SHA。
"""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from tools.baseline_protocol import read,save,sha
from tools.uncertainty_analysis import sigmoid,features
MANIFEST=Path(__file__).parent/'releases/proto_trident_best_1106.json'
def fusion(score,penalty):
    score=np.asarray(score,float)
    logit=np.log(np.clip(score,1e-12,1-1e-12)/(1-np.clip(score,1e-12,1-1e-12)))
    return sigmoid(logit+penalty)

def selected_uncertainty(model,raw,meta,recipe):
    import torch
    converted=__import__('tools.trident_uncertainty',fromlist=['to_seconds']).to_seconds(raw,meta)
    row={k:(v.numpy() if torch.is_tensor(v) else v) for k,v in converted.items()}
    _,u,signals=features(row)
    penalty=-(recipe.get('left_entropy_weight',0.)*u[:,0]+recipe.get('right_entropy_weight',0.)*u[:,1]
              +recipe.get('relative_std_weight',0.)*np.log1p(signals['relative_std']))
    lookup={}
    for i,(box,label,score) in enumerate(zip(row['segments'],row['labels'],row['scores'])):
        key=(float(box[0]),float(box[1]),int(label))
        old=lookup.get(key)
        if old is None or score>row['scores'][old]:
            lookup[key]=i
    model.test_nms_sigma=recipe['nms_sigma']
    pre=recipe['stage']=='pre_nms'
    scores=torch.from_numpy(fusion(raw['scores'].numpy(),penalty)).to(raw['scores']) if pre else raw['scores']
    result=model.postprocessing([dict(meta,segments=raw['segments'].clone(),scores=scores,labels=raw['labels'])])[0]
    ids=np.asarray([lookup[(float(box[0]),float(box[1]),int(label))]
                    for box,label in zip(result['segments'],result['labels'])],dtype=np.int64)
    output_scores=result['scores'].numpy().astype(float)
    if not pre:output_scores=fusion(output_scores,penalty[ids])
    return result,output_scores,torch.from_numpy(ids)


def rescore(record,parameters):
    scores=np.asarray(record['uncertainty_scores'],float)
    if parameters['readout']=='baseline':return scores.copy()
    logits=np.asarray(record['prototype_cosines'] if parameters['readout']=='prototype' else record['branch_classifier_logits'],float)/parameters['temperature']
    ready=np.asarray(record['source_prototype_ready'],bool)
    if parameters['readout']=='prototype':logits=logits.copy();logits[:,~ready]=-1e4
    logits=logits-logits.max(1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(1,keepdims=True)
    labels=np.asarray(record['labels'],int);q=prob[np.arange(len(labels)),labels]
    q=np.clip(np.where(ready[labels],q,.5),1e-6,1-1e-6)
    return fusion(scores,parameters['beta']*np.log(q/(1-q)))


@torch.no_grad()
def native_readout(model,fpn,raw,ids):
    branch=model.rq3_capacity.branch;unique,inverse=torch.unique(ids,sorted=True,return_inverse=True)
    embedded=[];classified=[];device=next(branch.parameters()).device
    for part in unique.split(64):
        result=branch.encode(fpn,[dict(segments=raw['segments'][part].to(device),levels=raw['level'][part].to(device))],model.fpn_strides)
        embedded.append(result['embedding']);classified.append(result['logits'])
    if embedded:
        embedding=torch.cat(embedded);logits=torch.cat(classified)
        bank=torch.nn.functional.normalize(branch.source_prototypes,dim=-1)
        cosine=(embedding@bank.T).cpu()[inverse].numpy();logits=logits.cpu()[inverse].numpy()
    else:
        cosine=np.empty((0,model.num_classes),np.float32);logits=cosine.copy()
    return cosine,logits


def evaluate_shift(args,shift):
    from tools import baseline_engine as base
    from libs.evaluation import ANETdetection
    from tools.native_ap_fast import install
    install();manifest=read(MANIFEST);setting=manifest['settings'][shift]
    reference=args.reference_root/(shift+'_trident');cfg=read(reference/'config.json')
    assert sha(reference/'config.json')==setting['config_sha256']
    assert sha(reference/'best.pt')==setting['checkpoint_sha256']
    dataset=base.dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],False)
    order=[row['id'] for row in dataset.data_list]
    parameters=manifest['shared_readout_parameters'] if args.shared else setting['readout']
    columns={k:[] for k in ['video-id','t-start','t-end','label','score']};model=None
    if args.cached_root:
        original=read(args.cached_root/'final.json')['results'][shift]
        assert original['passed'] and original['checkpoint_sha256']==setting['checkpoint_sha256']
    else:
        from libs.models.SADA import make_meta_arch
        from libs.utils import fix_random_seed
        from libs.trainer.da_trainer import move_to_device
        from tools.trident_uncertainty import DistributionObserver
        fix_random_seed(123,include_cuda=args.device.startswith('cuda'))
        if args.device.startswith('cuda'):torch.cuda.manual_seed(123)
        saved=torch.load(reference/'best.pt',map_location='cpu')
        assert saved['epoch']==setting['selected_epoch'] and saved['config']==cfg
        model=make_meta_arch('sada',deepcopy(cfg),device=args.device).to(args.device)
        model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval();del saved
        before=base.tensor_digest(model.state_dict());sigma=model.test_nms_sigma
    count=len(order) if args.limit_videos is None else min(args.limit_videos,len(order))
    for index,vid in enumerate(order[:count]):
        if args.cached_root:
            file=args.cached_root/'features'/shift/(vid+'.pt');assert sha(file)==original['feature_hashes'][vid]
            record=torch.load(file,map_location='cpu')
        else:
            with torch.no_grad():
                video=dataset[index];video['segments']=video['segments'][:0];video['labels']=video['labels'][:0]
                video=move_to_device([video],args.device);meta=video[0]['metadata'];assert meta['video_id']==vid
                captures=[];model.test_nms_sigma=sigma
                hook=model.neck_source.register_forward_hook(lambda module,inputs,value:captures.append(value[0]))
                try:
                    with DistributionObserver(model,1) as observer:
                        model(video_list_src=video,video_list_tgt=video,mode='test',domain='target',curr_epoch=setting['selected_epoch'])
                        raw=observer.record
                finally:hook.remove()
                prediction,scores,ids=selected_uncertainty(model,raw,meta,setting['uncertainty'])
                cosine,logits=native_readout(model,captures[1],raw,ids)
                record=dict(segments=prediction['segments'],labels=prediction['labels'],uncertainty_scores=scores,
                    prototype_cosines=cosine,branch_classifier_logits=logits,source_prototype_ready=model.rq3_capacity.branch.source_prototype_ready.cpu().numpy())
        adjusted=rescore(record,parameters);columns['video-id'].extend([vid]*len(adjusted))
        columns['t-start'].extend(record['segments'][:,0].tolist());columns['t-end'].extend(record['segments'][:,1].tolist())
        columns['label'].extend(record['labels'].tolist());columns['score'].extend(adjusted.tolist())
    if model is not None:assert before==base.tensor_digest(model.state_dict())
    output=args.output_root/shift;base.checkpoint(output/'predictions.pt',columns)
    report=dict(shift=shift,videos=count,full_dataset=count==len(order),optimizer_steps=0,parameters=parameters,model_state_unchanged=True,
        checkpoint_sha256=setting['checkpoint_sha256'],predictions_sha256=sha(output/'predictions.pt'))
    if count==len(order):
        ap,mean,*_=ANETdetection(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],num_workers=1,action_subsampling='10_majority_classes').evaluate({k:np.asarray(v) for k,v in columns.items()},verbose=False)
        expected=setting['map_percent'] if not args.shared else None
        if expected is not None:assert abs(100*float(mean)-expected)<1e-7
        report.update(map_percent=100*float(mean),by_iou=np.asarray(ap).tolist(),selected_score_reproduced=expected is not None)
    save(output/'result.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--cached-root',type=Path)
    p.add_argument('--shift',choices=['all']+['S'+str(i) for i in range(1,7)],default='all')
    p.add_argument('--device',default='cuda:0');p.add_argument('--shared',action='store_true');p.add_argument('--limit-videos',type=int)
    a=p.parse_args();shifts=list(read(MANIFEST)['settings']) if a.shift=='all' else [a.shift]
    results={s:evaluate_shift(a,s) for s in shifts};save(a.output_root/'summary.json',dict(results=results,all_six_complete=len(results)==6 and all(r['full_dataset'] for r in results.values()),target_validation_selected=True,single_seed=True))
