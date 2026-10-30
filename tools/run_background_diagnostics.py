"""六方向只读诊断：原模型复算、背景路由审计、冻结预测的GT背景oracle。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as p
from tools.background_diagnostics import SCHEMA,SPEC,error_counts,occupancy,routing_counts,sum_counts,iou


def worker(args):
    import numpy as np
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.evaluation import ANETdetection
    from tools.baseline_engine import dataset,tensor_digest
    from tools.calibrate_prototype_geometry import prepare_source_video,source_dense_observation
    from tools.baseline_gpu import first_cuda_guard
    args.output.mkdir(parents=True,exist_ok=False);identity=p.code_identity()
    ref=args.control/'formal/control'/args.shift;cfg=p.read(ref/'config.json');complete=p.read(ref/'complete.json')
    for name in ['best.pt','config.json','termination.json','evaluation/target_val.json']:
        if p.sha(ref/name)!=complete['artifacts'][str(ref/name)]:raise ValueError('参考工件改变: '+name)
    checkpoint_sha=p.sha(ref/'best.pt');annotation=Path(cfg['baseline']['annotation_for_offline_evaluation'])
    if p.sha(annotation)!=cfg['baseline']['annotation_sha256']:raise ValueError('离线标注改变')
    if cfg['model'].get('use_source_pseudo_labels',False):raise ValueError('本次source分组审计要求沿用源GT，不接受source伪标签变体')
    os.chdir(args.data_root);device='cpu' if args.cpu else 'cuda:0';guard=None if args.cpu else first_cuda_guard()
    torch.manual_seed(cfg['seed'])
    if not args.cpu:torch.cuda.manual_seed(cfg['seed'])
    saved=torch.load(ref/'best.pt',map_location='cpu')
    if saved['config']!=cfg or saved['epoch']!=p.termination(p.read(ref/'termination.json')['history'])['best_epoch']:raise ValueError('不是源选best')
    models={}
    for role,key in [('student','state_dict'),('ema','ema_state_dict')]:
        model=make_meta_arch('sada',deepcopy(cfg),device=device).to(device);model.load_state_dict(saved[key],strict=True);model.eval()
        if model.use_trident_head or cfg['model'].get('use_ce_cls',False):raise ValueError('本诊断仅支持冻结普通回归/独立sigmoid头')
        models[role]=model
    del saved
    before=tensor_digest({k:m.state_dict() for k,m in models.items()})
    plan=dict(schema=SCHEMA,spec=SPEC,code=identity,shift=args.shift,checkpoint_sha256=checkpoint_sha,
        annotation_sha256=p.sha(annotation),reference_complete_sha256=p.sha(ref/'complete.json'),backend=device,gpu_guard=guard,gpu_guard_sha256=None if guard is None else p.sha(guard),
        classification_threshold=cfg['model'].get('pl_threshold',.7),background_class_weight=cfg['model'].get('weight_no_class_adv',1.),
        target_gt_used_for_training=False,exploratory_target_gt_analysis=True,source_routing='source_gt',target_routing='score_pseudo_labels')
    p.save(args.output/'plan.json',plan)
    routing=[];point_index=[];predictions={name:[] for name in ['original','assigned_anchor_oracle','occupied_anchor_oracle','post_nms_background_oracle','full_backend_reference']}
    null_checked=False;video_ids=[];feature_inputs={}
    for domain,splits in [('source',cfg['src_val_split']),('target',cfg['tgt_val_split'])]:
        ds=dataset(cfg,annotation,splits,False)
        for i in range(len(ds)):
            original=ds[i];vid=original['metadata']['video_id'];clean,_=prepare_source_video(original,device)
            p.assert_no_target_gt([clean]);ema_data=None
            with torch.no_grad():
                for role,model in models.items():
                    obs,logits=source_dense_observation(model,clean)
                    point_data={}
                    # GT从此处开始仅用于诊断，不传给模型的特征前向。
                    classes,_=model.label_points(obs['points'],[original['segments'].to(device)],[original['labels'].to(device)])
                    sizes=[len(x) for x in obs['points']];assigned=[x.sum(-1)>0 for x in classes[0].split(sizes)]
                    occupied=[torch.as_tensor(occupancy(grid[:,0].cpu().numpy(),original['segments'].cpu().numpy()),device=device) for grid in obs['points']]
                    for level in range(len(logits)):
                        counts=routing_counts(logits[level][0].sigmoid().T.cpu().numpy(),obs['masks'][level][0].reshape(-1).cpu().numpy(),
                            assigned[level].cpu().numpy(),occupied[level].cpu().numpy(),plan['classification_threshold'],
                            background=(~assigned[level]).cpu().numpy() if domain=='source' else None)
                        routing.append(dict(video_id=vid,domain=domain,role=role,level=level,
                            active_alignment_level=level in cfg['model'].get('semantic_alignment_levels',list(range(6))),counts=counts))
                        scores=logits[level][0].sigmoid().T.cpu().numpy()
                        for key,value in dict(confidence=scores.max(-1),category=scores.argmax(-1).astype(np.int16),
                            valid=obs['masks'][level][0].reshape(-1).cpu().numpy().astype(bool),assigned=assigned[level].cpu().numpy(),inside=occupied[level].cpu().numpy()).items():
                            point_data['l%d_%s'%(level,key)]=value
                    point_file=args.output/'points'/domain/role/(vid+'.npz');point_file.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(point_file,**point_data)
                    point_index.append(dict(video_id=vid,domain=domain,role=role,path=str(point_file),sha256=p.sha(point_file)))
                    if role=='ema':ema_data=(obs,logits,assigned,occupied)
                if domain=='target':
                    obs,logits,assigned,occupied=ema_data;model=models['ema']
                    def infer(extra=None):
                        masks=[m.squeeze(1) for m in obs['masks']]
                        if extra is not None:masks=[m & x[None] for m,x in zip(masks,extra)]
                        return model.inference([clean],obs['points'],masks,[x.permute(0,2,1) for x in logits],obs['offsets'],None,None)[0]
                    base=infer()
                    if args.cpu or not null_checked:
                        full=model(video_list_src=[clean],video_list_tgt=[clean],mode='test')[1][0]
                        for key in ['segments','scores','labels']:
                            torch.testing.assert_close(base[key],full[key],rtol=1e-6,atol=1e-7)
                        null_checked=True
                        if args.cpu:
                            for segment,label,score in zip(full['segments'].tolist(),full['labels'].tolist(),full['scores'].tolist()):
                                predictions['full_backend_reference'].append(dict(video_id=vid,segment=segment,label=int(label),score=score))
                    for name,data in [('original',base),('assigned_anchor_oracle',infer(assigned)),('occupied_anchor_oracle',infer(occupied))]:
                        for segment,label,score in zip(data['segments'].tolist(),data['labels'].tolist(),data['scores'].tolist()):
                            predictions[name].append(dict(video_id=vid,segment=segment,label=int(label),score=score))
                    video_ids.append(vid)
            feature=Path(cfg['dataset']['feat_folder'])/((cfg['dataset']['file_prefix'] or '')+vid+cfg['dataset']['file_ext']);feature_inputs[str(feature)]=p.sha(feature)
            print('DIAGNOSTIC_VIDEO',args.shift,domain,i+1,len(ds),vid,flush=True)
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    frame=load_gt_seg_from_json(str(annotation),cfg['tgt_val_split'],action_subsampling='10_majority_classes')
    gt={vid:[] for vid in video_ids}
    for row in frame.to_dict('records'):gt[row['video-id']].append(dict(segment=[row['t-start'],row['t-end']],label=int(row['label'])))
    for row in predictions['original']:
        overlaps=iou(row['segment'],[g['segment'] for g in gt[row['video_id']]])
        if len(overlaps) and overlaps.max()>=SPEC['background_iou']:predictions['post_nms_background_oracle'].append(row)
    p.save(args.output/'predictions.json',predictions);p.save(args.output/'ground_truth_for_diagnosis.json',gt);p.save(args.output/'routing.json',routing);p.save(args.output/'point_index.json',point_index)
    maps={}
    for name,rows in predictions.items():
        columns={'video-id':np.asarray([r['video_id'] for r in rows]),'t-start':np.asarray([r['segment'][0] for r in rows]),
            't-end':np.asarray([r['segment'][1] for r in rows]),'label':np.asarray([r['label'] for r in rows]),'score':np.asarray([r['score'] for r in rows])}
        evaluator=ANETdetection(str(annotation),cfg['tgt_val_split'],num_workers=2,action_subsampling='10_majority_classes')
        ap,mean,*_=evaluator.evaluate(columns,verbose=False)
        maps[name]={'map_0.1_to_0.5':float(mean),'map_by_tiou':[float(x) for x in ap]}
    reference=p.read(ref/'evaluation/target_val.json')['map_0.1_to_0.5']
    backend_reference=maps['full_backend_reference']['map_0.1_to_0.5'] if args.cpu else reference
    if abs(maps['original']['map_0.1_to_0.5']-backend_reference)>1e-10:raise ValueError('同后端原完整模型与诊断mAP不同: '+str((backend_reference,maps)))
    if before!=tensor_digest({k:m.state_dict() for k,m in models.items()}):raise ValueError('只读诊断改变模型')
    aggregate={domain:{role:sum_counts([x['counts'] for x in routing if x['domain']==domain and x['role']==role and x['active_alignment_level']]) for role in models} for domain in ['source','target']}
    if args.cpu and torch.cuda.is_initialized():raise ValueError('CPU诊断意外初始化CUDA')
    result=dict(schema=SCHEMA,status='passed_read_only_audit',code=identity,backend=device,shift=args.shift,maps=maps,reference_map=reference,
        same_backend_reference_map=backend_reference,historical_gpu_map_difference=maps['original']['map_0.1_to_0.5']-reference,
        errors_by_tiou={str(t):error_counts(predictions['original'],gt,t) for t in SPEC['tiou']},routing=aggregate,
        reference_map_reproduced=True,single_stream_null_check_passed=null_checked,weights_unchanged=True,
        checkpoint_sha256=checkpoint_sha,feature_inputs=feature_inputs,
        limitations=['routing snapshots on validation data, not accumulated training gradients','point counts do not equal loss weights','oracles use target GT and are exploratory, not deployable methods','error counts are not additive AP contributions'],
        training_performed=False,artifacts={str(args.output/name):p.sha(args.output/name) for name in ['plan.json','predictions.json','ground_truth_for_diagnosis.json','routing.json','point_index.json']})
    from tools.background_routing_replay import replay
    result['routing_replay']=replay(args.output,plan)
    p.save(args.output/'result.json',result);print('DIAGNOSTIC_DONE',args.shift,maps,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--shift',required=True);parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--control',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    args=parser.parse_args()
    for name in ['output','data_root','control']:setattr(args,name,getattr(args,name).resolve())
    worker(args)
