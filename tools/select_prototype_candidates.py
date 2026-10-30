"""用完成UDA的seed123 Prototype最佳EMA重选候选；GT仅在候选冻结后离线评分。"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as p

SELECTION_SEED=6 # 与旧候选相同的哈希预算顺序；模型训练seed是123。
ACTIVE={}


def stable_rows(rows,seed=SELECTION_SEED):
    return sorted(rows,key=lambda row:hashlib.sha256('{}:{}'.format(seed,row['candidate_id']).encode()).hexdigest())


def limit_per_video(rows,max_per_video=30,max_proposals=200,seed=SELECTION_SEED):
    grouped=defaultdict(list)
    for row in stable_rows(rows,seed):grouped[row['video_id']].append(row)
    selected=[]
    for video in sorted(grouped):selected.extend(grouped[video][:max_per_video])
    return stable_rows(selected,seed)[:max_proposals]


def decode_candidates(video,levels):
    """CPU只解码已由原阈值选中的点；同坐标保留最高分的原FPN来源层。"""
    grouped=defaultdict(list);meta=video['metadata'];fps=meta['fps'];stride=meta['feat_stride'];offset=.5*meta['feat_num_frames'];duration=meta['duration']
    for level,points,scores,classes,offsets in levels:
        for grid,score,category,bounds in zip(points,scores,classes,offsets):
            coordinate=grid[0];index=round(coordinate)
            if abs(coordinate-index)>1e-6:raise ValueError('FPN点不是整数特征坐标')
            convert=lambda v:min(max((v*stride+offset)/fps,0.),duration)
            grouped[index].append(dict(fpn_level=level,score=score,teacher_top1_class=category,
                proposal_start_sec=convert(coordinate-bounds[0]*grid[3]),
                proposal_end_sec=convert(coordinate+bounds[1]*grid[3]),absolute_center_time=convert(coordinate)))
    rows=[]
    for index,members in sorted(grouped.items()):
        best=max(members,key=lambda x:x['score'])
        if best['proposal_end_sec']<=best['proposal_start_sec']:raise ValueError('新候选存在无效时间边界')
        rows.append(dict(candidate_id='{}-p{:07d}'.format(meta['video_id'],index),video_id=meta['video_id'],
            absolute_feature_index=index,absolute_center_time=best['absolute_center_time'],center_sec=best['absolute_center_time'],
            proposal_start_sec=best['proposal_start_sec'],proposal_end_sec=best['proposal_end_sec'],
            source_fpn_levels=[best['fpn_level']],teacher_top1_class=int(best['teacher_top1_class']),teacher_score=best['score']))
    return rows


def dense_target(model,video):
    x,mask=model.preprocessing([video],mode='validation')
    features,masks,_=model.backbone_target(x,mask);features,masks=model.neck_target(features,masks)
    logits,_=model.cls_head_target(features,masks);offsets,_=model.reg_head_target(features,masks)
    return model.point_generator(features),masks,logits,offsets


def worker(args):
    import torch
    from tools.baseline_engine import dataset,tensor_digest
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    from libs.trainer.da_trainer import move_to_device
    from tools.baseline_gpu import first_cuda_guard
    from tools.calibrate_prototype_geometry import source_dense_observation
    from tools.diagnose_vlm_candidates import evaluate_candidates,summarize
    entry=args.control/'formal/control'/args.shift;cfg=p.read(entry/'config.json');done=p.read(entry/'complete.json')
    for name in ['config.json','best.pt','termination.json']:
        if p.sha(entry/name)!=done['artifacts'][str(entry/name)]:raise ValueError('原版Prototype完成工件改变')
    if cfg['seed']!=123 or cfg['baseline']['regime']!='uda' or cfg['prototype_capacity']['variant']!='control' or cfg['baseline']['target_gt_used']:
        raise ValueError('必须使用完成UDA、无目标GT训练的seed123原版Prototype')
    target=Path(cfg['baseline']['target_manifest'])
    if p.sha(target)!=cfg['baseline']['target_manifest_sha256']:raise ValueError('目标无标注成员改变')
    guard=first_cuda_guard();os.chdir(args.data_root)
    saved=torch.load(entry/'best.pt',map_location='cpu');decision=p.termination(p.read(entry/'termination.json')['history'])
    if saved['config']!=cfg or saved['epoch']!=decision['best_epoch']:raise ValueError('不是源验证选定的UDA后best')
    fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').to('cuda:0');model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval();del saved
    if model.use_trident_head or cfg['model'].get('use_ce_cls',False):raise ValueError('本轮沿用原普通回归/sigmoid候选协议')
    before=tensor_digest(model.state_dict());ds=dataset(cfg,target,cfg['tgt_train_split'],False,True)
    members=p.read(target)['database']
    if set(members)!={x['id'] for x in ds.data_list}:raise ValueError('目标训练成员不全')
    all_rows=[];alpha=float(cfg['model']['pl_threshold']);feature_manifest={}
    with torch.no_grad():
        for i,raw in enumerate(ds):
            p.assert_no_target_gt([raw]);video=move_to_device([raw],'cuda:0')[0]
            points,masks,logits,offsets=dense_target(model,video)
            if i==0:
                if not (model.backbone_source is model.backbone_target and model.neck_source is model.neck_target and model.cls_head is model.cls_head_target and model.reg_head is model.reg_head_target):
                    raise ValueError('共享检测器身份不符')
                observation,reference_logits=source_dense_observation(model,video)
                for a,b in zip(logits,reference_logits):torch.testing.assert_close(a,b,rtol=0,atol=0)
                for a,b in zip(offsets,observation['offsets']):torch.testing.assert_close(a.permute(0,2,1),b,rtol=0,atol=0)
            levels=[]
            for level in (0,1,2):
                probabilities=logits[level][0].T.sigmoid();scores,classes=probabilities.max(-1)
                selected=masks[level][0,0].bool() & (scores>=alpha)
                levels.append((level,points[level][selected].cpu().tolist(),scores[selected].cpu().tolist(),
                    classes[selected].cpu().tolist(),offsets[level][0].T[selected].cpu().tolist()))
            all_rows.extend(decode_candidates(video,levels))
            feature=Path(cfg['dataset']['feat_folder'])/((cfg['dataset']['file_prefix'] or '')+raw['metadata']['video_id']+cfg['dataset']['file_ext'])
            feature_manifest[str(feature)]=p.sha(feature)
            p.save(args.output/'progress.json',dict(shift=args.shift,videos=i+1,total=len(ds),eligible_candidates=len(all_rows),target_gt_used_for_selection=False))
            print('EXPORTED',args.shift,i+1,len(ds),raw['metadata']['video_id'],len(all_rows),flush=True)
    if tensor_digest(model.state_dict())!=before:raise ValueError('只读导出改变模型')
    selected=limit_per_video(all_rows)
    if len(selected)!=200:raise ValueError('不足原200预算，不能静默改变候选数量')
    proposal=args.output/'proposals.jsonl';proposal.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in selected))
    meta=dict(schema='prototype-post-UDA-seed123-candidates-v1',shift=args.shift,model_training_seed=123,
        candidate_selection_seed=SELECTION_SEED,selection_levels=[0,1,2],selection_threshold=alpha,max_per_video=30,max_proposals=200,
        proposals_before_budget=len(all_rows),selected_count=len(selected),target_video_ids=sorted(members),target_gt_used_for_selection=False,
        source_selected_epoch=decision['best_epoch'],weights='ema_state_dict',training_regime='uda',variant='control',
        reference_complete=str(entry/'complete.json'),reference_complete_sha256=p.sha(entry/'complete.json'),
        checkpoint=str(entry/'best.pt'),checkpoint_sha256=p.sha(entry/'best.pt'),checkpoint_training_code=cfg['baseline']['code'],
        export_code=p.code_identity(),gpu_guard=guard,gpu_guard_sha256=p.sha(guard),target_manifest=str(target),target_manifest_sha256=p.sha(target),
        feature_inputs=feature_manifest,weights_unchanged=True,single_stream_equivalence='first-real-video-logits-and-offsets-exact',
        proposal_file_sha256=p.sha(proposal),vlm_scored=False,frames_extracted=False)
    p.save(args.output/'provenance.json',meta)
    # 先冻结候选，再读取目标GT评分；不依据GT重选或调整阈值。
    annotation=Path(cfg['baseline']['annotation_for_offline_evaluation'])
    if p.sha(annotation)!=cfg['baseline']['annotation_sha256']:raise ValueError('GT改变')
    db=p.read(annotation)['database'];database={v:db[v] for v in members}
    rows=[dict(x,student_class=x['teacher_top1_class'],student_score=x['teacher_score'],class_id=None,participation_selected=False) for x in selected]
    diagnosed=evaluate_candidates(rows,database);quality=summarize(diagnosed,database)
    # 新候选未评分VLM，不能把未评分误报成VLM准确率0。
    for key in ['vlm_majority_accuracy','vlm_correct_and_usable_fraction','vlm_fixes','vlm_breaks']:quality[key]=None
    for value in quality['thresholds'].values():value['vlm_joint_pass_fraction']=None
    p.save(args.output/'quality.json',dict(status='passed',quality=quality,rows=diagnosed,annotation_sha256=p.sha(annotation),
        target_gt_used='offline after selection only',vlm_scored=False))
    p.save(args.output/'complete.json',dict(status='passed',shift=args.shift,selected_count=200,source_selected_epoch=decision['best_epoch'],
        artifacts={str(args.output/n):p.sha(args.output/n) for n in ['proposals.jsonl','provenance.json','quality.json']}))
    print('CANDIDATES_DONE',args.shift,json.dumps(quality),flush=True)


def stop(signum,frame):
    for t in ACTIVE.values():
        if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
    raise SystemExit(128+signum)


def main(args):
    from tools.run_sada_baseline import acquire
    root=args.output;root.mkdir(parents=True,exist_ok=False);identity=p.code_identity();pending=['S1','S2','S3','S4','S5','S6'];completed=[]
    p.save(root/'plan.json',dict(code=identity,model='post-UDA seed123 Prototype control, source-best EMA',shifts=pending.copy(),gpus=args.gpus,
        selection_seed=6,candidates_per_shift=200,target_gt_for_selection=False,vlm_rescoring=False,new_training=False))
    try:
        while pending or ACTIVE:
            for shift,task in list(ACTIVE.items()):
                process=task['process']
                if process.poll() is None:continue
                if process.returncode:raise RuntimeError(shift+'导出失败；保留现场')
                record=p.read(root/shift/'complete.json')
                for path,digest in record['artifacts'].items():
                    if p.sha(path)!=digest:raise ValueError('新候选工件SHA改变')
                completed.append(record);task['context'].__exit__(None,None,None);del ACTIVE[shift]
            for shift in list(pending):
                found=acquire(args.gpus,args.data_root,root,8192,identity['commit'])
                if not found:break
                context,lease=found;out=root/shift;out.mkdir();pending.remove(shift)
                env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                    SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log=open(out/'export.log','x');cmd=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--worker','--shift',shift,'--output',str(out),'--control',str(args.control),'--data-root',str(args.data_root)]
                process=subprocess.Popen(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
                ACTIVE[shift]=dict(context=context,process=process,gpu=lease['index'])
            p.save(root/'state.json',dict(pid=os.getpid(),completed=completed,pending=pending,
                active={s:dict(pid=t['process'].pid,gpu=t['gpu']) for s,t in ACTIVE.items()}))
            time.sleep(5)
    except BaseException as error:
        p.save(root/'error.json',dict(error=repr(error),completed=completed,pending=pending));stop(signal.SIGTERM,None)
    comparison=[]
    for shift in ['S1','S2','S3','S4','S5','S6']:
        new=p.read(root/shift/'quality.json')['quality'];old=p.read(args.old_quality/(shift+'.json'))['all200']
        comparison.append(dict(shift=shift,new=new,old=old))
    p.save(root/'final_audit.json',dict(status='passed',count=6,results=completed,comparison=comparison,
        interpretation='same frozen selection protocol, new post-UDA Prototype generator; no new VLM scores'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--shift')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--control',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));parser.add_argument('--gpus',default='0,2,1,3')
    parser.add_argument('--old-quality',type=Path,default=Path('/vepfs/vlms/candidate_quality6_1031'));args=parser.parse_args()
    for name in ['output','control','data_root','old_quality']:setattr(args,name,getattr(args,name).resolve())
    if args.worker:worker(args)
    else:
        args.gpus=[int(x) for x in args.gpus.split(',')];signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main(args)
