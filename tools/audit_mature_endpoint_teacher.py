"""只读成熟学生/EMA的源端点比较；不加随机风险头，不查target，不选择新训练配置。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,termination

SCHEMA='source-mature-endpoint-teacher-diagnostic-v1'


def point_record(student,teacher,teacher_logits,classes,targets,segment,level,length):
    import torch
    a,b=segment;grid=student['points'][level]
    if not torch.equal(grid,teacher['points'][level]):raise ValueError('师生真实grid不同')
    common=student['masks'][level][0].reshape(-1).bool() & teacher['masks'][level][0].reshape(-1).bool()
    ids=(common & (grid[:,0]>=a) & (grid[:,0]<=b)).nonzero(as_tuple=True)[0]
    if not len(ids):return None,'no_real_point_inside_proposal'
    index=int(ids[(grid[ids,0]-(a+b)/2).abs().argmin()])
    if not bool(classes[level][index].sum()>0):return None,'not_source_assigned_positive'
    point=grid[index]
    def decode(offset):return torch.stack((point[0]-offset[0]*point[3],point[0]+offset[1]*point[3]))
    sb=decode(student['offsets'][level][0,index]);tb=decode(teacher['offsets'][level][0,index])
    gt=decode(targets[level][index]);predicted=int(teacher_logits[level][0,:,index].argmax())
    semantic=bool(classes[level][index,predicted]>0)
    allowed=torch.stack((tb>=0,tb<=length)).all(0) & semantic
    return dict(level=level,duration=float(b-a),absolute_point=float(point[0]),class_index=predicted,
        student_error=(sb-gt).abs().cpu().tolist(),teacher_error=(tb-gt).abs().cpu().tolist(),
        student_boundary=sb.cpu().tolist(),teacher_boundary=tb.cpu().tolist(),reference_gt=gt.cpu().tolist(),
        allowed=allowed.cpu().tolist()),None


def summary(rows,videos):
    import numpy as np
    by_video={};count=better=0
    for row in rows:
        if row['video_id'] not in videos:continue
        for side in (0,1):
            if not row['allowed'][side]:continue
            s=row['student_error'][side]/row['duration'];t=row['teacher_error'][side]/row['duration']
            by_video.setdefault(row['video_id'],[]).append((s,t));count+=1;better+=int(t<s)
    matrix=np.asarray([[np.mean([s for s,t in v]),np.mean([t for s,t in v]),
        np.mean([s-t for s,t in v]),np.mean([max(s-t,0.) for s,t in v])]
        for v in by_video.values()])
    result=dict(endpoints=count,teacher_better=better,teacher_win_rate=better/count if count else None,
        physical_videos=len(by_video),per_video={k:dict(endpoints=len(v),teacher_better=sum(t<s for s,t in v)) for k,v in by_video.items()},
        actual_positive_endpoints_exist=bool(better),oracle_uses_source_gt=True,
        unconditional_teacher_advantage_verified=False)
    if not len(matrix):return result
    result.update(video_equal_mean_student_error=float(matrix[:,0].mean()),
        video_equal_mean_teacher_error=float(matrix[:,1].mean()),
        video_equal_teacher_advantage=float(matrix[:,2].mean()),
        video_equal_oracle_positive_gain=float(matrix[:,3].mean()))
    if len(matrix)>=3:
        rng=np.random.RandomState(123)
        boot=np.array([matrix[rng.randint(0,len(matrix),len(matrix))].mean(0) for _ in range(2000)])
        ci=np.quantile(boot[:,2],[.025,.975]).tolist()
        result.update(teacher_advantage_ci95=ci,unconditional_teacher_advantage_verified=bool(ci[0]>0))
    return result


def collect(args):
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.prototype_capacity import feature_interval
    from tools.baseline_engine import dataset,tensor_digest
    from tools.validate_source_windows import source_inputs
    from tools.calibrate_prototype_geometry import source_dense_observation,prepare_source_video
    os.chdir(args.data_root)
    args.output.mkdir(parents=True,exist_ok=False)
    identity=code_identity();entry=args.control_root/'formal/control'/args.shift
    cfg,candidates,requests=source_inputs(args.control_root,args.source_pool,args.shift)
    complete=read(entry/'complete.json');terminal=termination(read(entry/'termination.json')['history'])
    for name in ('config.json','termination.json','best.pt','latest.pt'):
        if sha(entry/name)!=complete['artifacts'][str(entry/name)]:raise ValueError('原完整工件改变: '+name)
    holdout=read(args.holdout)
    if sha(requests)!=holdout['source_requests_sha256']:raise ValueError('不是原先固定的源候选')
    if set(holdout['fit_videos']) & set(holdout['audit_videos']):raise ValueError('物理视频交叉')
    if set(holdout['fit_videos']+holdout['audit_videos'])!={c['video_id'] for c in candidates}:
        raise ValueError('冻结源视频不完整')
    plan=dict(schema=SCHEMA,code=identity,shift=args.shift,checkpoints={'best':terminal['best_epoch'],'latest':terminal['epoch']},
        source_pool_sha256=sha(args.source_pool),holdout=holdout,target_data_used=False,target_map_used=False,
        new_risk_head_added=False,training_performed=False,checkpoint_selection='both source-best and legal-terminal, no selection by diagnostic results')
    save(args.output/'plan.json',plan)
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False)
    grouped={}
    for c in candidates:grouped.setdefault(c['video_id'],[]).append(c)
    if not set(grouped)<={x['id'] for x in ds.data_list}:raise ValueError('候选不属于源验证')
    reports={}
    for name,epoch in plan['checkpoints'].items():
        saved=torch.load(entry/(name+'.pt'),map_location='cpu')
        if saved['config']!=cfg or saved['epoch']!=epoch:raise ValueError('不是原源选模/停止保存点')
        student=make_meta_arch('sada',deepcopy(cfg),device='cpu').to('cpu')
        teacher=make_meta_arch('sada',deepcopy(cfg),device='cpu').to('cpu')
        student.load_state_dict(saved['state_dict'],strict=True);teacher.load_state_dict(saved['ema_state_dict'],strict=True)
        del saved
        student.eval();teacher.eval();before=tensor_digest((student.state_dict(),teacher.state_dict()))
        rng=torch.get_rng_state().clone();rows=[];excluded=[];features={}
        with torch.no_grad():
            for i,item in enumerate(ds.data_list):
                if item['id'] not in grouped:continue
                original=ds[i];clean,_=prepare_source_video(original,'cpu')
                so,_=source_dense_observation(student,clean);to,logits=source_dense_observation(teacher,clean)
                # GT只在两次预测结束后用于同一个原版正样本分配。
                classes,targets=student.label_points(so['points'],[original['segments']],[original['labels']])
                splits=[len(x) for x in so['points']];classes=classes[0].split(splits);targets=targets[0].split(splits)
                vid=item['id'];meta=original['metadata']
                feature=Path(cfg['dataset']['feat_folder'])/((cfg['dataset']['file_prefix'] or '')+vid+cfg['dataset']['file_ext'])
                features[str(feature)]=sha(feature)
                for candidate in grouped[vid]:
                    segment=feature_interval(dict(proposal_start_sec=candidate['proposal'][0],proposal_end_sec=candidate['proposal'][1]),meta)
                    row,reason=point_record(so,to,logits,classes,targets,segment,int(candidate['fpn_level']),original['feats'].shape[-1])
                    common=dict(candidate_id=candidate['candidate_id'],video_id=vid)
                    if row is None:excluded.append(dict(common,reason=reason))
                    else:rows.append(dict(common,**row))
                print('MATURE_SOURCE',name,epoch,vid,len(rows),flush=True)
        if not torch.equal(rng,torch.get_rng_state()) or before!=tensor_digest((student.state_dict(),teacher.state_dict())):
            raise ValueError('只读诊断修改参数或随机状态')
        save(args.output/(name+'_rows.json'),rows);save(args.output/(name+'_excluded.json'),excluded)
        reports[name]=dict(epoch=epoch,all_source=summary(rows,holdout['fit_videos']+holdout['audit_videos']),
            fit=summary(rows,holdout['fit_videos']),audit=summary(rows,holdout['audit_videos']),
            checkpoint=str(entry/(name+'.pt')),checkpoint_sha256=sha(entry/(name+'.pt')),
            rows_sha256=sha(args.output/(name+'_rows.json')),feature_inputs=features,
            model_and_rng_unchanged=True)
        del student,teacher
    if torch.cuda.is_initialized():raise ValueError('CPU诊断意外初始化CUDA')
    result=dict(status='completed_read_only_source_diagnostic',schema=SCHEMA,code=identity,reports=reports,
        cuda_initialized=False,target_map_used=False,risk_gate_validated=False,formal_training_started=False)
    save(args.output/'result.json',result)
    print('MATURE_DIAGNOSTIC_DONE',args.output/'result.json',flush=True)
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
