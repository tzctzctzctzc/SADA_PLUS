"""真实师生保存点的source验证；不拟合离线假教师，不读取target数据。"""
from copy import deepcopy
from pathlib import Path
import hashlib

from tools.baseline_protocol import read,save,sha
from tools.endpoint_risk_protocol import calibrate
from libs.models.SADA.endpoint_risk import SCHEMA,POLICY,decode_at


def collect(checkpoint, source_pool, output, margin, device, identity):
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.prototype_capacity import feature_interval
    from tools.baseline_engine import dataset,tensor_digest
    from tools.validate_source_windows import source_inputs
    from tools.calibrate_prototype_geometry import prepare_source_video

    checkpoint,source_pool,output=map(Path,(checkpoint,source_pool,output))
    saved=torch.load(checkpoint,map_location='cpu')
    cfg=saved['config']
    if saved.get('schema')!=SCHEMA or 'endpoint_risk' not in saved:
        raise ValueError('必须使用真实训练过误差头的当前保存点，不能给旧模型随机加头当证据')
    if not saved['endpoint_risk']['optimizer']['state']:
        raise ValueError('误差头没有实际源监督更新')
    if cfg['baseline']['code']!=identity or cfg['baseline']['target_gt_used']:
        raise ValueError('不是当前clean代码/严格无targetGT的保存点')
    shift=cfg['baseline']['shift'];pool=read(source_pool)
    # 复用冻结原source请求身份，不用target候选/GT反向挑源样本。
    control_root=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029')
    _,candidates,requests=source_inputs(control_root,source_pool,shift)
    if sha(cfg['dataset']['json_file'])!=pool['sources'][shift]['source_json_sha256']:
        raise ValueError('实际学生与冻结源候选manifest不匹配')
    student=make_meta_arch('sada',deepcopy(cfg),device=device).to(device)
    teacher=make_meta_arch('sada',deepcopy(cfg),device=device).to(device)
    student.load_state_dict(saved['state_dict'],strict=True)
    teacher.load_state_dict(saved['ema_state_dict'],strict=True)
    student.eval();teacher.eval()
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False)
    by_video={}
    for c in candidates:by_video.setdefault(c['video_id'],[]).append(c)
    records=[];excluded=[];features={}
    before=tensor_digest((student.state_dict(),teacher.state_dict()))
    with torch.no_grad():
        for i,item in enumerate(ds.data_list):
            if item['id'] not in by_video:continue
            original=ds[i];meta=original['metadata'];vid=meta['video_id']
            clean,_=prepare_source_video(original,device)
            so=student.endpoint_observation([clean],mode='test')
            to=teacher.endpoint_observation([clean],mode='test')
            # 预测完成后才计算GT分配，双方使用同一原版label_points，不各挑最优GT。
            classes,targets=student.label_points(so['points'],[original['segments'].to(device)],
                                                 [original['labels'].to(device)])
            splits=[len(x) for x in so['points']];classes=classes[0].split(splits);targets=targets[0].split(splits)
            feature=Path(cfg['dataset']['feat_folder'])/((cfg['dataset']['file_prefix'] or '')+vid+cfg['dataset']['file_ext'])
            features[str(feature)]=sha(feature)
            for c in by_video[vid]:
                segment=feature_interval(dict(proposal_start_sec=c['proposal'][0],proposal_end_sec=c['proposal'][1]),meta)
                level=int(c['fpn_level']);grid=so['points'][level]
                anchor=(c['anchor_sec']*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']
                index=int((grid[:,0]-anchor).abs().argmin())
                valid=bool(so['masks'][level][0].reshape(-1)[index] & to['masks'][level][0].reshape(-1)[index])
                if not valid or not bool(classes[level][index].sum()>0):
                    excluded.append(dict(candidate_id=c['candidate_id'],video_id=vid,reason='not_source_assigned_positive'));continue
                if not segment[0]<=float(grid[index,0])<=segment[1]:
                    excluded.append(dict(candidate_id=c['candidate_id'],video_id=vid,reason='original_anchor_outside_proposal'));continue
                sb,ss=decode_at(so,level,0,index);tb,ts=decode_at(to,level,0,index)
                point=grid[index];target=targets[level][index]
                gt=torch.stack((point[0]-target[0]*point[3],point[0]+target[1]*point[3]))
                records.append(dict(candidate_id=c['candidate_id'],video_id=vid,level=level,
                    duration=float(segment[1]-segment[0]),student_sigma=ss.cpu().tolist(),teacher_sigma=ts.cpu().tolist(),
                    student_error=(sb-gt).abs().cpu().tolist(),teacher_error=(tb-gt).abs().cpu().tolist(),
                    target_gt_used=False,reference_gt=gt.cpu().tolist(),absolute_point=float(point[0])))
            print('SOURCE_RISK',shift,vid,len(records),flush=True)
    if before!=tensor_digest((student.state_dict(),teacher.state_dict())):
        raise ValueError('源审计修改了模型状态')
    videos=sorted({r['video_id'] for r in records},key=lambda v:hashlib.sha256(('123:'+v).encode()).hexdigest())
    fit,audit=videos[::2],videos[1::2]
    certificate=calibrate(records,fit,audit,pool['membership'][shift]['train_videos'],margin)
    output.mkdir(parents=True,exist_ok=True)
    save(output/'source_rows.json',records);save(output/'excluded.json',excluded)
    certificate.update(code=identity,shift=shift,checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha(checkpoint),source_rows_sha256=sha(output/'source_rows.json'),
        source_pool=str(source_pool.resolve()),source_pool_sha256=sha(source_pool),
        source_manifest=cfg['dataset']['json_file'],source_manifest_sha256=sha(cfg['dataset']['json_file']),
        requests=str(requests),requests_sha256=sha(requests),feature_inputs=features,
        actual_student_and_ema=True,models_unchanged=True,device=str(device))
    save(output/'calibration.json',certificate)
    return certificate
