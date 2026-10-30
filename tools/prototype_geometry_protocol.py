"""源域校准和完成审计；所有门槛在读取target评估之前冻结。"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from tools import baseline_protocol as protocol
from libs.models.SADA.prototype_capacity import specification
from libs.models.SADA.prototype_geometry import SCHEMA, POLICY, VARIANT

read, save, sha = protocol.read, protocol.save, protocol.sha


def source_partitions(video_ids):
    ordered=sorted(set(video_ids),key=lambda v:hashlib.sha256(('123:'+v).encode()).hexdigest())
    if len(ordered)<2:
        raise ValueError('源域独立视频不足以划分校准与机制核验')
    return set(ordered[::2]),set(ordered[1::2])


def source_summary(rows):
    """相关性用源GT独立检查；tau只由校准视频的预测离散度确定。"""
    import numpy as np
    from scipy.stats import spearmanr
    fitting,checking=source_partitions([r['video_id'] for r in rows])
    valid=[r for r in rows if r['uncertainty'] is not None]
    fit=[r for r in valid if r['video_id'] in fitting]
    if not fit:
        raise ValueError('校准视频没有有效几何离散度')
    tau=float(np.median([r['uncertainty'] for r in fit]))
    if not math.isfinite(tau) or tau<=0:
        raise ValueError('校准视频的几何离散度中位数退化，不放宽尺度')
    audit=[r for r in valid if r['video_id'] in checking]
    # 在同视频/层内比较相近时长、原置信候选，避免只测跨域/长短差异。
    per_video={}
    for i,a in enumerate(audit):
        for b in audit[i+1:]:
            if a['video_id']!=b['video_id'] or a['level']!=b['level']:continue
            if max(a['duration'],b['duration'])/min(a['duration'],b['duration'])>2:continue
            if abs(a['confidence']-b['confidence'])>.1:continue
            du=a['uncertainty']-b['uncertainty'];di=a['iou']-b['iou']
            if du==0 or di==0:continue
            per_video.setdefault(a['video_id'],[]).append(float(du*di<0))
    concordance={v:float(np.mean(x)) for v,x in per_video.items()}
    rho=float(spearmanr([r['uncertainty'] for r in audit],[r['iou'] for r in audit])[0]) if len(audit)>2 else float('nan')
    return dict(tau=tau,fit_videos=sorted(fitting),audit_videos=sorted(checking),
        valid=len(valid),total=len(rows),missing=len(rows)-len(valid),
        audit_valid=len(audit),spearman_u_iou=rho if math.isfinite(rho) else None,
        matched_pair_video_concordance=concordance,
        audit_concordance=float(np.mean(list(concordance.values()))) if concordance else None)


def mechanism_gate(reports):
    import numpy as np
    values=[v for r in reports.values() for v in r['summary']['matched_pair_video_concordance'].values()]
    mean=float(np.mean(values)) if values else None
    # 仅作为进入训练的方向性源机制门禁，不冒充显著性/target可靠性保证。
    passed=set(reports)=={'S'+str(i) for i in range(1,7)} and bool(values) and mean>.5
    return dict(status='passed' if passed else 'no_source_mechanism_signal',
        source_only=True,video_equal_concordance=mean,contributing_shift_video_groups=len(values),
        rule='six_valid_calibrations_and_source_video_equal_matched_concordance_gt_0.5',
        matching=dict(same_video=True,same_fpn=True,duration_ratio_max=2.,confidence_gap_max=.1),
        not_a_target_risk_guarantee=True)


def config_for(control,calibration,identity,tmpdir):
    cfg=deepcopy(control)
    if cfg['prototype_capacity']['variant']!='control' or cfg['seed']!=123 or cfg['loader']['batch_size']!=2:
        raise ValueError('必须与seed123/batch2原版Prototype配对')
    cfg['baseline']['code']=identity;cfg['baseline']['tmpdir']=str(tmpdir)
    cfg['prototype_capacity'].update(variant=VARIANT,specification=specification(VARIANT),
        geometry_calibration=str(calibration),geometry_calibration_sha256=sha(calibration))
    return cfg


def assert_pair(cfg,control):
    expected=deepcopy(control)
    for key in ('code','tmpdir'):expected['baseline'][key]=cfg['baseline'][key]
    cal=cfg['prototype_capacity']
    expected['prototype_capacity'].update(variant=VARIANT,specification=specification(VARIANT),
        geometry_calibration=cal['geometry_calibration'],geometry_calibration_sha256=cal['geometry_calibration_sha256'])
    if cfg!=expected or cfg['baseline']['target_gt_used'] or cfg.get('oracle_auxiliary'):
        raise ValueError('配置超出几何权重/运行身份，或混入target GT oracle')
    proof=read(cal['geometry_calibration'])
    if proof['code']!=cfg['baseline']['code'] or proof['shift']!=cfg['baseline']['shift']:
        raise ValueError('源校准与当前代码/shift不绑定')


def verify_acceptance(root,cfg):
    proof=read(Path(root)/'acceptance.json')
    if proof['status']!='passed' or proof['code']!=cfg['baseline']['code']:
        raise ValueError('缺少当前clean代码真实验收')
    for path,digest in proof['reports'].items():
        if sha(path)!=digest:raise ValueError('真实GPU验收报告改变')
    if sha(Path(root)/'mechanism_gate.json')!=proof['mechanism_gate_sha256']:
        raise ValueError('源机制门禁改变')
    if read(Path(root)/'mechanism_gate.json')['status']!='passed':
        raise ValueError('源机制未通过，不能正式训练')


def audit_entry(entry):
    entry=Path(entry);record=read(entry/'complete.json');cfg=read(entry/'config.json')
    control=Path(record['control_complete']).parent
    if sha(control/'complete.json')!=record['control_complete_sha256']:
        raise ValueError('原版完成凭证改变')
    assert_pair(cfg,read(control/'config.json'))
    calibration=Path(cfg['prototype_capacity']['geometry_calibration'])
    if sha(calibration)!=cfg['prototype_capacity']['geometry_calibration_sha256']:
        raise ValueError('源校准改变')
    proof=read(calibration)
    if proof['policy']!=POLICY or proof['shift']!=cfg['baseline']['shift'] or proof['target_gt_used']:
        raise ValueError('几何方法身份或GT边界不符')
    for path,key in [(proof['source_pool'],'source_pool_sha256'),
                     (proof['source_requests'],'source_requests_sha256'),
                     (proof['source_manifest'],'source_manifest_sha256')]:
        if sha(path)!=proof[key]:raise ValueError('源校准输入改变')
    if sha(calibration.parent/'source_rows.json')!=proof['source_rows_sha256']:
        raise ValueError('源几何核验记录改变')
    verify_acceptance(entry.parents[2],cfg)
    out=protocol.audit_entry(entry,expected_schema=SCHEMA)
    import json
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if not any(r['auxiliary'].get('rq3_geometry_weighted_candidates',0)>0 and
               r['auxiliary'].get('rq3_target_gradient_norm',0)>0 for r in rows):
        raise ValueError('几何权重或真实target辅助梯度没有作用')
    if read(entry/'detector_initialization.json')['sha256']!=read(control/'detector_initialization.json')['sha256']:
        raise ValueError('共享检测器冷初始化不配对')
    out['variant']=VARIANT
    return out
