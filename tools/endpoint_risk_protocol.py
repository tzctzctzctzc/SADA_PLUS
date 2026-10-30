"""校准只用源视频；只有相对风险真的选出更好教师，才允许定位自训练。"""
from copy import deepcopy
import math
from pathlib import Path
import numpy as np

from libs.models.SADA.endpoint_risk import SCHEMA,POLICY,validate_options
from tools.baseline_protocol import read,save,sha


def config_for(control, options, identity, tmpdir):
    cfg=deepcopy(control);validate_options(options)
    cfg['endpoint_risk']=deepcopy(options)
    cfg['baseline']['code']=identity;cfg['baseline']['tmpdir']=str(tmpdir)
    assert_pair(cfg,control)
    return cfg


def audit_certificate(path):
    path=Path(path);cert=read(path);rows=read(path.parent/'source_rows.json')
    if cert['schema']!=SCHEMA or cert['policy']!=POLICY or cert['target_gt_used']:
        raise ValueError('源证据身份不符')
    if sha(path.parent/'source_rows.json')!=cert['source_rows_sha256']:
        raise ValueError('源逐端点记录改变')
    recomputed=calibrate(rows,cert['fit_videos'],cert['audit_videos'],cert['train_videos'],cert['margin'])
    for key in ('status','factors','source_audit'):
        if recomputed[key]!=cert[key]:raise ValueError('源教师优势证据不能独立复算')
    if not cert.get('actual_student_and_ema') or not cert.get('models_unchanged'):
        raise ValueError('不是实际训练保存点的独立源诊断')
    for key in ('checkpoint','source_manifest','source_pool','requests'):
        if sha(cert[key])!=cert[key+'_sha256']:raise ValueError('源机制绑定输入改变: '+key)
    for feature,digest in cert['feature_inputs'].items():
        if sha(feature)!=digest:raise ValueError('源诊断实际特征改变')
    return cert


def validate_installed_certificate(cert):
    if sha(cert['calibration_path'])!=cert['calibration_sha256']:
        raise ValueError('源校准工件改变')
    original=audit_certificate(cert['calibration_path'])
    if any(cert.get(k)!=v for k,v in original.items()):
        raise ValueError('保存点内风险尺度与原校准工件不一致')
    return original


def assert_pair(cfg, control):
    copy=deepcopy(cfg);options=copy.pop('endpoint_risk');validate_options(options)
    copy['baseline']['code']=control['baseline']['code']
    copy['baseline']['tmpdir']=control['baseline']['tmpdir']
    if copy!=control:raise ValueError('除端点风险/运行身份外改变原版Prototype配置')
    if (cfg['seed']!=123 or cfg['loader']['batch_size']!=2 or
        cfg['prototype_capacity']['variant']!='control' or cfg['baseline']['target_gt_used']):
        raise ValueError('必须沿用无targetGT的seed123/batch2原版Prototype')


def fit_factors(rows, fit_videos):
    result={}
    for level in sorted({x['level'] for x in rows}):
        group=[x for x in rows if x['video_id'] in fit_videos and x['level']==level]
        videos=sorted({x['video_id'] for x in group})
        item=dict(available=[False,False],student=[None,None],teacher=[None,None],videos=videos)
        if len(videos)>=2:
            for side in (0,1):
                okay=True
                for role in ('student','teacher'):
                    # 视频等权：让长视频/密集候选不支配尺度校准。
                    errors=[np.mean([x[role+'_error'][side] for x in group if x['video_id']==v]) for v in videos]
                    sigmas=[np.mean([x[role+'_sigma'][side] for x in group if x['video_id']==v]) for v in videos]
                    factor=float(np.mean(errors)/np.mean(sigmas))
                    if not math.isfinite(factor) or factor<=0:okay=False
                    item[role][side]=factor
                item['available'][side]=okay
        result[str(level)]=item
    return result


def check_rows(rows):
    keys=[(x['video_id'],x['candidate_id']) for x in rows]
    if len(set(keys))!=len(keys):raise ValueError('源候选重复')
    for row in rows:
        if row.get('target_gt_used',False) or row['duration']<=0:raise ValueError('源记录身份/时长非法')
        for key in ('student_error','teacher_error','student_sigma','teacher_sigma'):
            if len(row[key])!=2 or not all(math.isfinite(x) and x>=0 for x in row[key]):
                raise ValueError('误差/尺度记录非法')
        if min(*row['student_sigma'],*row['teacher_sigma'])<=0:raise ValueError('误差尺度必须正')


def gate_audit(rows, factors, margin):
    """主要效用是教师实际优于学生，并优于同分层覆盖率的随机选择。"""
    groups={}
    for row in rows:
        factor=factors.get(str(row['level']))
        if factor is None:continue
        for side in (0,1):
            if not factor['available'][side]:continue
            if not row.get('allowed',[True,True])[side]:continue
            duration=row['duration']
            predicted=(row['student_sigma'][side]*factor['student'][side]-
                       row['teacher_sigma'][side]*factor['teacher'][side])/duration
            gain=(row['student_error'][side]-row['teacher_error'][side])/duration
            key=(row['video_id'],row['level'],row.get('class_index',-1),side,math.floor(math.log2(duration)))
            groups.setdefault(key,[]).append(dict(selected=predicted>margin,gain=gain,
                teacher_risk=row['teacher_sigma'][side]*factor['teacher'][side]/duration))
    by_video={};selected=available=0
    for (video,level,category,side,duration_bin),values in groups.items():
        chosen=[x for x in values if x['selected']];n=len(chosen);available+=len(values);selected+=n
        if not n:continue
        gain=sum(x['gain'] for x in chosen)
        random_expected=n*float(np.mean([x['gain'] for x in values]))
        teacher_only=sum(x['gain'] for x in sorted(values,key=lambda x:x['teacher_risk'])[:n])
        entry=by_video.setdefault(video,dict(n=0,gain=0.,random_expected=0.,teacher_only=0.,better=0))
        entry['n']+=n;entry['gain']+=gain;entry['random_expected']+=random_expected
        entry['teacher_only']+=teacher_only;entry['better']+=sum(x['gain']>0 for x in chosen)
    matrix=np.asarray([[x['gain']/x['n'],(x['gain']-x['random_expected'])/x['n'],
                         (x['gain']-x['teacher_only'])/x['n']] for x in by_video.values()])
    result=dict(available_endpoints=available,selected_endpoints=selected,physical_videos=len(by_video),
                per_video=by_video,passed=False)
    if len(matrix)<3 or not selected:return result
    generator=np.random.RandomState(123)
    boot=np.stack([matrix[generator.randint(0,len(matrix),len(matrix))].mean(0) for _ in range(2000)])
    intervals=np.quantile(boot,[.025,.975],axis=0)
    result.update(mean_actual_advantage=float(matrix[:,0].mean()),
        mean_advantage_over_random=float(matrix[:,1].mean()),
        mean_advantage_over_teacher_only=float(matrix[:,2].mean()),
        advantage_ci95=intervals[:,0].tolist(),over_random_ci95=intervals[:,1].tolist(),
        over_teacher_only_ci95=intervals[:,2].tolist(),
        passed=bool(intervals[0,0]>0 and intervals[0,1]>0))
    return result


def calibrate(rows, fit_videos, audit_videos, train_videos, margin):
    check_rows(rows)
    fit,audit,train=map(set,(fit_videos,audit_videos,train_videos))
    if fit&audit or (fit|audit)&train or not fit or not audit:
        raise ValueError('源train/calibration/audit视频必须隔离')
    if {x['video_id'] for x in rows} != fit|audit:
        raise ValueError('源校准/审计成员不完整')
    if not math.isfinite(margin) or margin<0:raise ValueError('margin必须预先冻结且非负')
    factors=fit_factors(rows,fit)
    report=gate_audit([x for x in rows if x['video_id'] in audit],factors,margin)
    return dict(schema=SCHEMA,policy=POLICY,status='passed' if report['passed'] else 'rejected_no_teacher_advantage',
        target_gt_used=False,margin=margin,factors=factors,fit_videos=sorted(fit),
        audit_videos=sorted(audit),train_videos=sorted(train),source_audit=report,
        previously_used_for_source_model_selection=True,target_risk_guarantee=False)


def promote_checkpoint(source, calibration_path, output):
    """只给原保存点加源证据，原权重/历史/RNG/全部优化器保持逐字节同一状态。"""
    import torch
    from tools.baseline_engine import tensor_digest,checkpoint
    source,calibration_path,output=map(Path,(source,calibration_path,output))
    if output.exists() or source.resolve()==output.resolve():raise ValueError('禁止覆盖原保存点')
    cert=audit_certificate(calibration_path)
    if cert['status']!='passed' or cert['schema']!=SCHEMA or cert['checkpoint_sha256']!=sha(source):
        raise ValueError('源教师优势未通过或不对应这个真实师生保存点')
    if sha(calibration_path.parent/'source_rows.json')!=cert['source_rows_sha256']:
        raise ValueError('源风险记录改变')
    saved=torch.load(source,map_location='cpu')
    if saved.get('schema')!=SCHEMA or 'endpoint_risk' not in saved:raise ValueError('不是带实际误差头的保存点')
    if cert['code']!=saved['config']['baseline']['code']:raise ValueError('源诊断与训练不是同一clean代码')
    if saved['endpoint_risk']['calibration'] is not None:raise ValueError('不重复替换已冻结的校准')
    evidence=deepcopy(cert)
    evidence.update(calibration_path=str(calibration_path.resolve()),calibration_sha256=sha(calibration_path))
    before=tensor_digest({k:v for k,v in saved.items() if k!='endpoint_risk'})
    extra_before=tensor_digest({k:v for k,v in saved['endpoint_risk'].items() if k!='calibration'})
    saved['endpoint_risk']['calibration']=evidence
    output.parent.mkdir(parents=True,exist_ok=True);checkpoint(output,saved)
    loaded=torch.load(output,map_location='cpu')
    if tensor_digest({k:v for k,v in loaded.items() if k!='endpoint_risk'})!=before:
        raise ValueError('提升证据时改动了模型/历史/原优化器/RNG')
    if tensor_digest({k:v for k,v in loaded['endpoint_risk'].items() if k!='calibration'})!=extra_before:
        raise ValueError('提升证据时改动了误差头优化器/独立随机流')
    proof=dict(status='passed',source=cert['checkpoint'],source_sha256=sha(source),
        promoted_sha256=sha(output),calibration_sha256=sha(calibration_path),
        all_training_state_unchanged=True)
    save(output.with_suffix('.promotion.json'),proof)
    return proof


def assert_formal_ready(cfg, saved, acceptance):
    """CPU合成测试不能代替真实GPU验收；配置、源码和源证据均须绑定。"""
    from tools.baseline_engine import tensor_digest
    if saved.get('schema')!=SCHEMA or saved['config']!=cfg:raise ValueError('不是当前配置的严格保存点')
    extra=saved.get('endpoint_risk',{});cert=extra.get('calibration')
    if not cert or cert['status']!='passed' or cert['code']!=cfg['baseline']['code']:
        raise ValueError('缺少当前代码的源机制证据')
    validate_installed_certificate(cert)
    if not acceptance or acceptance.get('status')!='passed' or acceptance.get('device')!='cuda':
        raise ValueError('缺少真实GPU验收，CPU测试不能正式放行')
    if (acceptance['code']!=cfg['baseline']['code'] or acceptance['config_sha256']!=tensor_digest(cfg)
        or acceptance['calibration_sha256']!=cert['calibration_sha256']):
        raise ValueError('验收未绑定本配置/代码/校准')
    required=['target_gt_zero','risk_head_updated','teacher_detached','rng_roundtrip']
    required+=['head_only_no_pseudo'] if cfg['endpoint_risk']['mode']=='head_only' else ['pseudo_gradient_nonzero']
    if cfg['endpoint_risk']['mode']=='random':required.append('random_gate_actually_changed')
    for key in required:
        if acceptance.get(key) is not True:raise ValueError('GPU验收缺少真实检查: '+key)
    if len(acceptance['reports'])!=1:raise ValueError('必须绑定唯一原始GPU验收报告')
    for path,digest in acceptance['reports'].items():
        if sha(path)!=digest:raise ValueError('真实GPU诊断记录改变')
        report=read(path)
        if {k:v for k,v in acceptance.items() if k!='reports'}!=report:
            raise ValueError('放行凭证与原始GPU验收报告不一致')
    if sha(acceptance['checkpoint_path'])!=acceptance['checkpoint_sha256']:
        raise ValueError('真实验收的起始保存点改变')
    if sha(acceptance['gpu_guard'])!=acceptance['gpu_guard_sha256']:
        raise ValueError('首CUDA凭证改变')
    guard=read(acceptance['gpu_guard']);lease=read(guard['lease'])
    if (sha(guard['lease'])!=guard['lease_sha256'] or lease['commit']!=cfg['baseline']['code']['commit']
        or guard['observed']['uuid']!=lease['observed']['uuid']
        or guard['observed']['free_mib']<lease['minimum_free_mib']):
        raise ValueError('真实GPU准入不符')


def audit_entry(entry):
    import torch
    from tools import baseline_protocol as base
    entry=Path(entry);record=read(entry/'complete.json');cfg=read(entry/'config.json')
    reference=Path(record['control_complete'])
    if sha(reference)!=record['control_complete_sha256']:raise ValueError('原版配对完成凭证改变')
    assert_pair(cfg,read(reference.parent/'config.json'))
    if read(entry/'detector_initialization.json')['sha256']!=read(reference.parent/'detector_initialization.json')['sha256']:
        raise ValueError('新增头改变了原版检测器冷初始化')
    acceptance_path=Path(record['gpu_acceptance'])
    latest=torch.load(entry/'latest.pt',map_location='cpu')
    assert_formal_ready(cfg,latest,read(acceptance_path))
    for name in ('best','latest'):
        saved=torch.load(entry/(name+'.pt'),map_location='cpu')
        if saved.get('schema')!=SCHEMA or 'endpoint_risk' not in saved:
            raise ValueError('保存点缺少风险头/独立优化器/随机流')
    rows=[__import__('json').loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if not all('endpoint_risk' in r for r in rows):raise ValueError('缺少逐轮风险诊断')
    if not any(r['endpoint_risk'].get('head_gradient_norm',0)>0 for r in rows):
        raise ValueError('误差头未真实学习')
    if cfg['endpoint_risk']['mode']!='head_only' and not any(
        r['endpoint_risk'].get('pseudo_gradient_norm',0)>0 and r['endpoint_risk'].get('selected_endpoints',0)>0 for r in rows):
        raise ValueError('目标伪定位监督未生效')
    result=base.audit_entry(entry,expected_schema=SCHEMA,acceptance_path=acceptance_path)
    result['variant']=cfg['endpoint_risk']['mode']
    return result
