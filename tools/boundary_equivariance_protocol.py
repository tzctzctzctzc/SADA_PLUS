"""冻结边界分布配方，复用已通过的源支持证据；正式结果使用独立身份。"""
from copy import deepcopy
from pathlib import Path
import math
from libs.models.SADA.boundary_equivariance import SCHEMA,POLICY
from tools import baseline_protocol as p

SOURCE=Path('/vepfs/vlms/boundary_support_source6_0842ecb_1031')
SHIFTS={'S'+str(i) for i in range(1,7)}


def assert_pair(cfg,base):
    other=deepcopy(cfg)
    if other.pop('boundary_equivariance')!=dict(schema=SCHEMA,policy=POLICY):raise ValueError('边界配方改变')
    if base['seed']!=123 or base['loader']['batch_size']!=2 or base['prototype_capacity']['variant']!='control':raise ValueError('必须配对原B')
    if any(cfg.get(k) for k in ('context_transfer','endpoint_risk','phase_alignment','phase_control','background_oracle')):raise ValueError('不能混入其它探索')
    for k in ('code','tmpdir'):other['baseline'][k]=base['baseline'][k]
    if other!=base or cfg['baseline']['target_gt_used']:raise ValueError('配方以外配置改变')


def verify_source(root=SOURCE):
    final=p.read(root/'monitor/independent_final_audit.json');decision=p.read(root/'source_decision.json')
    if final['status']!='passed' or final['count']!=6 or decision['status']!='passed' or p.sha(root/'source_decision.json')!=final['source_decision_sha256']:raise ValueError('源支持未全六终审')
    if {r['shift'] for r in decision['completed']}!=SHIFTS:raise ValueError('源支持身份不完整')
    for row in decision['completed']:
        path=Path(row['result']);audit=path.parent/'source_audit.json'
        if p.sha(path)!=row['sha256'] or p.sha(audit)!=row['audit_sha256']:raise ValueError('源支持签名漂移')
        d=p.read(path)
        if not d['gpu_head_probe']['passed'] or d['model_sha256_before']!=d['model_sha256_after'] or d['target_data_read']:raise ValueError('源支持隔离或GPU检查失败')
    return dict(source_root=str(root),source_decision_sha256=p.sha(root/'source_decision.json'),independent_final_sha256=p.sha(root/'monitor/independent_final_audit.json'))


def prepare(root,control,identity):
    source=verify_source();final=p.read(control/'final_audit.json');old={r['shift']:r for r in final['results'] if r.get('variant')=='control'}
    if final['status']!='passed' or set(old)!=SHIFTS:raise ValueError('原B终审不全')
    configs={}
    for shift in sorted(SHIFTS):
        ref=control/'formal/control'/shift;done=p.read(ref/'complete.json');base=p.read(ref/'config.json')
        if p.sha(ref/'complete.json')!=old[shift]['complete_sha256']:raise ValueError('原B完成SHA漂移')
        for name in ('config.json','best.pt','evaluation/target_val.json','detector_initialization.json'):
            if p.sha(ref/name)!=done['artifacts'][str(ref/name)]:raise ValueError('原B工件漂移')
        for path,digest in [(base['dataset']['json_file'],base['baseline']['source_manifest_sha256']),
            (base['baseline']['target_manifest'],base['baseline']['target_manifest_sha256']),
            (base['prototype_capacity']['prepared_cache'],base['prototype_capacity']['prepared_cache_sha256'])]:
            if p.sha(path)!=digest:raise ValueError('冻结输入改变')
        for path,digest in p.read(base['prototype_capacity']['prepared_cache'])['source_artifacts'].items():
            if p.sha(path)!=digest:raise ValueError('VLM缓存来源改变')
        cfg=deepcopy(base);cfg['baseline'].update(code=identity,tmpdir=str(root/'tmp'));cfg['boundary_equivariance']=dict(schema=SCHEMA,policy=deepcopy(POLICY));assert_pair(cfg,base)
        configs[shift]=cfg;p.save(root/'configs'/(shift+'.json'),cfg)
    p.save(root/'source_gate.json',dict(status='passed',schema=SCHEMA,policy=POLICY,code=identity,source_proof=source))
    return configs


def verify_probe(report,initial_detector_sha):
    if report['status']!='passed' or report['initial_detector_sha']!=initial_detector_sha or report['diagnostic_epochs']!=[1,2,3,4,5] or report['extra_timing_warmup_steps']!=0:raise ValueError('真实连续五轮/冷初始化不符')
    if report['rng_roundtrip']['status']!='passed':raise ValueError('完整GPU状态恢复未通过')
    for phase in report['phases'].values():
        if phase['target_gt_losses']!=[0.,0.,0.] or not phase['boundary_head_updated'] or not math.isfinite(phase['source_gradient_norm']) or phase['source_gradient_norm']<=0:raise ValueError('源监督或新头未更新')
        if phase['boundary_equivariance']['source_distribution_loss']<=0:raise ValueError('源分布监督无效')
    active=report['phases']['5'];b=active['boundary_equivariance']
    for k in ('target_points','target_views','target_equivariance_loss','target_feature_gradient_norm','target_head_gradient_norm'):
        if not math.isfinite(b[k]) or b[k]<=0:raise ValueError('目标等变没有真实作用: '+k)
    for k in ('target_rng_restored','target_normalizers_unchanged','target_prototype_state_unchanged'):
        if b[k]!=1:raise ValueError('额外视图破坏原状态')
    if active['auxiliary']['rq3_target_gradient_norm']<=0 or not active['branch_parameters_updated']:raise ValueError('原Prototype未有效训练')


def audit(entry):
    import torch
    entry=Path(entry);done=p.read(entry/'complete.json');cfg=p.read(entry/'config.json');ref=Path(done['control_complete'])
    if p.sha(ref)!=done['control_complete_sha256']:raise ValueError('原B完成项改变')
    assert_pair(cfg,p.read(ref.parent/'config.json'))
    gate=p.read(done['source_gate'])
    if p.sha(done['source_gate'])!=done['source_gate_sha256'] or gate['source_proof']!=verify_source():raise ValueError('源支持证据改变')
    group=p.read(done['group_acceptance'])
    if p.sha(done['group_acceptance'])!=done['group_acceptance_sha256'] or len(group['acceptance'])!=6:raise ValueError('全六验收不存在')
    if {Path(path).parent.name for path in group['acceptance']}!=SHIFTS:raise ValueError('验收方向不全')
    for path,digest in group['acceptance'].items():
        if p.sha(path)!=digest:raise ValueError('验收SHA漂移')
    acceptance=p.read(entry/'acceptance.json')
    if acceptance['report_sha256']!=p.sha(entry/'probe_result.json'):raise ValueError('probe报告漂移')
    verify_probe(p.read(entry/'probe_result.json'),p.read(ref.parent/'detector_initialization.json')['sha256'])
    result=p.audit_entry(entry,expected_schema=SCHEMA,acceptance_path=entry/'acceptance.json')
    for name in ('best','latest'):
        saved=torch.load(entry/(name+'.pt'),map_location='cpu')
        if saved['schema']!=SCHEMA or not saved['state_dict']['boundary_equivariance.head.logits.weight'].abs().sum():raise ValueError('新头保存或身份错误')
    result.update(variant='boundary_distribution_equivariance',target_gt_used=False);return result
