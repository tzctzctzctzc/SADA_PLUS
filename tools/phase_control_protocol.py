"""角色盲/置换两个预定机制对照；保留全部结果，不重训原六项P或基线。"""
from copy import deepcopy
from pathlib import Path
import json,math
from libs.models.SADA.phase_alignment import CONTROL_SCHEMA,CONTROL_MODES,control_specification
from tools import baseline_protocol as p
from tools.phase_alignment_protocol import verify_source_gate,verify_probe


def assert_pair(cfg,semantic):
    other=deepcopy(cfg);control=other.pop('phase_control')
    if control!=control_specification(control['mode']):raise ValueError('机制对照配方改变')
    for key in ['code','tmpdir']:other['baseline'][key]=semantic['baseline'][key]
    if other!=semantic:raise ValueError('改变了角色以外的P配置')
    if cfg['seed']!=123 or cfg['loader']['batch_size']!=2 or cfg['baseline']['target_gt_used']:
        raise ValueError('对照的seed/batch/目标GT协议错误')


def prepare(root,semantic_root,identity):
    independent=p.read(semantic_root/'monitor/independent_final_audit.json')
    final=p.read(semantic_root/'final_audit.json')
    if independent['status']!='passed' or independent['count']!=6 or p.sha(semantic_root/'final_audit.json')!=independent['final_audit_sha256']:
        raise ValueError('原P没有独立六方向全审')
    if final['status']!='passed' or final['count']!=6 or {r['shift'] for r in final['results']}!={'S'+str(i) for i in range(1,7)}:
        raise ValueError('P不是精确六方向')
    verify_source_gate(semantic_root/'source_gate.json')
    configs={mode:{} for mode in CONTROL_MODES}
    for row in final['results']:
        entry=Path(row['complete']).parent;shift=row['shift'];done=p.read(entry/'complete.json')
        if p.sha(entry/'complete.json')!=row['complete_sha256']:raise ValueError('P complete改变')
        for name in ['config.json','detector_initialization.json','evaluation/target_val.json']:
            if p.sha(entry/name)!=done['artifacts'][str(entry/name)]:raise ValueError('P工件改变:'+name)
        base=p.read(entry/'config.json');cache=Path(base['prototype_capacity']['prepared_cache'])
        if p.sha(cache)!=base['prototype_capacity']['prepared_cache_sha256']:raise ValueError('旧40/200缓存改变')
        for mode in CONTROL_MODES:
            cfg=deepcopy(base);cfg['baseline'].update(code=identity,tmpdir=str(root/'tmp'))
            cfg['phase_control']=control_specification(mode);assert_pair(cfg,base)
            configs[mode][shift]=cfg;p.save(root/'configs'/mode/(shift+'.json'),cfg)
    p.save(root/'predecessor.json',dict(semantic_root=str(semantic_root),
        independent_final_audit_sha256=p.sha(semantic_root/'monitor/independent_final_audit.json'),
        final_audit_sha256=p.sha(semantic_root/'final_audit.json'),
        source_gate=str(semantic_root/'source_gate.json'),source_gate_sha256=p.sha(semantic_root/'source_gate.json'),
        source_diagnostic_reused=True,source_diagnostic_code='f71c0e146ba43501330dd1e5112398e9b76559e6',
        reason='诊断只查相同成熟源模型的坐标；此处未改变源数据或角色坐标定义，控制只作用训练DA',
        completed=[dict(shift=r['shift'],complete=r['complete'],sha256=r['complete_sha256']) for r in final['results']]))
    return configs


def audit(entry):
    import torch
    e=Path(entry);done=p.read(e/'complete.json');cfg=p.read(e/'config.json');mode=cfg['phase_control']['mode']
    if done['schema']!=CONTROL_SCHEMA or done['variant']!=mode:raise ValueError('机制对照身份错误')
    ref=Path(done['semantic_complete']);old=p.read(ref)
    if p.sha(ref)!=done['semantic_complete_sha256']:raise ValueError('P完成项改变')
    assert_pair(cfg,p.read(ref.parent/'config.json'))
    for name in ['evaluation/target_val.json','detector_initialization.json']:
        if p.sha(ref.parent/name)!=old['artifacts'][str(ref.parent/name)]:raise ValueError('P比较工件改变')
    if p.read(e/'detector_initialization.json')['sha256']!=p.read(ref.parent/'detector_initialization.json')['sha256']:
        raise ValueError('共同检测器冷初始化不配对')
    if p.sha(done['source_gate'])!=done['source_gate_sha256']:raise ValueError('复用源门禁改变')
    verify_source_gate(done['source_gate'])
    group=p.read(done['group_acceptance'])
    if p.sha(done['group_acceptance'])!=done['group_acceptance_sha256'] or group['status']!='passed' or group['mode']!=mode or set(group['shifts'])!={'S'+str(i) for i in range(1,7)}:
        raise ValueError('不是本模式精确六路验收')
    for path,digest in group['acceptance'].items():
        if p.sha(path)!=digest:raise ValueError('组验收工件改变')
    acceptance=p.read(e/'acceptance.json');report=p.read(e/'probe_result.json')
    if acceptance['report_sha256']!=p.sha(e/'probe_result.json') or acceptance['source_gate_sha256']!=done['source_gate_sha256']:
        raise ValueError('真实验收签名错误')
    verify_probe(report,p.read(ref.parent/'detector_initialization.json')['sha256'],mode)
    receipt=p.read(e/'train_gpu.json');lease=p.read(p.read(receipt['guard'])['lease'])
    if lease['minimum_free_mib']!=acceptance['minimum_free_mib'] or lease['minimum_free_mib']<max(8192,math.ceil(report['peak_reserved_mib']+2048)):
        raise ValueError('正式显存准入未覆盖峰值')
    rows=[json.loads(x) for x in (e/'metrics.jsonl').read_text().splitlines()]
    active=[x for x in rows if x['epoch']>=cfg['train_cfg']['warmup_adv']]
    if not active or not all(x['phase_discriminator_updated'] for x in active):raise ValueError('正式D没有实际更新')
    key='nonuniform_input' if mode=='role_blind' else 'permutation_l1'
    total=sum(sum(v for k,v in row['phase_alignment'].items() if k.startswith('phase_control_'+key+'_')) for row in active)
    if mode=='role_blind' and total!=0:raise ValueError('角色盲模式读取了角色sector')
    if mode=='role_permuted' and (total<=0 or active[-1]['phase_permutation_counter']<=0):raise ValueError('置换未执行')
    for name in ['best','latest']:
        saved=torch.load(e/(name+'.pt'),map_location='cpu')
        if saved['schema']!=CONTROL_SCHEMA or 'phase_permutation_counter' not in saved['model_controls']:
            raise ValueError('对照保存点缺少身份/独立置换随机控制状态')
    result=p.audit_entry(e,expected_schema=CONTROL_SCHEMA,acceptance_path=e/'acceptance.json')
    result.update(variant=mode,target_gt_used=False);return result
