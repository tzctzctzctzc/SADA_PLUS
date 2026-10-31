"""源GT上下文迁移的冻结配置、真实验收及完整结果身份。"""
from copy import deepcopy
import json
import math
from pathlib import Path
from libs.models.SADA.context_transfer import SCHEMA,POLICY
from tools import baseline_protocol as p


def assert_pair(cfg,base):
    other=deepcopy(cfg)
    if other.pop('context_transfer')!=dict(schema=SCHEMA,policy=POLICY):raise ValueError('上下文配方改变')
    if base['seed']!=123 or base['loader']['batch_size']!=2 or base['prototype_capacity']['variant']!='control':
        raise ValueError('只配对原Prototype seed123/batch2')
    if any(cfg.get(k) for k in ('phase_alignment','phase_control','endpoint_risk','background_oracle','instance_task')):
        raise ValueError('新方法不能混入其它探索分支')
    for k in ('code','tmpdir'):other['baseline'][k]=base['baseline'][k]
    if other!=base or cfg['baseline']['target_gt_used']:raise ValueError('声明机制以外配置改变')


def source_check(cfg):
    from tools.baseline_engine import dataset
    from libs.models.SADA.context_transfer import SourceContextTransfer,feature_ground_truth,paste_pair
    helper=SourceContextTransfer(cfg,10)
    source=dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],True)
    target=dataset(cfg,cfg['baseline']['target_manifest'],cfg['tgt_train_split'],True,True)
    if set(helper.annotations)!={v['id'] for v in source.data_list}:raise ValueError('训练GT混入源val或其它域')
    # 只做真实数据的坐标/成员/完整GT检查，无目标预测、无学习门限。
    for index in range(min(8,len(source),len(target))):
        src,tgt=source[index],target[index];p.assert_no_target_gt([tgt])
        gt,labels=feature_ground_truth(helper.annotations[src['metadata']['video_id']],src['metadata'],10,'cpu')
        mixed=paste_pair(src,tgt,gt,labels,5,index)
        if mixed is None:continue
        video,payload=mixed;p.assert_no_target_gt([video]);a,b=payload['original_source_patch'];c,d=payload['pasted']
        import torch
        if not torch.equal(src['feats'][:,a:b],video['feats'][:,c:d]):raise ValueError('真实复制特征改变')
        if not (len(payload['labels'])>0 and bool(payload['anchor_segment'][0]>=c+1) and bool(payload['anchor_segment'][1]<=d-1)
            and c>=1 and d<=tgt['feats'].shape[-1]-1):
            raise ValueError('混合GT不是完整内部区间')
        return dict(status='passed',source_video=src['metadata']['video_id'],target_context_video=tgt['metadata']['video_id'],
            source_gt_from_train_only=True,target_gt_used=False,exact_source_feature_patch=True,
            complete_instances=len(payload['labels']),partial_instances=len(payload['partial']),
            pasted=payload['pasted'],source_patch=payload['original_source_patch'],translation=payload['translation'],
            source_videos=len(source),target_videos=len(target))
    raise ValueError('真实源/目标样本中无法构造完整源动作patch；拒绝启动')


def prepare(root,control,identity):
    final=p.read(control/'final_audit.json')
    previous={r['shift']:r for r in final.get('results',[]) if r.get('variant')=='control'}
    if final['status']!='passed' or set(previous)!={'S'+str(i) for i in range(1,7)}:raise ValueError('原B六项终审不完整')
    configs={};checks={}
    for shift in ['S'+str(i) for i in range(1,7)]:
        ref=control/'formal/control'/shift;base=p.read(ref/'config.json');done=p.read(ref/'complete.json')
        if previous[shift]['complete_sha256']!=p.sha(ref/'complete.json'):raise ValueError('原B完成项改变')
        paths=[ref/'config.json',ref/'best.pt',ref/'evaluation/target_val.json',ref/'detector_initialization.json']
        for path in paths:
            if p.sha(path)!=done['artifacts'][str(path)]:raise ValueError('原B工件改变: '+str(path))
        for path,digest in [(base['dataset']['json_file'],base['baseline']['source_manifest_sha256']),
            (base['baseline']['target_manifest'],base['baseline']['target_manifest_sha256']),
            (base['prototype_capacity']['prepared_cache'],base['prototype_capacity']['prepared_cache_sha256'])]:
            if p.sha(path)!=digest:raise ValueError('冻结数据/缓存改变')
        for path,digest in p.read(base['prototype_capacity']['prepared_cache'])['source_artifacts'].items():
            if p.sha(path)!=digest:raise ValueError('原VLM缓存来源工件改变')
        cfg=deepcopy(base);cfg['baseline'].update(code=identity,tmpdir=str(root/'tmp'))
        cfg['context_transfer']=dict(schema=SCHEMA,policy=deepcopy(POLICY));assert_pair(cfg,base)
        checks[shift]=source_check(cfg);configs[shift]=cfg
        p.save(root/'configs'/(shift+'.json'),cfg)
        p.save(root/'references'/(shift+'.json'),dict(control_complete=str(ref/'complete.json'),
            complete_sha256=p.sha(ref/'complete.json'),best_checkpoint_sha256=p.sha(ref/'best.pt'),
            config_sha256=p.sha(ref/'config.json'),source_manifest_sha256=base['baseline']['source_manifest_sha256'],
            target_manifest_sha256=base['baseline']['target_manifest_sha256']))
    p.save(root/'source_gate.json',dict(status='passed',schema=SCHEMA,policy=POLICY,code=identity,checks=checks,
        nature='real_source_GT_geometry_and_isolation_check_not_target_effect_proof',target_gt_used=False))
    return configs


def verify_probe(report,initial_detector_sha):
    if report['status']!='passed' or report['initial_detector_sha']!=initial_detector_sha:
        raise ValueError('冷初始化/真实验收不配对')
    if report['diagnostic_epochs']!=[1,2,3,4,5] or report['extra_timing_warmup_steps']!=0:
        raise ValueError('必须连续完整1–5轮验收，不跳warmup或增加timing步骤')
    if report['rng_roundtrip']['status']!='passed':raise ValueError('完整CUDA/RNG恢复失败')
    for row in report['phases'].values():
        if row['target_gt_losses']!=[0.,0.,0.] or not row['model_updated'] or not math.isfinite(row['source_gradient_norm']) or row['source_gradient_norm']<=0:
            raise ValueError('真实监督/GT隔离失败')
    active=report['phases']['5'];aux=active['auxiliary'];mix=active['context_transfer']
    if aux['rq3_target_gradient_norm']<=0 or not active['branch_parameters_updated']:
        raise ValueError('原Prototype未真实更新')
    for key in ('pairs','complete_instances','copied_tokens','positive_points','known_negative_points',
                'cls_loss','reg_loss','feature_gradient_norm','cls_head_gradient_norm','reg_head_gradient_norm'):
        if not math.isfinite(mix[key]) or mix[key]<=0:raise ValueError('混合监督未真正生效: '+key)
    if mix['source_gt_only']!=1 or mix['target_gt_used']!=0:raise ValueError('混合监督使用了目标GT')


def audit(entry):
    import torch
    entry=Path(entry);done=p.read(entry/'complete.json');cfg=p.read(entry/'config.json')
    reference=Path(done['control_complete'])
    if p.sha(reference)!=done['control_complete_sha256']:raise ValueError('原B完成项改变')
    assert_pair(cfg,p.read(reference.parent/'config.json'))
    for path,digest in p.read(cfg['prototype_capacity']['prepared_cache'])['source_artifacts'].items():
        if p.sha(path)!=digest:raise ValueError('固定缓存来源工件改变')
    old_eval=reference.parent/'evaluation/target_val.json'
    if p.sha(old_eval)!=p.read(reference)['artifacts'][str(old_eval)]:raise ValueError('原B评估改变')
    if p.read(entry/'detector_initialization.json')['sha256']!=p.read(reference.parent/'detector_initialization.json')['sha256']:
        raise ValueError('共同检测器冷初始化不一致')
    gate=p.read(done['source_gate']);group=p.read(done['group_acceptance'])
    if p.sha(done['source_gate'])!=done['source_gate_sha256'] or gate['status']!='passed' or gate['policy']!=POLICY or set(gate['checks'])!={'S'+str(i) for i in range(1,7)}:
        raise ValueError('六向源GT完整性检查改变')
    if p.sha(done['group_acceptance'])!=done['group_acceptance_sha256'] or group['status']!='passed' or len(group['acceptance'])!=6:
        raise ValueError('正式前没有精确六向共同验收')
    for path,digest in group['acceptance'].items():
        if p.sha(path)!=digest:raise ValueError('共同验收SHA改变')
    acceptance=p.read(entry/'acceptance.json');probe=p.read(entry/'probe_result.json')
    if acceptance['report_sha256']!=p.sha(entry/'probe_result.json') or acceptance['source_gate_sha256']!=done['source_gate_sha256']:
        raise ValueError('验收报告/源证明改变')
    verify_probe(probe,p.read(reference.parent/'detector_initialization.json')['sha256'])
    rows=[json.loads(line) for line in (entry/'metrics.jsonl').read_text().splitlines()]
    active=[r['context_transfer'] for r in rows if r['epoch']>=cfg['train_cfg']['warmup_adv']]
    if not active or not any(r['positive_points']>0 and r['feature_gradient_norm']>0 for r in active):
        raise ValueError('正式训练没有有效上下文任务监督')
    if any(r['target_gt_used']!=0 or r['source_gt_only']!=1 for r in active):raise ValueError('正式混合GT来源错误')
    minimum=max(8192,math.ceil(probe['peak_reserved_mib']+2048))
    receipt=p.read(entry/'train_gpu.json');guard=p.read(receipt['guard']);lease=p.read(guard['lease'])
    if lease['minimum_free_mib']<minimum or lease['minimum_free_mib']!=acceptance['minimum_free_mib']:
        raise ValueError('显存准入不覆盖真实验收峰值')
    for name in ('best','latest'):
        checkpoint=torch.load(entry/(name+'.pt'),map_location='cpu')
        if checkpoint['schema']!=SCHEMA:raise ValueError('上下文方法checkpoint身份错误')
    result=p.audit_entry(entry,expected_schema=SCHEMA,acceptance_path=entry/'acceptance.json')
    result.update(variant='source_context_transfer',target_gt_used=False);return result
