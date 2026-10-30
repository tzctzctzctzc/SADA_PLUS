"""只增加一个冻结的前景对齐配方；保留原版六项完成工件，不重训对照。"""
from copy import deepcopy
from pathlib import Path
import json
import math
from libs.models.SADA.phase_alignment import SCHEMA,POLICY
from tools import baseline_protocol as p

SOURCE_GATE=dict(videos_per_shift=12, minimum_videos=2, minimum_positive_points=64,
                 minimum_true_phase_std=.03, mean_mae_advantage_over_constant_must_be_positive=True,
                 minimum_independent_videos_with_unit_mass_per_side_role=2,
                 target_gt_used=False, purpose='reject_uninformative_coordinate_not_target_gain_proof')


def assert_pair(cfg,base):
    other=deepcopy(cfg)
    if other.pop('phase_alignment')!=dict(schema=SCHEMA,policy=POLICY):raise ValueError('角色协议改变')
    if base['seed']!=123 or base['loader']['batch_size']!=2 or base['prototype_capacity']['variant']!='control':
        raise ValueError('只配对原版seed123/batch2 Prototype')
    if cfg['baseline']['target_gt_used'] or cfg.get('background_oracle') or cfg.get('endpoint_risk'):
        raise ValueError('角色实验不能混入oracle或风险门控')
    for k in ['code','tmpdir']:other['baseline'][k]=base['baseline'][k]
    if other!=base:raise ValueError('角色以外的训练配置改变')


def prepare(root,control,identity):
    configs={}
    final=p.read(control/'final_audit.json')
    previous={r['shift']:r for r in final.get('results',[]) if r.get('variant')=='control'}
    if final['status']!='passed' or set(previous)!={'S'+str(i) for i in range(1,7)}:
        raise ValueError('原版六方向没有完整终审')
    for shift in ['S'+str(i) for i in range(1,7)]:
        ref=control/'formal/control'/shift;base=p.read(ref/'config.json');done=p.read(ref/'complete.json')
        if previous[shift]['complete']!=str(ref/'complete.json') or previous[shift]['complete_sha256']!=p.sha(ref/'complete.json'):
            raise ValueError('原版完成项与原终审SHA不符')
        for path,digest in [(ref/'config.json',done['artifacts'][str(ref/'config.json')]),
            (ref/'evaluation/target_val.json',done['artifacts'][str(ref/'evaluation/target_val.json')]),
            (ref/'detector_initialization.json',done['artifacts'][str(ref/'detector_initialization.json')]),
            (Path(base['baseline']['target_manifest']),base['baseline']['target_manifest_sha256']),
            (Path(base['dataset']['json_file']),base['baseline']['source_manifest_sha256']),
            (Path(base['prototype_capacity']['prepared_cache']),base['prototype_capacity']['prepared_cache_sha256']),
            (ref/'best.pt',done['artifacts'][str(ref/'best.pt')])]:
            if p.sha(path)!=digest:raise ValueError('冻结原输入改变: '+str(path))
        cfg=deepcopy(base);cfg['baseline'].update(code=identity,tmpdir=str(root/'tmp'))
        cfg['phase_alignment']=dict(schema=SCHEMA,policy=deepcopy(POLICY));assert_pair(cfg,base)
        configs[shift]=cfg
        p.save(root/'configs'/(shift+'.json'),cfg)
        p.save(root/'references'/(shift+'.json'),dict(control_complete=str(ref/'complete.json'),
            complete_sha256=p.sha(ref/'complete.json'),best_checkpoint=str(ref/'best.pt'),
            checkpoint_sha256=p.sha(ref/'best.pt'),target_gt_used=False))
    return configs


def diagnose(cfg,reference,output):
    """原成熟source-best EMA在source留出集的坐标可用性；无优化/无target读取。"""
    import inspect,torch
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.phase_alignment import role_probabilities,expand_reference_state
    from libs.trainer.da_trainer import move_to_device
    from libs.utils import fix_random_seed
    from tools.baseline_engine import dataset,loader
    fix_random_seed(cfg['seed'],include_cuda=True);torch.cuda.manual_seed(cfg['seed'])
    saved=torch.load(reference/'best.pt',map_location='cpu')
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').to('cuda:0')
    model.load_state_dict(expand_reference_state(saved['ema_state_dict']),strict=True);model.eval()
    # 只需真实回归坐标；验证forward不运行Prototype或DA反传。
    original=model.losses;signature=inspect.signature(original);rows=[]
    observed=[]
    def observed_loss(*args,**kwargs):
        b=signature.bind(*args,**kwargs).arguments
        start=0
        for level,masks in enumerate(b['fpn_masks']):
            width=masks.shape[-1]
            labels=b['gt_cls_labels'][0][start:start+width]
            offsets=b['gt_offsets'][0][start:start+width]
            pos=(labels.sum(-1)>0)&masks[0]
            if pos.any():
                gtq,gtu=role_probabilities(offsets[pos]);pq,pu=role_probabilities(b['out_offsets'][level][0][pos])
                observed.append(dict(level=level,gt=gtu.cpu().tolist(),pred=pu.cpu().tolist(),
                    classes=labels[pos].argmax(-1).cpu().tolist(),
                    role_mass=gtq.sum(0).cpu().tolist(),role_sum_squared=gtq.square().sum(0).cpu().tolist()))
            start+=width
        return original(*args,**kwargs)
    model.losses=observed_loss
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False)
    dl=loader(ds,False,torch.Generator().manual_seed(cfg['seed']),1,cfg['loader']['num_workers'])
    with torch.no_grad():
        for i,video in enumerate(dl):
            if i>=SOURCE_GATE['videos_per_shift']:break
            observed.clear();name=video[0]['metadata']['video_id']
            source=move_to_device(video,'cuda:0');target=deepcopy(source)
            for item in target:
                item['labels']=torch.empty(0,dtype=torch.long,device='cuda:0');item['segments']=torch.empty(0,2,device='cuda:0')
            p.assert_no_target_gt(target)
            model(video_list_src=source,video_list_tgt=target,mode='valid',domain='src',curr_epoch=saved['epoch'])
            flat=[(g,v) for row in observed for g,v in zip(row['gt'],row['pred'])]
            if flat:
                rows.append(dict(video_id=name,points=len(flat),mae=sum(abs(g-v) for g,v in flat)/len(flat),
                    constant_mae=sum(abs(g-.5) for g,v in flat)/len(flat),levels=deepcopy(observed)))
            print('SOURCE_PHASE_AUDIT',cfg['baseline']['shift'],i,name,len(flat),flush=True)
    gt=[g for row in rows for level in row['levels'] for g in level['gt']]
    mean=sum(gt)/max(len(gt),1);std=math.sqrt(sum((v-mean)**2 for v in gt)/max(len(gt),1))
    report=dict(status='completed_source_only_diagnostic',schema=SCHEMA,shift=cfg['baseline']['shift'],
        target_gt_used=False,target_videos_read=0,source_gate=SOURCE_GATE,rows=rows,positive_points=len(gt),
        true_phase_std=std,video_equal_mae=sum(r['mae'] for r in rows)/max(len(rows),1),
        video_equal_constant_mae=sum(r['constant_mae'] for r in rows)/max(len(rows),1),
        checkpoint=str(reference/'best.pt'),checkpoint_sha256=p.sha(reference/'best.pt'),
        selected_epoch=saved['epoch'],diagnostic_checkpoint_written=False,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    report['side_role_support_videos']=[sum(sum(level['role_mass'][role] for level in row['levels'])>=1. for row in rows) for role in [0,2]]
    p.save(output,report)


def source_decision(reports):
    if len(reports)!=6 or {r['shift'] for r in reports}!={'S'+str(i) for i in range(1,7)}:
        raise ValueError('源预检不是精确六方向')
    support=all(len(r['rows'])>=SOURCE_GATE['minimum_videos'] and r['positive_points']>=SOURCE_GATE['minimum_positive_points']
                and r['true_phase_std']>=SOURCE_GATE['minimum_true_phase_std']
                and min(r['side_role_support_videos'])>=SOURCE_GATE['minimum_independent_videos_with_unit_mass_per_side_role'] for r in reports)
    advantage=sum(r['video_equal_constant_mae']-r['video_equal_mae'] for r in reports)/6
    return dict(status='passed' if support and advantage>0 else 'rejected_uninformative_source_phase',
                support_passed=support,equal_shift_mae_advantage=advantage,gate=SOURCE_GATE,
                target_gt_used=False,not_target_gain_proof=True)


def verify_source_gate(path):
    gate=p.read(path);reports=[]
    if len(gate['reports'])!=6 or len(gate['gpu_receipts'])!=6:raise ValueError('源门禁凭证不是六方向')
    for report_path,digest in gate['reports'].items():
        if p.sha(report_path)!=digest:raise ValueError('源诊断报告改变')
        reports.append(p.read(report_path))
        receipt_path=str(Path(report_path).parent/'sourcecheck_gpu.json')
        if p.sha(receipt_path)!=gate['gpu_receipts'][receipt_path]:raise ValueError('源诊断GPU凭证改变')
        receipt=p.read(receipt_path);guard=p.read(receipt['guard'])
        if p.sha(receipt['guard'])!=receipt['guard_sha256'] or p.sha(guard['lease'])!=guard['lease_sha256']:
            raise ValueError('源诊断锁/首CUDA凭证改变')
        if receipt['config_sha256']!=p.sha(Path(report_path).parent/'config.json'):
            raise ValueError('源诊断配置改变')
    decision=source_decision(reports)
    if decision['status']!='passed' or any(gate[k]!=v for k,v in decision.items()):
        raise ValueError('源角色门禁无法独立复算')
    return gate


def verify_probe(report,initial_detector_sha,role_mode='semantic'):
    """读取baseline_engine的真实报告命名，不去掉保留的rq3_前缀。"""
    if report['rng_roundtrip']['status']!='passed':raise ValueError('GPU完整恢复核验失败')
    if report['diagnostic_epochs']!=[1,2,3,4,5] or report['extra_timing_warmup_steps']!=0:
        raise ValueError('必须按正式顺序连续完整预热1至5轮，不能跳过2至4轮')
    if report['initial_detector_sha']!=initial_detector_sha:raise ValueError('共同检测器冷初始化不配对')
    phase=report['phases']['5']
    if not phase['branch_parameters_updated'] or phase['auxiliary']['rq3_target_gradient_norm']<=0:
        raise ValueError('原Prototype没有实际有效更新')
    if phase['target_gt_losses']!=[0.,0.,0.]:raise ValueError('目标GT监督不为0')
    if role_mode!='role_blind' and sum(phase['phase_weight_separation'])<=0:raise ValueError('角色D仍然恒等')
    if role_mode!='semantic':
        if not phase['phase_discriminator_updated']:raise ValueError('机制对照判别器未优化')
        stats=phase['phase_alignment']
        total=lambda key:sum(v for k,v in stats.items() if k.startswith('phase_control_'+key+'_'))
        if total('condition_calls')<=0:raise ValueError('机制对照没有真正执行')
        if role_mode=='role_blind' and total('nonuniform_input')!=0:raise ValueError('角色盲D仍读取角色信息')
        if role_mode=='role_permuted' and (total('permutation_l1')<=0 or total('nonuniform_input')<=0):
            raise ValueError('角色置换没有改变有效角色对应关系')
    if sum(v for k,v in phase['phase_alignment'].items() if k.startswith('phase_common_roles_'))<=0:
        raise ValueError('没有实际跨域角色对齐')
    if sum(v for k,v in phase['phase_alignment'].items() if k.startswith('phase_common_side_roles_'))<=0:
        raise ValueError('只有中间角色，不能冒充起中末对齐')


def audit(entry):
    import torch
    entry=Path(entry);record=p.read(entry/'complete.json');cfg=p.read(entry/'config.json')
    ref=Path(record['control_complete'])
    if p.sha(ref)!=record['control_complete_sha256']:raise ValueError('原对照完成工件改变')
    assert_pair(cfg,p.read(ref.parent/'config.json'))
    reference_eval=ref.parent/'evaluation/target_val.json'
    if p.sha(reference_eval)!=p.read(ref)['artifacts'][str(reference_eval)]:raise ValueError('原版目标评估改变')
    if p.read(entry/'detector_initialization.json')['sha256']!=p.read(ref.parent/'detector_initialization.json')['sha256']:
        raise ValueError('共同检测器冷初始化不配对')
    if p.sha(record['source_gate'])!=record['source_gate_sha256']:
        raise ValueError('源角色门禁未过或改变')
    verify_source_gate(record['source_gate'])
    group=p.read(record['group_acceptance'])
    if p.sha(record['group_acceptance'])!=record['group_acceptance_sha256'] or group['status']!='passed' or len(group['acceptance'])!=6:
        raise ValueError('缺少正式启动前六路全部验收证明')
    for path,digest in group['acceptance'].items():
        if p.sha(path)!=digest:raise ValueError('六路验收工件改变')
    acceptance=p.read(entry/'acceptance.json');probe=p.read(entry/'probe_result.json')
    if acceptance['report_sha256']!=p.sha(entry/'probe_result.json') or acceptance['source_gate_sha256']!=record['source_gate_sha256']:
        raise ValueError('真实验收未绑定当前GPU/source证明')
    minimum=max(8192,math.ceil(probe['peak_reserved_mib']+2048))
    receipt=p.read(entry/'train_gpu.json');lease=p.read(p.read(receipt['guard'])['lease'])
    if lease['minimum_free_mib']<minimum or lease['minimum_free_mib']!=acceptance['minimum_free_mib']:
        raise ValueError('正式显存准入未覆盖真实峰值')
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    active=[r for r in rows if r['epoch']>=cfg['train_cfg']['warmup_adv']]
    if not active or not any(sum(r['phase_weight_separation'])>0 for r in active):raise ValueError('角色判别器未更新')
    for name in ['best','latest']:
        saved=torch.load(entry/(name+'.pt'),map_location='cpu')
        if saved['schema']!=SCHEMA:raise ValueError('新方法checkpoint身份错误')
    result=p.audit_entry(entry,expected_schema=SCHEMA,acceptance_path=entry/'acceptance.json')
    result.update(variant='relative_phase',target_gt_used=False);return result
