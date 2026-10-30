"""O仅测GT正点被误送背景对齐的训练影响；结果不能计入正常UDA。"""
from copy import deepcopy
from pathlib import Path
from libs.models.SADA.background_oracle import SCHEMA,POLICY
from tools import baseline_protocol as p


def config_for(base,gt_path,identity,tmpdir):
    cfg=deepcopy(base)
    if base['seed']!=123 or base['loader']['batch_size']!=2 or base['prototype_capacity']['variant']!='control':raise ValueError('只配对原版seed123/batch2 Prototype')
    cfg['baseline'].update(code=identity,tmpdir=str(tmpdir),target_gt_used=True)
    cfg['background_oracle']=dict(schema=SCHEMA,policy=deepcopy(POLICY),target_train_gt=str(gt_path),target_train_gt_sha256=p.sha(gt_path))
    assert_pair(cfg,base);return cfg


def assert_pair(cfg,base):
    other=deepcopy(cfg);oracle=other.pop('background_oracle')
    if oracle['schema']!=SCHEMA or oracle['policy']!=POLICY or not cfg['baseline']['target_gt_used']:raise ValueError('oracle身份或GT使用声明不符')
    for key in ['code','tmpdir','target_gt_used']:other['baseline'][key]=base['baseline'][key]
    if other!=base:raise ValueError('oracle改变了背景排除以外的训练配置')


def prepare(root,control,identity):
    from tools.baseline_engine import dataset
    from libs.models.SADA.background_oracle import crop_annotations
    configs={}
    for shift in ['S1','S2','S3','S4','S5','S6']:
        ref=control/'formal/control'/shift;base=p.read(ref/'config.json');complete=p.read(ref/'complete.json')
        for path,digest in [(ref/'config.json',complete['artifacts'][str(ref/'config.json')]),
            (Path(base['baseline']['target_manifest']),base['baseline']['target_manifest_sha256']),
            (Path(base['dataset']['json_file']),base['baseline']['source_manifest_sha256']),
            (Path(base['prototype_capacity']['prepared_cache']),base['prototype_capacity']['prepared_cache_sha256'])]:
            if p.sha(path)!=digest:raise ValueError('冻结输入改变: '+str(path))
        annotation=Path(base['baseline']['annotation_for_offline_evaluation'])
        if p.sha(annotation)!=base['baseline']['annotation_sha256']:raise ValueError('原标注改变')
        original=p.read(annotation)['database'];target=p.read(base['baseline']['target_manifest'])['database']
        gt={vid:deepcopy(original[vid]) for vid in target}
        if any(row['subset'].lower() not in base['tgt_train_split'] for row in gt.values()):raise ValueError('GT旁路混入非训练视频')
        gt_path=root/'inputs'/shift/'target_train_oracle.json'
        p.save(gt_path,dict(schema=SCHEMA,shift=shift,database=gt),sort_keys=False)
        cfg=config_for(base,gt_path,identity,root/'tmp');configs[shift]=cfg
        p.save(root/'configs'/(shift+'.json'),cfg)
        # 同一无标注dataset入口，oracle关闭/启用不会参与getitem与crop。
        ds=dataset(base,base['baseline']['target_manifest'],base['tgt_train_split'],True,True)
        video=ds[0];p.assert_no_target_gt([video]);segments,labels=crop_annotations(gt[video['metadata']['video_id']],video['metadata'],set(range(10)),base['dataset']['trunc_thresh'])
        p.save(root/'inputs'/shift/'preflight.json',dict(status='passed',target_train_videos=len(gt),
            oracle_gt_sha256=p.sha(gt_path),original_target_manifest_sha256=p.sha(base['baseline']['target_manifest']),
            original_control_complete=str(ref/'complete.json'),original_control_complete_sha256=p.sha(ref/'complete.json'),
            target_input_gt_empty=True,oracle_crop_gt_count=len(labels),crop_start=video['metadata'].get('crop_start',0),
            crop_feat_len=video['feats'].shape[-1],target_gt_used=True,target_gt_scope=POLICY['scope']))
    return configs


def audit(entry):
    import json,torch
    entry=Path(entry);record=p.read(entry/'complete.json');cfg=p.read(entry/'config.json')
    reference=Path(record['control_complete'])
    if p.sha(reference)!=record['control_complete_sha256']:raise ValueError('原版对照完成项改变')
    assert_pair(cfg,p.read(reference.parent/'config.json'))
    if p.sha(cfg['background_oracle']['target_train_gt'])!=cfg['background_oracle']['target_train_gt_sha256']:raise ValueError('oracle训练旁路GT改变')
    if p.read(entry/'detector_initialization.json')['sha256']!=p.read(reference.parent/'detector_initialization.json')['sha256']:raise ValueError('冷初始化与原版不一致')
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if not all(x.get('target_gt_used_for_background_alignment') is True for x in rows):raise ValueError('未逐轮声明oracle的GT使用')
    if not any(sum(v for k,v in x.get('background_oracle',{}).items() if k.startswith('oracle_bg_removed_'))>0 for x in rows):raise ValueError('oracle未真正移除GT正点')
    for name in ['best','latest']:
        saved=torch.load(entry/(name+'.pt'),map_location='cpu')
        if saved['schema']!=SCHEMA or saved['config']!=cfg:raise ValueError('oracle保存点身份不符')
    result=p.audit_entry(entry,expected_schema=SCHEMA,acceptance_path=Path(record['gpu_acceptance']))
    result.update(target_gt_used=True,oracle_scope=POLICY['scope']);return result
