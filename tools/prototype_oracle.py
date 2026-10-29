"""固定原200候选/128维分支的GT类别oracle；主检测器不接收目标GT张量。"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from tools.baseline_protocol import read, save, sha

SCHEMA='rq3-prototype-gt-oracle-seed123-v1'
CACHE_SCHEMA='prototype-gt-oracle-cache-v1'
GROUPS=('fixed40','clean20','clean50','clean100')
LABELS=('TAKE','PUT','WASH','OPEN','CLOSE','INSERT','TURN_ON','CUT','TURN_OFF','POUR')
POLICY=dict(pool_size=200, original_budget=40, known_classes=10,
    truth='unique-known-class-union-coverage-strictly-over-half-proposal',
    ambiguous='abstain', unknown_or_background='abstain', boundaries='original_proposals_unchanged',
    clean_ranking='sha256-prototype-oracle-nested-seed6-video-anchor',
    clean_budgets={'clean20':40,'clean50':100,'clean100':200},
    student_seed=123, batch_size=2, prototype_variant='control', target_weight=.1,
    normalization='unchanged_mean_over_routed_candidates_no_inverse_coverage_scaling',
    target_gt_in_auxiliary=True, target_gt_in_detection=False, target_validation_for_selection=False)


def coverage(start,end,label,annotations):
    spans=sorted((max(start,a['segment'][0]),min(end,a['segment'][1])) for a in annotations
        if int(a['label_id'])==label and max(start,a['segment'][0])<min(end,a['segment'][1]))
    covered=0.;right=start
    for left,stop in spans:
        covered+=max(0.,stop-max(left,right));right=max(right,stop)
    return covered/(end-start)


def truth(row,annotations):
    start,end=row['proposal_start_sec'],row['proposal_end_sec']
    if not end>start:raise ValueError('原proposal区间非法')
    labels=[c for c in range(10) if coverage(start,end,c,annotations)>.5]
    reason='usable' if len(labels)==1 else 'no_known_majority' if not labels else 'multiple_known_majorities'
    return dict(label=labels[0] if len(labels)==1 else None,reason=reason)


def key(row):return row['video_id'],int(row['absolute_feature_index'])


def clean_order(rows):
    return sorted(rows,key=lambda x:hashlib.sha256(
        ('prototype-oracle-nested-seed6:'+x['video_id']+':'+str(x['absolute_feature_index'])).encode()).hexdigest())


def build_caches(original,database,target_members):
    rows=[x for values in original['by_video'].values() for x in values]
    if len(rows)!=200 or len({key(x) for x in rows})!=200:raise ValueError('必须是原200唯一候选')
    original_set={key(x) for x in rows if x['participation_selected']}
    if len(original_set)!=40:raise ValueError('旧参与名单不是精确40个')
    truths={}
    for row in rows:
        if row['video_id'] not in target_members:raise ValueError('oracle只能读取目标训练候选的标注')
        truths[key(row)]=truth(row,database[row['video_id']].get('annotations',[]))
    valid=[x for x in rows if truths[key(x)]['label'] is not None];ordered=clean_order(valid)
    if len(ordered)<100:raise ValueError('不足100个干净候选，不能悄悄缩小20/50%预算')
    chosen={'fixed40':original_set & {key(x) for x in valid},
        **{g:{key(x) for x in ordered[:n]} for g,n in POLICY['clean_budgets'].items()}}
    caches={}
    for group in GROUPS:
        by_video={}
        for original_row in rows:
            row=deepcopy(original_row);t=truths[key(row)];label=t['label']
            row['original_semantic_label']=row['semantic_label']
            row['original_participation_selected']=row['participation_selected']
            row.update(oracle_truth=t, class_id=label,semantic_label=LABELS[label] if label is not None else 'ABSTAIN',
                class_probabilities=[float(c==label) for c in range(10)], participation_selected=key(row) in chosen[group])
            by_video.setdefault(row['video_id'],[]).append(row)
        selected=[x for rr in by_video.values() for x in rr if x['participation_selected']]
        caches[group]=dict(schema=CACHE_SCHEMA,teacher='ground_truth_oracle_NOT_VLM',
            target_gt_used=True,target_gt_scope='target_training_annotations_auxiliary_cache_only',
            target_validation_gt_used=False,group=group,policy=POLICY,shift=original['shift'],
            original_budget=40 if group=='fixed40' else POLICY['clean_budgets'][group],
            candidate_count=200,usable_count=len(valid),selected_count=len(selected),
            rejection_counts=dict(Counter(x['reason'] for x in truths.values())),
            selected_class_counts=dict(Counter(x['semantic_label'] for x in selected)),
            selected_video_count=len({x['video_id'] for x in selected}),by_video=by_video)
    return caches


def prepare_shift(output,shift,base,old_prepared):
    cfg=read(base/'config.json');annotation=Path(cfg['baseline']['annotation_for_offline_evaluation'])
    if sha(annotation)!=cfg['baseline']['annotation_sha256']:raise ValueError('原标注SHA改变')
    source=read(cfg['dataset']['json_file'])['database'];target=read(cfg['baseline']['target_manifest'])['database']
    if set(source)&set(target):raise ValueError('源/目标成员交叉')
    db=read(annotation)['database'];target_members={v for v,d in db.items() if d['subset'].lower() in cfg['tgt_train_split']}
    if target_members!=set(target):raise ValueError('原目标训练成员不一致')
    caches=build_caches(read(old_prepared),db,target_members);records={}
    for group,cache in caches.items():
        cache['provenance']=dict(original_prepared=str(old_prepared),original_prepared_sha256=sha(old_prepared),
            annotation=str(annotation),annotation_sha256=sha(annotation),
            unlabeled_target_manifest=str(cfg['baseline']['target_manifest']),
            unlabeled_target_manifest_sha256=sha(cfg['baseline']['target_manifest']),
            target_train_members=sorted(target_members))
        path=output/'prepared'/group/(shift+'.json');save(path,cache)
        records[group]=dict(path=str(path),sha256=sha(path),selected_count=cache['selected_count'],
            usable_count=cache['usable_count'],selected_class_counts=cache['selected_class_counts'],
            selected_video_count=cache['selected_video_count'])
    return records


def validate_cache(path):
    cached=read(path)
    if cached.get('schema')!=CACHE_SCHEMA or cached.get('policy')!=POLICY:raise ValueError('oracle定义改变')
    p=cached['provenance']
    for f,h in [('original_prepared','original_prepared_sha256'),('annotation','annotation_sha256'),
                ('unlabeled_target_manifest','unlabeled_target_manifest_sha256')]:
        if sha(p[f])!=p[h]:raise ValueError('oracle来源工件改变: '+f)
    if set(read(p['unlabeled_target_manifest'])['database'])!=set(p['target_train_members']):raise ValueError('oracle成员不是冻结目标训练集合')
    rebuilt=build_caches(read(p['original_prepared']),read(p['annotation'])['database'],set(p['target_train_members']))[cached['group']]
    cached=deepcopy(cached);cached.pop('provenance')
    if cached!=rebuilt:raise ValueError('GT标签、候选边界或固定嵌套名单无法独立复算')
    return dict(status='passed',group=cached['group'],shift=cached['shift'],selected_count=cached['selected_count'])


def config_for(base,group,prepared,identity,tmpdir):
    from tools.run_prototype_capacity import paired_config
    if read(prepared)['group']!=group:raise ValueError('oracle配置组与缓存组不符')
    cfg=paired_config(base,'control',Path(prepared),identity,tmpdir)
    cfg['prototype_capacity']['oracle_auxiliary_ground_truth']=True
    cfg['baseline'].update(target_gt_used=True,target_gt_scope='oracle_auxiliary_cache_only')
    cfg['oracle_auxiliary']=dict(schema=SCHEMA,group=group,policy=POLICY,
        target_gt_for_auxiliary=True,target_gt_for_detector=False,target_val_for_selection=False)
    return cfg


def assert_pair(cfg,base):
    from tools.run_prototype_capacity import assert_pair as original_pair
    cfg=deepcopy(cfg);oracle=cfg.pop('oracle_auxiliary')
    if oracle!=dict(schema=SCHEMA,group=oracle['group'],policy=POLICY,
                   target_gt_for_auxiliary=True,target_gt_for_detector=False,target_val_for_selection=False):raise ValueError('oracle身份不符')
    if oracle['group'] not in GROUPS:raise ValueError('越界oracle组')
    if cfg['baseline'].pop('target_gt_scope')!='oracle_auxiliary_cache_only' or cfg['baseline']['target_gt_used'] is not True:raise ValueError('必须公开目标GT用于oracle')
    cfg['baseline']['target_gt_used']=base['baseline']['target_gt_used']
    if cfg['prototype_capacity'].pop('oracle_auxiliary_ground_truth') is not True or cfg['prototype_capacity']['variant']!='control':raise ValueError('必须是原版128维分支')
    original_pair(cfg,base)


def verify_acceptance(root,cfg):
    ready=read(Path(root)/'acceptance.json')
    if (ready['status']!='passed' or ready['schema']!=SCHEMA or ready['code']!=cfg['baseline']['code']
            or sha(Path(root)/'inputs.json')!=ready['inputs_sha256']):raise ValueError('oracle真实验收与代码/输入绑定失败')
    if not all(sha(p)==h for p,h in ready['report_sha256'].items()):raise ValueError('真实GPU验收报告改变')
    cap=cfg['prototype_capacity'];cache=read(cap['prepared_cache']);group=cfg['oracle_auxiliary']['group']
    if (sha(cap['prepared_cache'])!=cap['prepared_cache_sha256'] or cache['group']!=group
            or cache['schema']!=CACHE_SCHEMA or cache['policy']!=POLICY):raise ValueError('oracle缓存或组身份改变')
    frozen=read(Path(root)/'inputs.json')['prepared'][cfg['baseline']['shift']][group]
    if frozen['sha256']!=cap['prepared_cache_sha256'] or frozen['path']!=cap['prepared_cache']:raise ValueError('oracle不是已审计准备工件')


class ExposureTracker:
    """统计进入分支前可路由候选；不读取GT张量、不消耗RNG，不声称全部已对齐。"""
    def __init__(self,cache,strides,steps,path):
        self.cache,self.levels,self.steps,self.path=cache,len(strides),steps,Path(path)
        self.epochs={};self.global_ids=set()

    def observe(self,videos,epoch,expected_selected):
        from libs.models.SADA.prototype_capacity import fully_visible_cache
        kept,_,_=fully_visible_cache(self.cache,videos,0.)
        routed=[x for rows in kept for x in rows[:256] if x['source_fpn_levels'][0]<self.levels]
        if len(routed)!=expected_selected:raise ValueError('实际分支选择与独立路由观测不一致')
        record=self.epochs.setdefault(int(epoch),dict(batches=0,exposures=0,unique=set(),classes=Counter(),videos=set()))
        record['batches']+=1;record['exposures']+=len(routed)
        for row in routed:
            identity=row['candidate_id'];record['unique'].add(identity);self.global_ids.add(identity)
            record['classes'][row['semantic_label']]+=1;record['videos'].add(row['video_id'])
        if record['batches']==self.steps:self.flush()

    def flush(self):
        save(self.path,dict(scope='routable before source-prototype-readiness; not unique full-target coverage',
            unique_candidate_ids=sorted(self.global_ids),epochs={str(e):dict(batches=x['batches'],exposures=x['exposures'],
                unique_candidate_ids=sorted(x['unique']),class_exposures=dict(x['classes']),video_ids=sorted(x['videos']))
                for e,x in self.epochs.items()}))


def audit_entry(entry,verify_truth=True):
    from tools import baseline_protocol as protocol
    entry=Path(entry);cfg=read(entry/'config.json');done=read(entry/'complete.json')
    base=Path(done['baseline_complete']).parent;control=Path(done['control_complete']).parent
    for p,h in [(base/'complete.json',done['baseline_complete_sha256']),(control/'complete.json',done['control_complete_sha256'])]:
        if sha(p)!=h:raise ValueError('既有对照凭证改变')
    assert_pair(cfg,read(base/'config.json'))
    verify_acceptance(entry.parents[2],cfg)
    cap=cfg['prototype_capacity']
    if sha(cap['prepared_cache'])!=cap['prepared_cache_sha256']:raise ValueError('oracle缓存SHA改变')
    if verify_truth:validate_cache(cap['prepared_cache'])
    if done.get('target_gt_for_auxiliary') is not True or done['group']!=cfg['oracle_auxiliary']['group']:raise ValueError('oracle不能伪装为普通UDA')
    if read(entry/'detector_initialization.json')['sha256']!=read(base/'termination.json')['initial_state_sha']:raise ValueError('共享检测器初始化改变')
    result=protocol.audit_entry(entry,expected_schema=SCHEMA)
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if not any(x['auxiliary']['rq3_target_gradient_norm']>0 and x['auxiliary']['rq3_alignment_loss']>0 for x in rows):raise ValueError('oracle没有产生真实有限目标辅助梯度')
    exposure=read(entry/'oracle_exposure.json')
    if set(exposure['epochs'])!={str(x['epoch']) for x in rows}:raise ValueError('路由曝光缺少epoch')
    selected={x['candidate_id'] for rr in read(cap['prepared_cache'])['by_video'].values() for x in rr if x['participation_selected']}
    if not set(exposure['unique_candidate_ids'])<=selected:raise ValueError('路由到了预算外候选')
    for x in exposure['epochs'].values():
        if x['batches']!=cfg['baseline']['epoch_budget']['iterations_per_epoch']:raise ValueError('路由观测更新预算不符')
    result.update(group=done['group'],oracle=True,target_gt_for_auxiliary=True,
        selected_candidates=len(selected),unique_routable_candidates=len(exposure['unique_candidate_ids']))
    return result
