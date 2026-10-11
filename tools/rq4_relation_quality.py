"""复用源响应检查同实例关系；不拟合门槛、不访问目标GT、不查询教师。"""
import math
from pathlib import Path

from tools import rq4_protocol as p
from tools.rq4_artifacts import checked, reference, audit_source_chain
from tools.rq4_relational import core_interval, relation_weight, RULE


def original_grid_points(core, candidate, dataset, scale_factor):
    # 全视频原始特征grid，尚未模拟随机crop/mask，不能称训练实际覆盖。
    fps = dataset['default_fps']
    stride = dataset['feat_stride']*dataset['downsample_rate']
    frames = dataset['num_frames']
    step = scale_factor**candidate['fpn_level']
    lo, hi = [(x*fps-.5*frames)/stride for x in core]
    ids = list(range(max(0, math.floor(lo/step)+1), math.ceil(hi/step)))
    if len(ids) > RULE['max_points']:
        ids = [ids[0], ids[(len(ids)-1)//2], ids[-1]]
    return [(i*step*stride+.5*frames)/fps for i in ids]


def same_instance_pairs(points, annotations):
    # 任意同一标注动作实例（非仅同类别）；这是定位关系，不评分类。
    return [any(a['segment'][0] <= start < end <= a['segment'][1] for a in annotations)
            for start, end in zip(points, points[1:])]


def summarize(rows, mode):
    selected = [r for r in rows if len(r['points']) >= 2 and
                (mode == 'soft' or r['consistency'] >= .75)]
    weights = [r['consistency'] if mode == 'soft' else 1. for r in selected]
    numerator = sum(w*sum(r['same_instance_pairs']) for w,r in zip(weights,selected))
    denominator = sum(w*len(r['same_instance_pairs']) for w,r in zip(weights,selected))
    pairs = sum(len(r['same_instance_pairs']) for r in selected)
    return dict(candidates=len(selected), pairs=pairs, weight_sum=sum(weights),
                weighted_correct_pairs=numerator, weighted_pairs=denominator,
                weighted_pair_precision=numerator/denominator if denominator else None,
                pair_precision=sum(sum(r['same_instance_pairs']) for r in selected)/pairs if pairs else None)


def diagnose(source, teacher, baseline):
    # 再次复算原始源证据；只从source成员读取GT，不从target数据查标注。
    for item in source.values(): checked(item)
    audit_source_chain(source['requests']['path'], Path(source['identity']['path']).parent,
                       source['annotations']['path'], teacher)
    manifest = checked(source['requests'])
    database = checked(source['annotations'])['database']
    replies = {(r['video_id'],r['candidate_id']): r for r in checked(source['responses'])}
    groups = {}
    for request in manifest['requests']:
        row=request['candidate']
        if database[row['video_id']]['subset'].lower() not in {s.lower() for s in manifest['source_subsets']}:
            raise ValueError('relationship diagnostic outside source validation')
        groups.setdefault((row['video_id'],row['candidate_id'].rsplit('-context',1)[0]),[]).append(row)
    rows=[]
    for (video,cid), candidates in sorted(groups.items()):
        candidates.sort(key=lambda c:c['candidate_id'])
        if len(candidates)!=2: raise ValueError('source contexts missing')
        pair=[]
        for c in candidates:
            response=replies[(video,c['candidate_id'])]
            # 复用正式解码：原生缺失可无效，codec技术失败必须抛错。
            identity=dict(proposal_checkpoint_sha256=manifest['checkpoint_sha256'],
                          teacher_weights_sha256=p.digest('source-diagnostic'),
                          processor_sha256=p.digest('source-diagnostic'),
                          input_manifest_sha256=source['requests']['sha256'],
                          class_names_sha256=response['class_names_sha256'])
            cache=p.build_cache([c],[response],teacher=teacher,shift=manifest['shift'],
                                identity=identity,min_iou=0.,ambiguity_margin=0.)
            pair.append(cache['records'][0]['target'])
        candidate=candidates[0]
        core=core_interval(candidate['proposal'],pair)
        points=[]
        if core and core[0]<candidate['anchor_sec']<core[1]:
            points=original_grid_points(core,candidate,baseline['dataset'],baseline['model']['scale_factor'])
        rows.append(dict(video_id=video,candidate_id=cid,consistency=relation_weight(pair),
                         points=points,same_instance_pairs=same_instance_pairs(points,database[video]['annotations'])))
    return dict(schema='rq4-source-relation-diagnostic-v1',teacher=teacher,shift=manifest['shift'],
                source_evidence=source,target_gt_used=False,parameters_fitted=False,
                grid='original-full-video-no-crop-or-mask; coverage upper bound',
                summary={mode:summarize(rows,mode) for mode in ('hard','soft')},rows=rows)
