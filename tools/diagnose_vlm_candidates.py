"""离线审计实际送给VLM的冻结候选；GT不回流训练/预算/阈值。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha
from tools.prototype_oracle import truth

THRESHOLDS=(.1,.3,.5,.7)


def tiou(segment,other):
    overlap=max(0.,min(segment[1],other[1])-max(segment[0],other[0]))
    return overlap/(segment[1]-segment[0]+other[1]-other[0]-overlap)


def ratio(n,d):return n/d if d else None


def evaluate_candidates(rows,database):
    results=[]
    for row in rows:
        segment=[row['proposal_start_sec'],row['proposal_end_sec']]
        if not segment[1]>segment[0]:raise ValueError('候选时间区间非法')
        annotations=database[row['video_id']].get('annotations',[])
        task=[(j,a) for j,a in enumerate(annotations) if 0<=int(a['label_id'])<10]
        overlaps=[(tiou(segment,a['segment']),j,a) for j,a in task]
        best=max(overlaps,key=lambda x:x[0]) if overlaps else (0.,None,None)
        target=truth(row,annotations)
        if best[0]==0:
            kind='other_annotated_class_only' if any(tiou(segment,a['segment'])>0 for a in annotations) else 'no_annotated_action_overlap'
        else:kind=target['reason']
        item=dict(candidate_id=row['candidate_id'],video_id=row['video_id'],selected=row['participation_selected'],
            segment=segment,student_class=row['student_class'],vlm_class=row['class_id'],student_score=row['student_score'],
            truth=target,kind=kind,best_tiou=best[0],best_gt_index=best[1],
            best_gt_label=int(best[2]['label_id']) if best[2] else None,
            student_correct=target['label'] is not None and row['student_class']==target['label'],
            vlm_correct=target['label'] is not None and row['class_id']==target['label'],
            student_same_class_tiou=max((v for v,j,a in overlaps if int(a['label_id'])==row['student_class']),default=0.),
            vlm_same_class_tiou=max((v for v,j,a in overlaps if int(a['label_id'])==row['class_id']),default=0.),
            gt_matches={str(t):[[row['video_id'],j] for v,j,a in overlaps if v>=t] for t in THRESHOLDS})
        if best[2]:
            gt=best[2]['segment'];duration=gt[1]-gt[0]
            item.update(start_error_sec=segment[0]-gt[0],end_error_sec=segment[1]-gt[1],
                duration_ratio=(segment[1]-segment[0])/duration)
        results.append(item)
    return results


def summarize(rows,database):
    usable=[x for x in rows if x['truth']['label'] is not None]
    videos={x['video_id'] for x in rows}
    gt_total=sum(sum(0<=int(a['label_id'])<10 for a in d.get('annotations',[])) for d in database.values())
    gt_present=sum(sum(0<=int(a['label_id'])<10 for a in database[v].get('annotations',[])) for v in videos)
    output=dict(count=len(rows),represented_videos=len(videos),target_train_videos=len(database),
        task_gt_count_all_target_train=gt_total,task_gt_count_in_represented_videos=gt_present,
        kinds=dict(Counter(x['kind'] for x in rows)),unique_majority_count=len(usable),
        student_majority_accuracy=ratio(sum(x['student_correct'] for x in usable),len(usable)),
        vlm_majority_accuracy=ratio(sum(x['vlm_correct'] for x in usable),len(usable)),
        student_correct_and_usable_fraction=ratio(sum(x['student_correct'] for x in rows),len(rows)),
        vlm_correct_and_usable_fraction=ratio(sum(x['vlm_correct'] for x in rows),len(rows)),
        vlm_fixes=sum(not x['student_correct'] and x['vlm_correct'] for x in usable),
        vlm_breaks=sum(x['student_correct'] and not x['vlm_correct'] for x in usable),
        mean_best_tiou=statistics.mean(x['best_tiou'] for x in rows) if rows else None,
        median_best_tiou=statistics.median(x['best_tiou'] for x in rows) if rows else None,
        thresholds={})
    # 候选定位通过率与类别+定位联合通过率分开，重复候选覆盖同一GT只计一次。
    for t in THRESHOLDS:
        matched={tuple(pair) for x in rows for pair in x['gt_matches'][str(t)]}
        good=[x for x in rows if x['best_tiou']>=t]
        output['thresholds'][str(t)]=dict(localization_pass_count=len(good),localization_pass_fraction=ratio(len(good),len(rows)),
            student_joint_pass_fraction=ratio(sum(x['student_same_class_tiou']>=t for x in rows),len(rows)),
            vlm_joint_pass_fraction=ratio(sum(x['vlm_same_class_tiou']>=t for x in rows),len(rows)),
            covered_unique_gt=len(matched),task_gt_recall_all_target_train=ratio(len(matched),gt_total),
            task_gt_recall_in_represented_videos=ratio(len(matched),gt_present))
    return output


def inspect_shift(shift,configs):
    cfg=read(configs/(shift+'.json'));prepared_path=Path(cfg['prototype_capacity']['prepared_cache'])
    if sha(prepared_path)!=cfg['prototype_capacity']['prepared_cache_sha256']:raise ValueError('冻结教师缓存SHA改变')
    prepared=read(prepared_path)
    for path,digest in prepared['source_artifacts'].items():
        if sha(path)!=digest:raise ValueError('冻结输入改变: '+path)
    manifest_path=next(Path(p) for p in prepared['source_artifacts'] if p.endswith('/manifest.json') and 'shared_candidates' in p)
    manifest=read(manifest_path);raw_path=manifest_path.parents[1]/'proposals'/(shift+'.jsonl')
    if sha(raw_path)!=manifest['source_proposals_sha256']:raise ValueError('原始检测候选SHA改变')
    raw=[json.loads(x) for x in raw_path.read_text().splitlines()];original={x['candidate_id']:x for x in raw}
    cached=[x for values in prepared['by_video'].values() for x in values]
    if len(raw)!=200 or len(original)!=200 or {x['candidate_id'] for x in cached}!=set(original):raise ValueError('不是精确原200候选')
    meta_path=Path(str(raw_path)+'.meta.json');meta=read(meta_path)
    if meta['selection_teacher_provenance']['selection_teacher_checkpoint_sha256']!=manifest['teacher_checkpoint_sha256']:
        raise ValueError('候选生成模型身份不符')
    annotation=Path(cfg['baseline']['annotation_for_offline_evaluation'])
    if sha(annotation)!=cfg['baseline']['annotation_sha256']:raise ValueError('GT改变')
    target_path=Path(cfg['baseline']['target_manifest'])
    if sha(target_path)!=cfg['baseline']['target_manifest_sha256']:raise ValueError('目标成员改变')
    target=read(target_path)['database'];db=read(annotation)['database']
    if set(target)!=set(manifest['target_video_ids']) or set(target)!=set(meta['expected_video_ids']):raise ValueError('目标训练成员不一致')
    members={v:db[v] for v in target};joined=[]
    for row in cached:
        old=original[row['candidate_id']]
        for key in ['video_id','absolute_feature_index','proposal_start_sec','proposal_end_sec','source_fpn_levels']:
            if old[key]!=row[key]:raise ValueError('缓存候选坐标改变')
        if row['video_id'] not in target:raise ValueError('候选混入非训练视频')
        joined.append(dict(row,student_class=int(old['teacher_top1_class']),student_score=float(old['teacher_score'])))
    results=evaluate_candidates(joined,members);selected=[x for x in results if x['selected']]
    if len(selected)!=40:raise ValueError('不是原固定40名单')
    return dict(shift=shift,all200=summarize(results,members),selected40=summarize(selected,members),rows=results,
        provenance=dict(candidate_generation_seed=manifest['seed'],current_student_seed=cfg['seed'],
            generator=meta['selection_teacher_provenance'],frame_protocol=manifest['frame_protocol'],
            generation_selection_uses_gt=meta['selection_uses_gt'],evaluation_uses_target_gt=True,
            annotation=str(annotation),annotation_sha256=sha(annotation),prepared_cache=str(prepared_path),
            prepared_cache_sha256=sha(prepared_path),raw_proposals=str(raw_path),raw_proposals_sha256=sha(raw_path),
            manifest=str(manifest_path),manifest_sha256=sha(manifest_path)))


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);summaries=[]
    for shift in ['S1','S2','S3','S4','S5','S6']:
        result=inspect_shift(shift,args.configs);save(args.output/(shift+'.json'),result)
        summaries.append({k:v for k,v in result.items() if k!='rows'});print('DONE',shift,json.dumps(summaries[-1]),flush=True)
    mean={group:{key:statistics.mean(r[group][key] for r in summaries if r[group][key] is not None)
        for key in ['student_majority_accuracy','vlm_majority_accuracy','mean_best_tiou']}
        for group in ['all200','selected40']}
    save(args.output/'summary.json',dict(status='passed',scope='actual frozen VLM input candidates, not newly generated current-seed predictions',
        class_truth='one task class covers strictly over half proposal; ambiguous/no-majority excluded from conditional class accuracy',
        localization_truth='maximum tIoU to a single annotated task action irrespective of class; joint metrics require matching class',
        coverage='fixed candidate budget recall, not full detector recall',target_gt_usage='offline diagnosis only',new_training=False,
        thresholds=THRESHOLDS,count=6,script_sha256=sha(__file__),results=summaries,macro_shift_mean=mean))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',type=Path,default=Path('/vepfs/vlms/background_oracle6_2bd2ef5_1031/configs'))
    parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
