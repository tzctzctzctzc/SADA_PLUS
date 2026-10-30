"""六方向全链路离线审计；GT仅评价，不改变候选/参数/早停或现有模型。"""
import argparse
from collections import Counter,defaultdict
import json
import math
import os
from pathlib import Path
import sys

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.prototype_refresh import BASELINE,CONTROL,DIAGNOSTIC,SHIFTS,validate_cache

REFRESH=Path('/vepfs/vlms/proto_refresh6_20bedbb_1030')
LABELS=('TAKE','PUT','WASH','OPEN','CLOSE','INSERT','TURN_ON','CUT','TURN_OFF','POUR')
THRESHOLDS=(.1,.3,.5,.7)


def tiou(a,b):
    inter=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    return inter/max(1e-12,(a[1]-a[0])+(b[1]-b[0])-inter)


def union_length(spans):
    total=0.;right=None
    for a,b in sorted(spans):
        if b<=a:continue
        total+=b-max(a,right) if right is not None and b>right else b-a if right is None else 0.
        right=max(b,right) if right is not None else b
    return total


def truth_for(segment,annotations):
    a,b=segment;duration=b-a;by=defaultdict(list);all_spans=[];known_spans=[]
    best=0.;best_labels=set();best_index=[]
    for i,x in enumerate(annotations):
        label=int(x['label_id']);left=max(a,x['segment'][0]);right=min(b,x['segment'][1])
        if right>left:
            all_spans.append((left,right))
            if 0<=label<10:by[label].append((left,right));known_spans.append((left,right))
        if not 0<=label<10:continue
        overlap=tiou(segment,x['segment'])
        if overlap>best:best=overlap;best_labels={label};best_index=[i]
        elif overlap==best and overlap>0:best_labels.add(label);best_index.append(i)
    fractions={k:union_length(v)/duration for k,v in by.items()}
    majority=[k for k,v in fractions.items() if v>.5]
    unique=majority[0] if len(majority)==1 else None
    points=[a+(b-a)*(i+.5)/8 for i in range(8)]
    frame_hits=sum(any(int(x['label_id'])==unique and x['segment'][0]<=t<x['segment'][1]
                       for x in annotations) for t in points) if unique is not None else None
    return dict(unique_majority=unique,majority_labels=majority,known_fraction=union_length(known_spans)/duration,
        any_annotated_fraction=union_length(all_spans)/duration,
        best_known_tiou=best,clean_reference_label=(next(iter(best_labels)) if best>=.5 and len(best_labels)==1 else None),
        clean_reference_indices=best_index if best>=.5 and len(best_labels)==1 else [],
        ideal_eight_midpoints_on_majority=frame_hits,per_class_fraction=fractions)


def auc(scores,correct):
    pos=sum(correct);neg=len(correct)-pos
    if not pos or not neg:return None
    groups=defaultdict(lambda:[0,0])
    for s,y in zip(scores,correct):groups[s][0 if y else 1]+=1
    wins=0.;below=0
    for score,(p,n) in sorted(groups.items()):wins+=p*(below+.5*n);below+=n
    return wins/(pos*neg)


def coverage(rows,db):
    grouped=defaultdict(list)
    for r in rows:grouped[r['video_id']].append(r)
    gt_count=0;hit={t:0 for t in THRESHOLDS};class_hit=dict(hit)
    for v,record in db.items():
        for x in record.get('annotations',[]):
            if not 0<=int(x['label_id'])<10:continue
            gt_count+=1
            overlaps=[(tiou(r['interval'],x['segment']),r.get('class_id')) for r in grouped.get(v,[])]
            best=max((p for p,_ in overlaps),default=0.)
            same=max((p for p,c in overlaps if c==int(x['label_id'])),default=0.)
            for t in hit:hit[t]+=best>=t;class_hit[t]+=same>=t
    return dict(candidates=len(rows),videos=len(grouped),known_gt_instances=gt_count,
        class_agnostic_gt_recall={str(t):hit[t]/gt_count for t in hit},
        class_aware_gt_recall={str(t):class_hit[t]/gt_count for t in hit})


def semantic_summary(rows):
    n=len(rows);valid=[r for r in rows if r['truth']['unique_majority'] is not None]
    clean=[r for r in rows if r['truth']['clean_reference_label'] is not None]
    counts=Counter()
    for r in valid:
        label=r['truth']['unique_majority'];tc=r['vlm_class']==label
        if r.get('student_class') is not None:
            sc=r['student_class']==label
            counts['both_correct' if tc and sc else 'vlm_correct_student_wrong' if tc else
                   'student_correct_vlm_wrong' if sc else 'both_wrong']+=1
    return dict(count=n,videos=len({r['video_id'] for r in rows}),
        majority_evaluable=len(valid),vlm_correct_on_majority=sum(r['vlm_class']==r['truth']['unique_majority'] for r in valid),
        vlm_accuracy_on_majority=(sum(r['vlm_class']==r['truth']['unique_majority'] for r in valid)/len(valid) if valid else None),
        clean_tiou50_evaluable=len(clean),vlm_accuracy_on_clean_tiou50=(sum(r['vlm_class']==r['truth']['clean_reference_label'] for r in clean)/len(clean) if clean else None),
        no_known_majority=sum(r['truth']['unique_majority'] is None for r in rows),
        no_annotated_overlap=sum(r['truth']['any_annotated_fraction']==0 for r in rows),
        other_or_mixed=sum(r['truth']['unique_majority'] is None and r['truth']['any_annotated_fraction']>0 for r in rows),
        class_distribution=dict(Counter(LABELS[r['vlm_class']] for r in rows)),complementarity=dict(counts),
        ideal_midpoint_hits_histogram=dict(Counter(r['truth']['ideal_eight_midpoints_on_majority'] for r in valid)))


def teacher_rows(cache,db,student=None):
    result=[]
    for video,rr in cache['by_video'].items():
        for r in rr:
            interval=[r['proposal_start_sec'],r['proposal_end_sec']]
            result.append(dict(candidate_id=r['candidate_id'],video_id=video,interval=interval,
                vlm_class=int(r['class_id']),confidence=max(r['class_probabilities']),selected=r['participation_selected'],
                student_class=(student[r['candidate_id']]['class_id'] if student is not None else None),
                truth=truth_for(interval,db[video]['annotations'])))
    return result


def teacher_audit(rows,db):
    selected=[r for r in rows if r['selected']]
    confidence_bins=[]
    for lo,hi in [(0.,.5),(.5,.8),(.8,.9),(.9,.95),(.95,.99),(.99,1.000001)]:
        subset=[r for r in rows if lo<=r['confidence']<hi]
        confidence_bins.append(dict(low=lo,high=min(hi,1.),statistics=semantic_summary(subset)))
    # 类别正确度与动作是否存在分开，拒绝把closed-class置信度叫actionness。
    valid=[r for r in rows if r['truth']['unique_majority'] is not None]
    def pool(rr):return [dict(video_id=r['video_id'],interval=r['interval'],class_id=r['vlm_class']) for r in rr]
    return dict(all=semantic_summary(rows),top40=semantic_summary(selected),
        confidence_bins=confidence_bins,
        confidence_auc_for_known_majority=auc([r['confidence'] for r in rows],[r['truth']['unique_majority'] is not None for r in rows]),
        confidence_auc_for_class_correctness=auc([r['confidence'] for r in valid],[r['vlm_class']==r['truth']['unique_majority'] for r in valid]),
        all_coverage=coverage(pool(rows),db),selected_coverage=coverage(pool(selected),db))


def training_audit(entry):
    cfg=read(entry/'config.json');steps=cfg['baseline']['epoch_budget']['iterations_per_epoch']
    path=entry/'metrics.jsonl'
    if not path.exists():return dict(status='not_started')
    rows=[json.loads(x) for x in path.read_text().splitlines()];out=[]
    for x in rows:
        a=x['auxiliary'];n=a['rq3_selected']*steps
        # 每个非空批次至少包含一个候选，因此这是零入选批次比例的下界，不是实测比例。
        out.append(dict(epoch=x['epoch'],steps=steps,selected_exposures=n,
            minimum_fraction_batches_without_selected=max(0.,1.-a['rq3_selected']),
            candidates_seen=a['rq3_candidate_count']*steps,fully_visible_candidates=a['rq3_fully_visible_count']*steps,
            source_aux_ce=a['rq3_source_loss'],target_alignment_unweighted=a['rq3_alignment_loss'],
            target_feature_gradient_norm=a['rq3_target_gradient_norm'],
            effective_target_coefficient=.1*(1. if x['epoch']<cfg['train_cfg']['warmup_adv'] else cfg['model']['loss_weights']['weight_downstream'])))
    return dict(status='observed_complete_epochs',epochs=len(rows),epoch_statistics=out,
        unique_exposures_not_recorded=True,zero_batch_fraction_is_lower_bound=True,
        feature_gradient_and_clipped_parameter_gradient_must_not_be_compared=True)


def prototype_geometry(entry):
    import torch
    saved=torch.load(entry/'best.pt',map_location='cpu');state=saved['ema_state_dict']
    p=state['rq3_capacity.branch.source_prototypes'][:10];ready=state['rq3_capacity.branch.source_prototype_ready'][:10]
    normalized=torch.nn.functional.normalize(p,dim=-1);cos=(normalized@normalized.T).tolist()
    pairs=sorted([(cos[i][j],LABELS[i],LABELS[j]) for i in range(10) for j in range(i+1,10) if ready[i] and ready[j]],reverse=True)
    return dict(source_best_epoch=saved['epoch'],known_classes_ready=int(ready.sum()),cosine_matrix=cos,
        most_similar_class_pairs=pairs[:10],not_proof_of_collapse=True)


def run(root):
    root.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(code=code_identity(),scope='six_shift_existing_artifact_offline_chain_audit',
        no_training=True,no_new_VLM_queries=True,target_gt_for_evaluation_only=True,
        semantic_truth='single known class covers strictly more than half the interval by union length',
        clean_truth='maximum tIoU >=0.5 with unique known label',
        input_midpoints_are_ideal_not_actual_decoded_PTS=True))
    memo={};reports={};inputs={}
    for s in SHIFTS:
        control=CONTROL/'formal/control'/s;cfg=read(control/'config.json')
        annotation=Path(cfg['baseline']['annotation_for_offline_evaluation'])
        if sha(annotation)!=cfg['baseline']['annotation_sha256']:raise ValueError('离线GT身份改变')
        if str(annotation) not in memo:memo[str(annotation)]=read(annotation)['database']
        members=read(cfg['baseline']['target_manifest'])['database'];db={v:memo[str(annotation)][v] for v in members}
        old_path=Path(cfg['prototype_capacity']['prepared_cache'])
        if sha(old_path)!=cfg['prototype_capacity']['prepared_cache_sha256']:raise ValueError('原版缓存改变')
        old_rows=teacher_rows(read(old_path),db)
        new_path=REFRESH/'prepared'/(s+'.json');new_cache=validate_cache(new_path)
        inferred=DIAGNOSTIC/'entries'/(s+'_prototype');done=read(inferred/'inference_complete.json')
        for name in ['raw_pool.json','selected_pool.json']:
            if sha(inferred/name)!=done['outputs'][name]:raise ValueError('学生候选工件改变')
        new_pool=read(inferred/'selected_pool.json');lookup={r['candidate_id']:r for r in new_pool}
        new_rows=teacher_rows(new_cache,db,lookup)
        for row in new_rows:
            if row['interval']!=lookup[row['candidate_id']]['interval']:raise ValueError('学生/VLM没有比较同一区间')
        stages={}
        for m in ['so','uda','prototype']:
            e=DIAGNOSTIC/'entries'/(s+'_'+m);d=read(e/'inference_complete.json')
            if not d['target_gt_input_empty'] or not d['no_training']:raise ValueError('前序诊断输入含GT')
            for name in ['raw_pool.json','selected_pool.json']:
                if sha(e/name)!=d['outputs'][name]:raise ValueError('前序候选工件改变')
            raw=read(e/'raw_pool.json');pool=read(e/'selected_pool.json')
            stages[m]=dict(raw_threshold_pool=coverage(raw,db),sha_budget200=coverage(pool,db))
        outcomes={}
        for name,e in [('uda',BASELINE/'formal'/s/'uda'),('original',control),('refreshed',REFRESH/'formal/single_refresh'/s)]:
            if (e/'complete.json').exists():
                out=read(e/'evaluation/target_val.json');outcomes[name]=dict(mean_map=out['map_0.1_to_0.5'],
                    map_by_tiou=out['map_by_tiou'],selected_epoch=out['selected_epoch'],complete_sha256=sha(e/'complete.json'))
            else:outcomes[name]=dict(status='not_complete')
        report=dict(shift=s,target_videos=len(db),candidate_generation=stages,
            original_teacher=teacher_audit(old_rows,db),refreshed_teacher=teacher_audit(new_rows,db),
            original_training=training_audit(control),refreshed_training=training_audit(REFRESH/'formal/single_refresh'/s),
            source_prototype_geometry=prototype_geometry(control),outcomes=outcomes,
            graph=dict(vlm_only_selects_class_and_budget=True,auxiliary_classification_head_distinct_from_detector=True,
                direct_aux_gradient_to_detector_heads=False,shared_SGP_FPN_receives_aux_gradient=True,
                refreshed_cache_does_not_replace_SADA_pseudo_label_generator=True))
        save(root/(s+'.json'),report);reports[s]=dict(path=str(root/(s+'.json')),sha256=sha(root/(s+'.json')))
        inputs[s]=dict(annotation_sha256=sha(annotation),old_cache_sha256=sha(old_path),new_cache_sha256=sha(new_path),
            control_complete_sha256=sha(control/'complete.json'),candidate_inference_sha256=sha(inferred/'inference_complete.json'))
        print('AUDITED_CHAIN',s,flush=True)
    save(root/'audit.json',dict(status='passed',scope='CPU evidence audit; gradient direction and exact PTS require separate diagnostics',
        shifts=reports,inputs=inputs,no_training_or_hyperparameter_changes=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();os.chdir('/vepfs/vlms/SADA_PLUS');run(args.output.resolve())
