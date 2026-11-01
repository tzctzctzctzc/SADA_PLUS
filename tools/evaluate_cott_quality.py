"""仅CPU离线源GT评估；检测一对一匹配、重复/错类惩罚及全阶段成本审计。"""
import argparse
import os
import sys
from datetime import datetime,timezone
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.cott_protocol import *

def audit_video(root,v):
    entry=root/'videos'/v;done=read(entry/'complete.json')
    for name,h in done['artifacts'].items(): assert sha(entry/name)==h
    assert done['target_gt_used'] is False and done['training'] is False
    for name,h in done['call_result_sha256'].items():
        p=root/'calls'/v/name;assert sha(p/'result.json')==h
        r=read(p/'result.json');req=read(p/'request.json')
        assert sha(p/'request.json')==r['request_file_sha256']
        assert digest(req)==r['request_sha256']
        for f,a in r['attempt_sha256'].items(): assert sha(p/f)==a
        validate(r['parsed'],req['stage'],req['lo'],req['hi'],req['pair_count'])
    return read(entry/'predictions.json'),done

def compare(a,b,gt):
    out={}
    for t in SPEC['metrics_tiou']:
        ma,ua=match_predictions([p for p in a if p['confidence']>=.5],gt,t)
        mb,ub=match_predictions([p for p in b if p['confidence']>=.5],gt,t)
        out[str(t)]=dict(recovered_misses=len(ua-ub),lost_true_positives=len(ub-ua),
            delta_false_positives=(len(ma)-len(ua))-(len(mb)-len(ub)),
            candidate_a_tp=len(ua),reference_b_tp=len(ub))
    return out

def bootstrap_unique(rows,gt):
    import numpy as np
    ids=sorted(rows);rng=np.random.default_rng(123);values=[]
    # Cluster resampling uses independent video IDs, not repeated shift memberships.
    for _ in range(SPEC['bootstrap_video_replicates']):
        aa=[];bb=[];gg=[]
        for k,v in enumerate(rng.choice(ids,size=len(ids),replace=True)):
            aa += [dict(p,video_id=str(k)) for p in rows[v]['cott'] if p['confidence']>=.5]
            bb += [dict(p,video_id=str(k)) for p in rows[v]['direct'] if p['confidence']>=.5]
            gg += [dict(g,video_id=str(k)) for g in gt if g['video_id']==v]
        _,ua=match_predictions(aa,gg,.5);_,ub=match_predictions(bb,gg,.5)
        f=lambda n,p:2*n/(p+len(gg)) if p+len(gg) else 0.
        values.append(f(len(ua),len(aa))-f(len(ub),len(bb)))
    return dict(metric='CoTT minus direct class-aware F1@0.5 at score>=0.5',
        independent_videos=len(ids),replicates=len(values),seed=123,
        percentile_95=np.percentile(values,[2.5,97.5]).tolist(),
        caveat='exploratory small source subset, confidence interval is not target validation')

def evaluate(root,partial=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'evaluation CPU only'
    root=Path(root);plan=read(root/'plan.json');manifest=read(root/'manifest.json')
    assert sha(root/'manifest.json')==plan['manifest_sha256']
    assert sha(root/'evaluation_inputs/source_only.json')==plan['evaluation_inputs_sha256']
    inputs=read(root/'evaluation_inputs/source_only.json')
    expected={x['video_id'] for x in manifest['videos']};done={p.parent.name for p in (root/'videos').glob('*/complete.json')}
    assert done<=expected
    if not partial: assert done==expected and (root/'inference_complete.json').exists()
    if not done: raise ValueError('no complete videos')
    predictions={};receipts={};cost={}
    for v in sorted(done):
        predictions[v],receipt=audit_video(root,v);receipts[v]=sha(root/'videos'/v/'complete.json')
        cost[v]=read(root/'videos'/v/'cost.json')
    byshift={};unique_gt={}
    for s,selected in manifest['selection'].items():
        ids=sorted(set(selected)&done)
        if not ids: continue
        gt=[g for g in inputs[s]['ground_truth'] if g['video_id'] in ids]
        for v in ids:
            vg=[g for g in gt if g['video_id']==v]
            if v in unique_gt: assert unique_gt[v]==vg, 'overlap source annotations differ'
            unique_gt[v]=vg
        teacher=[p for p in inputs[s]['teacher_predictions'] if p['video_id'] in ids]
        cott=[p for v in ids for p in predictions[v]['cott']]
        direct=[p for v in ids for p in predictions[v]['direct']]
        byshift[s]=dict(videos=ids,complete_shift=len(ids)==len(selected),
            teacher=metrics(teacher,gt),direct=metrics(direct,gt),cott=metrics(cott,gt),
            cott_vs_teacher=compare(cott,teacher,gt),cott_vs_direct=compare(cott,direct,gt),
            actual_class_counts={c:sum(g['label']==c for g in gt) for c in CLASSES})
    gt=[g for v in sorted(unique_gt) for g in unique_gt[v]]
    cott=[p for v in sorted(done) for p in predictions[v]['cott']]
    direct=[p for v in sorted(done) for p in predictions[v]['direct']]
    out=dict(schema=SCHEMA,status='partial_source_pilot' if partial else 'complete_source_pilot',
        spec=SPEC,manifest_sha256=sha(root/'manifest.json'),video_complete_sha256=receipts,
        unique_videos=len(done),planned_unique_videos=len(expected),by_shift=byshift,
        unique_video_results=dict(cott=metrics(cott,gt),direct=metrics(direct,gt),
            cott_vs_direct=compare(cott,direct,gt),bootstrap=bootstrap_unique(predictions,gt)),
        cost=cost,model_identity_sha256=sha(root/'model_identity.json'),
        first_cuda_sha256=sha(root/'first_cuda.json'),
        conclusion_scope='source-only pilot; no target mAP, no student training, no proof of UDA gain')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    save(root/'evaluation'/('partial_'+stamp+'.json' if partial else 'final.json'),out)
    if not partial:
        save(root/'complete.json',dict(schema=SCHEMA,count=len(done),
            final_evaluation_sha256=sha(root/'evaluation/final.json'),video_complete_sha256=receipts,
            model_identity_sha256=sha(root/'model_identity.json'),target_gt_used=False))
        save(root/'state.json',dict(phase='complete_source_pilot',complete_videos=len(done)))
    print(json.dumps(dict(status=out['status'],videos=len(done),
        cott_f1_05=out['unique_video_results']['cott']['0.5']['f1'],
        direct_f1_05=out['unique_video_results']['direct']['0.5']['f1']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();evaluate(a.root,a.partial)
