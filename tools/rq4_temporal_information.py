"""固定源域视频留一诊断：VLM相对proposal的时间信息，不训练学生或查询教师。"""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor

from tools.rq4_protocol import file_sha, interval_iou
from tools.rq4_source_acceptance import save
from tools.run_rq4_temporal_probe import teacher_pairs, clean_head

POLICY = dict(seed=6, max_depth=2, min_samples_leaf=10,
              max_correction_durations=1., permutations=100, video_bootstraps=10000)
TEACHERS = ('timelens2_8b', 'timeple_8b')


def read(path):
    return json.loads(Path(path).read_text())


def scores(box, gt):
    if gt is None:
        return dict(iou=0., start_error=None, end_error=None)
    duration = gt[1]-gt[0]
    return dict(iou=interval_iou(box, gt), start_error=abs(box[0]-gt[0])/duration,
                end_error=abs(box[1]-gt[1])/duration)


def summarize(rows, boxes):
    values = [scores(box, row['gt']) for row, box in zip(rows, boxes)]
    videos = sorted({r['video_id'] for r in rows})
    per_video = {v:float(np.mean([s['iou'] for r,s in zip(rows,values) if r['video_id']==v]))
                 for v in videos}
    return dict(count=len(rows), matched_count=sum(r['gt'] is not None for r in rows),
        mean_iou=float(np.mean([s['iou'] for s in values])),
        video_balanced_iou=float(np.mean(list(per_video.values()))), per_video=per_video,
        start_error=float(np.mean([s['start_error'] for s in values if s['start_error'] is not None])),
        end_error=float(np.mean([s['end_error'] for s in values if s['end_error'] is not None])))


def direct_report(rows, pairs):
    outputs = {}
    for context in (0, 1):
        boxes = [p[context] if p[context] is not None else r['proposal'] for r,p in zip(rows,pairs)]
        result = summarize(rows, boxes)
        accepted = [(r,p[context]) for r,p in zip(rows,pairs) if p[context] is not None and r['gt'] is not None]
        result['coverage'] = sum(p[context] is not None for p in pairs)/len(rows)
        for index,name in enumerate(('start','end')):
            improved = worsened = moved = directed = direction_count = 0
            for row, box in accepted:
                old, new, gt = row['proposal'][index], box[index], row['gt'][index]
                improved += abs(new-gt) < abs(old-gt)-1e-8
                worsened += abs(new-gt) > abs(old-gt)+1e-8
                moved += abs(new-old)>1e-8
                if abs(new-old)>1e-8 and abs(gt-old)>1e-8:
                    direction_count += 1
                    directed += (new-old)*(gt-old)>0
            result[name+'_correction'] = dict(valid_matched=len(accepted),improved=improved,
                worsened=worsened,moved=moved,direction_evaluable=direction_count,
                correct_direction=directed)
        old = [scores(r['proposal'],r['gt'])['iou'] for r in rows]
        new = [scores(b,r['gt'])['iou'] for r,b in zip(rows,boxes)]
        result['delta_iou'] = float(np.mean(np.subtract(new,old)))
        outputs[str(context+2)+'x'] = result
    return outputs


def oracle_report(rows, pairs):
    whole, endpoints = [], []
    for row,pair in zip(rows,pairs):
        pool = [row['proposal']]+[p for p in pair if p is not None]
        whole.append(max(pool,key=lambda b:scores(b,row['gt'])['iou']))
        mixed = [list(b) for b in product([p[0] for p in pool],[p[1] for p in pool]) if b[0]<b[1]]
        endpoints.append(max(mixed,key=lambda b:scores(b,row['gt'])['iou']))
    return dict(whole_box=summarize(rows,whole), endpoint_mix=summarize(rows,endpoints),
                deployable=False, explanation='GT upper bound including unchanged proposal; not learned gain')


def geometry(rows):
    # 只用冻结proposal元数据，不含GT、video ID或真实类别。
    return np.array([[np.log(r['proposal'][1]-r['proposal'][0]),r['fpn_level'],
        (r['anchor_sec']-r['proposal'][0])/(r['proposal'][1]-r['proposal'][0]),
        sum(r['proposal'])/(2*r['duration'])] for r in rows],dtype=float)


def teacher_features(rows, pairs):
    result = []
    for row,pair in zip(rows,pairs):
        proposal = np.array(row['proposal']); duration=proposal[1]-proposal[0]
        offsets = [((np.array(p)-proposal)/duration).tolist() if p is not None else [0.,0.] for p in pair]
        result.append(offsets[0]+offsets[1]+[float(p is not None) for p in pair]+
            [interval_iou(*pair) if all(p is not None for p in pair) else 0.])
    return np.array(result)


def corrected(proposal, residual, duration):
    cap=POLICY['max_correction_durations']
    box=np.clip(np.array(proposal)+np.clip(residual,-cap,cap)*(proposal[1]-proposal[0]),0.,duration)
    return box.tolist() if box[0]<box[1] else list(proposal)


def predict_fold(rows, features, heldout):
    train=np.array([r['video_id']!=heldout and r['gt'] is not None for r in rows])
    test=np.array([r['video_id']==heldout for r in rows])
    assert not set(r['video_id'] for r,t in zip(rows,train) if t)&{heldout}
    fit_rows=[r for r,t in zip(rows,train) if t]
    targets=np.array([(np.array(r['gt'])-r['proposal'])/(r['proposal'][1]-r['proposal'][0]) for r in fit_rows])
    counts={v:sum(r['video_id']==v for r in fit_rows) for v in {r['video_id'] for r in fit_rows}}
    weights=np.array([1/counts[r['video_id']] for r in fit_rows]); weights*=len(weights)/weights.sum()
    model=DecisionTreeRegressor(max_depth=POLICY['max_depth'],min_samples_leaf=POLICY['min_samples_leaf'],random_state=POLICY['seed'])
    model.fit(features[train],targets,sample_weight=weights)
    pred=model.predict(features[test])
    boxes=[corrected(row['proposal'],p,row['duration']) for row,p in zip([r for r,t in zip(rows,test) if t],pred)]
    return np.where(test)[0],boxes,dict(heldout_video=heldout,train_videos=sorted(counts),
        train_count=int(train.sum()),test_count=int(test.sum()),leaf_count=int(model.get_n_leaves()))


def cross_validate(rows, extra):
    features=np.concatenate([geometry(rows),extra],axis=1)
    boxes=[None]*len(rows); folds=[]
    for video in sorted({r['video_id'] for r in rows}):
        indices,output,fold=predict_fold(rows,features,video)
        for i,box in zip(indices,output): boxes[i]=box
        folds.append(fold)
    return dict(summary=summarize(rows,boxes),predictions=boxes,folds=folds)


def shuffled_features(extra, rows, seed):
    rng=np.random.default_rng(seed); shuffled=extra.copy()
    # 在每个视频内置乱相对位移/有效性/一致性整组，保留各视频的边际分布。
    for vid in sorted({r['video_id'] for r in rows}):
        ids=np.array([i for i,r in enumerate(rows) if r['video_id']==vid])
        shuffled[ids]=extra[rng.permutation(ids)]
    return shuffled


def paired_uncertainty(a,b):
    keys=sorted(a);delta=np.array([a[k]-b[k] for k in keys])
    rng=np.random.default_rng(POLICY['seed'])
    bootstrap=delta[rng.integers(len(delta),size=(POLICY['video_bootstraps'],len(delta)))].mean(1)
    return dict(video_count=len(keys),mean_delta=float(delta.mean()),
        paired_video_bootstrap_95_interval=np.quantile(bootstrap,[.025,.975]).tolist(),
        videos_improved=int((delta>0).sum()),videos_worsened=int((delta<0).sum()),
        exploratory=True, caveat='seven reused source videos; not independent confirmation or causal proof')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-head',required=True)
    args=parser.parse_args();clean_head(args.expected_head)
    args.output.mkdir(exist_ok=False,parents=True)
    plan=read(args.probe_root/'plan.json')
    if plan['shift']!='S1' or plan['target_gt_used']:raise ValueError('S1 source only')
    for ref in [plan['inputs']['scoring_only.json'],plan['annotations']]:
        assert file_sha(ref['path'])==ref['sha256']
    truth=read(plan['inputs']['scoring_only.json']['path'])
    rows=truth['candidates'];pairs={};evidence={}
    for teacher in TEACHERS:
        source=plan['source_evidence'][teacher]
        pairs[teacher],evidence[teacher]=teacher_pairs(source['requests']['path'],
            Path(source['responses']['path']).parent,plan['annotations']['path'],teacher)
        for ref in source.values():assert file_sha(ref['path'])==ref['sha256']
        diagnostics=read(source['diagnostics']['path'])
        meta={(r['candidate']['video_id'],r['candidate']['candidate_id'].rsplit('-context',1)[0]):r['candidate'] for r in diagnostics['records']}
        assert set(pairs[teacher])=={(r['video_id'],r['candidate_id']) for r in rows}
        for row in rows:
            c=meta[(row['video_id'],row['candidate_id'])]
            assert np.allclose(c['proposal'],row['proposal'],atol=1e-6,rtol=0)
            row.update(fpn_level=c['fpn_level'],anchor_sec=c['anchor_sec'])
        pairs[teacher]=[pairs[teacher][(r['video_id'],r['candidate_id'])] for r in rows]
    policy=dict(schema='rq4-vlm-incremental-temporal-information-v1',code_commit=args.expected_head,
        settings=POLICY,source_evidence=evidence,probe_plan_sha256=file_sha(args.probe_root/'plan.json'),
        scope='S1 all95 cached source proposals, fixed previous class-agnostic instance match; no target data',
        comparator='same tree depth/leaf limit and residual targets; zero teacher slots for proposal-only; shuffled arm has identical feature count',
        no_hyperparameter_search=True,no_gpu=True,selection='all out-of-fold predictions, not best fold',
        libraries={'numpy':np.__version__,'sklearn':sklearn.__version__})
    save(args.output/'policy.json',policy)
    baseline=summarize(rows,[r['proposal'] for r in rows])
    proposal=cross_validate(rows,np.zeros((len(rows),7)))
    report=dict(policy=policy,status='complete',original=baseline,proposal_only=proposal,teachers={})
    for teacher in TEACHERS:
        extra=teacher_features(rows,pairs[teacher]); actual=cross_validate(rows,extra)
        null=[]
        for index in range(POLICY['permutations']):
            trial=cross_validate(rows,shuffled_features(extra,rows,POLICY['seed']+index))
            null.append(trial['summary']['video_balanced_iou'])
        observed=actual['summary']['video_balanced_iou']
        report['teachers'][teacher]=dict(direct=direct_report(rows,pairs[teacher]),oracle=oracle_report(rows,pairs[teacher]),
            learned=actual,increment=paired_uncertainty(actual['summary']['per_video'],proposal['summary']['per_video']),
            shuffled=dict(video_balanced_iou=null,mean=float(np.mean(null)),
                actual_minus_shuffled_mean=observed-float(np.mean(null)),
                fraction_at_least_actual=float((1+sum(x>=observed for x in null))/(len(null)+1)),
                caution='diagnostic permutation reference, not causal proof; proposal geometry correlation also disrupted'))
        print('INFORMATION_DIAGNOSTIC',teacher,json.dumps(report['teachers'][teacher]['increment']),flush=True)
    save(args.output/'rows.json',dict(rows=rows,pairs=pairs))
    save(args.output/'result.json',report)
    save(args.output/'complete.json',dict(status='passed',code_commit=args.expected_head,
        artifacts={p.name:file_sha(p) for p in args.output.iterdir() if p.is_file()},formal_rq4=False))


if __name__=='__main__':main()
