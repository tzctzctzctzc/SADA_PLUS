"""CPU-only, read-only analysis of the frozen RQ1/2/3 result inventory.

Writes only beneath --output. No training imports or CUDA calls. All rates are
fractions, except explicitly suffixed _pp/_pct. Missing evidence stays null.
DETAD taxonomy reference: HumamAlwassel/DETAD, action_detector_diagnosis.py.
This is an independent implementation, not a claim to reproduce all DETAD tools.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = ['TAKE', 'PUT', 'WASH', 'OPEN', 'CLOSE', 'INSERT', 'TURN-ON', 'CUT', 'TURN-OFF', 'POUR']
ERRORS = ['tp', 'duplicate', 'wrong_label', 'localization', 'confusion', 'background']
THRESHOLDS = np.linspace(.1, .5, 5)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(s) for s in stream if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return float(a / b) if b else None


def classification(y, pred, prob):
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm)
    f1 = np.divide(2*tp, cm.sum(0)+cm.sum(1), out=np.zeros(10), where=(cm.sum(0)+cm.sum(1))>0)
    conf = prob.max(1)
    correct = y == pred
    bins, ece = [], 0.
    for i in range(10):
        mask = (conf >= i/10) & ((conf < (i+1)/10) if i < 9 else (conf <= 1+1e-6))
        if mask.any():
            acc, c = float(correct[mask].mean()), float(conf[mask].mean())
            ece += mask.mean()*abs(acc-c)
            bins.append({'lower': i/10, 'upper': (i+1)/10, 'n': int(mask.sum()), 'accuracy': acc, 'confidence': c})
    # Stable order explicitly resolves tied confidence; full curves are saved.
    order = np.argsort(-conf, kind='stable')
    risk = np.cumsum(~correct[order])/np.arange(1, len(y)+1)
    curve = []
    for coverage in np.linspace(.05, 1., 20):
        k = max(1, int(np.ceil(len(y)*coverage)))
        curve.append({'coverage': k/len(y), 'n': k, 'risk': float(risk[k-1])})
    return {'n': len(y), 'accuracy': float(correct.mean()), 'macro_f1': float(f1.mean()),
            'macro_recall': float(np.divide(tp, cm.sum(1), out=np.zeros(10), where=cm.sum(1)>0).mean()),
            'ece_10_equal_width': float(ece), 'brier_sum_10_classes': float(((prob-np.eye(10)[y])**2).sum(1).mean()),
            'aurc_discrete_mean': float(risk.mean()), 'risk_coverage': curve, 'reliability_bins': bins,
            'confusion': cm.tolist(), 'per_class_f1': f1.tolist()}


def rq1(base, out):
    root = base/'SADA_PLUS/artifacts/rq1_semantic_20260915'
    index = read(root/'rq1_8vlm_main_table.json')
    manifest = lines(root/'rq1_val_gt_segments.jsonl')
    rows, per_class, curves, reliability, details, crosschecks = [], [], [], [], [], []
    for model in index['models']:
        pf = root/model['protocol_file'].replace('.protocol.json', '')
        old = read(root/model['metrics_file'])['models']
        matches = [v for v in old.values() if Path(v['prediction_file']).name == pf.name]
        assert len(matches) == 1, (model['model'], pf, list(old))
        old = matches[0]
        for seed in ['ensemble', 0, 1, 2, 3, 4]:
            path = pf if seed == 'ensemble' else pf.with_name(pf.stem+f'.seed{seed}'+pf.suffix)
            predictions = {(p['candidate_id'], p['shift']): p for p in lines(path)}
            prediction_digest = sha(path)
            for shift in [f'S{i}' for i in range(1, 7)]:
                for domain in ['source', 'target']:
                    expected = [r for r in manifest if {'shift': shift, 'domain': domain} in r['memberships']]
                    joined = [(r, predictions[(r['candidate_id'], shift)]) for r in expected if (r['candidate_id'], shift) in predictions]
                    assert len(joined) == len(expected) and all(p['status']=='ok' for _, p in joined)
                    y = np.array([r['gt_label_id'] for r, _ in joined])
                    pred = np.array([p['class_id'] for _, p in joined])
                    prob = np.array([p['class_probabilities'] for _, p in joined])
                    assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-5)
                    m = classification(y, pred, prob)
                    ident = {'model': model['model'], 'seed': seed, 'seed_note': 'probability ensemble seeds0-4' if seed=='ensemble' else 'single seed', 'shift': shift, 'domain': domain}
                    small = {k:v for k,v in m.items() if isinstance(v, (int, float))}
                    row = {**ident, **small, 'prediction_sha256': prediction_digest}
                    if seed == 'ensemble':
                        prev = old['shifts'][shift][domain]
                        consistent = abs(m['macro_f1']-prev['macro_f1']) < 1e-9
                        row['matches_existing_metrics'] = consistent
                        crosschecks.append({**ident,'recomputed_macro_f1':m['macro_f1'],'existing_macro_f1':prev['macro_f1'],'difference_pp':(m['macro_f1']-prev['macro_f1'])*100,'matches':consistent,'prediction_path':str(path),'metrics_path':str(root/model['metrics_file'])})
                        # Recompute intervals from the exact predictions being analyzed.
                        video_cms = defaultdict(lambda: np.zeros((10,10),dtype=int))
                        for (r,p),gt,pr in zip(joined,y,pred): video_cms[r['video_id']][gt,pr] += 1
                        matrices = np.array(list(video_cms.values()))
                        rng = np.random.default_rng(20261009)
                        bs = []
                        for _ in range(1000):
                            cm = matrices[rng.integers(len(matrices),size=len(matrices))].sum(0)
                            denom=cm.sum(0)+cm.sum(1)
                            bs.append(float(np.divide(2*np.diag(cm),denom,out=np.zeros(10),where=denom>0).mean()))
                        ci = np.percentile(bs,[2.5,97.5]).tolist()
                        row.update({'macro_f1_video_ci_low':ci[0],'macro_f1_video_ci_high':ci[1],'ci_unit':'1000 video bootstrap; not seed uncertainty'})
                        details.append({**ident, **m, 'recomputed_video_bootstrap_95ci':ci,'matches_existing_metrics':consistent})
                        for c in range(10):
                            cm = np.array(m['confusion'])
                            per_class.append({**ident, 'class_id': c, 'class': LABELS[c], 'f1': m['per_class_f1'][c], 'recall': ratio(cm[c,c], cm[c].sum()), 'precision': ratio(cm[c,c], cm[:,c].sum()), 'support': int(cm[c].sum())})
                        curves.extend({**ident, **r} for r in m['risk_coverage'])
                        reliability.extend({**ident, **r} for r in m['reliability_bins'])
                    rows.append(row)
        print('RQ1', model['model'], flush=True)
    table(out/'rq1_classification.csv', rows)
    table(out/'rq1_class_metrics.csv', per_class)
    table(out/'rq1_risk_coverage.csv', curves)
    table(out/'rq1_reliability.csv', reliability)
    dump(out/'rq1_confusions_and_bootstrap.json', details)
    table(out/'rq1_existing_metrics_crosscheck.csv',crosschecks)
    return rows


def inventory(base):
    runs = []
    def add(path, rq, protocol, model, regime, seed, shift, complete=None):
        assert path.is_dir(), str(path)
        runs.append({'path': str(path), 'rq': rq, 'protocol': protocol, 'model': model, 'regime': regime, 'seed': seed, 'shift': shift, 'complete': str(complete) if complete else None})
    for model in ['slowfast', 'egovideo_hod', 'internvideo2_clip_1b', 'qwen3_vl_8b']:
        for seed in ([6] if model == 'qwen3_vl_8b' else [6,7,8]):
            for n in range(1,7):
                for regime in ['so','uda']:
                    suffix = f'S{n}_seed{seed}'
                    if model == 'slowfast':
                        path = base/f'rq2_slowfast_{regime}_cc8_0924'/model/suffix
                    elif model == 'qwen3_vl_8b':
                        root = 'rq2_qwen_uda_seed6_0926' if regime=='uda' else ('rq2_qwen_so_5a8041d_0925' if n<=2 else 'rq2_qwen_so_5a8041d_seed6resume_0925')
                        path = base/root/suffix
                    else:
                        root = 'rq2_uda_runs_7902377_0924' if regime=='uda' else ('rq2_so_eval_9a4b0f_0924' if n<=2 and seed==6 else 'rq2_so_full_dee1b_0924')
                        path = base/root/model/suffix
                    add(path, 'RQ2', 'rq2-source-step', model, regime, seed, f'S{n}')
    for n in range(1,7):
        for regime in ['so','uda']:
            add(base/f'rq2_matched_slowfast_S{n}_seed6_7067306_0927'/regime/f'S{n}_seed6', 'baseline', 'rq3-matched', 'slowfast', regime, 6, f'S{n}')
    old = read(base/'rq3_immediate_a2b4be5_1007/final_audit_16_seed6_1007.json')
    completes = [Path(r['complete']) for r in old['rows']]
    for root in ['rq3_intern_prototype_S1_batch2_9fe8195_1007', 'rq3_intern_batch2_remaining7_1008', 'rq3_qwen_batch2_remaining12_1008', 'rq3_intern_batch2_gpu2_remaining3_1008', 'rq3_qwen_batch2_cachefix9_b91b347_1008', 'rq3_intern_earlystop_refresh_b65023d_1008', 'rq3_earlystop15_gpu3_b65023d_1008', 'rq3_qwen_S6_gpu2_b65023d_1008']:
        completes.extend((base/root).rglob('complete.json'))
    seen = {}
    for f in completes:
        c = read(f)
        key = (c['teacher'], c['strategy'], c['shift'], c['seed'])
        if key in seen:
            assert seen[key] == str(f), (key, seen[key], f)
            continue
        seen[key] = str(f)
        add(f.parent/'evaluation', 'RQ3', 'rq3-matched', c['teacher'], c['strategy'], c['seed'], c['shift'], f)
    assert len(seen) == 36 and len(runs) == 168, (len(seen),len(runs))
    return runs


def ground_truth(database, splits):
    splits = {s.lower() for s in ([splits] if isinstance(splits,str) else splits)}
    result = {}
    for vid, v in database.items():
        if v['subset'].lower() not in splits:
            continue
        kept = []
        for a in v['annotations']:
            label = int(a['label_id'])
            if label not in range(10):
                continue
            s,e = a['segment']
            if any(label==p[2] and abs(s-p[0])<=.001 and abs(e-p[1])<=.001 for p in kept):
                continue
            kept.append((s,e,label))
        if kept:
            result[vid] = np.asarray(kept, dtype=float)
    return result


def error_code(best_iou, same_label, threshold):
    if best_iou >= threshold:
        return 1 if same_label else 2
    if best_iou >= .1:
        return 3 if same_label else 4
    return 5


def ap_from_tp(tp, n_gt):
    if not len(tp) or not n_gt:
        return 0.
    recall = np.r_[0., np.cumsum(tp)/n_gt, 1.]
    precision = np.r_[0., np.cumsum(tp)/np.arange(1,len(tp)+1), 0.]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall)*precision[1:]))


def detection(predictions, gt):
    """Full-prediction matching + DETAD error counts at full/1x/5x GT budgets.

    Duration recall bins are descriptive fixed seconds, not DETAD normalized mAP.
    Corrected-AP after removing an error type is a diagnostic upper bound only.
    """
    pred_by_video = defaultdict(list)
    for i,p in enumerate(predictions):
        pred_by_video[p['video-id']].append(i)
    labels = np.array([int(p['label']) for p in predictions])
    scores = np.array([float(p['score']) for p in predictions])
    n = len(predictions)
    assert np.isfinite(scores).all() and np.isin(labels, np.arange(10)).all()
    matched = np.full((5,n), -1, dtype=int)
    errors = np.full((5,n), 5, dtype=np.int8)
    gt_counts = np.zeros(10, dtype=int)
    durations, gt_labels, offsets = [], [], {}
    offset = 0
    for vid,g in gt.items():
        offsets[vid] = offset
        offset += len(g)
        durations.extend(g[:,1]-g[:,0]); gt_labels.extend(g[:,2].astype(int))
        gt_counts += np.bincount(g[:,2].astype(int), minlength=10)
    durations, gt_labels = np.array(durations), np.array(gt_labels)
    for vid, ids in pred_by_video.items():
        g = gt.get(vid)
        if g is None:
            continue
        # Matching is independent across videos; preserve global score tie order.
        ids = np.array(ids)[np.argsort(-scores[ids], kind='stable')]
        used = np.zeros((5,len(g)), dtype=bool)
        for start in range(0,len(ids),512):
            chunk = ids[start:start+512]
            seg = np.array([[predictions[i]['t-start'], predictions[i]['t-end']] for i in chunk])
            # Legacy inference can clamp proposals to a zero-length video edge.
            # Keep them as zero-IoU false positives, as the original evaluator
            # does; do not silently remove them and improve AP.
            assert np.isfinite(seg).all() and np.all(seg[:,1]>=seg[:,0])
            intersection = np.maximum(0, np.minimum(seg[:,None,1],g[None,:,1])-np.maximum(seg[:,None,0],g[None,:,0]))
            ious = intersection/(seg[:,None,1]-seg[:,None,0]+g[None,:,1]-g[None,:,0]-intersection)
            for k,i in enumerate(chunk):
                io = ious[k]; best = int(io.argmax())
                candidates = np.flatnonzero((g[:,2]==labels[i]) & (io>=.1-1e-12))
                candidates = candidates[np.argsort(io[candidates])[::-1]]
                for ti,t in enumerate(THRESHOLDS):
                    eligible = candidates[(io[candidates]>=t) & ~used[ti,candidates]]
                    if len(eligible):
                        j = eligible[0]; used[ti,j] = True
                        matched[ti,i] = offsets[vid]+j; errors[ti,i] = 0
                    else:
                        errors[ti,i] = error_code(io[best], g[best,2]==labels[i], t)
    ap = np.zeros((5,10))
    class_orders = {}
    oracle = np.zeros((5,5,10))
    for c in range(10):
        ids = np.flatnonzero(labels==c)
        ids = ids[np.argsort(-scores[ids],kind='stable')]
        class_orders[c] = ids
        for ti in range(5):
            tp = matched[ti,ids]>=0
            ap[ti,c] = ap_from_tp(tp, gt_counts[c])
            for err in range(1,6):
                oracle[ti,err-1,c] = ap_from_tp(tp[errors[ti,ids]!=err],gt_counts[c])
    summaries = []
    for budget in ['all',1,5]:
        ids = np.concatenate([idx if budget=='all' else idx[:int(budget*gt_counts[c])] for c,idx in class_orders.items()])
        for ti,t in enumerate(THRESHOLDS):
            counts = np.bincount(errors[ti,ids], minlength=6)
            hit = np.zeros(len(durations), dtype=bool)
            hit[matched[ti,ids][matched[ti,ids]>=0]] = True
            summary = {'tiou': round(float(t),1), 'prediction_budget': str(budget), 'n_predictions': len(ids), 'n_gt': len(durations), 'recall': float(hit.mean()), 'miss_rate': float(1-hit.mean())}
            summary.update({name+'_count': int(counts[k]) for k,name in enumerate(ERRORS)})
            summary.update({name+'_fraction': ratio(counts[k],len(ids)) for k,name in enumerate(ERRORS)})
            for name,mask in [('short_le2s',durations<=2),('medium_2to5s',(durations>2)&(durations<=5)),('long_gt5s',durations>5)]:
                summary[name+'_n_gt'] = int(mask.sum())
                summary[name+'_recall'] = float(hit[mask].mean()) if mask.any() else None
            summaries.append(summary)
    return {'recomputed_map': float(ap.mean()), 'ap_by_tiou_class': ap.tolist(), 'summaries': summaries,
            'remove_error_ap_gain_pp': {ERRORS[k+1]: float((oracle[:,k,:].mean()-ap.mean())*100) for k in range(5)},
            'gt_class_counts': gt_counts.tolist(), 'prediction_score_ties': int(len(scores)-len(np.unique(scores)))}


def tal(base, out, runs, limit=0):
    annotation = base/'SADA_PLUS/data/ek100_slowfast/annotations/epic_kitchens_100_verb_detailed_split.json'
    ann_hash = sha(annotation); database = read(annotation)['database']
    rows, classes, errors, epoch_rows, supervision, audit = [], [], [], [], [], []
    for index, run in enumerate(runs):
        ident = {k:v for k,v in run.items() if k not in ['path','complete']}
        path = Path(run['path']); manifest = read(path/'manifest.json')
        complete = read(run['complete']) if run['complete'] else {}
        for domain in ['source','target']:
            f = path/f'{domain}_val.json'; ev = read(f)
            digest = sha(f)
            assert digest == manifest['outputs'][f.name], str(f)
            assert ev['seed']==run['seed'] and ev['shift']==run['shift']
            assert ev['annotation_sha256']==ann_hash
            assert ev['checkpoint_sha256']==manifest['checkpoint_sha256']
            if complete:
                assert ev['checkpoint_sha256']==complete['checkpoint_sha256']
            small = {**ident,'domain':domain,'map_pct':ev['map_0.1_to_0.5']*100,'best_epoch':ev['checkpoint_epoch'],'evaluation_sha256':digest,'path':str(f)}
            small.update({f'map_tiou_{float(k):.1f}_pct':v*100 for k,v in ev['map_by_tiou'].items()})
            rows.append(small)
            classes.extend({**ident,'domain':domain,'class_id':int(c),'class':LABELS[int(c)],'ap_pct':float(v)*100} for c,v in ev['ap_by_class'].items())
            if domain=='source':
                continue
            key = '_'.join(str(ident[k]) for k in ['rq','protocol','model','regime','shift','seed'])
            cache = out/'detection'/f'{key}.json'
            if not cache.exists() and (not limit or index < limit):
                diag = detection(ev['predictions'], ground_truth(database,ev['split']))
                diag['evaluation_sha256'] = digest
                diag['map_difference_pp'] = (diag['recomputed_map']-ev['map_0.1_to_0.5'])*100
                dump(cache, diag)
            if cache.exists():
                diag = read(cache)
                assert diag['evaluation_sha256']==digest
                # AP ties follow stable order here; log differences, never rewrite original mAP.
                audit.append({**ident,'map_difference_pp':diag['map_difference_pp'],'prediction_score_ties':diag['prediction_score_ties']})
                errors.extend({**ident,**r} for r in diag['summaries'])
            if complete:
                logdir = Path(ev['checkpoint']).parents[2]
                epochs = lines(logdir/'benchmark_epoch_metrics.jsonl')
                sums = defaultdict(float)
                for ep in epochs:
                    keep = {k:v for k,v in ep.items() if k=='epoch' or k.startswith(('t_rq3_', 't_target_gt_','lr_'))}
                    assert all(np.isfinite(v) for v in keep.values() if isinstance(v,(float,int)))
                    assert all(ep[k]==0 for k in ['t_target_gt_cls_loss','t_target_gt_reg_loss','t_target_gt_downstream_loss'])
                    epoch_rows.append({**ident,**keep})
                    for k,v in keep.items():
                        if k.startswith('t_rq3_'): sums[k] += v
                metrics = {**ident, 'epochs_recorded':len(epochs),'best_epoch':ev['checkpoint_epoch'], 'aggregation':'ratio of sums of epoch per-step means; repeated exposures, not unique candidates', 'early_stopping':bool(complete.get('early_stopping')),'stopping_budget_matched_baseline':complete.get('stopping_budget_matched_baseline',True),'gt_three_losses_all_zero':True}
                for prefix,num,den in [('verify_eligible_rate','verify_eligible','verify_observed'),('verify_reject_of_observed','verify_rejected','verify_observed'),('verify_reject_of_eligible','verify_rejected','verify_eligible'),('verify_uncertain_rate','verify_uncertain','verify_observed'),('verify_ambiguous_rate','verify_ambiguous','verify_observed'),('prototype_selected_of_visible','selected','fully_visible_count'),('prototype_visible_of_candidates','fully_visible_count','candidate_count')]:
                    metrics[prefix] = ratio(sums['t_rq3_'+num],sums['t_rq3_'+den])
                for k in ['loss','source_loss','alignment_loss','gradient_norm','target_gradient_norm','aligned_classes']:
                    metrics[k+'_epoch_mean'] = ratio(sums['t_rq3_'+k],len(epochs)) if 't_rq3_'+k in epochs[0] else None
                metrics['nonzero_target_gradient_epochs'] = sum(ep.get('t_rq3_target_gradient_norm',0)>0 for ep in epochs) if 't_rq3_target_gradient_norm' in epochs[0] else None
                metrics['verify_rejection_nonzero_epochs'] = sum(ep.get('t_rq3_verify_rejected',0)>0 for ep in epochs) if run['regime']=='verify' else None
                supervision.append(metrics)
        print('TAL',index+1,len(runs),key,flush=True)
    table(out/'tal_results.csv', rows); table(out/'tal_class_ap.csv', classes)
    table(out/'detection_errors.csv', errors); table(out/'detection_map_crosscheck.csv',audit)
    table(out/'rq3_epoch_trajectories.csv',epoch_rows); table(out/'rq3_supervision.csv',supervision)
    return rows,supervision


def aggregate(out, rq1_rows, tal_rows, supervision):
    target = [r for r in tal_rows if r['domain']=='target']
    pairs, summary = [], []
    lookup = {(r['protocol'],r['model'],r['regime'],r['shift'],r['seed']):r for r in target}
    for r in target:
        if r['rq']=='RQ3':
            b = lookup[('rq3-matched','slowfast','uda',r['shift'],6)]
        elif r['rq']=='RQ2' and r['regime']=='uda':
            b = lookup[(r['protocol'],r['model'],'so',r['shift'],r['seed'])]
        else:
            continue
        pairs.append({**{k:r[k] for k in ['rq','protocol','model','regime','shift','seed']},'target_map_pct':r['map_pct'],'reference_map_pct':b['map_pct'],'delta_pp':r['map_pct']-b['map_pct'],'reference':'matched SlowFast UDA' if r['rq']=='RQ3' else 'same-model SO'})
    groups = defaultdict(list)
    for r in pairs: groups[(r['rq'],r['model'],r['regime'])].append(r)
    for (rq,model,regime), rs in groups.items():
        # First average shifts within each training seed, then describe seeds.
        seeds = sorted(set(r['seed'] for r in rs))
        means = [np.mean([r['delta_pp'] for r in rs if r['seed']==s]) for s in seeds]
        summary.append({'rq':rq,'model':model,'regime':regime,'seeds':','.join(map(str,seeds)),'n_seeds':len(seeds),'seed_note':'single seed; no seed CI' if len(seeds)==1 else 'sample SD across per-seed six-shift means','target_map_pct':float(np.mean([r['target_map_pct'] for r in rs])),'reference_map_pct':float(np.mean([r['reference_map_pct'] for r in rs])),'delta_pp':float(np.mean(means)),'delta_seed_sd_pp':float(np.std(means,ddof=1)) if len(means)>1 else None,'positive_run_shift_pairs':sum(r['delta_pp']>0 for r in rs),'n_run_shift_pairs':len(rs),'worst_delta_pp':min(r['delta_pp'] for r in rs)})
    table(out/'paired_deltas.csv',pairs); table(out/'main_summary.csv',summary)
    dump(out/'availability.json',{'computed':['RQ1 confusion/per-class F1/recall/precision, Macro-F1/accuracy, ECE/Brier, risk-coverage/AURC, ensemble and five classifier seeds, video bootstrap','RQ2/3 mAP by tIoU/class, paired deltas, DETAD-style FP categories, fixed-duration recall and misses','RQ3 36-run epoch supervision/gradient/GT-isolation traces'], 'reused_with_crosscheck':[], 'not_available':['RQ1 UDA: no completed implementation/run in inspected inventory','RQ3 per-candidate historical correction/false-rejection correctness','per-epoch full embeddings, attention and gradient tensors','multiple training seeds for RQ3 or Qwen RQ2'], 'not_computed':['new GPU replay/oracle experiments','TAL video-bootstrap mAP intervals (not replaced by seed or shift intervals)','full DETAD sensitivity suite or error-repair retraining'], 'scope':'RQ1 source-trained classifier on GT segments; RQ2 source-step protocol separately from matched SlowFast; RQ3 revised Verify/Prototype only. No Hard/batch8/pre-fix mixing.', 'audit_scope':'evaluation JSON hashes checked against manifests; annotation hash checked; checkpoint hashes cross-referenced, not reread/rehashed; no new full checkpoint audit', 'interpretation':'target GT is used for offline analysis only; diagnostic oracle AP gains and correlations are not causal or deployable gains'})
    print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,default=Path('/vepfs/vlms')); p.add_argument('--output',type=Path,required=True); p.add_argument('--diagnostic-limit',type=int,default=0); p.add_argument('--skip-rq1',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    runs=inventory(a.base); dump(a.output/'inventory.json',runs)
    r1=[] if a.skip_rq1 else rq1(a.base,a.output)
    results,sup=tal(a.base,a.output,runs,a.diagnostic_limit)
    aggregate(a.output,r1,results,sup)
    dump(a.output/'analysis_provenance.json',{'script_sha256':sha(__file__),'runs':len(runs),'rq3_runs':sum(r['rq']=='RQ3' for r in runs),'diagnostic_limit':a.diagnostic_limit,'target_gt_role':'offline analysis only','device':'CPU','completed':not bool(a.diagnostic_limit)})


if __name__=='__main__':
    main()
