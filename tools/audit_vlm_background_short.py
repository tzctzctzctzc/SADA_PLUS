"""只读CPU终审：独立重算三态决策、真实BG、官方AP及同候选置信度对照。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ternary(values, order):
    assert len(values) == 3 and sorted(order) == [0, 1, 2]
    assert all(math.isfinite(x) for x in values)
    high = max(values)
    winner = 2 if values.count(high) > 1 else order[values.index(high)]
    weights = [math.exp(v-high) for v in values]
    probs = {state: w/sum(weights) for state, w in zip(order, weights)}
    return winner, probs[1]


def audit_group(root, name, job, rows):
    import numpy as np
    import torch
    from tools.foreground_oracle_ap import prediction_frame
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json, compute_average_precision_detection
    evaluation_path = root/'evaluations'/(name+'.json')
    evaluation = read(evaluation_path)
    for path, h in evaluation['artifacts'].items(): assert sha(path) == h
    cfg_path = root/'groups'/name/'config.json'
    assert sha(cfg_path) == job['config_sha256']
    cfg = read(cfg_path)
    source = job['domain'] == 'source_val'
    annotation = cfg['dataset']['json_file'] if source else cfg['baseline']['annotation_for_offline_evaluation']
    splits = cfg['src_val_split'] if source else cfg['tgt_train_split']
    gt = load_gt_seg_from_json(annotation, splits, action_subsampling='10_majority_classes')
    gt = gt[(gt['video-id'] == job['video']) & (gt['t-start'] != gt['t-end'])].reset_index(drop=True)
    original = torch.load(job['original'], map_location='cpu')
    frame = prediction_frame([original])
    boxes = frame[['t-start','t-end']].to_numpy(dtype=float)
    truth = gt[['t-start','t-end']].to_numpy(dtype=float)
    if len(truth):
        intersection = np.maximum(0., np.minimum(boxes[:,None,1],truth[None,:,1])-np.maximum(boxes[:,None,0],truth[None,:,0]))
        union = (boxes[:,1]-boxes[:,0])[:,None]+(truth[:,1]-truth[:,0])[None]-intersection
        overlaps = intersection/union
        maximum = overlaps.max(axis=1)
    else: maximum = np.zeros(len(frame))
    bg = maximum < .01
    def metric(mask):
        aps = []; tps = []
        for c in range(10):
            g = gt[gt.label == c].reset_index(drop=True)
            p = frame[mask & (frame.label.to_numpy() == c)].reset_index(drop=True)
            if not len(g) or not len(p):
                aps.append(np.zeros(5)); tps.append(np.zeros(5)); continue
            result = compute_average_precision_detection(g, p)
            aps.append(np.nan_to_num(result[0])); tps.append(result[1])
        return {'map': float(np.mean(aps)), 'AP_by_tiou': np.mean(aps,axis=0).tolist(), 'TP': np.sum(tps,axis=0).astype(int).tolist()}
    baseline = metric(np.ones(len(frame),dtype=bool))
    assert np.allclose(baseline['AP_by_tiou'],evaluation['baseline']['AP_by_tiou'],atol=1e-10,rtol=0)
    selected = sorted(map(int,job['candidate_map']))
    assert len(selected) == 8
    masks = {m: np.ones(len(frame),dtype=bool) for m in ('presence','verify','contrast')}
    details = []; response_paths = set()
    for index in selected:
        cid = job['candidate_map'][str(index)]; row = rows[cid]
        case_path = root/'cases'/(cid+'.json'); case = read(case_path)
        assert case['input_sha256'] == digest(row)
        responses = {}
        for kind, pointer in case['response_files'].items():
            path = pointer['path']; assert sha(path) == pointer['sha256']
            response = read(path); req = response['request']; response_paths.add(path)
            assert response['request_id'] == digest(req)
            assert req['times'] == row['times']
            expected = [f['sha256'] for f in row['frame_records']]
            if kind == 'static': expected = [expected[6]] * len(expected)
            assert req['frame_sha256'] == expected
            assert req['model_identity_sha256'] == sha(root/'model_identity.json')
            assert max(response['logits']) <= response['logsumexp'] + 1e-5
            guard = read(response['guard']); assert sha(response['guard']) == response['guard_sha256']
            lease = read(guard['lease']); assert sha(guard['lease']) == guard['lease_sha256']
            assert guard['owner_pid'] == lease['owner_pid']
            assert [x['uuid'] for x in guard['observed']] == [x['uuid'] for x in lease['cards']]
            assert min(x['free_mib'] for x in guard['observed']) >= 20480
            responses[kind] = response
        if row['observable']:
            p = responses['presence']; v = responses['predicted']; static = responses['static']
            ps, pb = ternary(p['logits'],p['request']['option_order'])
            vs, vb = ternary(v['logits'],v['request']['option_order'])
            os_, ob = (2,0.)
            if vs == 1:
                other = responses['others']; os_, ob = ternary(other['logits'],other['request']['option_order'])
            assert static['request']['option_order'] == p['request']['option_order']
            cs, cb = ternary([1.5*a-.5*b for a,b in zip(p['logits'],static['logits'])],p['request']['option_order'])
            delete = dict(presence=ps == 1,verify=vs == 1 and os_ == 1,contrast=cs == 1)
            scores = dict(presence=pb,verify=vb*ob if vs == 1 else 0.,contrast=cb)
        else:
            delete = {m:False for m in masks}; scores = {m:0. for m in masks}
        for m in masks:
            assert delete[m] == case['results'][m]['delete']
            assert abs(scores[m]-case['results'][m]['BG_score']) < 1e-12
            masks[m][index] = not delete[m]
        masses = {k:sum(math.exp(v-r['logsumexp']) for v in r['logits']) for k,r in responses.items()}
        details.append(dict(index=index,cid=cid,score=float(frame.score.iloc[index]),label=int(frame.label.iloc[index]),
            anyclass_max_IoU=float(maximum[index]),BG=bool(bg[index]),delete=delete,BG_scores=scores,option_mass=masses))
    results = {}
    labels = frame.label.to_numpy(); confidence = frame.score.to_numpy()
    for mode, keep in masks.items():
        assert keep[confidence < .3].all()
        assert set(np.flatnonzero(~keep)) <= set(selected)
        actual = metric(keep)
        reported = evaluation['results'][mode]
        assert np.allclose(actual['AP_by_tiou'],reported['actual']['AP_by_tiou'],atol=1e-10,rtol=0)
        assert actual['TP'] == reported['actual']['TP']
        control = np.ones(len(frame),dtype=bool)
        for c in range(10):
            ids = [i for i in selected if labels[i] == c]
            count = sum(not keep[i] for i in ids)
            chosen = sorted(ids,key=lambda i:(confidence[i],i))[:count]
            control[chosen] = False
        matched = metric(control)
        assert int((~keep).sum()) == int((~control).sum())
        results[mode] = dict(delta_pp=100*(actual['map']-baseline['map']),actual=actual,
            removed=int((~keep).sum()),BG_removed=int((~keep & bg).sum()),nonBG_removed=int((~keep & ~bg).sum()),
            same_selected_pool_confidence_delta_pp=100*(matched['map']-baseline['map']),
            confidence_BG_removed=int((~control & bg).sum()),confidence_nonBG_removed=int((~control & ~bg).sum()))
        assert results[mode]['BG_removed'] == reported['BG_removed']
        assert results[mode]['nonBG_removed'] == reported['nonBG_removed']
    return dict(passed=True,group=name,domain=job['domain'],video=job['video'],selected_BG=int(bg[selected].sum()),
        selected_count=8,results=results,candidates=details,response_paths=sorted(response_paths),
        evaluation_sha256=sha(evaluation_path),config_sha256=sha(cfg_path),truth_annotation_sha256=sha(annotation),
        scope='full video AP under fixed 8-candidate partial intervention; not exhaustive filtering',target_val_read=False)


def run(root, watch):
    groups = read(root/'groups.json'); rows = {r['candidate_id']:r for r in read(root/'inference_manifest.json')['cases']}
    assert len(groups) == 12 and len(rows) == 96
    output = root/'independent_results'; output.mkdir(exist_ok=True)
    while True:
        for name, job in groups.items():
            evaluation = root/'evaluations'/(name+'.json'); out = output/(name+'.json')
            if not evaluation.exists(): continue
            if out.exists():
                assert read(out)['evaluation_sha256'] == sha(evaluation); continue
            result = audit_group(root,name,job,rows)
            temp = out.with_suffix('.tmp'); temp.write_text(json.dumps(result,indent=2)); temp.replace(out)
            print(name, json.dumps(result['results']), flush=True)
        reports = [read(output/(name+'.json')) for name in groups if (output/(name+'.json')).exists()]
        if len(reports) == 12 and (root/'final.json').exists(): break
        if not watch: return
        assert not (root/'error.json').exists() and not (root/'stop.json').exists()
        time.sleep(10)
    summary = {}
    for domain in ('source_val','target_train'):
        subset = [r for r in reports if r['domain']==domain]
        summary[domain] = {'selected_candidates':sum(r['selected_count'] for r in subset),'selected_BG':sum(r['selected_BG'] for r in subset),'modes':{}}
        for mode in ('presence','verify','contrast'):
            summary[domain]['modes'][mode] = {k:sum(r['results'][mode][k] for r in subset) for k in ('removed','BG_removed','nonBG_removed','confidence_BG_removed','confidence_nonBG_removed')}
            for key in ('delta_pp','same_selected_pool_confidence_delta_pp'):
                summary[domain]['modes'][mode]['selected_video_macro_'+key] = sum(r['results'][mode][key] for r in subset)/len(subset)
    final = read(root/'final.json'); assert set(final['groups']) == set(groups) and final['exact_cases'] == 96
    for name in groups: assert final['groups'][name]['sha256'] == sha(root/'evaluations'/(name+'.json'))
    requests = {p for report in reports for p in report['response_paths']}
    costs = [read(p) for p in requests]
    report = dict(passed=True,exact_cases=96,exact_groups=12,summary=summary,
        final_sha256=sha(root/'final.json'),auditor_sha256=sha(__file__),
        group_artifacts={str(output/(name+'.json')):sha(output/(name+'.json')) for name in groups},
        selected_unique_requests=len(requests),selected_request_forward_count=sum(r['forward_count'] for r in costs),
        selected_request_prefill_tokens=sum(r['actual_prefill_tokens'] for r in costs),
        discarded_long_queue_cost_not_included_here=True,
        scope='fixed short mechanism pilot; 6 selected videos per domain; not full-domain or final UDA results')
    out=root/'independent_final_audit.json'; temp=out.with_suffix('.tmp'); temp.write_text(json.dumps(report,indent=2));temp.replace(out)
    print('FINAL',json.dumps(report),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();os.chdir('/vepfs/vlms/SADA_PLUS');run(args.root,args.watch)
