"""完整模型的CPU误检、类别AP、参数漂移诊断；不选择目标epoch或改训练。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha


def iou_matrix(boxes, truth):
    inter = np.maximum(0, np.minimum(boxes[:, None, 1], truth[None, :, 1]) - np.maximum(boxes[:, None, 0], truth[None, :, 0]))
    return inter / np.maximum(1e-12, np.diff(boxes, axis=1) + np.diff(truth, axis=1).T - inter)


def error_partition(iou, predicted_labels, truth_labels, scores, threshold):
    """固定分数顺序的一对一TP；其余分类为互斥计数，不冒称可加AP贡献。"""
    counts = dict(tp=0, duplicate=0, classification=0, localization=0,
                  class_and_localization=0, background=0, near_background=0)
    matched = set()
    for index in np.argsort(-scores, kind='stable'):
        row = iou[index]
        if not len(row):
            counts['background'] += 1
            continue
        same = truth_labels == predicted_labels[index]
        available = [j for j in np.argsort(-row, kind='stable') if same[j] and row[j] >= threshold and j not in matched]
        if available:
            matched.add(available[0]);counts['tp'] += 1
        elif np.any(same & (row >= threshold)):
            counts['duplicate'] += 1
        elif row.max() < .01:
            counts['background'] += 1
        elif row.max() < .1:
            counts['near_background'] += 1
        elif np.any((~same) & (row >= threshold)):
            counts['classification'] += 1
        elif np.any(same & (row >= .1)):
            counts['localization'] += 1
        else:
            counts['class_and_localization'] += 1
    assert sum(counts.values()) == len(scores)
    return counts


def analyze_prediction(job):
    from libs.evaluation import ANETdetection
    from libs.evaluation.evaluator_anet import remove_duplicate_annotations
    entry = Path(job['entry']);cfg = read(entry / 'config.json')
    row = next(r for r in read(entry / 'inference_manifest.json')['reports'] if r['domain'] == job['domain'] and r['epoch'] == job['epoch'])
    assert sha(row['prediction']) == row['prediction_sha256']
    annotation = cfg['baseline']['annotation_for_offline_evaluation']
    assert sha(annotation) == cfg['baseline']['annotation_sha256']
    splits = cfg['src_val_split'] if job['domain'] == 'source' else cfg['tgt_val_split']
    database = read(annotation)['database']
    gt = {v: [a for a in remove_duplicate_annotations(d.get('annotations', [])) if a['label_id'] < 10]
          for v, d in database.items() if d['subset'].lower() in {s.lower() for s in splits}}
    with np.load(row['prediction'], allow_pickle=False) as stream:
        pred = {k: stream[k] for k in stream.files}
    assert set(pred['video-id']) <= set(gt)
    reports = []
    for score in (.1, .3, .5):
        for tiou in (.1, .3, .5):
            reports.append(dict(rule='score', value=score, tiou=tiou, counts={}, GT_count=sum(map(len, gt.values()))))
    for k in (50, 100, 200):
        for tiou in (.1, .3, .5):
            reports.append(dict(rule='topK_per_video', value=k, tiou=tiou, counts={}, GT_count=sum(map(len, gt.values()))))
    for video, annotations in gt.items():
        ids = np.flatnonzero(pred['video-id'] == video)
        boxes = np.stack([pred['t-start'][ids], pred['t-end'][ids]], -1)
        truth = np.asarray([a['segment'] for a in annotations], dtype=float).reshape(-1, 2)
        labels = np.asarray([a['label_id'] for a in annotations])
        matrix = iou_matrix(boxes, truth)
        order = np.argsort(-pred['score'][ids], kind='stable')
        for report in reports:
            selected = np.flatnonzero(pred['score'][ids] >= report['value']) if report['rule'] == 'score' else order[:report['value']]
            counts = error_partition(matrix[selected], pred['label'][ids][selected], labels,
                                     pred['score'][ids][selected], report['tiou'])
            for name, count in counts.items():
                report['counts'][name] = report['counts'].get(name, 0) + count
    for r in reports:
        n = sum(r['counts'].values());tp = r['counts'].get('tp', 0)
        r.update(prediction_count=n, precision=tp/n if n else None,
                 recall=tp/r['GT_count'] if r['GT_count'] else None,
                 background_fraction=r['counts'].get('background', 0)/n if n else None)
    evaluator = ANETdetection(annotation, splits, num_workers=1, action_subsampling='10_majority_classes')
    ap, mean, class_ap, *_ = evaluator.evaluate(pred, verbose=False)
    saved = read(entry / ('evaluation/%s_e%03d.json' % (job['domain'], job['epoch'])))
    assert abs(mean - saved['map']) < 1e-7
    result = dict(job, prediction_sha256=row['prediction_sha256'], annotation_sha256=cfg['baseline']['annotation_sha256'],
                  map=float(mean), map_by_tiou=ap.tolist(), class_AP={str(evaluator.idx_to_activity_index[i]):float(x) for i,x in enumerate(class_ap)},
                  diagnostic_grid=reports, target_GT_CPU_only=job['domain']=='target',
                  caveat='Counts depend on fixed rule; they are not additive AP causes. Target analysis is exploratory and never checkpoint selection.')
    return result


def model_drift(original, shift):
    import torch
    e = original / 'formal/control' / shift
    initial = torch.load(e / 'epoch000.pt', map_location='cpu')
    reference = {k:v for k,v in initial['ema_state_dict'].items() if torch.is_floating_point(v)}
    lr0 = [[g['lr'] for g in opt['param_groups']] for opt in initial['optimizers']]
    del initial
    reports = {}
    for mode in ('control', 'gate'):
        checkpoint = torch.load(original / 'formal' / mode / shift / 'epoch020.pt', map_location='cpu')
        groups = {}
        for group in ('backbone_source', 'neck_source', 'cls_head', 'reg_head'):
            numerator = denominator = 0.;count = 0
            for name, old in reference.items():
                if group not in name:
                    continue
                current = checkpoint['ema_state_dict'][name]
                numerator += float((current.double() - old.double()).square().sum())
                denominator += float(old.double().square().sum());count += old.numel()
            groups[group] = dict(relative_L2=(numerator/max(denominator,1e-30))**.5, elements=count)
        reports[mode] = dict(EMA_drift=groups, initial_optimizer_LRs=lr0,
                            epoch20_optimizer_LRs=[[g['lr'] for g in opt['param_groups']] for opt in checkpoint['optimizers']],
                            original_schedule_not_compressed=True)
        del checkpoint
    return reports


def main(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    a.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for shift in ('S1', 'S3'):
        for mode, epoch in (('control',0),('control',20),('gate',20)):
            jobs.append(dict(shift=shift, mode=mode, epoch=epoch, domain='target', entry=str(a.preview/'formal'/mode/shift)))
    save(a.output/'plan.json',dict(jobs=jobs, source=str(a.preview), new_training=False,
         scope='completed two predeclared pairs; target-informed CPU exploratory error analysis, no model/epoch choice'))
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(analyze_prediction, jobs):
            name='%s_%s_e%03d'%(result['shift'],result['mode'],result['epoch'])
            save(a.output/(name+'.json'),result);results.append(result)
            print('DIAGNOSTIC_COMPLETE',name,result['map'],flush=True)
    drift={s:model_drift(a.original,s) for s in ('S1','S3')}
    source={}
    for path in a.candidate.glob('formal/*/phase/source_candidate_history.json'):
        history=read(path);assert [x['epoch'] for x in history]==list(range(len(history)))
        best=min(history,key=lambda x:(x['source_candidate_score'],x['epoch']))
        q=best['quality']
        source[path.parts[-3]]=dict(completed_epoch=history[-1]['epoch'],best_so_far=best,
            refinement_gain=q['raw_error']-q['selected_error'],history_sha256=sha(path),
            selected_class_NLL_drop=history[0]['quality']['class_nll']-q['class_nll'],
            warning='Source head still training if epoch<15; provisional source-only mechanism evidence, not detection AP.')
    save(a.output/'final.json',dict(count=len(results),results=results,model_drift=drift,source_head=source,
         target_informed_development=True,formal_UDA_result=False,new_training=False,group_mean_reported=False))
    print('ANALYSIS_COMPLETE',str(a.output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--preview',type=Path,required=True);p.add_argument('--original',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
