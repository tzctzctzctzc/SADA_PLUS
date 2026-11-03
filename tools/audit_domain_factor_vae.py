"""单CPU等待各任务完成，独立核协议/状态和完整预测NumPy AP，不接触训练进程。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np
import pandas as pd
import torch
from tools.baseline_protocol import read, save, sha, termination, decision, code_identity
from tools.run_domain_factor_vae import TASKS, pair_check
from tools.audit_background_ap_cost import IndependentAP, overlaps


def truth(annotation, splits):
    """复核原GT去重约定，保留原label 0..9与原秒单位，不重编号。"""
    database = read(annotation)['database']
    members, rows = set(), []
    for video, item in database.items():
        if item['subset'].lower() not in {s.lower() for s in splits}:
            continue
        members.add(video)
        accepted = []
        for event in item['annotations']:
            start, end = map(float, event['segment'])
            label = int(event['label_id'])
            if any(old[2] == label and abs(start-old[0]) <= 1e-3 and abs(end-old[1]) <= 1e-3
                   for old in accepted):
                continue
            accepted.append((start, end, label))
            if 0 <= label < 10 and end != start:
                rows.append((video, start, end, label))
    return members, pd.DataFrame(rows, columns=['video-id', 't-start', 't-end', 'label'])


def ap_and_background(gt, pred):
    aps = []
    # 原ANET定义是GT实际出现类别的均值；完整域另确认10类均出现。
    classes = sorted(gt['label'].unique())
    for label in classes:
        g = gt[gt['label'] == label].reset_index(drop=True)
        p = pred[pred['label'] == label].reset_index(drop=True)
        aps.append(IndependentAP(g, p).ap(np.ones(len(p), dtype=bool)))
    ap = np.asarray(aps).mean(0)
    high = pred[pred['score'] >= .3].reset_index(drop=True)
    bg = np.ones(len(high), dtype=bool)
    for video, ids in high.groupby('video-id', sort=False).indices.items():
        g = gt[gt['video-id'] == video]
        if len(g):
            matrix = overlaps(high.iloc[ids][['t-start', 't-end']].to_numpy(float),
                              g[['t-start', 't-end']].to_numpy(float))
            bg[ids] = matrix.max(1) < .01
    counts = dict(total=int(len(high)), pure_task_background=int(bg.sum()), nonBG=int((~bg).sum()),
                  definition='any-task-class tIoU<.01; score>=.3',
                  caveat='new detector outputs are not the same candidate pool as historical baseline')
    return ap, counts, classes


def audit(root, task):
    entry = root/task
    cfg, binding = read(entry/'config.json'), read(entry/'origin_binding.json')
    complete = read(entry/'complete.json')
    for path, digest in complete['artifacts'].items():
        assert sha(entry/path) == digest, (task, path)
    for path, digest in binding['inputs'].items():
        assert sha(path) == digest, path
    source = read(cfg['dataset']['json_file'])['database']
    target = read(cfg['baseline']['target_manifest'])['database']
    assert not set(source) & set(target)
    assert all('annotations' not in item for item in target.values())
    assert cfg['model']['adv_method'] == ['None'] and cfg['model']['loss_weights']['weight_adv'] == 0.
    assert cfg['seed'] == cfg['init_rand_seed'] == 123 and cfg['loader']['batch_size'] == 2
    state = read(root/'state.json')
    guards = {}
    for stage in ('train', 'evaluate'):
        proof = read(entry/(stage+'_complete.json'))
        assert sha(proof['guard']) == proof['guard_sha256']
        guard = read(proof['guard'])
        assert sha(guard['lease']) == guard['lease_sha256']
        lease = read(guard['lease'])
        expected_owner = state['owner_pid']
        if stage == 'evaluate' and (entry/'evaluate_recovery_launch.json').exists():
            recovery = read(entry/'evaluate_recovery_launch.json')
            assert recovery['original_training_owner'] == state['owner_pid']
            assert recovery['source_commit'] == cfg['baseline']['code']['commit']
            assert not recovery['training_repeated']
            for name, digest in recovery['before'].items():
                assert sha(entry/name) == digest, name
            assert guard['child_pid'] == recovery['child_pid']
            assert guard['lease'] == recovery['lease'] and sha(guard['lease']) == recovery['lease_sha256']
            expected_owner = recovery['owner_pid']
            recovered = read(entry/'evaluate_recovery_complete.json')
            assert recovered['passed'] and recovered['receipt_sha256'] == sha(entry/'evaluate_recovery_launch.json')
            assert complete['recovery_sha256'] == sha(entry/'evaluate_recovery_complete.json')
        assert guard['owner_pid'] == lease['owner_pid'] == expected_owner
        assert lease['commit'] == cfg['baseline']['code']['commit']
        assert guard['observed']['uuid'] == lease['observed']['uuid']
        assert guard['observed']['free_mib'] >= lease['minimum_free_mib'] == 16384
        guards[stage] = sha(proof['guard'])
    probe = read(entry/'probe.json')
    assert probe['passed'] and probe['complete_next_step_replay_passed']
    assert sha(entry/'gpu_probe_state.pt') == probe['probe_checkpoint_sha256']
    initial = read(entry/'formal_initialization.json')
    assert initial['passed'] and not initial['loaded_trained_checkpoint']
    assert initial['sha256'] == probe['initial_state_sha']
    term = read(entry/'termination.json')
    selected = termination(term['history'])
    assert term['initial_state_sha'] == initial['sha256']
    rows = [json.loads(line) for line in (entry/'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows] == list(range(1, selected['epoch']+1))
    assert [[r['epoch'], r['source_validation_loss']] for r in rows if 'source_validation_loss' in r] == term['history']
    assert all(r['target_gt_cls_loss'] == r['target_gt_reg_loss'] == r['target_gt_downstream_loss'] == 0 for r in rows)
    for row in rows:
        assert row['domain_factor']['factor_source_fg'] > 0
        assert np.isfinite(list(row['domain_factor'].values())).all()
    for name, epoch in (('best.pt', selected['best_epoch']), ('latest.pt', selected['epoch'])):
        saved = torch.load(entry/name, map_location='cpu')
        assert saved['epoch'] == epoch and saved['config'] == cfg
        assert len(saved['optimizers']) == len(saved['schedulers']) == 3
        assert set(saved['rng']) == {'python', 'numpy', 'cpu', 'cuda', 'loaders'}
        assert len(saved['rng']['cuda']) == 1
        assert 'domain_factor.sampling_state' in saved['state_dict']
        assert 'domain_factor.sampling_state' in saved['ema_state_dict']
        assert saved['state_dict']['domain_factor.action_mean.weight'].shape[0] == 128
        assert saved['state_dict']['domain_factor.domain_mean.weight'].shape[0] == 32
        assert all(saved['optimizers'][i]['state'] for i in (0, 1))
        del saved
    results = {}
    for domain, splits in (('source', cfg['src_val_split']), ('target', cfg['tgt_val_split'])):
        report = read(entry/'evaluation'/(domain+'_val.json'))
        assert report['checkpoint_sha256'] == sha(entry/'best.pt')
        assert report['selected_epoch'] == selected['best_epoch']
        columns = torch.load(entry/'evaluation'/(domain+'_predictions.pt'), map_location='cpu')
        pred = pd.DataFrame(columns)
        members, gt = truth(cfg['baseline']['annotation_for_offline_evaluation'], splits)
        assert set(pred['video-id']) <= members
        assert np.isfinite(pred[['t-start', 't-end', 'score']].to_numpy(float)).all()
        assert len(pred) == report['prediction_count']
        ap, counts, classes = ap_and_background(gt, pred)
        assert classes == list(range(10)), ('unexpected missing GT classes', classes)
        assert np.max(np.abs(ap-np.asarray(report['map_by_tiou']))) < 1e-7
        assert abs(ap.mean()-report['map_0.1_to_0.5']) < 1e-7
        results[domain] = dict(map_by_tiou=ap.tolist(), map=float(ap.mean()), highscore_pool=counts,
            raw_prediction_sha256=sha(entry/'evaluation'/(domain+'_predictions.pt')))
    save(root/'independent_results'/(task+'.json'), dict(passed=True, task=task, results=results,
        selected=selected, complete_sha256=sha(entry/'complete.json'), guard_sha256=guards,
        delta_vs_historical_SADA_Prototype_pp=100*(results['target']['map']-binding['historical_target_map']),
        shared_initialization='producer real-GPU hash assertion; not independently reconstructed on CPU',
        independent_AP=True, target_GT_training=False))
    print('FACTOR_AUDITED', task, results['target']['map'], flush=True)


def run(root):
    audited = set()
    while True:
        if (root/'stop.json').exists():
            save(root/'independent_audit_stopped.json', dict(reason='user or scheduler stop; no restart'))
            return
        for task in TASKS:
            if task not in audited and (root/task/'complete.json').exists():
                audit(root, task)
                audited.add(task)
        save(root/'independent_audit_progress.json', dict(audited=sorted(audited), owner_pid=os.getpid()))
        if (root/'final.json').exists():
            final = read(root/'final.json')
            queue=root/'evaluation_repair_queue.json'
            if queue.exists() and read(queue)['status'] != 'complete':
                time.sleep(5)
                continue
            if any((root/task/'evaluate_recovery_launch.json').exists()
                   and not (root/task/'evaluate_recovery_complete.json').exists()
                   and not (root/task/'evaluate_recovery_error.json').exists()
                   for task in final['failed']):
                time.sleep(5)
                continue
            recovered = {task for task in final['failed']
                         if (root/task/'evaluate_recovery_complete.json').exists()}
            if any(task not in audited for task in recovered):
                time.sleep(5)
                continue
            assert set(final['completed']) | recovered == audited
            pairs = {}
            for shift in ('S1', 'S2'):
                ae, vae = shift+'_ae', shift+'_vae'
                pair_check(read(root/ae/'config.json'), read(root/vae/'config.json'))
                if ae in audited and vae in audited:
                    assert read(root/ae/'probe.json')['common_branch_initial_sha'] == read(root/vae/'probe.json')['common_branch_initial_sha']
                    a, v = [read(root/'independent_results'/(x+'.json')) for x in (ae, vae)]
                    pairs[shift] = dict(vae_minus_ae_pp=100*(v['results']['target']['map']-a['results']['target']['map']))
            save(root/'independent_final_audit.json', dict(passed=len(audited) == 4, code=code_identity(),
                producer_final_sha256=sha(root/'final.json'), results={task:sha(root/'independent_results'/(task+'.json')) for task in sorted(audited)},
                pairs=pairs, failed=final['failed'], recovered_evaluations=sorted(recovered),
                scope='S1/S2 development, not six-direction final mean',
                selection='original source validation only; all four results retained'))
            return
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    run(args.root)
