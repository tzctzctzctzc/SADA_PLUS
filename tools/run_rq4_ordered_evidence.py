"""E2有限两教师CPU队列；完整复用E1特征，不占用CUDA。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import numpy as np
from tools.rq4_protocol import file_sha
from tools.rq4_source_acceptance import save
from tools.run_rq4_teacher_queries import clean_head
from tools.run_rq4_local_evidence import read
from tools import rq4_local_evidence as e1
from tools import rq4_ordered_evidence as e2

CODE = Path(__file__).resolve().parents[1]
TEACHERS = ['timelens2_8b', 'timeple_8b']


def load_rows(source, teacher):
    complete = read(source/'complete.json')
    assert complete['status'] == 'complete' and complete['scope'] == 'E1-source-crossfit-only'
    for info in complete['results'].values(): assert file_sha(info['path']) == info['sha256']
    m = read(source/'manifest.json'); prepared = read(source/'prepared.json')
    assert file_sha(source/'manifest.json') == prepared['manifest_sha256']
    assert file_sha(source/'source_labels.json') == prepared['labels_sha256']
    proof_path = source/teacher/'features_complete.json'; proof = read(proof_path)
    result = read(source/teacher/'result.json')
    assert file_sha(proof_path) == result['feature_proof_sha256']
    assert proof['manifest_sha256'] == file_sha(source/'manifest.json')
    assert proof['identity_sha256'] == file_sha(source/teacher/'identity.json')
    records = {r['id']: r for r in proof['records']}
    assert len(records) == len(proof['records']) == len(m['rows'])
    assert set(records) == {r['id'] for r in m['rows']}
    truth = read(source/'source_labels.json'); rows = []; sf_cache = {}
    for row in m['rows']:
        path = source/teacher/(row['id']+'.npz')
        assert file_sha(path) == records[row['id']]['sha256']
        with np.load(path) as data: visual, times = data['features'], data['times']
        sf_path = row['slowfast_path']
        if sf_path not in sf_cache:
            assert file_sha(sf_path) == row['slowfast_sha256']
            with np.load(sf_path) as data: sf_cache[sf_path] = data['feats'].astype('float32')
        rows.append(dict(row, vlm=visual, times=times,
            slowfast=e1.slowfast_at(sf_cache[sf_path], times, row['dataset']),
            labels=e1.labels(times, truth[row['candidate']['video_id']], row['candidate']['class_id'])))
    assert m['folds'] == e1.folds([r['candidate']['video_id'] for r in rows])
    return rows, m


def evaluate(args):
    clean_head(args.expected_head)
    frozen = read(args.root/'protocol.json')
    assert frozen['policy'] == e2.POLICY and frozen['code_commit'] == args.expected_head
    assert frozen['source_complete_sha256'] == file_sha(args.source/'complete.json')
    out = args.root/args.teacher; out.mkdir(exist_ok=False)
    rows, manifest = load_rows(args.source, args.teacher)
    x, y, videos, identities = e2.design(rows)
    # 第三轮真实全池审计：静态路径逐元素复现E1，时间点/标签/分组完全相同。
    old_x, old_y, old_videos, old_ids = e1.design(rows)
    assert np.array_equal(y, old_y) and np.array_equal(videos, old_videos) and identities == old_ids
    for a, b in [('geometry','geometry'), ('static_sf','slowfast'), ('static_fusion','fusion')]:
        assert np.array_equal(x[a], old_x[b])
    assert all(v.shape[1] == 142 for k, v in x.items() if k != 'geometry')
    assert all(np.isfinite(v).all() for v in x.values())
    save(out/'input_audit.json', dict(status='passed', candidates=len(rows), videos=len(set(videos)),
        timepoints=len(y), exact_static_design=True, code_commit=args.expected_head))
    del old_x, rows
    predictions = {}; diagnostics = {}
    for name in e2.CONDITIONS:
        print('E2_CROSSFIT', args.teacher, name, flush=True)
        predictions[name], diagnostics[name] = e1.crossfit(x[name], y, videos, manifest['folds'])
    np.savez_compressed(out/'out_of_fold.npz', labels=y, videos=videos, **predictions)
    save(out/'prediction_index.json', identities)
    print('E2_BOOTSTRAP', args.teacher, flush=True)
    result = e2.summarize(y, predictions, videos)
    save(out/'result.json', dict(result, policy=e2.POLICY, code_commit=args.expected_head,
        diagnostics=diagnostics, teacher=args.teacher,
        protocol_sha256=file_sha(args.root/'protocol.json'),
        feature_proof_sha256=file_sha(args.source/args.teacher/'features_complete.json'),
        predictions_sha256=file_sha(out/'out_of_fold.npz'), index_sha256=file_sha(out/'prediction_index.json')))
    print('E2_COMPLETE', args.teacher, result['positive_signal'], flush=True)


def driver(args):
    clean_head(args.expected_head); args.root.mkdir(exist_ok=False)
    save(args.root/'protocol.json', dict(policy=e2.POLICY, source=str(args.source),
        source_complete_sha256=file_sha(args.source/'complete.json'), code_commit=args.expected_head))
    children = []
    def stop(*_): raise RuntimeError('explicit stop')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        for teacher in TEACHERS:
            with (args.root/(teacher+'.log')).open('x') as log:
                cmd = [sys.executable, '-u', '-m', 'tools.run_rq4_ordered_evidence', 'evaluate',
                    '--source', str(args.source), '--root', str(args.root),
                    '--expected-head', args.expected_head, '--teacher', teacher]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
                children.append(subprocess.Popen(cmd, cwd=CODE, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True))
                print('E2_CHILD', teacher, children[-1].pid, flush=True)
        for child in children:
            if child.wait(): raise RuntimeError('E2 child failed; preserve outputs')
        results = {t: dict(path=str(args.root/t/'result.json'),
            sha256=file_sha(args.root/t/'result.json')) for t in TEACHERS}
        save(args.root/'complete.json', dict(status='complete', scope='E2-source-crossfit-only', results=results))
    except BaseException:
        for child in children:
            if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        for child in children: child.wait()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['driver','evaluate'])
    p.add_argument('--source', type=Path, required=True); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--expected-head', required=True); p.add_argument('--teacher', choices=TEACHERS)
    args = p.parse_args(); args.source = args.source.resolve(); args.root = args.root.resolve()
    globals()[args.mode](args)


if __name__ == '__main__': main()
