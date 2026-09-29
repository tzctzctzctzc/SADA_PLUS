"""冻结旧完整结果身份，生成仅Verify/Prototype的fresh重跑清单；不使用mAP筛选。"""
import argparse
import json
from pathlib import Path

from tools.rq2_dense_protocol import sha256_file
from tools.run_rq3_matrix import completed_entries, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, action='append', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    completed = completed_entries(args.result_root)
    snapshots, reruns = [], []
    for identity in matrix():
        if identity not in completed:
            continue
        path = completed[identity]
        record = json.loads(path.read_text())
        hashes = {}
        for relative in ('complete.json', 'preflight.json', 'baseline_pair.json', 'supervision_audit.json',
                         'evaluation/manifest.json', 'evaluation/source_val.json', 'evaluation/target_val.json'):
            artifact = path.parent / relative
            hashes[str(artifact)] = sha256_file(artifact)
        snapshots.append(dict(identity=identity, record=record, artifact_sha256=hashes))
        if identity[1] != 'hard':
            reruns.append(dict(teacher=identity[0], strategy=identity[1], shift=identity[2],
                               original_complete=str(path), original_complete_sha256=sha256_file(path),
                               artifact_sha256=hashes))
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / 'old_results_snapshot.json').write_text(json.dumps(snapshots, indent=2, sort_keys=True))
    plan = dict(schema='rq3-revised-completed-only-v1', seed=6, fresh=True,
                excludes='Hard and unfinished/unevaluated old runs', entries=reruns)
    (args.output_root / 'rerun_plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True))
    print(json.dumps(dict(frozen_complete=len(snapshots), revised_reruns=len(reruns),
                          plan_sha256=sha256_file(args.output_root / 'rerun_plan.json'))))


if __name__ == '__main__':
    main()
