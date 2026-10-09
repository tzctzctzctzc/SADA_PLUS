"""只写新增双教师36项的固定计划/模型身份；本工具不启动GPU。"""
import argparse
import json
from pathlib import Path

from tools.rq3_extra_teachers import EXTRA, frozen_protocol, model_path
from tools.run_rq3_matrix import load_entry_plan
from tools.run_rq3_trusted_queue import RUNTIME
from tools.score_rq3_vlm_candidates import check_code


def write_identity(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('existing teacher identity differs: ' + str(path))
    else:
        with path.open('x') as stream:
            json.dump(value, stream, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    args = parser.parse_args()
    check_code(args.expected_head)
    args.root.mkdir(parents=True, exist_ok=False)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for model in EXTRA:
        write_identity(RUNTIME / (model + '_anchor.json'), frozen_protocol(model))
        write_identity(RUNTIME / (model + '_settings.json'), dict(model_path=str(model_path(model))))
        rows = [dict(teacher=model, strategy=strategy, shift='S'+str(shift))
                for shift in range(1, 7) for strategy in ('hard', 'verify', 'prototype')]
        plan = dict(schema='rq3-expanded-fresh-matrix-v1', fresh=True, seed=6, entries=rows,
                    batch_size=2, early_stop_patience_epochs=15, max_epochs=50,
                    hard_policy='original-all-common-candidates-source-loss-cap-0.05',
                    verify_prototype_participation=.2)
        path = args.root / (model + '.json')
        write_identity(path, plan)
        assert len(load_entry_plan(path)) == 18
        print('RQ3_EXTRA_PLAN_READY', model, path, flush=True)


if __name__ == '__main__':
    main()
