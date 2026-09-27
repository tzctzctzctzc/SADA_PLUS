"""隔离验收真实train_da/Trainer一步更新；不产正式checkpoint或实验成绩。"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import runpy
import sys

from tools.score_rq3_vlm_candidates import check_code, idle_gpu_uuid
from tools.rq3_protocol import audit_baseline
from tools.run_rq3_matrix import completed_baseline


def crop_with_candidate(dataset, video, coordinate):
    index = next(i for i, row in enumerate(dataset.data_list) if row['id'] == video)
    for _ in range(200):
        item = dataset[index]
        start = item['metadata']['crop_start']
        if start <= coordinate < start + item['feats'].shape[-1] - 4:
            if item['labels'].numel() or item['segments'].numel():
                raise ValueError('smoke target batch contains GT')
            return item
    raise ValueError('no ordinary target crop contains the audited candidate')


def source_with_class(dataset, label):
    for index, row in enumerate(dataset.data_list):
        if row['labels'] is None or label not in row['labels']:
            continue
        for _ in range(20):
            item = dataset[index]
            if label in item['labels'].tolist():
                return item
    raise ValueError('source-only smoke could not find its class')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('smoke report exists')
    training_args = args.training_args
    if training_args[:1] == ['--']:
        training_args = training_args[1:]
    if '--rq3-strategy' not in training_args or '--rq2-audit-output' in training_args:
        raise ValueError('smoke requires the real RQ3 training entry')
    check_code(args.expected_head)
    complete = json.loads((args.baseline_root / 'complete.json').read_text())
    baseline, checkpoint = completed_baseline(args.baseline_root, complete['shift'])
    import fcntl
    with Path('/vepfs/vlms/SADA_PLUS/.rq2_gpu3.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES'] = idle_gpu_uuid(3)
        import torch
        import libs.trainer.da_trainer as trainer_module
        original_train = trainer_module.Trainer.train
        original_length = trainer_module.epoch_length_from_config

        def single_step(trainer):
            cfg = trainer.cfg.config
            audit_baseline(baseline, cfg)
            if str(trainer.model.device) != 'cuda:0':
                raise ValueError('smoke must use only the mapped physical GPU3')
            # 只为验收加载已核验SO EMA；正式54项仍fresh训练，禁止复用此状态。
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)['ema_state_dict']
            loaded = trainer.model.load_state_dict(state, strict=False)
            if loaded.unexpected_keys or any(not key.startswith('rq3_auxiliary.') for key in loaded.missing_keys):
                raise ValueError('smoke SO initialization does not match the real student')
            del state
            trainer.model_ema.set(trainer.model)
            auxiliary = trainer.model.rq3_auxiliary
            video = next(iter(auxiliary.cache))
            entry = auxiliary.cache[video][0]
            source = source_with_class(trainer.train_loader_src.dataset, entry['class_id'])
            target = crop_with_candidate(trainer.train_loader_tgt.dataset, video, entry['absolute_feature_index'])
            # 同一真实样本复制到batch2，仅用于执行路径验收；不记为正式数据采样。
            trainer.train_loader_src = [[copy.deepcopy(source), copy.deepcopy(source)]]
            trainer.train_loader_tgt = [[copy.deepcopy(target), copy.deepcopy(target)]]
            trainer_module.epoch_length_from_config = lambda *unused: 1
            epoch = max(1, trainer.model.warmup_adv)
            before = {name: p.detach().clone() for name, p in trainer.model.named_parameters() if p.requires_grad}
            trainer._train_epoch(epoch)
            metrics = {key: float(value) for key, value in trainer.train_metrics.result().items()}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError('nonfinite full-trainer smoke metrics')
            for key in ('t_target_gt_cls_loss', 't_target_gt_reg_loss', 't_target_gt_downstream_loss'):
                if metrics[key] != 0:
                    raise ValueError('target GT entered the smoke objective')
            changed = [name for name, p in trainer.model.named_parameters()
                       if name in before and not torch.equal(p, before[name])]
            if not changed or metrics['t_rq3_eligible'] <= 0:
                raise ValueError('real trainer did not update or candidate did not reach the model')
            effective = metrics['t_rq3_selected'] > 0 and metrics['t_rq3_loss'] > 0 and metrics['t_rq3_gradient_norm'] > 0
            if auxiliary.strategy != 'verify' and not effective:
                raise ValueError('hard/prototype auxiliary gradient did not activate')
            report = {'schema': 'rq3-real-trainer-smoke-v1', 'formal_result': False,
                      'strategy': auxiliary.strategy, 'code_commit': args.expected_head,
                      'initialization': str(checkpoint), 'initialization_kind': 'audited-so-ema-smoke-only',
                      'source_video': source['metadata']['video_id'], 'target_video': video,
                      'target_crop': target['metadata'], 'epoch_for_branch_test': epoch,
                      'steps': 1, 'batch_size': 2, 'gpu': 3, 'metrics': metrics,
                      'updated_parameter_count': len(changed), 'auxiliary_effective': effective,
                      'status': 'passed' if effective else 'verify-rejected-all-real-candidates'}
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open('x') as stream:
                json.dump(report, stream, sort_keys=True)
            print('RQ3_REAL_TRAINER_SMOKE', json.dumps(report, sort_keys=True))

        trainer_module.Trainer.train = single_step
        original_argv = sys.argv
        try:
            sys.argv = ['train_da.py', *training_args]
            runpy.run_path(str(Path(__file__).resolve().parents[1] / 'train_da.py'), run_name='__main__')
        finally:
            sys.argv = original_argv
            trainer_module.Trainer.train = original_train
            trainer_module.epoch_length_from_config = original_length


if __name__ == '__main__':
    main()
