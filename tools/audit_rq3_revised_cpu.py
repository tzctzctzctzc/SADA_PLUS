"""真实Trainer的CPU两步反事实验收；不占GPU、不产正式checkpoint。"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', choices=('verify', 'prototype'), required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--candidate-manifest', type=Path, required=True)
    parser.add_argument('--confidence-calibration', type=Path, required=True)
    parser.add_argument('--teacher', choices=('egovideo_hod', 'internvideo2_clip_1b', 'qwen3_vl_8b'), default='egovideo_hod')
    args = parser.parse_args()
    args.report = args.report.resolve()
    args.cache = args.cache.resolve()
    args.candidate_manifest = args.candidate_manifest.resolve()
    args.confidence_calibration = args.confidence_calibration.resolve()
    if args.report.exists():
        raise FileExistsError(args.report)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    import libs.trainer.da_trainer as tm
    from libs.models.SADA.optimizer import make_optimizer, make_scheduler
    from tools.smoke_rq3_training import crop_with_candidate, source_with_class
    from tools.run_rq3_matrix import completed_baseline
    from tools.rq3_protocol import audit_baseline, load_cache, proposal_evidence
    # 反事实验收要求逐位复现，避免CPU并行归约的浮点顺序差异。
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.autograd.set_multithreading_enabled(False)
    torch.backends.mkldnn.enabled = False
    root = Path(__file__).resolve().parents[1]
    os.chdir('/vepfs/vlms/SADA_PLUS')

    def audit(trainer):
        assert str(trainer.model.device) == 'cpu'
        baseline, checkpoint = completed_baseline(
            Path('/vepfs/vlms/rq2_matched_slowfast_S1_seed6_7067306_0927'), 'S1')
        audit_baseline(baseline, trainer.cfg.config)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['ema_state_dict']
        loaded = trainer.model.load_state_dict(state, strict=False)
        assert not loaded.unexpected_keys
        assert all(k.startswith('rq3_prototype.') for k in loaded.missing_keys)
        del state
        trainer.model_ema.set(trainer.model)
        cfg = trainer.cfg['rq3_auxiliary']
        payload, _ = load_cache(cfg['cache_path'])
        evidence = proposal_evidence(payload, cfg['manifest_path'])
        from libs.models.SADA.rq3_revised import confident, feature_interval
        fixture = None
        candidates = [(video, entry) for video, entries in evidence.items() for entry in entries]
        candidates.sort(key=lambda pair: not confident(pair[1], cfg['confidence_threshold']))
        for video, entry in candidates:
            for attempt in range(20):
                target = crop_with_candidate(trainer.train_loader_tgt.dataset, video, entry['absolute_feature_index'])
                start, end = feature_interval(entry, target['metadata'])
                crop_start = target['metadata']['crop_start']
                matched_level = any((entry['absolute_feature_index'] - crop_start) % trainer.model.fpn_strides[level] == 0
                                    for level in entry['source_fpn_levels'])
                if matched_level and crop_start <= start < end <= crop_start + target['feats'].shape[-1]:
                    fixture = (video, entry, target)
                    break
            if fixture is not None:
                break
        if fixture is None:
            raise ValueError('no complete-window CPU fixture; do not modify evidence to force activation')
        video, entry, target = fixture
        source = source_with_class(trainer.train_loader_src.dataset, entry['class_id'])
        trainer.train_loader_src = [[copy.deepcopy(source), copy.deepcopy(source)]]
        trainer.train_loader_tgt = [[copy.deepcopy(target), copy.deepcopy(target)]]
        scheduler_length = trainer.cfg['rq2_formal']['epoch_budget']['iterations_per_epoch']
        tm.epoch_length_from_config = lambda *unused: 2
        initial = copy.deepcopy((trainer.model, trainer.model_ema))
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        outputs, baseline_state, repeat_state = {}, None, None
        # FP32两步优化的数值验收界限；保留全部逐位差异，不把末位舍入计为策略效应。
        parameter_atol = 1e-8
        for variant in ('baseline', 'baseline_repeat', args.strategy):
            trainer.model, trainer.model_ema = copy.deepcopy(initial)
            trainer.train_loader_src = [[copy.deepcopy(source), copy.deepcopy(source)]]
            trainer.train_loader_tgt = [[copy.deepcopy(target), copy.deepcopy(target)]]
            opt_cfg = trainer.cfg['opt']
            exclude = ['disc_dom_adv', 'disc_dom_conditional_adv', 'learnable_embeddings']
            for oname, sname, discriminator in (
                ('optimizer_gen_source', 'lr_scheduler_gen_source', False),
                ('optimizer_gen_source_alignment', 'lr_scheduler_gen_source_alignment', False),
                ('optimizer_disc', 'lr_scheduler_disc', True)):
                kwargs = {'include_params': exclude} if discriminator else {'exclude_params': exclude}
                optimizer = make_optimizer(trainer.model, opt_cfg,
                                           opt_cfg['disc_learning_rate' if discriminator else 'learning_rate'], **kwargs)
                setattr(trainer, oname, optimizer)
                setattr(trainer, sname, make_scheduler(optimizer, opt_cfg, scheduler_length))
            if variant.startswith('baseline'):
                trainer.model.rq3_prototype = None
                trainer.model.rq3_verify_cache = None
                trainer.model_ema.module.rq3_prototype = None
                trainer.model_ema.module.rq3_verify_cache = None
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            trainer.train_metrics.reset()
            # 仅CPU测试兼容原Trainer的原地scalar +=；数值/梯度与GPU非叶to一致。
            real_tensor = torch.tensor
            def nonleaf_scalar(*values, **kwargs):
                value = real_tensor(*values, **kwargs)
                return value.clone() if kwargs.get('requires_grad') and value.ndim == 0 else value
            torch.tensor = nonleaf_scalar
            try:
                trainer._train_epoch(max(1, trainer.model.warmup_adv))
            finally:
                torch.tensor = real_tensor
            metrics = {k: float(v) for k, v in trainer.train_metrics.result().items()}
            assert all(np.isfinite(v) for v in metrics.values())
            assert all(metrics[k] == 0 for k in (
                't_target_gt_cls_loss', 't_target_gt_reg_loss', 't_target_gt_downstream_loss'))
            current = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
            if baseline_state is None:
                baseline_state = current
            changed = {n: float((p - baseline_state[n]).abs().max()) for n, p in current.items()
                       if n in baseline_state and not torch.equal(p, baseline_state[n])}
            outputs[variant] = {'metrics': metrics, 'changed_base_parameters': changed}
            if variant == 'baseline_repeat':
                repeat_state = current
            elif variant == args.strategy:
                outputs[variant]['changed_repeat_parameters'] = {
                    n: float((p - repeat_state[n]).abs().max()) for n, p in current.items()
                    if n in repeat_state and not torch.equal(p, repeat_state[n])}
            print('RQ3_REVISED_CPU_VARIANT', variant, len(changed),
                  json.dumps(sorted(changed.items(), key=lambda pair: -pair[1])[:5]), flush=True)
        assert max(outputs['baseline_repeat']['changed_base_parameters'].values(), default=0) <= parameter_atol, 'CPU baseline exceeds FP32 reproducibility tolerance'
        tested = outputs[args.strategy]
        meaningful_change = max(tested['changed_base_parameters'].values(), default=0) > parameter_atol
        if args.strategy == 'verify':
            assert tested['metrics']['t_rq3_verify_observed'] > 0, 'fixture did not exercise actual SADA foreground verification'
            intervened = tested['metrics']['t_rq3_verify_rejected'] > 0
            assert meaningful_change == intervened, 'Verify intervention/student update mismatch'
            if not intervened:
                # 第二次独立基线是直接控制；不干预Verify仍要求与其逐位一致。
                assert not tested['changed_repeat_parameters'], 'no-op Verify differs from repeated baseline'
        else:
            assert meaningful_change, 'source prototype branch did not update original student beyond roundoff'
            intervened = tested['metrics']['t_rq3_selected'] > 0
            assert (tested['metrics']['t_rq3_target_gradient_norm'] > 0) == intervened
            assert (tested['metrics']['t_rq3_alignment_loss'] > 0) == intervened
        report = dict(status='passed', formal_result=False, device='cpu', strategy=args.strategy,
                      vlm_intervened=intervened, confidence_threshold=cfg['confidence_threshold'],
                      fixture='audited SO EMA; two duplicated real crops; two actual Trainer steps',
                      initialization=str(checkpoint), source_video=source['metadata']['video_id'],
                      parameter_atol=parameter_atol, no_op_vs_repeat='bitwise-exact',
                      target_crop=target['metadata'], results=outputs,
                      cpu_shim='nonleaf scalar clone for original inplace CPU loss addition')
        with args.report.open('x') as stream:
            json.dump(report, stream, sort_keys=True)
        print('RQ3_REVISED_CPU_PASS', args.report, flush=True)

    tm.Trainer.train = audit
    sys.argv = ['train_da.py', '-c', str(root / 'configs/dark_vs_all_new_videos_full/version_final_ckpt.yaml'),
                '-s', '6', '-i', '-1', '--rq2-shift', 'S1', '--rq2-registry',
                '/vepfs/vlms/SADA_PLUS/configs/rq2/shift_registry.yaml', '--rq2-reference-root',
                '/vepfs/vlms/SADA_PLUS/data/ek100_slowfast/features', '-n', args.report.stem,
                '--debug', '--benchmark-audit-log', '--rq2-regime', 'uda', '--supervised-upper-bound', 'off',
                '--rq2-target-manifest', '/vepfs/vlms/rq2_uda_target_manifests_0924/S1.json',
                '--rq3-strategy', args.strategy, '--rq3-teacher', args.teacher, '--rq3-cache', str(args.cache),
                '--rq3-revision', 'revised', '--rq3-candidate-manifest', str(args.candidate_manifest),
                '--rq3-confidence-calibration', str(args.confidence_calibration)]
    runpy.run_path(str(root / 'train_da.py'), run_name='__main__')


if __name__ == '__main__':
    main()
