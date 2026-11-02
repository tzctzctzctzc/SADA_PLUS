"""独立CPU重算真实probe标签、源作用域、账本和GPU准入，不复用标签实现。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.baseline_protocol import read, save, sha
from tools.proposal_foreground import BASE_COMMIT, SPEC


def audit(root):
    import torch
    plan = read(root/'plan.json')
    assert plan['base'] == BASE_COMMIT and plan['specification'] == SPEC
    accepted = read(root/'acceptance.json')
    assert accepted['count'] == 6 and not accepted['formal_started']
    reports = {}
    for shift in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6'):
        entry = root/'probes'/shift
        report = read(entry/'result.json'); cfg = read(entry/'config.json')
        assert sha(entry/'result.json') == accepted['reports'][shift]
        assert report['config_sha256'] == sha(entry/'config.json')
        assert report['code'] == cfg['baseline']['code'] == plan['code']
        payload = read(cfg['proposal_foreground']['source_truth'])
        assert sha(cfg['proposal_foreground']['source_truth']) == cfg['proposal_foreground']['source_truth_sha256']
        assert not set(payload['roles']['fit']) & set(payload['roles']['calibration'])
        assert set(payload['roles']['fit']) | set(payload['roles']['calibration']) == set(payload['records'])
        assert not set(payload['records']) & set(payload['source_val_ids'])
        guard = read(report['guard']); lease = read(guard['lease'])
        assert sha(report['guard']) == report['guard_sha256']
        assert sha(guard['lease']) == guard['lease_sha256']
        assert guard['owner_pid'] == lease['owner_pid'] and guard['observed']['uuid'] == lease['uuid']
        assert guard['observed']['free_mib'] >= lease['minimum_free_mib'] >= 12288
        assert lease['commit'] == plan['code']['commit']
        for phase in report['phases'].values():
            assert all(phase['updated'].values())
            gradients = phase['auxiliary_gradients']
            assert gradients['SGP'] > 0 and gradients['foreground'] > 0 and gradients['cls'] == gradients['reg'] == 0
        observed = entry/'probe_source_observations.pt'
        assert sha(observed) == report['source_observation_sha256']
        counts = np.zeros(3, dtype=np.int64)
        for row in torch.load(observed, map_location='cpu'):
            assert row['video'] in payload['roles']['fit']
            boxes, gt = row['boxes'].numpy(), row['truth'].numpy()
            inter = np.maximum(0., np.minimum(boxes[:, None, 1], gt[None, :, 1]) - np.maximum(boxes[:, None, 0], gt[None, :, 0]))
            union = (boxes[:, 1]-boxes[:, 0])[:, None] + (gt[:, 1]-gt[:, 0])[None] - inter
            maximum = np.max(inter / np.maximum(union, 1e-8), axis=1) if len(gt) else np.zeros(len(boxes))
            labels = np.where(maximum < .01, 0, np.where(maximum >= .5, 2, 1))
            np.testing.assert_array_equal(labels, row['states'].numpy())
            full = np.asarray(payload['records'][row['video']]['segments'], np.float32).reshape(-1, 2) - row['crop_start']
            assert all(any(np.array_equal(box, original) for original in full) for box in gt)
            counts += np.bincount(labels, minlength=3)
        ledger = entry/'probe_events.jsonl'
        assert sha(ledger) == report['ledger_sha256']
        digest = '0'*64
        for number, text in enumerate(ledger.read_text().splitlines(), 1):
            event = json.loads(text); recorded = event.pop('digest')
            assert event['step'] == number and event['target_new_loss'] == 0
            for row in event['source']:
                assert row['video'] in payload['roles']['fit'] and sum(row['counts']) == row['candidates']
            assert set(event['calibration_excluded']) <= set(payload['roles']['calibration'])
            digest = hashlib.sha256((digest + json.dumps(event, sort_keys=True)).encode()).hexdigest()
            assert digest == recorded
        assert number == 4
        for key in ('original_cold_identity_exact', 'disabled_all_native_losses_exact',
                    'original_complete_NMS_voting_predictions_exact', 'full_state_and_all_RNG_restore',
                    'reconstruct_context_restore_exact'):
            assert report[key]
        artifacts = {str(p):sha(p) for p in (entry/'config.json', entry/'result.json', observed, ledger, Path(report['guard']), Path(guard['lease']))}
        reports[shift] = dict(passed=True, actual_candidate_label_counts=counts.tolist(),
                             artifacts=artifacts, head_parameters=report['parameters'])
    result = dict(passed=True, count=6, code=plan['code'], reports=reports,
        mechanism_quality_or_mAP_not_established=True, formal_training_started=False)
    save(root/'independent_implementation_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    args = parser.parse_args(); print(json.dumps(audit(args.root)))
