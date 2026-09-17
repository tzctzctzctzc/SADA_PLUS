"""Analyze one frozen P036 stage with component-cluster uncertainty."""
import ast
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata

ROOT = Path('/vepfs/vlms/1')
assert Path.cwd().resolve() == ROOT
assert len(sys.argv) == 2 and sys.argv[1] in {'development', 'confirmation'}
stage = sys.argv[1]
config_path = Path('configs/P036_risk_controlled_source_veto.json')
config = json.loads(config_path.read_text())
stage_config = config['stages'][stage]
rows = stage_config['records']
predictions_path = Path(f'runs/P036_{stage}/predictions.jsonl')
predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
assert len(predictions) == len({row['key'] for row in predictions}) == stage_config['expected_records']
assert all(math.isfinite(row['margin']) and all(math.isfinite(x) for x in row['true_false_logits']) for row in predictions)
counts = collections.Counter(row['condition'] for row in predictions)
assert counts == {condition: len(rows) * config['topk_rerank'] for condition in config['conditions']}
lookup = {(row['id'], row['condition'], row['candidate']): float(row['margin']) for row in predictions}

metric_source = Path('third_party/pinpoint-dataset/utils/metrics.py')
manifest = json.loads(Path('literature/PinPoint_source_manifest.json').read_text())
assert hashlib.sha256(metric_source.read_bytes()).hexdigest() == next(
    row['sha256'] for row in manifest['files'] if row['path'] == 'utils/metrics.py'
)
namespace = {'Dict': Dict, 'List': List}
exec(
    compile(
        ast.Module(
            body=[node for node in ast.parse(metric_source.read_text()).body if isinstance(node, ast.FunctionDef)],
            type_ignores=[],
        ),
        str(metric_source),
        'exec',
    ),
    namespace,
)

def metrics(rank, row):
    official = namespace['calculate_metrics_at_k'](rank, row['positive_set'], 10)
    return {
        'ap@10': float(official['ap']),
        'precision@1': float(rank[0] in row['positive_set']),
        'precision@10': float(sum(x in row['positive_set'] for x in rank[:10]) / 10),
        'negative_precision@10': float(sum(x in row['negative_set'] for x in rank[:10]) / 10),
    }

def auc(positive, negative):
    if not positive or not negative:
        return None
    scores = np.asarray(positive + negative, dtype=np.float64)
    labels = np.asarray([1] * len(positive) + [0] * len(negative))
    ranks = rankdata(scores, method='average')
    return float((ranks[labels == 1].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))

blocks_by_component = collections.defaultdict(list)
for index, row in enumerate(rows):
    blocks_by_component[row['reference_component']].append(index)
blocks = list(blocks_by_component.values())
assert len(blocks) == stage_config['n_components']
rng = np.random.default_rng(17036 if stage == 'development' else 27036)
bootstrap = [
    np.concatenate([blocks[index] for index in rng.integers(len(blocks), size=len(blocks))])
    for _ in range(5000)
]

names = ['joint', 'risk_source_veto', 'zero_source_veto', 'weak_min', 'log_prob_and']
values = {name: [] for name in names}
mechanism_auc = {name: [] for name in names}
ties = collections.Counter()
details = []
threshold = float(config['method']['calibration']['threshold'])
row_veto_counts = []

for row in rows:
    candidates = row['candidates_to_score']
    joint = np.asarray([lookup[row['id'], 'labelled_dual', candidate] for candidate in candidates], dtype=np.float64)
    source1 = np.asarray([lookup[row['id'], 'source1', candidate] for candidate in candidates], dtype=np.float64)
    source2 = np.asarray([lookup[row['id'], 'source2', candidate] for candidate in candidates], dtype=np.float64)
    weak = np.minimum(source1, source2)
    log_and = -np.logaddexp(0, -source1) - np.logaddexp(0, -source2)
    veto = weak < threshold
    zero_veto = weak < 0
    orders = {
        'joint': np.argsort(-joint, kind='stable'),
        'risk_source_veto': np.asarray(sorted(range(len(candidates)), key=lambda i: (bool(veto[i]), -joint[i], i))),
        'zero_source_veto': np.asarray(sorted(range(len(candidates)), key=lambda i: (bool(zero_veto[i]), -joint[i], i))),
        'weak_min': np.argsort(-weak, kind='stable'),
        'log_prob_and': np.argsort(-log_and, kind='stable'),
    }
    score_vectors = {
        'joint': joint,
        'risk_source_veto': np.where(veto, 0.0, 1.0),
        'zero_source_veto': np.where(zero_veto, 0.0, 1.0),
        'weak_min': weak,
        'log_prob_and': log_and,
    }
    row_result = {
        'id': row['id'],
        'reference_component': row['reference_component'],
        'query_category': row['query_category'],
        'metrics': {},
        'ranks': {},
        'joint': joint.tolist(),
        'source1': source1.tolist(),
        'source2': source2.tolist(),
        'weak': weak.tolist(),
        'risk_veto': veto.tolist(),
    }
    positives = set(row['positive_set'])
    negatives = set(row['negative_set'])
    row_veto_counts.append({
        'positive_total': sum(candidate in positives for candidate in candidates),
        'positive_veto': sum(candidate in positives and bool(flag) for candidate, flag in zip(candidates, veto)),
        'negative_total': sum(candidate in negatives for candidate in candidates),
        'negative_veto': sum(candidate in negatives and bool(flag) for candidate, flag in zip(candidates, veto)),
    })
    for name, order in orders.items():
        rank = [candidates[index] for index in order] + row['original_candidates'][config['topk_rerank']:]
        assert len(rank) == len(set(rank)) == 50
        result = metrics(rank, row)
        values[name].append(result)
        row_result['metrics'][name] = result
        row_result['ranks'][name] = rank
        vector = score_vectors[name]
        ties[name] += int(len(set(vector.tolist())) < len(vector))
        positive_scores = [float(score) for candidate, score in zip(candidates, vector) if candidate in positives]
        negative_scores = [float(score) for candidate, score in zip(candidates, vector) if candidate in negatives]
        mechanism_auc[name].append(auc(positive_scores, negative_scores))
    details.append(row_result)

summaries = {
    name: {metric: float(np.mean([result[metric] for result in results])) for metric in results[0]}
    for name, results in values.items()
}
eligible_indices = [index for index, row in enumerate(rows) if set(row['positive_set']) & set(row['candidates_to_score'])]
eligible_summaries = {
    name: {
        metric: float(np.mean([values[name][index][metric] for index in eligible_indices]))
        for metric in values[name][0]
    }
    for name in names
}
paired = {}
for method in names[1:]:
    comparison = method + ' minus joint'
    paired[comparison] = {}
    for metric in values['joint'][0]:
        delta = np.asarray([x[metric] for x in values[method]]) - np.asarray([x[metric] for x in values['joint']])
        paired[comparison][metric] = {
            'difference_pp': float(delta.mean() * 100),
            'component_cluster_95ci_pp': np.percentile([delta[index].mean() * 100 for index in bootstrap], [2.5, 97.5]).tolist(),
            'query_gains': int((delta > 0).sum()),
            'query_losses': int((delta < 0).sum()),
        }

def ratio(indices, numerator, denominator):
    top = sum(row_veto_counts[index][numerator] for index in indices)
    bottom = sum(row_veto_counts[index][denominator] for index in indices)
    return float(top / bottom) if bottom else float('nan')
all_indices = np.arange(len(rows))
positive_veto_rate = ratio(all_indices, 'positive_veto', 'positive_total')
negative_veto_rate = ratio(all_indices, 'negative_veto', 'negative_total')
positive_boot = [ratio(index, 'positive_veto', 'positive_total') for index in bootstrap]
negative_boot = [ratio(index, 'negative_veto', 'negative_total') for index in bootstrap]
difference_boot = [negative - positive for negative, positive in zip(negative_boot, positive_boot)]
veto_report = {
    'threshold': threshold,
    'positive_candidates': int(sum(x['positive_total'] for x in row_veto_counts)),
    'positive_vetoed': int(sum(x['positive_veto'] for x in row_veto_counts)),
    'positive_veto_rate': positive_veto_rate,
    'positive_veto_rate_component_95ci': np.nanpercentile(positive_boot, [2.5, 97.5]).tolist(),
    'explicit_negative_candidates': int(sum(x['negative_total'] for x in row_veto_counts)),
    'explicit_negative_vetoed': int(sum(x['negative_veto'] for x in row_veto_counts)),
    'explicit_negative_veto_rate': negative_veto_rate,
    'explicit_negative_veto_rate_component_95ci': np.nanpercentile(negative_boot, [2.5, 97.5]).tolist(),
    'negative_minus_positive_veto_rate': negative_veto_rate - positive_veto_rate,
    'negative_minus_positive_component_95ci': np.nanpercentile(difference_boot, [2.5, 97.5]).tolist(),
}
macro_auc = {}
for name in names:
    eligible = [value for value in mechanism_auc[name] if value is not None]
    macro_auc[name] = {'n_eligible_queries': len(eligible), 'macro_positive_vs_explicit_negative_auc': float(np.mean(eligible)) if eligible else None}
primary = paired['risk_source_veto minus joint']
gate_checks = {
    'rsv_minus_joint_ap10_gt_0': primary['ap@10']['difference_pp'] > 0,
    'rsv_minus_joint_negative_precision_at10_le_0': primary['negative_precision@10']['difference_pp'] <= 0,
    'positive_candidate_veto_rate_le_0_05': positive_veto_rate <= 0.05,
    'negative_veto_rate_minus_positive_gt_0': negative_veto_rate > positive_veto_rate,
}
report = {
    'complete': True,
    'stage': stage,
    'evidence_status': config['evidence_status'],
    'n_queries': len(rows),
    'n_reference_components': len(blocks),
    'n_queries_with_positive_top20': len(eligible_indices),
    'summaries_all_queries': summaries,
    'summaries_positive_top20_queries': eligible_summaries,
    'paired': paired,
    'source_veto': veto_report,
    'mechanism_auc': macro_auc,
    'ties_anywhere_in_top20_scores': dict(ties),
    'continuation_gate_evaluation': {
        'checks': gate_checks,
        'all_pass': all(gate_checks.values()),
        'meaning': config['development_gate']['meaning'] if stage == 'development' else 'Frozen replication criteria on the untouched confirmation stage.',
    },
    'score_file_sha256': hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'limits': config['limits'],
}
Path(f'runs/P036_{stage}_details.json').write_text(json.dumps(details, indent=2) + '\n')
result_path = Path(f'results/P036_{stage}.json')
result_path.write_text(json.dumps(report, indent=2) + '\n')
for name in names:
    print(name, 'AP10', round(summaries[name]['ap@10'], 6), 'P1', round(summaries[name]['precision@1'], 6), 'negP10', round(summaries[name]['negative_precision@10'], 6))
print('VETO', veto_report)
print('PRIMARY', primary)
print('GATE', report['continuation_gate_evaluation'])
