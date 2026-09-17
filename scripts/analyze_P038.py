"""Analyze the frozen P038 semantic-role interleaving pilot."""
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata


ROOT = Path('/vepfs/vlms/1')
assert Path.cwd().resolve() == ROOT
config_path = Path('configs/P038_semantically_interleaved_role_cards.json')
config = json.loads(config_path.read_text())
rows = config['records']
predictions_path = Path('runs/P038_semantically_interleaved_role_cards/predictions.jsonl')
predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
assert len(predictions) == len({row['key'] for row in predictions}) == config['expected_records']
assert all(math.isfinite(row['margin']) for row in predictions)
counts = collections.Counter(row['condition'] for row in predictions)
assert counts == {condition: len(rows) * config['topk_rerank'] for condition in config['conditions']}
new = {(row['id'], row['condition'], row['candidate']): float(row['margin']) for row in predictions}

p030 = [json.loads(line) for line in Path('runs/P030_pinpoint_references/predictions.jsonl').read_text().splitlines()]
original = {
    (row['id'], row['candidate']): float(row['margin'])
    for row in p030
    if row['condition'] == 'labelled_dual'
}

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
    positives = set(row['positive_set'])
    negatives = set(row['negative_set'])
    return {
        'ap@10': float(official['ap']),
        'precision@1': float(rank[0] in positives),
        'precision@10': float(sum(x in positives for x in rank[:10]) / 10),
        'negative_precision@10': float(sum(x in negatives for x in rank[:10]) / 10),
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
assert len(blocks) == config['n_reference_components']
rng = np.random.default_rng(17038)
bootstrap = [
    np.concatenate([blocks[index] for index in rng.integers(len(blocks), size=len(blocks))])
    for _ in range(5000)
]

names = ['original_dual', 'sirc_ab', 'sirc_ba_equivalent', 'sirc_orbit_min']
values = {name: [] for name in names}
macro_auc_values = {name: [] for name in names}
positive_contrasts = []
negative_contrasts = []
row_positive_contrasts = []
row_negative_contrasts = []
equivalence_gaps = []
details = []

for row in rows:
    candidates = row['candidates_to_score']
    base = np.asarray([original[row['id'], candidate] for candidate in candidates], dtype=np.float64)
    ab = np.asarray([new[row['id'], 'sirc_ab', candidate] for candidate in candidates], dtype=np.float64)
    ba = np.asarray([new[row['id'], 'sirc_ba_equivalent', candidate] for candidate in candidates], dtype=np.float64)
    wrong = np.asarray([new[row['id'], 'sirc_ba_wrong_binding', candidate] for candidate in candidates], dtype=np.float64)
    orbit_min = np.minimum(ab, ba)
    contrast = ba - wrong
    vectors = {
        'original_dual': base,
        'sirc_ab': ab,
        'sirc_ba_equivalent': ba,
        'sirc_orbit_min': orbit_min,
    }
    positives = set(row['positive_set'])
    negatives = set(row['negative_set'])
    row_pos = [float(score) for candidate, score in zip(candidates, contrast) if candidate in positives]
    row_neg = [float(score) for candidate, score in zip(candidates, contrast) if candidate in negatives]
    positive_contrasts.extend(row_pos)
    negative_contrasts.extend(row_neg)
    row_positive_contrasts.append(row_pos)
    row_negative_contrasts.append(row_neg)
    equivalence_gaps.extend(np.abs(ab - ba).tolist())
    row_result = {
        'id': row['id'],
        'reference_component': row['reference_component'],
        'metrics': {},
        'ranks': {},
        'original': base.tolist(),
        'sirc_ab': ab.tolist(),
        'sirc_ba_equivalent': ba.tolist(),
        'sirc_ba_wrong_binding': wrong.tolist(),
        'binding_contrast': contrast.tolist(),
        'sirc_orbit_min': orbit_min.tolist(),
    }
    for name, vector in vectors.items():
        order = np.argsort(-vector, kind='stable')
        rank = [candidates[index] for index in order] + row['original_candidates'][config['topk_rerank']:]
        assert len(rank) == len(set(rank)) == 50
        result = metrics(rank, row)
        values[name].append(result)
        row_result['metrics'][name] = result
        row_result['ranks'][name] = rank
        pos_scores = [float(score) for candidate, score in zip(candidates, vector) if candidate in positives]
        neg_scores = [float(score) for candidate, score in zip(candidates, vector) if candidate in negatives]
        macro_auc_values[name].append(auc(pos_scores, neg_scores))
    details.append(row_result)

assert len(positive_contrasts) == config['n_positive_top20']
assert len(negative_contrasts) == config['n_negative_top20']
summaries = {
    name: {metric: float(np.mean([result[metric] for result in results])) for metric in results[0]}
    for name, results in values.items()
}
paired = {}
for method in names[1:]:
    comparison = method + ' minus original_dual'
    paired[comparison] = {}
    for metric in values['original_dual'][0]:
        delta = np.asarray([x[metric] for x in values[method]]) - np.asarray([x[metric] for x in values['original_dual']])
        paired[comparison][metric] = {
            'difference_pp': float(delta.mean() * 100),
            'component_cluster_95ci_pp': np.percentile(
                [delta[index].mean() * 100 for index in bootstrap], [2.5, 97.5]
            ).tolist(),
            'query_gains': int((delta > 0).sum()),
            'query_losses': int((delta < 0).sum()),
        }


def mean_from_rows(indices, values_by_row):
    values = [value for index in indices for value in values_by_row[index]]
    return float(np.mean(values)) if values else float('nan')


all_indices = np.arange(len(rows))
positive_mean = mean_from_rows(all_indices, row_positive_contrasts)
negative_mean = mean_from_rows(all_indices, row_negative_contrasts)
positive_boot = [mean_from_rows(indices, row_positive_contrasts) for indices in bootstrap]
negative_boot = [mean_from_rows(indices, row_negative_contrasts) for indices in bootstrap]
contrast_auc_values = [
    auc(row_pos, row_neg)
    for row_pos, row_neg in zip(row_positive_contrasts, row_negative_contrasts)
]
eligible_contrast_auc = [value for value in contrast_auc_values if value is not None]
macro_auc = {
    name: {
        'n_eligible_queries': len([value for value in macro_auc_values[name] if value is not None]),
        'macro_positive_vs_explicit_negative_auc': float(np.mean([value for value in macro_auc_values[name] if value is not None])),
    }
    for name in names
}
observed_equivalence_gap = float(np.mean(equivalence_gaps))
reference_equivalence_gap = float(config['reference_equivalence_gap']['mean_absolute_margin_gap'])
binding_macro_auc = float(np.mean(eligible_contrast_auc)) if eligible_contrast_auc else None
mechanism = {
    'mean_absolute_sirc_ab_minus_ba_margin': observed_equivalence_gap,
    'p037_reference_mean_absolute_gap': reference_equivalence_gap,
    'equivalence_gap_ratio': observed_equivalence_gap / reference_equivalence_gap,
    'positive_candidates': len(positive_contrasts),
    'positive_mean_correct_minus_wrong_binding': positive_mean,
    'positive_mean_component_95ci': np.nanpercentile(positive_boot, [2.5, 97.5]).tolist(),
    'positive_correct_binding_win_rate': float(np.mean(np.asarray(positive_contrasts) > 0)),
    'explicit_negative_candidates': len(negative_contrasts),
    'explicit_negative_mean_correct_minus_wrong_binding': negative_mean,
    'explicit_negative_mean_component_95ci': np.nanpercentile(negative_boot, [2.5, 97.5]).tolist(),
    'binding_contrast_macro_positive_vs_explicit_negative_auc': binding_macro_auc,
    'n_binding_auc_queries': len(eligible_contrast_auc),
    'score_macro_auc': macro_auc,
}

primary = paired['sirc_ab minus original_dual']
gate_checks = {
    'sirc_ab_minus_original_dual_ap10_gt_0': primary['ap@10']['difference_pp'] > 0,
    'sirc_ab_minus_original_dual_negative_precision_at10_le_0': primary['negative_precision@10']['difference_pp'] <= 0,
    'sirc_equivalence_gap_below_p037': observed_equivalence_gap < reference_equivalence_gap,
    'positive_correct_minus_wrong_binding_mean_gt_0': positive_mean > 0,
    'binding_contrast_macro_auc_gt_0_5': binding_macro_auc is not None and binding_macro_auc > 0.5,
}
report = {
    'complete': True,
    'evidence_status': config['evidence_status'],
    'n_queries': len(rows),
    'n_reference_components': len(blocks),
    'summaries': summaries,
    'paired': paired,
    'mechanism': mechanism,
    'continuation_gate_evaluation': {
        'checks': gate_checks,
        'all_pass': all(gate_checks.values()),
        'meaning': config['development_gate']['meaning'],
    },
    'score_file_sha256': hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'limits': config['limits'],
}
Path('runs/P038_semantically_interleaved_role_cards/details.json').write_text(json.dumps(details, indent=2) + '\n')
Path('results/P038_semantically_interleaved_role_cards.json').write_text(json.dumps(report, indent=2) + '\n')
for name in names:
    print(name, 'AP10', round(summaries[name]['ap@10'], 6), 'P1', round(summaries[name]['precision@1'], 6), 'negP10', round(summaries[name]['negative_precision@10'], 6))
print('MECHANISM', mechanism)
print('PRIMARY', primary)
print('GATE', report['continuation_gate_evaluation'])
