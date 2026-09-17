"""Analyze the frozen P039 Shadow Binding Bridges mechanism screen."""
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path('/vepfs/vlms/1')
assert Path.cwd().resolve() == ROOT
config_path = Path('configs/P039_shadow_binding_bridges.json')
config = json.loads(config_path.read_text())
rows = config['records']
predictions_path = Path('runs/P039_shadow_binding_bridges/predictions.jsonl')
predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
assert len(predictions) == len({row['key'] for row in predictions}) == config['expected_records']
assert all(math.isfinite(row['margin']) for row in predictions)
counts = collections.Counter(row['intervention'] for row in predictions)
assert counts == {item['id']: config['expected_base_tasks'] for item in config['interventions']}
scores = {
    (row['id'], row['candidate'], row['condition'], row['intervention']): float(row['margin'])
    for row in predictions
}


def auc(positive, negative):
    assert positive and negative
    values = np.asarray(positive + negative, dtype=np.float64)
    labels = np.asarray([1] * len(positive) + [0] * len(negative))
    ranks = rankdata(values, method='average')
    return float((ranks[labels == 1].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


intervention_ids = [item['id'] for item in config['interventions']]
per_intervention = {}
details = []
for intervention in intervention_ids:
    query_score_aucs = []
    query_binding_aucs = []
    positive_contrasts = []
    negative_contrasts = []
    equivalence_gaps = []
    for row in rows:
        candidates = row['candidates_to_score']
        positives = set(row['pilot_positive_candidates'])
        negatives = set(row['pilot_explicit_negative_candidates'])
        correct = np.asarray([
            scores[row['id'], candidate, 'correct_ab', intervention] for candidate in candidates
        ])
        equivalent = np.asarray([
            scores[row['id'], candidate, 'equivalent_ba', intervention] for candidate in candidates
        ])
        wrong = np.asarray([
            scores[row['id'], candidate, 'wrong_binding_ba', intervention] for candidate in candidates
        ])
        contrast = equivalent - wrong
        positive_scores = [float(value) for candidate, value in zip(candidates, correct) if candidate in positives]
        negative_scores = [float(value) for candidate, value in zip(candidates, correct) if candidate in negatives]
        row_positive_contrasts = [float(value) for candidate, value in zip(candidates, contrast) if candidate in positives]
        row_negative_contrasts = [float(value) for candidate, value in zip(candidates, contrast) if candidate in negatives]
        query_score_aucs.append(auc(positive_scores, negative_scores))
        query_binding_aucs.append(auc(row_positive_contrasts, row_negative_contrasts))
        positive_contrasts.extend(row_positive_contrasts)
        negative_contrasts.extend(row_negative_contrasts)
        equivalence_gaps.extend(np.abs(correct - equivalent).tolist())
        details.append({
            'id': row['id'],
            'reference_component': row['reference_component'],
            'intervention': intervention,
            'candidates': candidates,
            'correct_ab': correct.tolist(),
            'equivalent_ba': equivalent.tolist(),
            'wrong_binding_ba': wrong.tolist(),
            'binding_contrast': contrast.tolist(),
            'score_auc': query_score_aucs[-1],
            'binding_contrast_auc': query_binding_aucs[-1],
        })
    assert len(positive_contrasts) == config['n_positive_rows']
    assert len(negative_contrasts) == config['n_explicit_negative_rows']
    per_intervention[intervention] = {
        'correct_condition_score_macro_auc': float(np.mean(query_score_aucs)),
        'binding_contrast_macro_auc': float(np.mean(query_binding_aucs)),
        'positive_correct_minus_wrong_binding_mean': float(np.mean(positive_contrasts)),
        'positive_correct_binding_win_rate': float(np.mean(np.asarray(positive_contrasts) > 0)),
        'explicit_negative_correct_minus_wrong_binding_mean': float(np.mean(negative_contrasts)),
        'equivalent_order_mean_absolute_margin_gap': float(np.mean(equivalence_gaps)),
        'n_query_aucs': len(query_score_aucs),
    }

soft = per_intervention['soft']
gate_checks = {}
eligible = []
for intervention in config['candidate_windows']:
    result = per_intervention[intervention]
    checks = {
        'positive_binding_mean_gt_soft': (
            result['positive_correct_minus_wrong_binding_mean']
            > soft['positive_correct_minus_wrong_binding_mean']
        ),
        'binding_macro_auc_gt_0_55': result['binding_contrast_macro_auc'] > 0.55,
        'binding_macro_auc_at_least_soft_plus_0_05': (
            result['binding_contrast_macro_auc'] >= soft['binding_contrast_macro_auc'] + 0.05
        ),
        'correct_score_macro_auc_within_0_02_of_soft': (
            result['correct_condition_score_macro_auc']
            >= soft['correct_condition_score_macro_auc'] - 0.02
        ),
        'equivalent_order_gap_lt_soft': (
            result['equivalent_order_mean_absolute_margin_gap']
            < soft['equivalent_order_mean_absolute_margin_gap']
        ),
    }
    gate_checks[intervention] = checks
    if all(checks.values()):
        eligible.append(intervention)

window_order = {name: index for index, name in enumerate(config['candidate_windows'])}
selected = None
if eligible:
    selected = sorted(
        eligible,
        key=lambda name: (
            -per_intervention[name]['binding_contrast_macro_auc'],
            -per_intervention[name]['correct_condition_score_macro_auc'],
            window_order[name],
        ),
    )[0]

all_layer_pass = (
    selected is not None
    and per_intervention['bridge_all_layers']['binding_contrast_macro_auc']
    < per_intervention[selected]['binding_contrast_macro_auc']
)
all_pass = selected is not None and all_layer_pass
report = {
    'complete': True,
    'evidence_status': config['evidence_status'],
    'n_queries': config['n_queries'],
    'n_reference_components': config['n_reference_components'],
    'n_candidate_rows': config['n_candidate_rows'],
    'per_intervention': per_intervention,
    'selection': {
        'eligible_windows': eligible,
        'selected_window': selected if all_pass else None,
        'provisional_best_before_all_layer_control': selected,
        'all_layer_control_pass': all_layer_pass,
        'all_pass': all_pass,
        'gate_checks': gate_checks,
        'meaning': config['development_gate']['meaning'],
    },
    'score_file_sha256': hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'limits': config['limits'],
}
Path('runs/P039_shadow_binding_bridges/details.json').write_text(json.dumps(details, indent=2) + '\n')
Path('results/P039_shadow_binding_bridges.json').write_text(json.dumps(report, indent=2) + '\n')
for intervention in intervention_ids:
    value = per_intervention[intervention]
    print(
        intervention,
        'score_auc', round(value['correct_condition_score_macro_auc'], 6),
        'binding_auc', round(value['binding_contrast_macro_auc'], 6),
        'positive_binding_mean', round(value['positive_correct_minus_wrong_binding_mean'], 6),
        'equivalence_gap', round(value['equivalent_order_mean_absolute_margin_gap'], 6),
    )
print('SELECTION', report['selection'])
