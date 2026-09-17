"""Freeze a staged source-veto transfer test before any reserved-image scoring."""
import collections
import datetime
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path('/vepfs/vlms/1')
assert Path.cwd().resolve() == ROOT
SEED = 'P036-source-veto-v1'
TOPK = 20

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def qid(value):
    return str(value).removeprefix('query_').zfill(5)

def valid_signature(value):
    return bool(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{32}', value))

p030_path = Path('configs/P030_pinpoint_references.json')
p035_path = Path('configs/P035_source_bound_predicates.json')
p035_scores_path = Path('runs/P035_source_bound_predicates/predictions.jsonl')
p035_result_path = Path('results/P035_source_bound_predicates.json')
parquet_path = Path('data/PinPoint/pinpoint_licensed.parquet')
candidates_path = Path('data/PinPoint/bge_vl_mllm_s1_retrieval_results_licensed.json')
p030 = json.loads(p030_path.read_text())
p035 = json.loads(p035_path.read_text())
p035_result = json.loads(p035_result_path.read_text())
assert p035_result['complete'] and not p035_result['continuation_gate_evaluation']['all_pass']
assert sha(p035_scores_path) == p035_result['score_file_sha256']

# P035 froze min(source1, source2) before scores. P036 changes only how that
# source evidence is used: it vetoes extreme failures while retaining joint order.
p035_scores = [json.loads(line) for line in p035_scores_path.read_text().splitlines()]
source = {(r['id'], r['condition'], r['candidate']): float(r['margin']) for r in p035_scores}
calibration_scores = []
for row in p030['records']:
    for candidate in row['candidates_to_score']:
        if candidate in row['positive_set']:
            calibration_scores.append(min(source[row['id'], 'source1', candidate], source[row['id'], 'source2', candidate]))
assert len(calibration_scores) == 21 and all(math.isfinite(x) for x in calibration_scores)
alpha = 0.05
# Candidate-level split-conformal p=(1 + count(calibration <= score))/(n+1).
# At alpha=.05 and n=21, only a score strictly below the calibration minimum is vetoed.
threshold = min(calibration_scores)
assert 1 / (len(calibration_scores) + 1) <= alpha < 2 / (len(calibration_scores) + 1)
assert threshold == -3.875

frame = pd.read_parquet(parquet_path)
raw = json.loads(candidates_path.read_text())
assert set(raw) == set(frame.query_id.map(qid))
assert all(len(v['retrieved_items']) == len(set(v['retrieved_items'])) == 50 for v in raw.values())
multi = frame[frame.query_image_signature2.map(valid_signature)]
parent = {}
def find(value):
    parent.setdefault(value, value)
    if parent[value] != value:
        parent[value] = find(parent[value])
    return parent[value]
for _, row in multi.iterrows():
    left, right = find(row.query_image_signature), find(row.query_image_signature2)
    parent[max(left, right)] = min(left, right)
groups = collections.defaultdict(list)
for _, row in multi.iterrows():
    groups[find(row.query_image_signature)].append(row)
conflicts = {
    qid(row.query_id)
    for _, row in frame.iterrows()
    if set(row.positive_candidates) & set(row.negative_candidates)
}
selected = {row['reference_component'] for row in p030['records']}
reserved = set(p030['reserved_reference_components'])
assert not selected & reserved and selected | reserved == set(groups)

component_records = {}
component_stats = {}
for component in sorted(reserved):
    records = []
    for row in groups[component]:
        identifier = qid(row.query_id)
        if identifier in conflicts:
            continue
        positives = sorted(x for x in set(row.positive_candidates) if valid_signature(x))
        negatives = sorted(x for x in set(row.negative_candidates) if valid_signature(x))
        assert not set(positives) & set(negatives)
        candidates = raw[identifier]['retrieved_items']
        records.append({
            'id': identifier,
            'source_query_id': row.query_id,
            'reference_component': component,
            'query_category': row.query_category,
            'references': [row.query_image_signature, row.query_image_signature2],
            'instruction': row.instruction,
            'positive_set': positives,
            'negative_set': negatives,
            'original_candidates': candidates,
            'candidates_to_score': candidates[:TOPK],
        })
    if not records:
        continue
    component_records[component] = records
    component_stats[component] = {
        'n_rows': len(records),
        'n_positive_top20': sum(len(set(r['positive_set']) & set(r['candidates_to_score'])) for r in records),
        'n_negative_top20': sum(len(set(r['negative_set']) & set(r['candidates_to_score'])) for r in records),
        'categories': dict(collections.Counter(r['query_category'] for r in records)),
    }
assert len(component_records) == 54
assert sum(len(v) for v in component_records.values()) == 518
assert sum(v['n_positive_top20'] for v in component_stats.values()) == 262
assert sum(v['n_negative_top20'] for v in component_stats.values()) == 518

# Metadata-only stratification. Each stage gets eight positive-bearing components;
# the large Context-Fit component is held for confirmation, and the other rare-
# category component stays in development. Remaining zero-positive components are
# hash ordered and greedily row balanced. No model score enters this split.
positive = sorted(
    [c for c, x in component_stats.items() if x['n_positive_top20'] > 0],
    key=lambda c: (component_stats[c]['n_positive_top20'], digest(SEED + '|positive|' + c)),
)
assert len(positive) == 16
giant = max(positive, key=lambda c: component_stats[c]['n_positive_top20'])
confirmation_components = {giant, *positive[:7]}
development_components = set(positive) - confirmation_components
rare = [c for c, x in component_stats.items() if set(x['categories']) - {'Composition'}]
assert len(rare) == 2 and giant in rare
small_rare = next(c for c in rare if c != giant)
development_components.add(small_rare)
zeros = sorted(
    [c for c, x in component_stats.items() if x['n_positive_top20'] == 0 and c != small_rare],
    key=lambda c: digest(SEED + '|zero|' + c),
)
def row_count(components):
    return sum(component_stats[c]['n_rows'] for c in components)
for component in zeros:
    if len(development_components) >= 27:
        confirmation_components.add(component)
    elif len(confirmation_components) >= 27:
        development_components.add(component)
    elif row_count(development_components) <= row_count(confirmation_components):
        development_components.add(component)
    else:
        confirmation_components.add(component)
assert len(development_components) == len(confirmation_components) == 27
assert not development_components & confirmation_components
assert development_components | confirmation_components == set(component_records)

def flatten(components):
    return [row for c in sorted(components) for row in component_records[c]]

def stage_summary(records, components):
    return {
        'n_components': len(components),
        'n_queries': len(records),
        'n_positive_top20': sum(len(set(r['positive_set']) & set(r['candidates_to_score'])) for r in records),
        'n_negative_top20': sum(len(set(r['negative_set']) & set(r['candidates_to_score'])) for r in records),
        'n_positive_components': sum(component_stats[c]['n_positive_top20'] > 0 for c in components),
        'categories': dict(collections.Counter(r['query_category'] for r in records)),
    }

development_records = flatten(development_components)
confirmation_records = flatten(confirmation_components)
stages = {}
for name, records, components in [
    ('development', development_records, development_components),
    ('confirmation', confirmation_records, confirmation_components),
]:
    signatures = sorted({x for r in records for x in r['references'] + r['candidates_to_score']})
    stages[name] = {
        **stage_summary(records, components),
        'components': sorted(components),
        'records': records,
        'image_signatures': signatures,
        'expected_records': len(records) * TOPK * 3,
        'new_model_scores_seen': False,
    }
assert stages['development']['n_queries'] == 233
assert stages['confirmation']['n_queries'] == 285
assert stages['development']['n_positive_top20'] == 127
assert stages['confirmation']['n_positive_top20'] == 135

config = {
    'id': 'P036',
    'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'evidence_status': 'prospective_staged_transfer_after_P035',
    'candidate_method_name': 'Risk-Controlled Source Veto (RSV)',
    'purpose': 'Preserve the joint multi-reference VLM ranking and use source-bound judges only to veto candidates that fail a source more extremely than every development positive.',
    'transfer_origin': 'Selective prediction and conformal risk control, adapted to source-specific constraint verification in multi-reference composed retrieval.',
    'direct_prior_boundaries': [
        'Constraint-verification reranking already exists for single-reference fashion CIR.',
        'Conformal candidate sets and uncertainty calibration already exist for single-reference CIR and generic multimodal retrieval.',
        'The candidate contribution is therefore limited to source-bound multi-reference vetoing, interaction-preserving ranking, explicit-negative evaluation, and staged component-disjoint validation.',
    ],
    'method': {
        'joint_score': 'raw labelled_dual True-minus-False margin with both references and candidate visible',
        'source_score': 'weak=min(m_source1,m_source2), using the frozen P035 source prompts',
        'calibration': {
            'source': '21 positive candidates in the P030 development top20',
            'alpha': alpha,
            'candidate_level_exchangeability_caveat': 'Candidates share queries and reference components; call this risk-inspired calibration, not a formal distribution-free guarantee.',
            'p_value': '(1 + count(calibration_weak <= candidate_weak)) / (n_calibration + 1)',
            'veto_rule': 'veto iff p_value <= alpha, equivalently weak < threshold',
            'n_positive_candidates': len(calibration_scores),
            'threshold': threshold,
            'strict_comparison': '<',
            'calibration_score_sha256': hashlib.sha256(json.dumps(sorted(calibration_scores)).encode()).hexdigest(),
        },
        'ranking': 'Place non-vetoed top20 candidates first and vetoed candidates second; preserve descending joint_score within each block; append original ranks 21-50 unchanged.',
        'no_weight_or_threshold_tuning': True,
    },
    'conditions': ['labelled_dual', 'source1', 'source2'],
    'model': p035['model'],
    'model_revision': p035['model_revision'],
    'gpu_uuid': p035['gpu_uuid'],
    'backend': p035['backend'],
    'dtype': p035['dtype'],
    'min_pixels': p035['min_pixels'],
    'max_pixels': p035['max_pixels'],
    'batch_size': p035['batch_size'],
    'topk_rerank': TOPK,
    'split': {
        'unit': 'exact-RGB-connected reference component',
        'procedure': 'metadata-only positive-component balancing plus hash/row balancing; no reserved model scores',
        'development_first': True,
        'confirmation_execution_rule': 'Score confirmation images only if every frozen development gate passes.',
    },
    'development_gate': {
        'all_required': True,
        'rsv_minus_joint_ap10': 'strictly > 0',
        'rsv_minus_joint_explicit_negative_precision_at10': 'less than or equal to 0',
        'positive_candidate_veto_rate': 'less than or equal to 0.05',
        'explicit_negative_veto_rate_minus_positive_veto_rate': 'strictly > 0',
        'meaning': 'Eligibility to access confirmation images; not significance, novelty, or SOTA.',
    },
    'primary_metrics': ['set-normalized AP@10', 'precision@1', 'precision@10', 'explicit-negative precision@10'],
    'mechanism_metrics': ['positive candidate veto rate', 'explicit-negative candidate veto rate', 'component-cluster bootstrap intervals'],
    'fixed_secondary_controls': [
        'pure weak-source min-margin ranking (P035 direct copy)',
        'pure probabilistic AND ranking (P035 direct copy)',
        'uncalibrated zero-threshold source veto preserving joint order',
    ],
    'source_sha256': {
        str(p030_path): sha(p030_path),
        str(p035_path): sha(p035_path),
        str(p035_scores_path): sha(p035_scores_path),
        str(p035_result_path): sha(p035_result_path),
        str(parquet_path): sha(parquet_path),
        str(candidates_path): sha(candidates_path),
    },
    'stages': stages,
    'limits': [
        'The top20 reranking protocol reuses the released candidate lists and is not a full-corpus retriever.',
        'Candidate-level calibration points are dependent within queries/components, so no formal marginal-coverage claim is made.',
        'Only two reserved components contain non-Composition categories; category generalization remains descriptive.',
        'The confirmation stage remains untouched until every development gate passes.',
    ],
}
out = Path('configs/P036_risk_controlled_source_veto.json')
assert not out.exists()
out.write_text(json.dumps(config, indent=2) + '\n')
print(json.dumps({
    'config': str(out),
    'sha256': sha(out),
    'threshold': threshold,
    'development': {k: v for k, v in stages['development'].items() if k not in {'records', 'image_signatures', 'components'}},
    'confirmation': {k: v for k, v in stages['confirmation'].items() if k not in {'records', 'image_signatures', 'components'}},
    'development_images': len(stages['development']['image_signatures']),
    'confirmation_images': len(stages['confirmation']['image_signatures']),
}, indent=2))
