"""Freeze a source-binding intervention probe on the already observed P030 cohort."""
import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path('/vepfs/vlms/1')
assert Path.cwd().resolve() == ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


p030_path = Path('configs/P030_pinpoint_references.json')
p030_scores_path = Path('runs/P030_pinpoint_references/predictions.jsonl')
p030_result_path = Path('results/P030_pinpoint_references.json')
p030_manifest_path = Path('data/PinPoint/P030_image_manifest.json')
p030 = json.loads(p030_path.read_text())
p030_result = json.loads(p030_result_path.read_text())
manifest = json.loads(p030_manifest_path.read_text())
assert p030_result['complete']
assert sha(p030_scores_path) == p030_result['score_file_sha256']
assert manifest['complete'] and manifest['n_failed'] == 0

records = p030['records']
assert len(records) == 48
assert all(row['instruction'].count('<img1>') == 1 for row in records)
assert all(row['instruction'].count('<img2>') == 1 for row in records)
assert all(row['references'][0] != row['references'][1] for row in records)
assert all(len(row['candidates_to_score']) == 20 for row in records)

positive_top20 = sum(
    len(set(row['positive_set']) & set(row['candidates_to_score'])) for row in records
)
negative_top20 = sum(
    len(set(row['negative_set']) & set(row['candidates_to_score'])) for row in records
)
assert positive_top20 == 21 and negative_top20 == 68

config = {
    'id': 'P037',
    'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'evidence_status': 'prospective_exploratory_probe_on_observed_P030_cohort',
    'candidate_method_name': 'Position-Controlled Source Intervention (PCSI)',
    'purpose': (
        'Test whether a multi-reference VLM binds each visual source to the textual role that names it, '
        'using a matched intervention that changes only img1/img2 token assignments.'
    ),
    'innovation_boundary': [
        'Generic MLLM reranking, constraint verification, and calibrated risk control are prior art.',
        'This probe tests source-role binding through a matched counterfactual with identical images, image order, and prompt length.',
        'A positive pilot supports a later held-out source-interventional reranker; it does not establish novelty or generalization.',
    ],
    'method': {
        'original': 'P030 labelled_dual margin with references [A,B] and the original instruction.',
        'equivalent_reindexed': (
            'Display references [B,A] and swap <img1>/<img2> in the instruction. '
            'This preserves the requested semantics while changing reference position.'
        ),
        'binding_swapped': (
            'Display references [B,A] and keep the original instruction. '
            'This changes source-role binding while matching equivalent_reindexed in images, order, layout, and length.'
        ),
        'binding_contrast': 'margin(equivalent_reindexed) - margin(binding_swapped)',
        'pcsi_score': 'margin(original) + binding_contrast',
        'weights_or_thresholds_tuned': False,
    },
    'conditions': ['equivalent_reindexed', 'binding_swapped'],
    'model': p030['model'],
    'model_revision': p030['model_revision'],
    'gpu_uuid': p030['gpu_uuid'],
    'backend': p030['backend'],
    'dtype': p030['dtype'],
    'min_pixels': p030['min_pixels'],
    'max_pixels': p030['max_pixels'],
    'batch_size': p030['batch_size'],
    'topk_rerank': 20,
    'n_queries': len(records),
    'n_components': len({row['reference_component'] for row in records}),
    'n_positive_top20': positive_top20,
    'n_negative_top20': negative_top20,
    'records': records,
    'image_signatures': p030['image_signatures'],
    'expected_records': len(records) * 20 * 2,
    'development_gate': {
        'all_required': True,
        'positive_candidate_mean_binding_contrast': 'strictly > 0',
        'binding_contrast_macro_positive_vs_explicit_negative_auc': 'strictly > 0.5',
        'pcsi_minus_original_ap10': 'strictly > 0',
        'pcsi_minus_original_explicit_negative_precision_at10': 'less than or equal to 0',
        'meaning': (
            'Eligibility to design a larger component-disjoint held-out test. '
            'This exploratory gate is not significance, novelty, or SOTA evidence.'
        ),
    },
    'source_sha256': {
        str(p030_path): sha(p030_path),
        str(p030_scores_path): sha(p030_scores_path),
        str(p030_result_path): sha(p030_result_path),
        str(p030_manifest_path): sha(p030_manifest_path),
    },
    'limits': [
        'P030 was previously observed and contains only 21 positive and 68 explicit-negative top20 candidates.',
        'The probe uses one query per reference component and is exploratory.',
        'Swapping source roles can leave semantics unchanged when the two referenced attributes overlap.',
        'Only a prospectively frozen component-disjoint follow-up can support a general method claim.',
    ],
}

path = Path('configs/P037_position_controlled_source_intervention.json')
assert not path.exists()
path.write_text(json.dumps(config, indent=2) + '\n')
print(path, sha(path), config['expected_records'])
