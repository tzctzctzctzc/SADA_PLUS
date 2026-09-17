"""Freeze the bounded P039 Shadow Binding Bridges mechanism screen."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/vepfs/vlms/1')
assert Path.cwd().resolve() == ROOT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source_paths = [
    'configs/P038_semantically_interleaved_role_cards.json',
    'runs/P038_semantically_interleaved_role_cards/predictions.jsonl',
    'results/P038_semantically_interleaved_role_cards.json',
    'data/PinPoint/P030_image_manifest.json',
]
source = json.loads(Path(source_paths[0]).read_text())
p038_result = json.loads(Path(source_paths[2]).read_text())
assert p038_result['complete']
assert not p038_result['continuation_gate_evaluation']['all_pass']

rows = []
for row in source['records']:
    candidates = sorted(set(row['pilot_positive_candidates']) | set(row['pilot_explicit_negative_candidates']))
    positives = sorted(set(candidates) & set(row['positive_set']))
    negatives = sorted(set(candidates) & set(row['negative_set']))
    assert positives and negatives
    frozen = dict(row)
    frozen['candidates_to_score'] = candidates
    frozen['pilot_positive_candidates'] = positives
    frozen['pilot_explicit_negative_candidates'] = negatives
    rows.append(frozen)

assert len(rows) == 10
assert sum(len(row['candidates_to_score']) for row in rows) == 34
assert sum(len(row['pilot_positive_candidates']) for row in rows) == 18
assert sum(len(row['pilot_explicit_negative_candidates']) for row in rows) == 16

conditions = ['correct_ab', 'equivalent_ba', 'wrong_binding_ba']
interventions = [
    {'id': 'soft', 'layers_zero_based': []},
    {'id': 'bridge_layers_01_07', 'layers_zero_based': list(range(0, 7))},
    {'id': 'bridge_layers_08_14', 'layers_zero_based': list(range(7, 14))},
    {'id': 'bridge_layers_15_21', 'layers_zero_based': list(range(14, 21))},
    {'id': 'bridge_layers_22_28', 'layers_zero_based': list(range(21, 28))},
    {'id': 'bridge_all_layers', 'layers_zero_based': list(range(28))},
]
candidate_windows = [item['id'] for item in interventions if item['id'] not in {'soft', 'bridge_all_layers'}]
image_signatures = sorted({
    signature
    for row in rows
    for signature in row['references'] + row['candidates_to_score']
})

config = {
    'id': 'P039',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'evidence_status': 'prospective_bounded_mechanism_screen_on_observed_P038_development_queries',
    'candidate_method_name': 'Shadow Binding Bridges (SBB)',
    'purpose': (
        'Preserve source-private, role-conditioned high-dimensional evidence in shadow bridge tokens, '
        'then inject only those bridge states into an otherwise untouched joint forward pass.'
    ),
    'failure_model': (
        'P014 showed that isolating the main stream destroys useful global evidence, while P035 showed '
        'that reducing each source to a scalar before composition loses useful joint evidence. P038 '
        'improved order stability but its binding contrast did not discriminate positives. SBB keeps '
        'the global stream and introduces a narrow high-dimensional private path before joint reasoning.'
    ),
    'literature_grounding': [
        {
            'work': 'DreamRenderer (ICCV 2025)',
            'primitive': 'Private bridge-token paths can bind instances while the main stream retains global composition.',
            'transfer_delta': 'Use source-private bridge states for discriminative multi-reference retrieval rather than image generation.',
        },
        {
            'work': 'MultimodalCausalTrace (NeurIPS 2024)',
            'primitive': 'A small layer/token subset can carry causally important visual information.',
            'transfer_delta': 'Screen preregistered layer windows instead of modifying all visual tokens or all layers.',
        },
        {
            'work': 'Mechanisms of Object Localization (CVPR 2026) and Visual Retrieval Heads (arXiv:2608.27417)',
            'primitive': 'Visual grounding is concentrated in a small causal subset of model computation.',
            'transfer_delta': 'Test whether narrow bridge injection improves binding evidence without erasing the joint stream.',
        },
        {
            'work': 'MultiBind (arXiv:2603.21937)',
            'primitive': 'Matched wrong binding separates source confusion from ordinary degradation.',
            'transfer_delta': 'Select a layer window only when correct-minus-wrong binding separates positives from explicit negatives.',
        },
    ],
    'innovation_boundary': [
        'Bridge tokens, masked attention, causal activation intervention, and multi-vector representations are prior art.',
        'The candidate delta is a training-free retrieval intervention that caches source-private role anchors and injects them into an intact joint multi-reference forward pass.',
        'P039 is a selected development mechanism screen. A pass would justify only a prospective component-disjoint P040 pilot, not a novelty or SOTA claim.',
    ],
    'conditions': conditions,
    'interventions': interventions,
    'candidate_windows': candidate_windows,
    'prompt_protocol': {
        'shared_instruction': 'Judge whether the candidate jointly satisfies both source-local requirements.',
        'source_block': 'Each role requirement is placed immediately before its attached reference image.',
        'bridge_marker': '[SOURCE_PRIVATE_EVIDENCE_ANCHOR]',
        'correct_ab': 'Role A with reference A, then Role B with reference B.',
        'equivalent_ba': 'Role B with reference B, then Role A with reference A; semantics unchanged.',
        'wrong_binding_ba': 'Role A with reference B, then Role B with reference A; image order matches equivalent_ba.',
        'shadow_path': 'Block every token after the first bridge from attending to the first reference image or first bridge until the second bridge is formed.',
        'joint_path': 'Keep the original causal mask and replace only bridge-marker hidden states in the preregistered layer window.',
    },
    'model': source['model'],
    'model_revision': source['model_revision'],
    'gpu_uuid': 'GPU-277ce254-c418-8feb-d0eb-e3929d6a6f9f',
    'backend': source['backend'],
    'dtype': source['dtype'],
    'min_pixels': source['min_pixels'],
    'max_pixels': source['max_pixels'],
    'batch_size': 4,
    'n_queries': len(rows),
    'n_reference_components': len({row['reference_component'] for row in rows}),
    'n_candidate_rows': sum(len(row['candidates_to_score']) for row in rows),
    'n_positive_rows': sum(len(row['pilot_positive_candidates']) for row in rows),
    'n_explicit_negative_rows': sum(len(row['pilot_explicit_negative_candidates']) for row in rows),
    'records': rows,
    'image_signatures': image_signatures,
    'expected_base_tasks': sum(len(row['candidates_to_score']) for row in rows) * len(conditions),
    'expected_records': sum(len(row['candidates_to_score']) for row in rows) * len(conditions) * len(interventions),
    'selection_rule': {
        'eligible': 'A non-control window must pass every development gate.',
        'primary': 'Highest binding-contrast macro AUC.',
        'tie_breaks': ['Higher correct-condition score macro AUC.', 'Earlier layer window.'],
        'all_layer_control': 'If bridge_all_layers has binding AUC greater than or equal to the selected narrow window, stop rather than selecting a window.',
    },
    'development_gate': {
        'all_required': True,
        'positive_correct_minus_wrong_binding_mean': 'strictly greater than soft',
        'binding_contrast_macro_auc': 'strictly greater than 0.55 and at least soft + 0.05',
        'correct_condition_score_macro_auc': 'no more than 0.02 below soft',
        'equivalent_order_mean_absolute_margin_gap': 'strictly lower than soft',
        'all_layer_control': 'binding AUC strictly below the selected narrow window',
        'meaning': 'Eligibility to design P040 on frozen, component-disjoint observed data; no full-scale or held-out access.',
    },
    'source_sha256': {path: sha256(path) for path in source_paths},
    'limits': [
        'The same ten observed development queries used by P038 are reused for a mechanism screen.',
        'Only 18 positives and 16 explicit negatives are scored; uncertainty is high and no confirmatory claim is allowed.',
        'Layer windows are coarse and chosen only by the frozen rule.',
        'No reserved or held-out component may be accessed in P039.',
        'Any later full-scale run requires a separate prospective pilot whose frozen gates all pass.',
    ],
}

out = Path('configs/P039_shadow_binding_bridges.json')
out.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({
    'path': str(out),
    'n_queries': config['n_queries'],
    'n_candidate_rows': config['n_candidate_rows'],
    'expected_base_tasks': config['expected_base_tasks'],
    'expected_records': config['expected_records'],
    'gpu_uuid': config['gpu_uuid'],
}, indent=2))
