"""Freeze the bounded P038 semantic-role interleaving pilot."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/vepfs/vlms/1')
assert Path.cwd().resolve() == ROOT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source_paths = [
    'configs/P037_position_controlled_source_intervention.json',
    'runs/P030_pinpoint_references/predictions.jsonl',
    'runs/P037_position_controlled_source_intervention/details.json',
    'data/PinPoint/P030_image_manifest.json',
]
source = json.loads(Path(source_paths[0]).read_text())
details = json.loads(Path(source_paths[2]).read_text())
details_by_id = {row['id']: row for row in details}

# These source-local roles use only the released instruction. They are frozen
# before any P038 model score and do not use candidate labels or images.
roles = {
    '05098': {
        'A': 'The candidate should preserve the proportions and smart-casual structure shown by this reference.',
        'B': 'The candidate should use the night-sky color palette shown by this reference.',
    },
    '05885': {
        'A': 'The candidate should incorporate natural gray pebbles like those shown by this reference.',
        'B': 'The candidate should preserve the clean minimal silhouette and elegance of the vases shown by this reference.',
    },
    '05161': {
        'A': 'The candidate should use the green wood appearance shown by this reference.',
        'B': 'The candidate should use the terracotta appearance shown by this reference.',
    },
    '05921': {
        'A': 'The candidate should depict a natural outdoor ceremony setting resembling this reference.',
        'B': 'The candidate should carry the pink and coral tones shown by this reference through flowers, fabrics, or accents.',
    },
    '05555': {
        'A': 'The candidate should use wood accents like those shown by this reference.',
        'B': 'The candidate should preserve the traditional brick-house type and structure shown by this reference.',
    },
    '05697': {
        'A': 'The candidate should use the chunky-knit texture shown by this reference.',
        'B': 'The candidate should preserve the oversized zip cut shown by this reference.',
    },
    '05432': {
        'A': 'The candidate should include the subtle pearl-and-stone shimmer shown by this reference.',
        'B': 'The candidate should include the ivory flowers and gold foliage shown by this reference.',
    },
    '05968': {
        'A': 'The candidate should preserve the flowing ruffled maxi silhouette shown by this reference.',
        'B': 'The candidate should use the sunset color tones shown by this reference.',
    },
    '06100': {
        'A': 'The candidate should preserve the striped pattern and button-up silhouette shown by this reference.',
        'B': 'The candidate should use the soft muted sage-green tones shown by this reference.',
    },
    '05370': {
        'A': 'The candidate should preserve the cut shown by this reference.',
        'B': 'The candidate should use the beige color shown by this reference.',
    },
}

rows = []
for row in source['records']:
    positives = set(row['candidates_to_score']) & set(row['positive_set'])
    negatives = set(row['candidates_to_score']) & set(row['negative_set'])
    if not (positives and negatives):
        continue
    assert row['id'] in roles
    frozen = dict(row)
    frozen['role_cards'] = roles[row['id']]
    frozen['pilot_positive_candidates'] = sorted(positives)
    frozen['pilot_explicit_negative_candidates'] = sorted(negatives)
    rows.append(frozen)

assert len(rows) == len(roles) == 10
assert all(len(row['candidates_to_score']) == 20 for row in rows)

equivalence_differences = []
for row in rows:
    d = details_by_id[row['id']]
    equivalence_differences.extend(
        abs(float(a) - float(b)) for a, b in zip(d['original'], d['equivalent'])
    )
reference_gap = sum(equivalence_differences) / len(equivalence_differences)

image_signatures = sorted({
    signature
    for row in rows
    for signature in row['references'] + row['candidates_to_score']
})

config = {
    'id': 'P038',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'evidence_status': 'prospective_bounded_pilot_on_observed_P030_development_queries',
    'candidate_method_name': 'Semantically Interleaved Role Cards (SIRC)',
    'purpose': (
        'Transfer explicit reference-token placement from multi-reference video editing '
        'to multi-image composed retrieval: bind each reference directly to its source-local '
        'semantic role before joint candidate verification.'
    ),
    'literature_grounding': [
        {
            'work': 'ReBind (arXiv:2607.14681)',
            'primitive': 'Place reference tokens at their semantic positions to make reference relationships explicit.',
            'transfer_delta': 'Use source-local role cards for multi-image retrieval scoring and test block-order invariance plus binding swaps.',
        },
        {
            'work': 'Janossy Pooling (ICLR 2019)',
            'primitive': 'Average or compare a sequence model over permutations to obtain set-level invariance.',
            'transfer_delta': 'Treat jointly relabelled reference-role blocks as a semantic orbit rather than raw independent images.',
        },
        {
            'work': 'MultiBind (arXiv:2603.21937)',
            'primitive': 'Separate self-degradation from off-diagonal cross-source confusion.',
            'transfer_delta': 'Measure semantic-equivalence gap separately from correct-versus-swapped binding contrast.',
        },
    ],
    'innovation_boundary': [
        'Generic position-bias mitigation, constraint factorization, local matching, and MLLM reranking are prior art.',
        'The candidate delta is the paired retrieval formulation: source-local semantic interleaving, a joint-relabeling orbit test, and a matched wrong-binding intervention.',
        'A positive pilot would justify only a larger held-out design; it would not establish novelty or SOTA.',
    ],
    'conditions': ['sirc_ab', 'sirc_ba_equivalent', 'sirc_ba_wrong_binding'],
    'prompt_protocol': {
        'role_prefix': 'List the two source-local requirements once before the visual blocks.',
        'sirc_ab': 'Role A text with reference A, followed by Role B text with reference B.',
        'sirc_ba_equivalent': 'Role B text with reference B, followed by Role A text with reference A; semantics unchanged.',
        'sirc_ba_wrong_binding': 'Role A text with reference B, followed by Role B text with reference A; image order matches the equivalent condition but the binding is wrong.',
        'decision': 'Score True versus False for whether the candidate jointly satisfies both source-local roles.',
    },
    'model': source['model'],
    'model_revision': source['model_revision'],
    'gpu_uuid': source['gpu_uuid'],
    'backend': source['backend'],
    'dtype': source['dtype'],
    'min_pixels': source['min_pixels'],
    'max_pixels': source['max_pixels'],
    'batch_size': 4,
    'topk_rerank': 20,
    'n_queries': len(rows),
    'n_reference_components': len({row['reference_component'] for row in rows}),
    'n_positive_top20': sum(len(row['pilot_positive_candidates']) for row in rows),
    'n_negative_top20': sum(len(row['pilot_explicit_negative_candidates']) for row in rows),
    'records': rows,
    'image_signatures': image_signatures,
    'expected_records': len(rows) * 3 * 20,
    'reference_equivalence_gap': {
        'source': 'P037 original-dual versus semantically equivalent reindexing on the same ten queries',
        'mean_absolute_margin_gap': reference_gap,
        'n_candidate_pairs': len(equivalence_differences),
    },
    'development_gate': {
        'all_required': True,
        'sirc_ab_minus_original_dual_ap10': 'strictly > 0',
        'sirc_ab_minus_original_dual_explicit_negative_precision_at10': 'less than or equal to 0',
        'sirc_equivalence_mean_absolute_gap': f'strictly < {reference_gap}',
        'positive_candidate_correct_minus_wrong_binding_mean': 'strictly > 0',
        'binding_contrast_macro_positive_vs_explicit_negative_auc': 'strictly > 0.5',
        'meaning': 'Eligibility to design a component-disjoint held-out pilot; not significance, novelty, or SOTA evidence.',
    },
    'source_sha256': {path: sha256(path) for path in source_paths},
    'limits': [
        'Only ten previously observed Composition queries have both a positive and an explicit negative in the frozen top20.',
        'Role cards are human-written from released instructions and may introduce authoring bias.',
        'The pilot is a selected top20 diagnostic, not a full-corpus retrieval evaluation.',
        'No confirmation component may be accessed unless every frozen gate passes.',
    ],
}

out = Path('configs/P038_semantically_interleaved_role_cards.json')
out.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({
    'path': str(out),
    'n_queries': config['n_queries'],
    'expected_records': config['expected_records'],
    'n_positive_top20': config['n_positive_top20'],
    'n_negative_top20': config['n_negative_top20'],
    'reference_mean_absolute_gap': reference_gap,
}, indent=2))
