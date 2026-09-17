"""Run the bounded P039 Shadow Binding Bridges screen on physical GPU 1."""
import concurrent.futures
import functools
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ROOT = Path('/vepfs/vlms/1')
assert Path.cwd().resolve() == ROOT
config_path = Path('configs/P039_shadow_binding_bridges.json')
config = json.loads(config_path.read_text())
rows = config['records']
for path, expected in config['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
assert os.environ['CUDA_VISIBLE_DEVICES'] == config['gpu_uuid']
assert torch.cuda.device_count() == 1
assert 'A100' in torch.cuda.get_device_name(0)
torch.set_num_threads(8)
torch.manual_seed(17039)
torch.backends.cuda.matmul.allow_tf32 = False

manifest_path = Path('data/PinPoint/P030_image_manifest.json')
manifest = json.loads(manifest_path.read_text())
assert manifest['complete'] and manifest['n_failed'] == 0
images = {entry['signature']: entry for entry in manifest['entries']}
assert set(config['image_signatures']) <= set(images)

model_path = ROOT / 'models' / config['model']
assert json.loads((model_path / 'source_manifest.json').read_text())['revision'] == config['model_revision']
processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True,
    min_pixels=config['min_pixels'],
    max_pixels=config['max_pixels'],
)
processor.tokenizer.padding_side = 'left'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map={'': 0},
    attn_implementation=config['backend'],
).eval()
assert len(model.model.layers) == 28
token_ids = [processor.tokenizer.encode(token, add_special_tokens=False) for token in ['True', 'False']]
assert all(len(ids) == 1 for ids in token_ids)
token_ids = [ids[0] for ids in token_ids]
bridge_marker = config['prompt_protocol']['bridge_marker']
# Exclude the final bracket because Qwen merges it with the following newlines;
# the stable marker prefix is identical in both source blocks.
bridge_marker_ids = processor.tokenizer.encode(bridge_marker[:-1], add_special_tokens=False)
assert bridge_marker_ids


def role_messages(row, condition, candidate):
    references = row['references']
    roles = row['role_cards']
    if condition == 'correct_ab':
        blocks = [('A', references[0]), ('B', references[1])]
    elif condition == 'equivalent_ba':
        blocks = [('B', references[1]), ('A', references[0])]
    else:
        assert condition == 'wrong_binding_ba'
        blocks = [('A', references[1]), ('B', references[0])]
    content = [{
        'type': 'text',
        'text': (
            'Judge whether the candidate image jointly satisfies both source-local requirements. '
            'Use each reference only for the role attached to its visual block. The evidence anchor '
            'after a reference closes that source block. Answer True only if both roles are clearly '
            'satisfied.\n'
        ),
    }]
    for role, signature in blocks:
        content.extend([
            {
                'type': 'text',
                'text': f'\nRole {role} requirement: {roles[role]}\nReference for Role {role}:\n',
            },
            {'type': 'image_url', 'image_url': {'url': signature}},
            {'type': 'text', 'text': f'\n{bridge_marker}\n'},
        ])
    content.extend([
        {'type': 'text', 'text': '\nCandidate image:\n'},
        {'type': 'image_url', 'image_url': {'url': candidate}},
        {
            'type': 'text',
            'text': (
                '\nDoes the candidate clearly satisfy both Role A and Role B using the evidence '
                'from their attached references? Answer exactly one word: True or False'
            ),
        },
    ])
    return [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': content},
    ]


@functools.lru_cache(maxsize=96)
def image_for(signature):
    entry = images[signature]
    data = (ROOT / entry['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
    return Image.open(io.BytesIO(data)).convert('RGB')


def payload(item):
    row = item['row']
    messages = role_messages(row, item['condition'], item['candidate'])
    signatures = []
    converted = []
    for message in messages:
        if isinstance(message['content'], str):
            converted.append(message)
            continue
        content = []
        for part in message['content']:
            if part['type'] == 'image_url':
                signatures.append(part['image_url']['url'])
                content.append({'type': 'image'})
            else:
                content.append(part)
        converted.append({'role': message['role'], 'content': content})
    if item['condition'] == 'correct_ab':
        assert signatures == [row['references'][0], row['references'][1], item['candidate']]
    else:
        assert signatures == [row['references'][1], row['references'][0], item['candidate']]
    return converted, messages, signatures


def subsequences(values, pattern):
    n = len(pattern)
    return [index for index in range(len(values) - n + 1) if values[index:index + n] == pattern]


def prepare_cpu(items):
    texts = []
    flat_images = []
    counts = []
    details = []
    for item in items:
        messages, hash_payload, signatures = payload(item)
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        flat_images.extend(image_for(signature) for signature in signatures)
        counts.append(len(signatures))
        details.append({
            'image_signatures': signatures,
            'payload_sha256': hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode()).hexdigest(),
        })
    batch = processor(text=texts, images=flat_images, padding=True, return_tensors='pt')
    grids = batch['image_grid_thw']
    assert len(grids) == sum(counts)
    offset = 0
    for index, count in enumerate(counts):
        assert count == 3
        segment = grids[offset:offset + count]
        offset += count
        image_tokens = int((segment.prod(dim=1) // 4).sum())
        ids = batch['input_ids'][index].tolist()
        starts = [i for i, token in enumerate(ids) if token == model.config.vision_start_token_id]
        ends = [i for i, token in enumerate(ids) if token == model.config.vision_end_token_id]
        anchors = subsequences(ids, bridge_marker_ids)
        assert len(starts) == len(ends) == 3
        assert len(anchors) == 2
        anchor_spans = [[start, start + len(bridge_marker_ids)] for start in anchors]
        assert ends[0] < anchor_spans[0][0] < anchor_spans[0][1] <= starts[1]
        assert ends[1] < anchor_spans[1][0] < anchor_spans[1][1] <= starts[2]
        assert int((batch['input_ids'][index] == model.config.image_token_id).sum()) == image_tokens
        assert int(batch['attention_mask'][index, -1]) == 1
        details[index].update(
            image_grids=segment.tolist(),
            image_tokens=image_tokens,
            input_tokens=int(batch['attention_mask'][index].sum()),
            image_starts=starts,
            image_ends=ends,
            bridge_spans=anchor_spans,
            bridge_token_count=2 * len(bridge_marker_ids),
        )
    return dict(batch), details


def to_device(batch):
    return {
        key: value.to('cuda:0', dtype=model.dtype if value.is_floating_point() else None)
        for key, value in batch.items()
    }


def prepared_batches(item_batches):
    if not item_batches:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(prepare_cpu, item_batches[0])
        for index, items in enumerate(item_batches):
            tick = time.monotonic()
            cpu_batch, details = future.result()
            if index + 1 < len(item_batches):
                future = pool.submit(prepare_cpu, item_batches[index + 1])
            yield items, to_device(cpu_batch), details, tick


def bridge_positions(details):
    return [
        [position for start, end in detail['bridge_spans'] for position in range(start, end)]
        for detail in details
    ]


def replace_hidden(args, kwargs, hidden):
    if 'hidden_states' in kwargs:
        updated = dict(kwargs)
        updated['hidden_states'] = hidden
        return args, updated
    updated = list(args)
    updated[0] = hidden
    return tuple(updated), kwargs


def states_for(batch, details, isolate):
    positions = bridge_positions(details)
    length = batch['input_ids'].shape[1]
    captures = [None] * len(model.model.layers)
    handles = []
    edited_masks = [None] * len(model.model.layers)

    def make_hook(layer_index):
        def hook(module, args, kwargs):
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            if hidden.shape[1] != length:
                return None
            captures[layer_index] = torch.stack([
                hidden[batch_index, row_positions].detach().clone()
                for batch_index, row_positions in enumerate(positions)
            ])
            if not isolate:
                return None
            mask = kwargs.get('attention_mask')
            if edited_masks[layer_index] is None:
                if mask is None:
                    floor = torch.finfo(hidden.dtype).min
                    mask = torch.full(
                        (hidden.shape[0], 1, length, length),
                        floor,
                        device=hidden.device,
                        dtype=hidden.dtype,
                    ).triu(1)
                    valid_keys = batch['attention_mask'].to(device=hidden.device, dtype=torch.bool)
                    mask = mask.masked_fill(~valid_keys[:, None, None, :], floor)
                assert mask.ndim == 4
                edited = mask.clone()
                floor = torch.finfo(hidden.dtype).min
                for batch_index, detail in enumerate(details):
                    first_anchor_end = detail['bridge_spans'][0][1]
                    second_anchor_end = detail['bridge_spans'][1][1]
                    first_image_start = detail['image_starts'][0]
                    edited[
                        batch_index,
                        :,
                        first_anchor_end:second_anchor_end,
                        first_image_start:first_anchor_end,
                    ] = floor
                edited_masks[layer_index] = edited
            updated = dict(kwargs)
            updated['attention_mask'] = edited_masks[layer_index]
            return args, updated
        return hook

    try:
        for index, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_pre_hook(make_hook(index), with_kwargs=True))
        model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert all(state is not None for state in captures)
    assert all(torch.isfinite(state).all() for state in captures)
    return captures


def logits_for(batch, injected_states=None, layers=()):
    if not layers:
        return model(**batch, use_cache=False).logits[:, -1, token_ids].float().cpu()
    assert injected_states is not None
    positions = None
    handles = []

    def make_hook(layer_index):
        def hook(module, args, kwargs):
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            updated_hidden = hidden.clone()
            for batch_index, row_positions in enumerate(positions):
                updated_hidden[batch_index, row_positions] = injected_states[layer_index][batch_index]
            return replace_hidden(args, kwargs, updated_hidden)
        return hook

    # positions are identical to the order stored in the injected tensors.
    positions = current_bridge_positions
    try:
        for index in layers:
            handles.append(model.model.layers[index].register_forward_pre_hook(make_hook(index), with_kwargs=True))
        return model(**batch, use_cache=False).logits[:, -1, token_ids].float().cpu()
    finally:
        for handle in handles:
            handle.remove()


base_tasks = [
    {
        'row': row,
        'condition': condition,
        'candidate': candidate,
        'base_key': row['id'] + '|' + condition + '|' + candidate,
    }
    for row in rows
    for condition in config['conditions']
    for candidate in row['candidates_to_score']
]
assert len(base_tasks) == config['expected_base_tasks']
interventions = {item['id']: item['layers_zero_based'] for item in config['interventions']}

out = Path('runs/P039_shadow_binding_bridges')
out.mkdir(exist_ok=True)
metadata = {
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'image_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'torch': torch.__version__,
    'transformers': transformers.__version__,
    'model_revision': config['model_revision'],
    'token_ids': token_ids,
    'bridge_marker_ids': bridge_marker_ids,
    'batch_size': config['batch_size'],
}
metadata_path = out / 'metadata.json'
if metadata_path.exists():
    assert json.loads(metadata_path.read_text()) == metadata
else:
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
predictions_path = out / 'predictions.jsonl'
existing_predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()] if predictions_path.exists() else []
done = {row['key'] for row in existing_predictions}
assert len(done) == len(existing_predictions)
expected_keys = {
    task['base_key'] + '|' + intervention
    for task in base_tasks
    for intervention in interventions
}
assert done <= expected_keys
pending = [
    task for task in base_tasks
    if any(task['base_key'] + '|' + intervention not in done for intervention in interventions)
]
item_batches = [pending[start:start + config['batch_size']] for start in range(0, len(pending), config['batch_size'])]

current_bridge_positions = None
smoke_written = (out / 'protocol_smoke.json').exists()
with torch.inference_mode():
    lm_head_hook = model.lm_head.register_forward_pre_hook(lambda module, arguments: (arguments[0][:, -1:, :],))
    try:
        with predictions_path.open('a', buffering=1) as handle:
            for items, batch, details, tick in prepared_batches(item_batches):
                current_bridge_positions = bridge_positions(details)
                soft_logits = logits_for(batch)
                if not smoke_written:
                    ordinary_states = states_for(batch, details, isolate=False)
                    identity_logits = logits_for(batch, ordinary_states, tuple(range(28)))
                    identity_error = float((soft_logits - identity_logits).abs().max())
                    del identity_logits
                shadow_states = states_for(batch, details, isolate=True)
                if not smoke_written:
                    state_delta = float(torch.stack([
                        (ordinary - shadow).float().abs().mean()
                        for ordinary, shadow in zip(ordinary_states[1:], shadow_states[1:])
                    ]).mean())
                    smoke = {
                        'identity_replay_max_logit_error': identity_error,
                        'shadow_bridge_mean_absolute_state_delta': state_delta,
                        'shadow_states_finite': all(bool(torch.isfinite(state).all()) for state in shadow_states),
                        'bridge_tokens_per_example': len(current_bridge_positions[0]),
                        'image_count_per_example': len(details[0]['image_starts']),
                    }
                    assert identity_error <= 0.5
                    assert state_delta > 0
                    assert smoke['shadow_states_finite']
                    (out / 'protocol_smoke.json').write_text(json.dumps(smoke, indent=2) + '\n')
                    print('protocol smoke', smoke, flush=True)
                    smoke_written = True
                    del ordinary_states

                all_logits = {'soft': soft_logits}
                for intervention, layers in interventions.items():
                    if intervention == 'soft':
                        continue
                    all_logits[intervention] = logits_for(batch, shadow_states, tuple(layers))
                elapsed = time.monotonic() - tick
                for batch_index, (item, detail) in enumerate(zip(items, details)):
                    for intervention, values in all_logits.items():
                        key = item['base_key'] + '|' + intervention
                        if key in done:
                            continue
                        value = values[batch_index].tolist()
                        assert all(math.isfinite(x) for x in value)
                        record = {
                            'key': key,
                            'id': item['row']['id'],
                            'reference_component': item['row']['reference_component'],
                            'condition': item['condition'],
                            'candidate': item['candidate'],
                            'intervention': intervention,
                            'layers_zero_based': interventions[intervention],
                            'true_false_logits': value,
                            'margin': value[0] - value[1],
                            'pipeline_seconds_per_base_task': elapsed / len(items),
                            'runner_sha256': metadata['runner_sha256'],
                            'execution_pipeline': 'shared_shadow_then_layer_window_injection_v1',
                            **detail,
                        }
                        handle.write(json.dumps(record) + '\n')
                        done.add(key)
                del batch, soft_logits, shadow_states, all_logits
                if len(done) % 102 == 0:
                    print('completed', len(done), '/', len(expected_keys), flush=True)
    finally:
        lm_head_hook.remove()

assert len(done) == len(expected_keys) == config['expected_records']
(out / 'summary.json').write_text(json.dumps({
    'complete': True,
    'n_records': len(done),
    'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
}, indent=2) + '\n')
print('P039 scoring complete', len(done), flush=True)
