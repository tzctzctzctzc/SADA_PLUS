"""Score the frozen joint and source views for one P036 stage on physical GPU 3."""
import ast
import copy
import functools
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path('/vepfs/vlms/1')
assert Path.cwd().resolve() == ROOT
assert len(sys.argv) == 2 and sys.argv[1] in {'development', 'confirmation'}
stage = sys.argv[1]
config_path = Path('configs/P036_risk_controlled_source_veto.json')
config = json.loads(config_path.read_text())
if stage == 'confirmation':
    development_result = Path('results/P036_development.json')
    assert development_result.exists()
    assert json.loads(development_result.read_text())['continuation_gate_evaluation']['all_pass']
stage_config = config['stages'][stage]
rows = stage_config['records']
assert len(rows) == stage_config['n_queries']
for path, expected in config['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
assert os.environ['CUDA_VISIBLE_DEVICES'] == config['gpu_uuid']
assert torch.cuda.device_count() == 1
assert 'A100' in torch.cuda.get_device_name(0)
torch.set_num_threads(8)
torch.manual_seed(17036)
torch.backends.cuda.matmul.allow_tf32 = False

manifest_path = Path(f'data/PinPoint/P036_{stage}_image_manifest.json')
manifest = json.loads(manifest_path.read_text())
assert manifest['complete'] and manifest['stage'] == stage
assert manifest['config_sha256'] == hashlib.sha256(config_path.read_bytes()).hexdigest()
images = {entry['signature']: entry for entry in manifest['entries']}
assert set(images) == set(stage_config['image_signatures'])

source_manifest = json.loads(Path('literature/PinPoint_source_manifest.json').read_text())
official_source = Path('third_party/pinpoint-dataset/rerank.py')
assert hashlib.sha256(official_source.read_bytes()).hexdigest() == next(
    entry['sha256'] for entry in source_manifest['files'] if entry['path'] == 'rerank.py'
)
tree = ast.parse(official_source.read_text())
namespace = {'Dict': Dict, 'List': List}
nodes = [
    node for node in tree.body
    if (isinstance(node, ast.FunctionDef) and node.name == 'build_messages')
    or (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in {'SYSTEM_PROMPT', 'BASE_INSTRUCTIONS'} for target in node.targets)
    )
]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(official_source), 'exec'), namespace)

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
token_ids = [processor.tokenizer.encode(token, add_special_tokens=False) for token in ['True', 'False']]
assert all(len(ids) == 1 for ids in token_ids)
token_ids = [ids[0] for ids in token_ids]

SOURCE_SYSTEM = """You are an expert visual search evaluator. You will judge one source-bound requirement in a multi-reference visual search request. Be strict and answer True only when the candidate clearly satisfies the visual requirement contributed by the specified reference image. Answer False when the requirement is absent, partial, ambiguous, or contradicted."""
SOURCE_INSTRUCTIONS = """Evaluate exactly one source-bound predicate.
1. Read the full request and identify only the visual property, role, style, shape, material, color, or context contributed by the specified reference image.
2. Compare that specified reference image with the candidate.
3. Do not require properties contributed by the other reference image; it is intentionally not shown.
4. Return True only if the candidate clearly satisfies the specified source's contribution.
Answer exactly one word: True or False."""

def source_messages(row, source_index):
    source_number = source_index + 1
    return [
        {'role': 'system', 'content': SOURCE_SYSTEM},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': SOURCE_INSTRUCTIONS},
                {
                    'type': 'text',
                    'text': (
                        '\n**Full multi-reference request:** ' + row['instruction']
                        + '\n**Predicate to judge:** Does the candidate satisfy the visual requirement contributed by Reference Image <img'
                        + str(source_number) + '>?' + '\n**Specified Reference Image <img'
                        + str(source_number) + '>:**'
                    ),
                },
                {'type': 'image'},
                {'type': 'text', 'text': '\n**Candidate Image:**'},
                {'type': 'image'},
                {
                    'type': 'text',
                    'text': '\n**Decision:** Judge only the specified source-bound predicate. Answer exactly one word: True or False',
                },
            ],
        },
    ]

def dual_messages(row, candidate):
    messages = copy.deepcopy(namespace['build_messages'](row['instruction'], row['references'][0], candidate))
    content = []
    ordinal = 0
    for item in messages[1]['content']:
        if item['type'] == 'text':
            item['text'] = item['text'].replace('**Reference Image:**', '**Reference Image <img1>:**')
            content.append(item)
        elif item['type'] == 'image_url':
            if ordinal == 0:
                content.append(item)
                content.extend([
                    {'type': 'text', 'text': '\n**Reference Image <img2>:**\n'},
                    {'type': 'image_url', 'image_url': {'url': row['references'][1]}},
                ])
            else:
                content.extend([{'type': 'text', 'text': '\n**Candidate Image:**\n'}, item])
            ordinal += 1
    assert ordinal == 2
    messages[1]['content'] = content
    return messages

@functools.lru_cache(maxsize=96)
def image_for(signature):
    entry = images[signature]
    data = (ROOT / entry['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
    return Image.open(io.BytesIO(data)).convert('RGB')

def payload(item):
    row = item['row']
    condition = item['condition']
    candidate = item['candidate']
    if condition == 'labelled_dual':
        messages = dual_messages(row, candidate)
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
        assert signatures == [row['references'][0], row['references'][1], candidate]
        return converted, messages, signatures
    source_index = 0 if condition == 'source1' else 1
    messages = source_messages(row, source_index)
    signatures = [row['references'][source_index], candidate]
    return messages, messages, signatures

def prepare(items):
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
        segment = grids[offset:offset + count]
        offset += count
        image_tokens = int((segment.prod(dim=1) // 4).sum())
        assert int((batch['input_ids'][index] == model.config.image_token_id).sum()) == image_tokens
        assert int(batch['attention_mask'][index, -1]) == 1
        details[index].update(
            image_grids=segment.tolist(),
            image_tokens=image_tokens,
            input_tokens=int(batch['attention_mask'][index].sum()),
        )
    return {
        key: value.to('cuda:0', dtype=model.dtype if value.is_floating_point() else None)
        for key, value in batch.items()
    }, details

def logits_for(batch):
    return model(**batch, use_cache=False).logits[:, -1, token_ids].float().cpu()

tasks = [
    {
        'row': row,
        'condition': condition,
        'candidate': candidate,
        'key': row['id'] + '|' + condition + '|' + candidate,
    }
    for row in rows
    for condition in config['conditions']
    for candidate in row['candidates_to_score']
]
assert len(tasks) == stage_config['expected_records']
out = Path(f'runs/P036_{stage}')
out.mkdir(exist_ok=True)
metadata = {
    'stage': stage,
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'image_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'torch': torch.__version__,
    'transformers': transformers.__version__,
    'model_revision': config['model_revision'],
    'token_ids': token_ids,
    'batch_size': config['batch_size'],
}
metadata_path = out / 'metadata.json'
if metadata_path.exists():
    assert json.loads(metadata_path.read_text()) == metadata
else:
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
predictions_path = out / 'predictions.jsonl'
done = {row['key'] for row in map(json.loads, predictions_path.read_text().splitlines())} if predictions_path.exists() else set()
assert done <= {task['key'] for task in tasks}

with torch.inference_mode():
    single, single_details = prepare(tasks[:1])
    full_logits = logits_for(single)
    hook = model.lm_head.register_forward_pre_hook(lambda module, arguments: (arguments[0][:, -1:, :],))
    short_logits = logits_for(single)
    batch, batch_details = prepare(tasks[:config['batch_size']])
    batch_logits = logits_for(batch)
    source_task = next(task for task in tasks if task['condition'] == 'source1')
    source_batch, source_details = prepare([source_task])
    source_logits = logits_for(source_batch)
    assert single_details[0] == batch_details[0]
    smoke = {
        'full_vs_last_head_max_error': float((full_logits - short_logits).abs().max()),
        'single_vs_batch_max_error_first_example': float((short_logits - batch_logits[:1]).abs().max()),
        'joint_image_count': len(single_details[0]['image_grids']),
        'source_image_count': len(source_details[0]['image_grids']),
        'source_logits_finite': bool(torch.isfinite(source_logits).all()),
    }
    assert max(smoke['full_vs_last_head_max_error'], smoke['single_vs_batch_max_error_first_example']) <= 0.5
    assert smoke['joint_image_count'] == 3 and smoke['source_image_count'] == 2 and smoke['source_logits_finite']
    (out / 'protocol_smoke.json').write_text(json.dumps(smoke, indent=2) + '\n')
    print('protocol smoke', smoke, flush=True)
    del single, batch, source_batch, full_logits, short_logits, batch_logits, source_logits
    with predictions_path.open('a', buffering=1) as handle:
        for start in range(0, len(tasks), config['batch_size']):
            items = [task for task in tasks[start:start + config['batch_size']] if task['key'] not in done]
            if not items:
                continue
            tick = time.monotonic()
            batch, details = prepare(items)
            values = logits_for(batch)
            elapsed = time.monotonic() - tick
            for item, detail, value in zip(items, details, values.tolist()):
                assert all(math.isfinite(x) for x in value)
                record = {
                    'key': item['key'],
                    'id': item['row']['id'],
                    'condition': item['condition'],
                    'candidate': item['candidate'],
                    'true_false_logits': value,
                    'margin': value[0] - value[1],
                    'pipeline_seconds_per_item': elapsed / len(items),
                    **detail,
                }
                handle.write(json.dumps(record) + '\n')
                done.add(item['key'])
            del batch, values
            if len(done) % 120 == 0:
                print('completed', len(done), '/', len(tasks), flush=True)
    hook.remove()
assert len(done) == len(tasks)
(out / 'summary.json').write_text(json.dumps({
    'complete': True,
    'stage': stage,
    'n_records': len(done),
    'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
}, indent=2) + '\n')
print('P036', stage, 'scoring complete', len(done), flush=True)
