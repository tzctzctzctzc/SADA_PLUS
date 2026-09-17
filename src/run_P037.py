"""Score the two matched source interventions for P037 on physical GPU 3."""
import ast
import concurrent.futures
import copy
import functools
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ROOT = Path('/vepfs/vlms/1')
assert Path.cwd().resolve() == ROOT
config_path = Path('configs/P037_position_controlled_source_intervention.json')
config = json.loads(config_path.read_text())
rows = config['records']
for path, expected in config['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
assert os.environ['CUDA_VISIBLE_DEVICES'] == config['gpu_uuid']
assert torch.cuda.device_count() == 1
assert 'A100' in torch.cuda.get_device_name(0)
torch.set_num_threads(8)
torch.manual_seed(17037)
torch.backends.cuda.matmul.allow_tf32 = False

manifest_path = Path('data/PinPoint/P030_image_manifest.json')
manifest = json.loads(manifest_path.read_text())
assert manifest['complete'] and manifest['n_failed'] == 0
images = {entry['signature']: entry for entry in manifest['entries']}
assert set(config['image_signatures']) <= set(images)

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


def swap_placeholders(text):
    assert text.count('<img1>') == text.count('<img2>') == 1
    return text.replace('<img1>', '<source_tmp>').replace('<img2>', '<img1>').replace('<source_tmp>', '<img2>')


def dual_messages(instruction, references, candidate):
    messages = copy.deepcopy(namespace['build_messages'](instruction, references[0], candidate))
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
                    {'type': 'image_url', 'image_url': {'url': references[1]}},
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
    references = [row['references'][1], row['references'][0]]
    instruction = row['instruction']
    if item['condition'] == 'equivalent_reindexed':
        instruction = swap_placeholders(instruction)
    else:
        assert item['condition'] == 'binding_swapped'
    messages = dual_messages(instruction, references, item['candidate'])
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
    assert signatures == [row['references'][1], row['references'][0], item['candidate']]
    return converted, messages, signatures


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
    return dict(batch), details


def to_device(batch):
    return {
        key: value.to('cuda:0', dtype=model.dtype if value.is_floating_point() else None)
        for key, value in batch.items()
    }


def prepare(items):
    batch, details = prepare_cpu(items)
    return to_device(batch), details


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
assert len(tasks) == config['expected_records']
out = Path('runs/P037_position_controlled_source_intervention')
out.mkdir(exist_ok=True)
metadata = {
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
existing_predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()] if predictions_path.exists() else []
done = {row['key'] for row in existing_predictions}
assert len(done) == len(existing_predictions)
assert done <= {task['key'] for task in tasks}

with torch.inference_mode():
    single, single_details = prepare(tasks[:1])
    full_logits = logits_for(single)
    hook = model.lm_head.register_forward_pre_hook(lambda module, arguments: (arguments[0][:, -1:, :],))
    short_logits = logits_for(single)
    batch, batch_details = prepare(tasks[:config['batch_size']])
    batch_logits = logits_for(batch)
    matched_items = [tasks[0], next(task for task in tasks if task['condition'] == 'binding_swapped')]
    matched_batch, matched_details = prepare(matched_items)
    matched_logits = logits_for(matched_batch)
    assert single_details[0] == batch_details[0]
    assert matched_details[0]['image_signatures'] == matched_details[1]['image_signatures']
    assert matched_details[0]['image_grids'] == matched_details[1]['image_grids']
    assert matched_details[0]['image_tokens'] == matched_details[1]['image_tokens']
    assert matched_details[0]['input_tokens'] == matched_details[1]['input_tokens']
    smoke = {
        'full_vs_last_head_max_error': float((full_logits - short_logits).abs().max()),
        'single_vs_batch_max_error_first_example': float((short_logits - batch_logits[:1]).abs().max()),
        'matched_image_signatures_equal': True,
        'matched_image_grids_equal': True,
        'matched_input_tokens_equal': True,
        'matched_logits_finite': bool(torch.isfinite(matched_logits).all()),
    }
    assert max(smoke['full_vs_last_head_max_error'], smoke['single_vs_batch_max_error_first_example']) <= 0.5
    assert smoke['matched_logits_finite']
    (out / 'protocol_smoke.json').write_text(json.dumps(smoke, indent=2) + '\n')
    print('protocol smoke', smoke, flush=True)
    del single, batch, matched_batch, full_logits, short_logits, batch_logits, matched_logits
    runner_sha256 = metadata['runner_sha256']
    item_batches = []
    for start in range(0, len(tasks), config['batch_size']):
        items = [task for task in tasks[start:start + config['batch_size']] if task['key'] not in done]
        if items:
            item_batches.append(items)
    with predictions_path.open('a', buffering=1) as handle:
        for items, batch, details, tick in prepared_batches(item_batches):
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
                    'runner_sha256': runner_sha256,
                    'execution_pipeline': 'cpu_prefetch_v1',
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
    'n_records': len(done),
    'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
}, indent=2) + '\n')
print('P037 scoring complete', len(done), flush=True)
