"""Frozen source-bound predicate scoring on physical GPU 3."""
import functools
import hashlib
import io
import json
import os
import time
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path("/vepfs/vlms/1")
assert Path.cwd().resolve() == ROOT
cfg_path = Path("configs/P035_source_bound_predicates.json")
cfg = json.loads(cfg_path.read_text())
for path, expected in cfg["source_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
cohort = json.loads(Path(cfg["cohort_config"]).read_text())
rows = cohort["records"]
assert len(rows) == 48 and all(row["query_category"] if "query_category" in row else True for row in rows)
assert os.environ["CUDA_VISIBLE_DEVICES"] == cfg["gpu_uuid"]
assert torch.cuda.device_count() == 1
assert "A100" in torch.cuda.get_device_name(0)
torch.set_num_threads(8)
torch.manual_seed(17035)
torch.backends.cuda.matmul.allow_tf32 = False

image_manifest_path = Path("data/PinPoint/P030_image_manifest.json")
image_manifest = json.loads(image_manifest_path.read_text())
assert image_manifest["complete"]
images = {row["signature"]: row for row in image_manifest["entries"]}
model_path = ROOT / "models" / cfg["model"]
assert json.loads((model_path / "source_manifest.json").read_text())["revision"] == cfg["model_revision"]

processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True,
    min_pixels=cfg["min_pixels"],
    max_pixels=cfg["max_pixels"],
)
processor.tokenizer.padding_side = "left"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation=cfg["backend"],
).eval()
token_ids = [
    processor.tokenizer.encode(token, add_special_tokens=False)[0]
    for token in ["True", "False"]
]
assert all(
    len(processor.tokenizer.encode(token, add_special_tokens=False)) == 1
    for token in ["True", "False"]
)

SYSTEM = """You are an expert visual search evaluator. You will judge one source-bound requirement in a multi-reference visual search request. Be strict and answer True only when the candidate clearly satisfies the visual requirement contributed by the specified reference image. Answer False when the requirement is absent, partial, ambiguous, or contradicted."""
INSTRUCTIONS = """Evaluate exactly one source-bound predicate.
1. Read the full request and identify only the visual property, role, style, shape, material, color, or context contributed by the specified reference image.
2. Compare that specified reference image with the candidate.
3. Do not require properties contributed by the other reference image; it is intentionally not shown.
4. Return True only if the candidate clearly satisfies the specified source's contribution.
Answer exactly one word: True or False."""


def messages(row, source_index, candidate):
    source_number = source_index + 1
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": INSTRUCTIONS},
                {
                    "type": "text",
                    "text": (
                        chr(10)
                        + "**Full multi-reference request:** "
                        + row["instruction"]
                        + chr(10)
                        + "**Predicate to judge:** Does the candidate satisfy the visual requirement contributed by Reference Image <img"
                        + str(source_number)
                        + ">?"
                        + chr(10)
                        + "**Specified Reference Image <img"
                        + str(source_number)
                        + ">:**"
                    ),
                },
                {"type": "image"},
                {"type": "text", "text": chr(10) + "**Candidate Image:**"},
                {"type": "image"},
                {
                    "type": "text",
                    "text": chr(10)
                    + "**Decision:** Judge only the specified source-bound predicate. Answer exactly one word: True or False",
                },
            ],
        },
    ]


@functools.lru_cache(maxsize=96)
def image_for(signature):
    entry = images[signature]
    data = (ROOT / entry["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    return Image.open(io.BytesIO(data)).convert("RGB")


def prepare(items):
    texts = []
    flat_images = []
    details = []
    for item in items:
        message = messages(item["row"], item["source_index"], item["candidate"])
        texts.append(
            processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
        )
        signatures = [
            item["row"]["references"][item["source_index"]],
            item["candidate"],
        ]
        flat_images.extend(image_for(signature) for signature in signatures)
        details.append(
            {
                "image_signatures": signatures,
                "payload_sha256": hashlib.sha256(
                    json.dumps(message, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    batch = processor(
        text=texts,
        images=flat_images,
        padding=True,
        return_tensors="pt",
    )
    grids = batch["image_grid_thw"]
    assert len(grids) == 2 * len(items)
    for index in range(len(items)):
        segment = grids[2 * index : 2 * index + 2]
        image_tokens = int((segment.prod(dim=1) // 4).sum())
        assert int(
            (batch["input_ids"][index] == model.config.image_token_id).sum()
        ) == image_tokens
        assert int(batch["attention_mask"][index, -1]) == 1
        details[index].update(
            image_grids=segment.tolist(),
            image_tokens=image_tokens,
            input_tokens=int(batch["attention_mask"][index].sum()),
        )
    return {
        key: value.to(
            "cuda:0", dtype=model.dtype if value.is_floating_point() else None
        )
        for key, value in batch.items()
    }, details


def logits_for(batch):
    return model(**batch, use_cache=False).logits[:, -1, token_ids].float().cpu()


tasks = [
    {
        "row": row,
        "source_index": source_index,
        "condition": "source" + str(source_index + 1),
        "candidate": candidate,
        "key": row["id"]
        + "|source"
        + str(source_index + 1)
        + "|"
        + candidate,
    }
    for row in rows
    for source_index in [0, 1]
    for candidate in row["candidates_to_score"]
]
assert len(tasks) == cfg["expected_records"]

out = Path("runs/P035_source_bound_predicates")
out.mkdir(exist_ok=True)
metadata = {
    "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "image_manifest_sha256": hashlib.sha256(image_manifest_path.read_bytes()).hexdigest(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "model_revision": cfg["model_revision"],
    "token_ids": token_ids,
    "batch_size": cfg["batch_size"],
}
metadata_path = out / "metadata.json"
if metadata_path.exists():
    assert json.loads(metadata_path.read_text()) == metadata
else:
    metadata_path.write_text(json.dumps(metadata, indent=2) + chr(10))

predictions_path = out / "predictions.jsonl"
done = (
    {
        row["key"]
        for row in map(json.loads, predictions_path.read_text().splitlines())
    }
    if predictions_path.exists()
    else set()
)
assert done <= {task["key"] for task in tasks}

with torch.inference_mode():
    single, single_details = prepare(tasks[:1])
    full_logits = logits_for(single)
    hook = model.lm_head.register_forward_pre_hook(
        lambda module, arguments: (arguments[0][:, -1:, :],)
    )
    short_logits = logits_for(single)
    batch, batch_details = prepare(tasks[: cfg["batch_size"]])
    batch_logits = logits_for(batch)
    assert single_details[0] == batch_details[0]
    smoke = {
        "full_vs_last_head_max_error": float((full_logits - short_logits).abs().max()),
        "single_vs_batch_max_error_first_example": float(
            (short_logits - batch_logits[:1]).abs().max()
        ),
        "first_batch_source_conditions": [
            task["condition"] for task in tasks[: cfg["batch_size"]]
        ],
        "first_batch_image_counts": [len(row["image_grids"]) for row in batch_details],
    }
    assert max(
        smoke["full_vs_last_head_max_error"],
        smoke["single_vs_batch_max_error_first_example"],
    ) <= 0.5
    (out / "protocol_smoke.json").write_text(json.dumps(smoke, indent=2) + chr(10))
    print("protocol smoke", smoke, flush=True)
    del single, batch, full_logits, short_logits, batch_logits

    with predictions_path.open("a", buffering=1) as handle:
        for start in range(0, len(tasks), cfg["batch_size"]):
            items = [
                item
                for item in tasks[start : start + cfg["batch_size"]]
                if item["key"] not in done
            ]
            if not items:
                continue
            begin = time.monotonic()
            batch, details = prepare(items)
            values = logits_for(batch)
            seconds = (time.monotonic() - begin) / len(items)
            for item, detail, value in zip(items, details, values.tolist()):
                record = {
                    "key": item["key"],
                    "id": item["row"]["id"],
                    "condition": item["condition"],
                    "candidate": item["candidate"],
                    "true_false_logits": value,
                    "margin": value[0] - value[1],
                    "pipeline_seconds_per_item": seconds,
                    **detail,
                }
                handle.write(json.dumps(record) + chr(10))
                done.add(item["key"])
            del batch, values
            if len(done) % 80 == 0:
                print("completed", len(done), "/", len(tasks), flush=True)
    hook.remove()

assert len(done) == len(tasks)
(out / "summary.json").write_text(
    json.dumps(
        {
            "complete": True,
            "n_records": len(done),
            "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        indent=2,
    )
    + chr(10)
)
print("P035 scoring complete", len(done), flush=True)
