"""原Qwen闭集读出接口的独立子集；8B/32B共用，不引入目标数据或生成式自评。"""


import hashlib
import json
import math
import random
from pathlib import Path


LABELS = (
    "TAKE", "PUT", "WASH", "OPEN", "CLOSE", "INSERT", "TURN_ON",
    "CUT", "TURN_OFF", "POUR",
)


ACTION_PHRASES = (
    "taking or picking up an object",
    "putting or placing an object down",
    "washing or cleaning an object",
    "opening an object",
    "closing an object",
    "inserting an object into a container",
    "turning on a device, light, flame, or flow",
    "cutting or slicing an object",
    "turning off a device, light, flame, or flow",
    "pouring contents from a container",
)


LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


DIGITS = tuple(str(index) for index in range(len(LABELS)))


PHASE_ACTION_DESCRIPTIONS = (
    "the main object starts resting or not held, the hand grasps it, and it ends held or moved away",
    "the main object starts held, the hand places it, and it ends resting or released; ignore opening or closing a container when that is only an enabling sub-action",
    "the object is cleaned through a sustained sequence such as rinsing, rubbing, or washing",
    "the container, door, lid, or barrier itself changes from closed to open; do not choose this when opening is only incidental to moving another object",
    "the container, door, lid, or barrier itself changes from open to closed; do not choose this when closing is only incidental to moving another object",
    "the main object starts outside a slot, opening, or container and ends inserted inside it",
    "a device, light, flame, or flow changes from inactive or off to active or on",
    "a cutting tool repeatedly contacts an object and divides or slices it",
    "a device, light, flame, or flow changes from active or on to inactive or off",
    "a container tilts and its contents visibly flow or are transferred out",
)


CONTRASTIVE_TRANSITION_RULES = (
    "TAKE versus PUT: choose TAKE only for a visible resting-to-held change, and "
    "PUT only for a visible held-to-resting or held-to-released change.",
    "OPEN versus CLOSE: follow the visible state change of the door, lid, or container itself.",
    "TURN_ON versus TURN_OFF: follow the visible inactive-to-active or active-to-inactive change.",
    "Do not relabel INSERT as PUT when an object visibly moves into a slot or container. "
    "Do not relabel POUR as PUT when contents visibly flow from a tilted container.",
)


def digit_to_label(candidate_id):
    """Deterministically shuffle label positions to cancel digit-position bias."""
    seed = int.from_bytes(
        hashlib.sha256(str(candidate_id).encode("utf-8")).digest()[:8], "big"
    )
    labels = list(LABELS)
    random.Random(seed).shuffle(labels)
    return tuple(labels)


def build_prompt(candidate_id, frame_count=8, prompt_mode="baseline",
                 input_mode="images"):
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    mapping = digit_to_label(candidate_id)
    if input_mode not in ("images", "native_video"):
        raise ValueError("unknown input mode: {}".format(input_mode))
    if prompt_mode in ("baseline", "contrastive_transition"):
        phrases = ACTION_PHRASES
    elif prompt_mode == "phase_transition":
        phrases = PHASE_ACTION_DESCRIPTIONS
    else:
        raise ValueError("unknown prompt mode: {}".format(prompt_mode))
    definitions = {label: phrase for label, phrase in zip(LABELS, phrases)}
    options = [
        "{}: {} ({})".format(digit, label, definitions[label])
        for digit, label in zip(DIGITS, mapping)
    ]
    guidance = ""
    subject = (
        "The video above contains {} frames sampled uniformly from one temporal proposal"
        if input_mode == "native_video"
        else "The {} images above are frames sampled uniformly from one temporal proposal"
    ).format(frame_count)
    if prompt_mode == "baseline":
        intro = (
            "{} in chronological order. Classify the single dominant hand-object action. "
            "Choose exactly one option below.\n"
        ).format(subject)
    elif prompt_mode == "phase_transition":
        intro = (
            "{} in chronological order. Infer the dominant goal-level hand-object action from "
            "the change between the early, middle, and late frames. Track the main object's "
            "state before and after the interaction. Ignore visually salient opening, closing, "
            "or hand motions when they are only incidental steps toward moving another object. "
            "Choose the option whose full state transition best matches the sequence.\n"
        ).format(subject)
    else:
        intro = (
            "{} in chronological order. Classify the single dominant hand-object action that "
            "is visibly supported. The proposal may contain only part of an action, so do "
            "not invent an unseen intention or require a complete ideal action sequence. "
            "Use temporal order only to resolve the specific confusing pairs below.\n"
        ).format(subject)
        guidance = "\nTemporal disambiguation rules:\n" + "\n".join(
            "- " + rule for rule in CONTRASTIVE_TRANSITION_RULES
        )
    return (
        intro
        + "\n".join(options)
        + guidance
        + "\nAnswer immediately with exactly one digit from 0 to 9. Do not add "
        "words, punctuation, explanation, or formatting."
    )


def build_messages(paths, prompt, min_pixels, max_pixels):
    content = [{"type": "text", "text": "Chronological proposal frames:"}]
    for index, path in enumerate(paths, 1):
        content.append({"type": "text", "text": "FRAME {}".format(index)})
        content.append({
            "type": "image",
            "image": Path(path).resolve().as_uri(),
            "min_pixels": int(min_pixels),
            "max_pixels": int(max_pixels),
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_native_video_messages(paths, prompt, min_pixels, max_pixels, sample_fps):
    return [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [Path(path).resolve().as_uri() for path in paths],
                "sample_fps": float(sample_fps),
                "min_pixels": int(min_pixels),
                "max_pixels": int(max_pixels),
            },
            {"type": "text", "text": prompt},
        ],
    }]


def resolve_digit_token_ids(tokenizer):
    token_ids = []
    for digit in DIGITS:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                "digit {!r} is not one tokenizer token: {}".format(digit, encoded)
            )
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("digit token IDs are not unique: {}".format(token_ids))
    return tuple(token_ids)


def normalize_digit_logits(digit_logits, mapping, temperature=1.0):
    if temperature <= 0:
        raise ValueError("score temperature must be positive")
    if len(digit_logits) != len(DIGITS) or len(mapping) != len(DIGITS):
        raise ValueError("expected ten digit logits and ten mapped labels")
    scaled = [float(value) / temperature for value in digit_logits]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    digit_probabilities = [value / total for value in weights]
    class_probabilities = [0.0] * len(LABELS)
    class_logits = [0.0] * len(LABELS)
    for digit_index, label in enumerate(mapping):
        class_id = LABEL_TO_ID[label]
        class_probabilities[class_id] = digit_probabilities[digit_index]
        class_logits[class_id] = float(digit_logits[digit_index])
    order = sorted(range(len(LABELS)), key=class_probabilities.__getitem__, reverse=True)
    top1, top2 = order[:2]
    entropy = -sum(
        probability * math.log(max(probability, 1e-12))
        for probability in class_probabilities
    )
    return {
        "prediction": LABELS[top1],
        "class_id": top1,
        "confidence": class_probabilities[top1],
        "margin": class_probabilities[top1] - class_probabilities[top2],
        "entropy": entropy,
        "class_probabilities": class_probabilities,
        "class_logits": class_logits,
        "digit_probabilities": digit_probabilities,
    }


def mixed_frame_grid(paths, min_pixels, max_pixels):
    """混合恢复分辨率使用同一官方网格；不写回、替换或裁剪任何源帧。"""
    from PIL import Image
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    if len(set(sizes)) <= 1:
        return {}
    from qwen_vl_utils.vision_process import smart_resize
    width, height = min(sizes, key=lambda size: size[0] * size[1])
    height, width = smart_resize(height, width, factor=32,
                                 min_pixels=min_pixels, max_pixels=max_pixels)
    return dict(resized_height=height, resized_width=width)


def infer_one(runtime, paths, candidate_id, args, sample_fps=None):
    torch = runtime["torch"]
    prompt_mode = getattr(args, "prompt_mode", "baseline")
    input_mode = getattr(args, "input_mode", "images")
    prompt = build_prompt(candidate_id, args.frame_count, prompt_mode, input_mode)
    mapping = digit_to_label(candidate_id)
    if input_mode == "native_video":
        if sample_fps is None:
            sample_fps = float(args.sample_fps)
        messages = build_native_video_messages(
            paths, prompt, args.min_pixels, args.max_pixels, sample_fps
        )
        if getattr(args, 'uniform_mixed_frame_grid', False):
            messages[0]['content'][0].update(mixed_frame_grid(paths, args.min_pixels, args.max_pixels))
    else:
        messages = build_messages(paths, prompt, args.min_pixels, args.max_pixels)
    chat_template_kwargs = runtime.get("chat_template_kwargs", {})
    text = runtime["processor"].apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs
    )
    if input_mode == "native_video":
        image_inputs, video_inputs, video_kwargs = runtime["process_vision_info"](
            messages, image_patch_size=16, return_video_kwargs=True,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            video_inputs, video_metadata = zip(*video_inputs)
            video_inputs, video_metadata = list(video_inputs), list(video_metadata)
        else:
            video_metadata = None
        inputs = runtime["processor"](
            text=[text], images=image_inputs, videos=video_inputs,
            video_metadata=video_metadata, padding=True, return_tensors="pt",
            do_resize=False, **video_kwargs
        ).to(runtime["device"])
    else:
        image_inputs, video_inputs = runtime["process_vision_info"](
            messages, image_patch_size=16
        )
        inputs = runtime["processor"](
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(runtime["device"])
    with torch.inference_mode():
        output = runtime["model"](
            **inputs, use_cache=False, logits_to_keep=1, return_dict=True
        )
    next_logits = output.logits[0, -1].float()
    token_ids = torch.tensor(
        runtime["digit_token_ids"], device=next_logits.device, dtype=torch.long
    )
    digit_logits_tensor = next_logits.index_select(0, token_ids)
    candidate_log_mass = (
        torch.logsumexp(digit_logits_tensor, dim=0)
        - torch.logsumexp(next_logits, dim=0)
    )
    result = normalize_digit_logits(
        digit_logits_tensor.detach().cpu().tolist(),
        mapping,
        args.score_temperature,
    )
    result.update({
        "candidate_mass": float(candidate_log_mass.exp().clamp(0, 1).cpu()),
        "digit_to_label": list(mapping),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    })
    return result


ORIGINAL_INTERFACE_SHA256 = '811480a0bfb43ee0e030535c4f04b5795dc0c797c810aa125d24e4f00e0203f8'
