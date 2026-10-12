"""额外官方生成式教师；固定权重身份，沿用Qwen的候选盲十类读出。"""
import hashlib
import json
from pathlib import Path
import tempfile
import types

import numpy as np
from PIL import Image

EXTRA = {
    'timelens2_8b': {
        'repo': 'MCG-NJU/TimeLens2-8B',
        'revision': 'afb728d176c1ab919b3ee12e581c1620cbdfa5c0',
        'weights': {
            'model-00001-of-00004.safetensors': '8c9b8f3693764451f19205a5903e86d7adcbad47b0606292b27e462ecdfd2901',
            'model-00002-of-00004.safetensors': '268eb57412e2390592e552f7aa89cb55a7b933ce5c256941535552143baf0471',
            'model-00003-of-00004.safetensors': 'ebbd8974bfd0c7049007d1b595b20192c41596d094ebeba9c49dc3f22872f4ba',
            'model-00004-of-00004.safetensors': 'd7487cf611578016a95403978efba3c62040da5a90c6383e88a836f79f14e4e5',
        },
    },
    'timeple_8b': {
        'repo': 'KlingTeam/TimePLE',
        'revision': 'f583ef50ea9fc3498d537ff14438af25d5613d38',
        'weights': {
            'model-00001-of-00004.safetensors': '0ebdf1905d9e72a95510362a90f1a07ee9c4f3cf59b43d913898af0745fc72b8',
            'model-00002-of-00004.safetensors': '73dacaf181b9e0f094459f46ff4bbf72153ca4b9c18372fca453772a82c0a58b',
            'model-00003-of-00004.safetensors': 'a8a6c25fcbfcf1cfdca0cc20b026e3a3dc2ff8476395e7a3768c49e1aaefbb1a',
            'model-00004-of-00004.safetensors': '4be0c7653dac89633c511e9436565b84c4724fc2feac634e2fcb626755f94759',
        },
    },
    'videochat3_4b': {
        'repo': 'MCG-NJU/VideoChat3-4B',
        'revision': '37fa901ec5913f84bc31108ebc1e60ad1903634c',
        'weights': {
            'model-0001-others-save_rank0.safetensors': '4c173466ec6a2c5e1544859021fbeac825067ccd54e80b887372d83b1c281654',
            'model-0002-others-save_rank0.safetensors': '322af91f1ba81f290cee20d66bf0841be4826091e6e942688243b38bbc26e597',
            'model-0003-others-save_rank0.safetensors': '1c16eb013f0224c018db06148e28ee60bc1e2b47b1fffde0e311ed9debd63b90',
        },
    },
    'moss_vl_instruct_0708': {
        'repo': 'OpenMOSS-Team/MOSS-VL-Instruct-0708',
        'revision': '26db199e60a15bbb4ef61831906c0e4a46f547c0',
        'weights': {
            'model-00001-of-00005.safetensors': 'f5f1f72dd4bfacc389e9d560764230e07f8467aefbbe9f511ddf771231b177f9',
            'model-00002-of-00005.safetensors': 'ce762b47f03b4c1d5e8eaceb0939da1bc731122a975c69b21cd3c44b73320beb',
            'model-00003-of-00005.safetensors': 'f5160ec8be44b74fccfc8486e6fc8b2eac880d0b2e90da27f6fe839a62cf3ccb',
            'model-00004-of-00005.safetensors': '3dbf131bf99837ba8c3a23a6f8bdd178aa0fd5a0451a8f0f725d782472b2e7d6',
            'model-00005-of-00005.safetensors': 'e2dcab669fb66feeca21b3f83cdc7ceba5ffa6918c5abdc60ccc42494a490fc2',
        },
    },
}


def frozen_protocol(model):
    spec = EXTRA[model]
    payload = dict(schema='rq3-official-teacher-anchor-v1', model_name=model,
                   repository=spec['repo'], revision=spec['revision'], checkpoint_sha256=spec['weights'])
    payload['protocol_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return payload


def model_path(model):
    spec = EXTRA[model]
    return Path('/vepfs/vlms/cache/huggingface') / ('models--' + spec['repo'].replace('/', '--')) / 'snapshots' / spec['revision']


def uniform_frames(frames):
    """仅混合尺寸统一到最小帧网格；不裁剪、不换帧，原缓存不写回。"""
    shapes = {frame.shape[:2] for frame in frames}
    if len(shapes) == 1:
        return frames
    height, width = min(shapes, key=lambda shape: shape[0] * shape[1])
    return [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BICUBIC)) for frame in frames]


def write_lossless_video(frames, path, sample_fps):
    import av
    from fractions import Fraction
    rate = Fraction(sample_fps).limit_denominator(100000)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=rate)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = 'bgr0'
        for pixels in frames:
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        decoded = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    if len(decoded) != len(frames) or any(not np.array_equal(a, b) for a, b in zip(decoded, frames)):
        raise ValueError('MOSS lossless video transport changed candidate pixels')


def load_extra_teacher(model, settings):
    if model in ('timelens2_8b', 'timeple_8b'):
        from tools.rq3_temporal_teachers import load_temporal_teacher
        return load_temporal_teacher(model, settings)
    import torch
    from tools.run_qwen3_vl_classification import build_prompt, digit_to_label, resolve_digit_token_ids, normalize_digit_logits
    from tools.rq3_gpu import guard_before_cuda
    args = types.SimpleNamespace(model_path=settings['model_path'], offline=True, dtype='bfloat16',
                                 vision_attention='sdpa', pre_cuda_guard=guard_before_cuda)
    # CPU加载完成后，模型构造器第一次迁移CUDA之前复核锁内选中的物理卡。
    if model == 'videochat3_4b':
        from tools.extract_rq1_videochat3_features import load_videochat3
        runtime = load_videochat3(args)
    else:
        from tools.extract_rq1_moss_vl_features import load_moss_vl
        runtime = load_moss_vl(args)
    processor, network = runtime['processor'], runtime['model']
    token_ids = resolve_digit_token_ids(processor.tokenizer)

    def score(frames, paths, row):
        frames = uniform_frames(frames)
        duration = float(row['proposal_end_sec']) - float(row['proposal_start_sec'])
        if duration <= 0:
            raise ValueError('invalid candidate duration')
        sample_fps = 8.0 / duration
        prompt = build_prompt(row['candidate_id'], 8, 'baseline', 'native_video')
        if model == 'videochat3_4b':
            video = np.stack(frames)
            messages = [{'role': 'user', 'content': [{'type': 'video', 'video': 'candidate.mp4'}, {'type': 'text', 'text': prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            metadata = dict(total_num_frames=8, fps=sample_fps, duration=duration, height=video.shape[1], width=video.shape[2])
            inputs = processor(text=[text], videos=[video], video_metadata=[metadata], num_frames=8, fps=None,
                               do_resize=True, size=dict(shortest_edge=224*224, longest_edge=448*448),
                               padding=False, return_tensors='pt').to(runtime['device'])
            extra = {}
        else:
            with tempfile.TemporaryDirectory(prefix='rq3_moss_lossless_') as directory:
                video_path = Path(directory) / 'candidate.mkv'
                write_lossless_video(frames, video_path, sample_fps)
                query = dict(messages=[{'role': 'user', 'content': [{'type': 'video', 'video': str(video_path)}, {'type': 'text', 'text': prompt}]}],
                             media_kwargs=dict(min_pixels=224*224, max_pixels=448*448, video_max_pixels=8*448*448,
                                               video_fps=sample_fps, min_frames=8, max_frames=8))
                prepared = network.offline_prepare_query_cpu(processor, query, padding=False)
            inputs = network._offline_move_inputs_to_devices(prepared['inputs_cpu'])
            extra = dict(vision_chunked_length=64)
        with torch.inference_mode():
            output = network(**inputs, use_cache=False, logits_to_keep=1, return_dict=True, **extra)
        logits = output.logits[0, -1].float()
        selected = logits.index_select(0, torch.tensor(token_ids, device=logits.device)).cpu().tolist()
        return normalize_digit_logits(selected, digit_to_label(row['candidate_id']))['class_probabilities']
    return score
