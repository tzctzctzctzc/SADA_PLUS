"""定位专长模型的RQ3类别接口；原生完整模型十类logits，不使用生成端点。"""
import os
from pathlib import Path
import subprocess
import sys
import types

TEACHERS = ('timelens2_8b', 'timeple_8b')
TIMEPLE_REVISION = '05728f40bb34478811bcf86532e38e6499ce4462'


def official_timeple_root(settings):
    root = Path(settings['official_repo']).resolve()
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain'], cwd=root, text=True).strip()
    if head != TIMEPLE_REVISION or dirty:
        raise ValueError('TimePLE official implementation must match pinned clean revision')
    return root


def check_loading(info):
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('incomplete native teacher loading: '+str(info))


def load_temporal_teacher(teacher, settings):
    import torch
    from tools.rq3_gpu import guard_before_cuda
    from tools.run_qwen3_vl_classification import infer_one, resolve_digit_token_ids
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    if teacher == 'timeple_8b':
        root = official_timeple_root(settings)
        sys.path.insert(0, str(root/'src'))
        import timeple  # 注册官方模型与processor，不用普通Qwen吞掉codec权重。
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
    network, loading = AutoModelForImageTextToText.from_pretrained(settings['model_path'],
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        output_loading_info=True)
    check_loading(loading)
    if teacher == 'timeple_8b' and not getattr(network,'use_timeple_codec',False):
        raise ValueError('TimePLE codec architecture missing')
    processor = AutoProcessor.from_pretrained(settings['model_path'], local_files_only=True)
    guard_before_cuda()
    network = network.eval().requires_grad_(False).to('cuda:0')
    runtime = dict(torch=torch, model=network, processor=processor,
        process_vision_info=process_vision_info, device=network.device,
        digit_token_ids=resolve_digit_token_ids(processor.tokenizer))
    args = types.SimpleNamespace(frame_count=8, prompt_mode='baseline', input_mode='native_video',
        min_pixels=224*224, max_pixels=448*448, score_temperature=1., uniform_mixed_frame_grid=True)
    def score(frames, paths, row):
        duration = row['proposal_end_sec']-row['proposal_start_sec']
        if duration <= 0: raise ValueError('nonpositive proposal duration')
        return infer_one(runtime, paths, row['candidate_id'], args,
                         sample_fps=8./duration)['class_probabilities']
    return score
