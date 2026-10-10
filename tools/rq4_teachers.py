"""原生定位适配器；模型装载/独占GPU锁由调用方负责，不自动抢卡。"""
import copy
import time
from pathlib import Path
from tools.rq4_protocol import digest, file_sha, query_for, validate_candidate

TIMELENS_PROMPT = ('Given the query: "{query}", return ALL time spans (in seconds) '
                  'where the query is relevant.\n'
                  'Output format MUST be a JSON array of [start, end] pairs.\n')


class TimeLens2Adapter:
    """官方AutoModelForImageTextToText + processor；不将dense特征当边界。"""
    teacher = 'timelens2_8b'

    def __init__(self, model, processor, video_options, max_new_tokens):
        self.model = model.eval().requires_grad_(False)
        self.processor = processor
        self.video_options = dict(video_options)
        if set(self.video_options) - {'fps', 'min_pixels', 'max_pixels', 'total_pixels', 'max_frames', 'nframes'}:
            raise ValueError('unknown video sampling option')
        self.max_new_tokens = int(max_new_tokens)
        if self.max_new_tokens <= 0:
            raise ValueError('invalid generation length')

    def predict(self, sample_id, clip_path, query):
        import torch
        from qwen_vl_utils import process_vision_info
        messages = [dict(role='user', content=[
            dict(type='video', video=Path(clip_path).resolve().as_uri(), **self.video_options),
            dict(type='text', text=TIMELENS_PROMPT.format(query=query))])]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, kwargs = process_vision_info(messages, image_patch_size=16,
                                                   return_video_kwargs=True, return_video_metadata=True)
        videos, metadata = zip(*videos)
        inputs = self.processor(text=text, images=images, videos=list(videos),
                                video_metadata=list(metadata), do_resize=False,
                                return_tensors='pt', **kwargs).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        trimmed = [out[len(ids):] for ids, out in zip(inputs.input_ids, generated)]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        return dict(raw_text=raw, metadata=dict(video_metadata=list(metadata)), error='')


class TimePLEAdapter:
    """注入官方evaluation/src/runners/qwen_family.QwenTimePLERunner实例。

    使用其vLLM hidden-state side channel + TIMESPAN codec，不能换成普通generate。
    官方runner环境及权重核验由独立装载入口管理，构造本适配器不会初始化CUDA。
    """
    teacher = 'timeple_8b'

    def __init__(self, official_runner, sample_type):
        if type(official_runner).__name__ != 'QwenTimePLERunner':
            raise ValueError('TimePLE requires the official codec runner')
        prompt = official_runner.prompt_cfg
        # 拒绝官方通用模板可插入的GT字段；只允许query占位符。
        import string
        for key in ('user_prompt', 'system_prompt', 'format_prompt'):
            fields = [name for _, name, _, _ in string.Formatter().parse(prompt.get(key, '')) if name]
            if any(name != 'query' for name in fields):
                raise ValueError('RQ4 prompt must not reference annotations/sample metadata')
        generation = official_runner.generation_cfg
        if generation.get('temperature') != 0:
            raise ValueError('RQ4 requires deterministic teacher generation')
        self.runner, self.sample_type = official_runner, sample_type

    def predict(self, sample_id, clip_path, query):
        sample = self.sample_type(global_index=0, sample_id=sample_id, query=query,
                                  video_path=str(clip_path), gt_timestamps=[], raw_sample={})
        results = self.runner.predict_batch([sample])
        if len(results) != 1 or results[0].sample_id != sample_id:
            raise ValueError('official runner response identity mismatch')
        response = results[0]
        if response.error:
            raise RuntimeError(response.error)
        # codec缺失属于接口故障，不允许静默退化为文本解析。
        if response.metadata.get('codec_predictions') is None:
            raise RuntimeError('official TimePLE codec did not return intervals')
        return dict(raw_text=response.raw_text, metadata=response.metadata, error='')


def score_requests(adapter, requests, class_names):
    """clip必须事先按window切出；保存其SHA及切片声明供输入审计。

    本接口不读取目标标注、不构造GT裁剪；切片正确性仍需真实视频验收。
    """
    outputs = []
    for request in requests:
        if set(request) != {'candidate', 'clip_path', 'clip_sha256', 'clip_start_sec', 'clip_end_sec'}:
            raise ValueError('unexpected request fields')
        row = validate_candidate(request['candidate'])
        if [request['clip_start_sec'], request['clip_end_sec']] != row['window']:
            raise ValueError('clip origin/duration differ from context')
        if file_sha(request['clip_path']) != request['clip_sha256']:
            raise ValueError('query clip changed')
        started = time.monotonic()
        response = adapter.predict(row['candidate_id'], request['clip_path'], query_for(row, class_names))
        if response.get('error'):
            raise RuntimeError(response['error'])
        outputs.append(dict(video_id=row['video_id'], candidate_id=row['candidate_id'],
                            candidate_sha256=digest(row), teacher=adapter.teacher,
                            request_sha256=digest(request), class_names_sha256=digest(class_names),
                            seconds=time.monotonic() - started, **copy.deepcopy(response)))
    return outputs
