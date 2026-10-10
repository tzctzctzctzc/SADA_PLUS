"""源域双上下文真实教师验收；不生成正式训练缓存或冻结门控阈值。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.rq4_protocol import digest, file_sha, decode_response, match_response, interval_iou


def timeple_prompt_variant(prompt, variant):
    result = dict(prompt)
    if variant == 'native-only':
        result['user_prompt'] = ('Localize the event "{query}" in the video. '
            'Output only <|TIMESPAN|>. Do not output a description or numerical timestamps.')
    elif variant != 'official':
        raise ValueError('unknown TimePLE diagnostic prompt')
    return result


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,r_frame_rate:format=duration', '-of', 'json', str(path)], text=True))


def context(proposal, duration, scale):
    center = sum(proposal) / 2
    width = max(scale * (proposal[1] - proposal[0]), 2 * scale)
    return [max(0., center - width / 2), min(duration, center + width / 2)]


def prepare(args):
    evaluation = json.loads(Path(args.evaluation).read_text())
    if evaluation['domain'] != 'source' or evaluation['seed'] != 123:
        raise ValueError('acceptance requires seed123 source predictions')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    groups = {}
    for index, pred in enumerate(evaluation['predictions']):
        # 只从预测读取，禁止用真实类别或边界构造查询/切片。
        groups.setdefault(pred['video-id'], []).append((index, pred))
    requests, videos = [], {}
    for vid, rows in sorted(groups.items()):
        video = Path(args.videos) / (vid + '.MP4')
        info = probe(video)
        duration = float(info['format']['duration'])
        fps_num, fps_den = map(float, info['streams'][0]['r_frame_rate'].split('/'))
        fps = fps_num / fps_den
        videos[vid] = dict(path=str(video), probe=info)
        # 固定SHA抽样，仅用于源域接口诊断，不按GT或教师效果选样。
        rows.sort(key=lambda r: digest([123, vid, r[0]]))
        for index, pred in rows[:args.per_video]:
            proposal = [float(pred['t-start']), float(pred['t-end'])]
            if not 0 <= proposal[0] < proposal[1] <= duration:
                raise ValueError('source prediction outside actual video')
            for scale in (2, 3):
                window = context(proposal, duration, scale)
                cid = '%s-p%d-context%d' % (vid, index, scale)
                clip = root / (cid + '.mkv')
                subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-ss', str(window[0]),
                    '-i', str(video), '-t', str(window[1] - window[0]), '-map', '0:v:0', '-an',
                    '-vf', 'scale=640:-2', '-c:v', 'ffv1', '-threads', '2', str(clip)], check=True)
                actual = float(probe(clip)['format']['duration'])
                if abs(actual - (window[1] - window[0])) > 2 / fps + .01:
                    raise ValueError('clip duration differs from declared time origin')
                row = dict(candidate_id=cid, video_id=vid, class_id=int(pred['label']),
                    proposal=proposal, window=window, video_duration=duration,
                    anchor_sec=sum(proposal)/2, fpn_level=0)
                requests.append(dict(candidate=row, clip_path=str(clip), clip_sha256=file_sha(clip),
                    clip_start_sec=window[0], clip_end_sec=window[1]))
    save(root / 'requests.json', dict(diagnostic_only=True, split='source_validation', source_subsets=evaluation['split'],
        seed=123, prediction_file_sha256=file_sha(args.evaluation),
        checkpoint_sha256=evaluation['checkpoint_sha256'], videos=videos,
        anchor_note='diagnostic midpoint only; not exported as a formal SADA anchor',
        contexts=[2, 3], requests=requests))
    print('SOURCE_REQUESTS_READY', len(requests), flush=True)


def weights(root):
    root = Path(root)
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    names = sorted(set(index['weight_map'].values()))
    if set(names) != {p.name for p in root.glob('model-*.safetensors')}:
        raise ValueError('weight index/shards mismatch')
    hashes = {}
    for name in names:
        path = root / name
        actual = file_sha(path)
        # HF LFS blob的文件名为其内容SHA；检查实际文件，不只记录自报hash。
        expected = path.resolve().name
        if len(expected) != 64 or expected != actual:
            raise ValueError('HF checkpoint content SHA mismatch: ' + name)
        hashes[name] = actual
    for path in sorted(root.iterdir()):
        if path.suffix in ('.json', '.jinja', '.txt', '.pth'):
            hashes[path.name] = file_sha(path)
    return hashes


def guard_teacher_cuda(selected, teacher):
    from tools.rq3_gpu import idle_gpu_uuid
    # vLLM 0.19的官方TimePLE路径将CVD解析为int；仍限制为同一物理卡。
    expected = str(selected['index']) if teacher == 'timeple_8b' else selected['uuid']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected:
        raise RuntimeError('teacher visibility differs from reserved GPU')
    if idle_gpu_uuid(selected['index']) != selected['uuid']:
        raise RuntimeError('reserved GPU identity changed')
    print('RQ4_PRE_CUDA_IDLE_CHECK_PASS', selected, flush=True)


def score(args):
    from tools.rq3_gpu import gpu_slot
    from tools.rq4_teachers import TimeLens2Adapter, TimePLEAdapter, NativeIntervalMissing, score_requests
    from tools.run_clip_target_adapter import ACTION_PHRASES
    manifest = json.loads(Path(args.requests).read_text())
    if manifest.get('diagnostic_only') is not True or manifest.get('split') != 'source_validation':
        raise ValueError('source diagnostic input required')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    identities = weights(args.model)
    save(root/'identity.json', dict(weights=identities, requests_sha256=file_sha(args.requests),
        code_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        diagnostic_only=True, teacher=args.teacher))
    with gpu_slot(args.gpu, wait=False) as selected:
        visible = str(selected['index']) if args.teacher == 'timeple_8b' else selected['uuid']
        os.environ.update(CUDA_VISIBLE_DEVICES=visible, RQ3_GPU_INDEX=str(selected['index']),
            RQ3_GPU_UUID=selected['uuid'], TOKENIZERS_PARALLELISM='false')
        if args.teacher == 'timelens2_8b':
            from transformers import AutoModelForImageTextToText, AutoProcessor
            import torch
            processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
            guard_teacher_cuda(selected, args.teacher)
            torch.manual_seed(123)
            model, loading = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True,
                torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', output_loading_info=True)
            if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
                raise ValueError('incomplete native TimeLens2 loading: ' + str(loading))
            save(root/'loading.json', loading)
            adapter = TimeLens2Adapter(model, processor,
                dict(fps=2., min_pixels=32*32, max_pixels=480*480, max_frames=200), 256)
        else:
            # 官方codec运行时注册需与离线引擎同进程，spawn不会继承该注册。
            os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
            import yaml
            repo = Path(args.official_repo)
            sys.path[:0] = [str(repo/'evaluation/src'), str(repo/'src')]
            from runners.qwen_family import QwenTimePLERunner
            from benchmark_loader import BenchmarkSample
            inference = yaml.safe_load((repo/'evaluation/configs/models/timeple_8b.yaml').read_text())['model']['eval']['config']['inference']
            prompt = yaml.safe_load((repo/'evaluation/configs/prompts/common.yaml').read_text())['prompt']['by_runner_family']['qwen3_vl_timeple']
            prompt = timeple_prompt_variant(prompt, args.timeple_prompt)
            inference.update(model_path=args.model, allowed_local_media_path=str(Path(args.requests).parent))
            # 单条验收降低预分配，不改原生codec、精度或生成规则。
            inference['engine'].update(max_num_seqs=1, enforce_eager=True, seed=123,
                kv_cache_memory_bytes=4*1024**3)
            if args.timeple_sync:
                inference['engine'].setdefault('extra_llm_kwargs', {})['async_scheduling'] = False
            config = dict(model=dict(name='timeple_8b'), inference=inference, prompt=prompt)
            save(root/'official_config.json', config)
            save(root/'runtime.json', dict(VLLM_ENABLE_V1_MULTIPROCESSING='0',
                CUDA_VISIBLE_DEVICES=visible, gpu_uuid=selected['uuid'],
                note='native codec in-process; 4GiB cache covers official 20480-token context'))
            runner = QwenTimePLERunner(config, config_path=root/'official_config.json')
            adapter = TimePLEAdapter(runner, BenchmarkSample)
            guard_teacher_cuda(selected, args.teacher)
        # TimePLE在第一次predict时才真正加载模型，不把adapter构造当加载成功。
        print('TEACHER_ADAPTER_READY', args.teacher, flush=True)
        responses = []
        for request in manifest['requests']:
            started = time.monotonic()
            try:
                output = score_requests(adapter, [request], list(ACTION_PHRASES))[0]
            except NativeIntervalMissing as exc:
                # 仅本源域诊断允许收集无原生区间样本；不做文本回退或训练缓存。
                row = request['candidate']
                output = dict(video_id=row['video_id'], candidate_id=row['candidate_id'],
                    candidate_sha256=digest(row), teacher=args.teacher,
                    request_sha256=digest(request), class_names_sha256=digest(list(ACTION_PHRASES)),
                    seconds=time.monotonic()-started, **exc.response)
            responses.append(output)
            save(root/(request['candidate']['candidate_id']+'.json'), output)
            print('SOURCE_RESPONSE', request['candidate']['candidate_id'], round(time.monotonic()-started, 3), flush=True)
        save(root/'responses.json', responses)
        print('SOURCE_DIAGNOSTIC_COMPLETE', args.teacher, len(responses),
              'native_missing', sum(bool(r.get('error')) for r in responses), flush=True)


def summarize(args):
    manifest = json.loads(Path(args.requests).read_text())
    responses = json.loads(Path(args.responses).read_text())
    lookup = {(r['video_id'], r['candidate_id']): r for r in responses}
    records, pairs = [], {}
    for request in manifest['requests']:
        row = request['candidate']; r = lookup[(row['video_id'], row['candidate_id'])]
        if r['candidate_sha256'] != digest(row):
            raise ValueError('mismatched response')
        try:
            spans = decode_response(r['teacher'], r['raw_text'], r['metadata'], row['window'][1]-row['window'][0])
            target, reason = match_response(row, spans, min_iou=0., ambiguity_margin=0.)
        except (ValueError, TypeError):
            target, reason = None, 'invalid_output'
        entry = dict(candidate=row, target=target, reason=reason)
        records.append(entry)
        pairs.setdefault(row['candidate_id'].rsplit('-context', 1)[0], []).append(entry)
    stability = []
    for key, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError('missing paired context')
        valid = all(x['target'] is not None for x in pair)
        stability.append(dict(candidate_id=key, score=interval_iou(pair[0]['target'],pair[1]['target']) if valid else None))
    save(args.output, dict(diagnostic_only=True, threshold_frozen=False, records=records,
        context_consistency=stability, counts={reason:sum(r['reason']==reason for r in records)
            for reason in ('matched','no_match','ambiguous','invalid_output')}))


def source_quality(record, annotations):
    """事后按原候选确定GT实例，不按教师边界择优匹配。"""
    row = record['candidate']
    eligible = [a['segment'] for a in annotations if int(a['label_id']) == row['class_id']]
    ranked = sorted(((interval_iou(row['proposal'], s), s) for s in eligible), reverse=True)
    if not ranked or ranked[0][0] <= 0:
        return dict(status='no_same_class_overlap', old_iou=None, new_iou=None, delta=None)
    old, gt = ranked[0]
    target = record['target']
    new = interval_iou(target, gt) if target is not None else old
    return dict(status='matched_source_gt' if target is not None else 'teacher_abstained',
        old_iou=old, new_iou=new, delta=new-old)


def calibrate(args):
    """源域可靠性曲线；不从这批小样本自动冻结或导出训练门控。"""
    manifest = json.loads(Path(args.requests).read_text())
    diagnostics = json.loads(Path(args.diagnostics).read_text())
    if manifest.get('split') != 'source_validation' or not diagnostics.get('diagnostic_only'):
        raise ValueError('source-only diagnostic artifacts required')
    database = json.loads(Path(args.annotations).read_text())['database']
    allowed = set(manifest['source_subsets'])
    for vid in manifest['videos']:
        if database[vid]['subset'] not in allowed:
            raise ValueError('calibration video is not in source validation')
    stability = {r['candidate_id']:r['score'] for r in diagnostics['context_consistency']}
    rows = []
    for record in diagnostics['records']:
        candidate = record['candidate']; cid = candidate['candidate_id']
        if not cid.endswith('-context2'):
            continue
        rows.append(dict(candidate_id=cid, consistency=stability[cid.rsplit('-context',1)[0]],
            **source_quality(record, database[candidate['video_id']]['annotations'])))
    curves = []
    for threshold in (0., .25, .5, .75, 1.):
        accepted = [r for r in rows if r['consistency'] is not None and r['consistency'] >= threshold]
        matched = [r for r in accepted if r['delta'] is not None]
        curves.append(dict(threshold=threshold, accepted=len(accepted),
            no_same_class_overlap=sum(r['delta'] is None for r in accepted),
            mean_delta=sum(r['delta'] for r in matched)/len(matched) if matched else None,
            improved=sum(r['delta']>0 for r in matched), harmed=sum(r['delta']<0 for r in matched)))
    save(args.output, dict(diagnostic_only=True, threshold_frozen=False, target_gt_used=False,
        score_name='paired-context-absolute-tiou', source_video_ids=list(manifest['videos']),
        annotation_sha256=file_sha(args.annotations), request_sha256=file_sha(args.requests),
        diagnostics_sha256=file_sha(args.diagnostics), rows=rows, curves=curves,
        limitation='small source pilot; no independent gate validation or loss-weight calibration yet'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--evaluation', required=True); p.add_argument('--videos', required=True)
    p.add_argument('--output', required=True); p.add_argument('--per-video', type=int, default=1)
    p = sub.add_parser('score'); p.add_argument('--requests', required=True); p.add_argument('--output', required=True)
    p.add_argument('--teacher', choices=['timelens2_8b','timeple_8b'], required=True)
    p.add_argument('--model', required=True); p.add_argument('--gpu', choices=['0','1'], required=True)
    p.add_argument('--official-repo')
    p.add_argument('--timeple-sync', action='store_true', help='原生codec单变量同步调度诊断')
    p.add_argument('--timeple-prompt', choices=['official','native-only'], default='official')
    p = sub.add_parser('summarize'); p.add_argument('--requests', required=True)
    p.add_argument('--responses', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('calibrate'); p.add_argument('--requests', required=True)
    p.add_argument('--diagnostics', required=True); p.add_argument('--annotations', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    {'prepare':prepare, 'score':score, 'summarize':summarize, 'calibrate':calibrate}[args.command](args)


if __name__ == '__main__':
    main()
