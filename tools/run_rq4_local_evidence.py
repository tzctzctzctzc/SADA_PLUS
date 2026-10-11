"""E1：冻结全源候选池→单卡官方视觉子模块→完整视频交叉拟合。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import numpy as np
from tools.rq4_protocol import file_sha
from tools.rq4_source_acceptance import save, weights
from tools.run_rq4_teacher_queries import BASE, MODELS, clean_head
from tools.rq4_local_evidence import POLICY, digest, folds, labels, temporal_pool, slowfast_at, design, crossfit, summarize

CODE = Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text())


def prepare(root, head):
    clean_head(head); root.mkdir(parents=True, exist_ok=False)
    rows, sources, seen = [], {}, {}
    annotation = BASE/'SADA_PLUS/data/ek100_slowfast/annotations/epic_kitchens_100_verb_detailed_split.json'
    database = read(annotation)['database']
    truth = {}
    for shift in range(1, 7):
        prefix = 'rq4_source_seed6_ec7882a_1011' if shift == 1 else 'rq4_source_seed6_b08c147_1011'
        path = BASE/prefix/'inputs'/f'S{shift}'/'requests.json'; m = read(path)
        assert m['split'] == 'source_validation' and m['seed'] == 6 and m['shift'] == f'S{shift}'
        cfg_path = BASE/('sada_unified_f9ac276_1009_gpu2' if shift % 2 else 'sada_unified_f9ac276_1009_gpu3')/f'S{shift}'/'uda_preflight.json'
        cfg = read(cfg_path); sources[str(path)] = file_sha(path); sources[str(cfg_path)] = file_sha(cfg_path)
        src = {x.lower() for x in cfg['src_val_split']}
        forbidden = {x.lower() for k in ('src_train_split', 'tgt_train_split') for x in cfg[k]}
        for request in m['requests']:
            c = request['candidate']
            if not c['candidate_id'].endswith('-context3'): continue
            assert database[c['video_id']]['subset'].lower() in src - forbidden
            assert [request['clip_start_sec'], request['clip_end_sec']] == c['window']
            assert file_sha(request['clip_path']) == request['clip_sha256']
            identity = digest([c['video_id'], c['window'], c['proposal'], c['class_id']])
            if identity in seen:
                seen[identity]['origins'].append(c['candidate_id']); continue
            d = cfg['dataset']; feature = Path(d['feat_folder']) / ((d['file_prefix'] or '')+c['video_id']+d['file_ext'])
            if not feature.is_absolute(): feature = BASE/'SADA_PLUS'/feature
            if d['default_fps'] is None: raise ValueError('explicit SlowFast fps required')
            row = dict(id=identity, candidate=c, origins=[c['candidate_id']], request=request,
                       slowfast_path=str(feature), slowfast_sha256=file_sha(feature), dataset=d)
            seen[identity] = row; rows.append(row)
            truth[c['video_id']] = database[c['video_id']]['annotations']
    mapping = folds([r['candidate']['video_id'] for r in rows])
    save(root/'manifest.json', dict(policy=POLICY, code_commit=head, rows=rows, folds=mapping,
                                   sources=sources, annotations_sha256=file_sha(annotation)))
    save(root/'source_labels.json', truth)
    save(root/'prepared.json', dict(count=len(rows), videos=len(mapping), manifest_sha256=file_sha(root/'manifest.json'),
                                   labels_sha256=file_sha(root/'source_labels.json'), code_commit=head))
    print('E1_PREPARED', len(rows), len(mapping), flush=True)


def check(root, head):
    clean_head(head); m = read(root/'manifest.json'); p = read(root/'prepared.json')
    assert m['policy'] == POLICY and m['code_commit'] == head
    assert p['manifest_sha256'] == file_sha(root/'manifest.json')
    assert p['labels_sha256'] == file_sha(root/'source_labels.json')
    return m


def load_visual(model_path):
    """只加载完整官方visual子树，严格逐key恢复；不声称运行语言头/TimePLE codec。"""
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    cfg = Qwen3VLVisionConfig(**read(model_path/'config.json')['vision_config'])
    cfg._attn_implementation = 'sdpa'
    model = Qwen3VLVisionModel(cfg).to(dtype=torch.bfloat16)
    index = read(model_path/'model.safetensors.index.json')['weight_map']
    selected = {k: v for k, v in index.items() if k.startswith('model.visual.')}
    state = {}
    for shard in sorted(set(selected.values())):
        with safe_open(model_path/shard, framework='pt', device='cpu') as handle:
            for key, value in selected.items():
                if value == shard: state[key.removeprefix('model.visual.')] = handle.get_tensor(key)
    model.load_state_dict(state, strict=True)
    import hashlib
    tensor_sha = {k: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k, v in state.items()}
    return model.eval().requires_grad_(False), cfg, tensor_sha


def official_inputs(request, processor):
    from qwen_vl_utils import process_vision_info
    content = dict(type='video', video=Path(request['clip_path']).as_uri(), fps=8., max_frames=200,
                   min_pixels=32*32, max_pixels=480*480, total_pixels=400*480*480)
    _, videos, _ = process_vision_info([dict(role='user', content=[content])], image_patch_size=16,
                                       return_video_kwargs=True, return_video_metadata=True)
    tensor, metadata = videos[0]
    actual = [float(i)/metadata['fps'] for i in metadata['frames_indices']]
    assert len(actual) == tensor.shape[0]
    inputs = processor(videos=[tensor], video_metadata=[metadata], do_resize=False,
                       do_sample_frames=False, return_tensors='pt')
    grid = inputs['video_grid_thw']; assert tuple(grid.shape) == (1, 3)
    assert int(grid[0, 0])*2 == len(actual)
    assert actual[-1] <= request['clip_end_sec']-request['clip_start_sec']+.1
    return inputs, actual


def audit_inputs(root, head):
    """两权重严格恢复及同一官方输入接口的最短/最长真实clip，禁CUDA。"""
    m = check(root, head)
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    ordered = sorted(m['rows'], key=lambda r:r['request']['clip_end_sec']-r['request']['clip_start_sec'])
    reports = {}
    for teacher in MODELS:
        path = Path(MODELS[teacher]['model'])
        model, cfg, sha = load_visual(path)
        processor = Qwen3VLVideoProcessor.from_pretrained(path, local_files_only=True)
        assert processor.temporal_patch_size == cfg.temporal_patch_size == 2
        assert processor.merge_size == cfg.spatial_merge_size == 2
        samples = []
        for row in (ordered[0], ordered[-1]):
            inputs, actual = official_inputs(row['request'], processor)
            samples.append(dict(id=row['id'], grid=inputs['video_grid_thw'].tolist(), frames=len(actual),
                                pixels=list(inputs['pixel_values_videos'].shape), actual_times=actual))
        reports[teacher] = dict(visual_tensor_sha256=sha, samples=samples, visual_config=cfg.to_dict())
        del model
        print('E1_CPU_INPUT_AUDIT', teacher, len(sha), flush=True)
    save(root/'input_audit.json', dict(status='passed', code_commit=head,
         manifest_sha256=file_sha(root/'manifest.json'), teachers=reports,
         identical_visual_weights=reports['timelens2_8b']['visual_tensor_sha256']==reports['timeple_8b']['visual_tensor_sha256']))


def extract(root, head, teacher):
    m = check(root, head); out = root/teacher; out.mkdir(exist_ok=False)
    audit = read(root/'input_audit.json')
    assert audit['status'] == 'passed' and audit['code_commit'] == head
    assert audit['manifest_sha256'] == file_sha(root/'manifest.json')
    model_path = Path(MODELS[teacher]['model'])
    identity = dict(teacher=teacher, code_commit=head, weights=weights(model_path),
                    scope='official frozen visual subtree only; no language model or interval codec',
                    manifest_sha256=file_sha(root/'manifest.json'))
    from tools.rq3_gpu import gpu_slot, idle_gpu_uuid
    with gpu_slot('3', BASE/'SADA_PLUS', wait=True) as card:
        os.environ['CUDA_VISIBLE_DEVICES'] = card['uuid']
        import torch
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        model, config, tensor_sha = load_visual(model_path)
        assert tensor_sha == audit['teachers'][teacher]['visual_tensor_sha256']
        processor = Qwen3VLVideoProcessor.from_pretrained(model_path, local_files_only=True)
        assert processor.temporal_patch_size == config.temporal_patch_size == 2
        assert processor.merge_size == config.spatial_merge_size == 2
        identity.update(visual_tensor_sha256=tensor_sha, visual_config=config.to_dict(),
                        processor=processor.to_dict(), device=card, torch=torch.__version__)
        save(out/'identity.json', identity)
        if idle_gpu_uuid(3) != card['uuid']: raise RuntimeError('GPU changed before CUDA')
        torch.manual_seed(6); model = model.to('cuda'); records = []
        for index, row in enumerate(m['rows']):
            started = time.monotonic(); request = row['request']
            assert file_sha(request['clip_path']) == request['clip_sha256']
            inputs, actual = official_inputs(request, processor)
            grid = inputs['video_grid_thw']
            with torch.inference_mode():
                tokens, _ = model(inputs['pixel_values_videos'].to('cuda', dtype=torch.bfloat16),
                                  grid_thw=grid.to('cuda'))
            feature, times = temporal_pool(tokens.float().cpu().numpy(), grid[0].tolist(), actual)
            times += request['clip_start_sec']
            assert times[-1] <= request['clip_end_sec']+.1
            path = out/(row['id']+'.npz')
            np.savez_compressed(path, features=feature, times=times)
            record = dict(id=row['id'], sha256=file_sha(path), grid=grid[0].tolist(),
                          frames=actual, seconds=time.monotonic()-started)
            records.append(record); save(out/(row['id']+'.json'), record)
            print('E1_VISUAL', teacher, index+1, len(m['rows']), feature.shape, record['seconds'], flush=True)
        save(out/'features_complete.json', dict(code_commit=head, identity_sha256=file_sha(out/'identity.json'),
                                              manifest_sha256=file_sha(root/'manifest.json'), records=records))


def evaluate(root, head, teacher):
    m = check(root, head); out = root/teacher; proof = read(out/'features_complete.json')
    assert proof['manifest_sha256'] == file_sha(root/'manifest.json')
    assert proof['identity_sha256'] == file_sha(out/'identity.json')
    records = {r['id']: r for r in proof['records']}
    assert len(records) == len(m['rows']) and set(records) == {r['id'] for r in m['rows']}
    truth = read(root/'source_labels.json'); rows = []; sf_cache = {}
    for row in m['rows']:
        path = out/(row['id']+'.npz'); assert file_sha(path) == records[row['id']]['sha256']
        with np.load(path) as feature: visual, times = feature['features'], feature['times']
        source = row['slowfast_path']
        if source not in sf_cache:
            assert file_sha(source) == row['slowfast_sha256']
            with np.load(source) as raw: sf_cache[source] = raw['feats'].astype('float32')
        sf = slowfast_at(sf_cache[source], times, row['dataset'])
        target = labels(times, truth[row['candidate']['video_id']], row['candidate']['class_id'])
        rows.append(dict(row, vlm=visual, times=times, slowfast=sf, labels=target))
    x, y, videos, identities = design(rows); predictions = {}; diagnostics = {}
    for name in POLICY['conditions']:
        print('E1_CROSSFIT', teacher, name, flush=True)
        predictions[name], diagnostics[name] = crossfit(x[name], y, videos, m['folds'])
    prediction_path = out/'out_of_fold.npz'
    np.savez_compressed(prediction_path, labels=y, videos=videos, **predictions)
    save(out/'prediction_index.json', identities)
    print('E1_BOOTSTRAP', teacher, flush=True)
    report = summarize(y, predictions, videos)
    save(out/'result.json', dict(report, code_commit=head, policy=POLICY, folds=m['folds'],
         diagnostics=diagnostics, feature_proof_sha256=file_sha(out/'features_complete.json'),
         predictions_sha256=file_sha(prediction_path), labels_sha256=file_sha(root/'source_labels.json'),
         index_sha256=file_sha(out/'prediction_index.json'), teacher=teacher))
    print('E1_COMPLETE', teacher, report['positive_signal'], flush=True)


def driver(args):
    check(args.root, args.expected_head)
    child = None
    def stop(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        raise RuntimeError('explicit stop; preserved completed and partial artifacts')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    # 有限两教师串行，各抽取结束释放GPU；CPU评分不阻塞下一教师抽取前的资源检查。
    for teacher in ('timelens2_8b', 'timeple_8b'):
        for mode in ('extract', 'evaluate'):
            with (args.root/(teacher+'_'+mode+'.log')).open('x') as log:
                cmd = [sys.executable, '-u', '-m', 'tools.run_rq4_local_evidence', mode,
                       '--root', str(args.root), '--expected-head', args.expected_head, '--teacher', teacher]
                env = dict(os.environ)
                if mode == 'evaluate': env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                child = subprocess.Popen(cmd, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                print('E1_CHILD', mode, teacher, child.pid, flush=True)
                code = child.wait()
                if code: raise RuntimeError(f'{mode} {teacher} exit={code}; preserve artifacts')
    results = {teacher: dict(path=str(args.root/teacher/'result.json'),
                            sha256=file_sha(args.root/teacher/'result.json')) for teacher in ('timelens2_8b', 'timeple_8b')}
    save(args.root/'complete.json', dict(status='complete', results=results, scope='E1-source-crossfit-only'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'audit_inputs', 'extract', 'evaluate', 'driver'])
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--expected-head', required=True)
    parser.add_argument('--teacher', choices=['timelens2_8b', 'timeple_8b'])
    args = parser.parse_args(); args.root = args.root.resolve()
    if args.mode == 'driver': driver(args)
    elif args.mode == 'prepare': prepare(args.root, args.expected_head)
    elif args.mode == 'audit_inputs': audit_inputs(args.root, args.expected_head)
    else: globals()[args.mode](args.root, args.expected_head, args.teacher)


if __name__ == '__main__': main()
