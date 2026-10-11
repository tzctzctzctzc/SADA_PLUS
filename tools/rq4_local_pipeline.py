"""E1等价CPU预取/断点接续；科学计算仍调用冻结原实现。"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
from tools.run_rq4_local_evidence import (BASE, CODE, MODELS, read, check, clean_head,
    save, file_sha, official_inputs, load_visual, temporal_pool, weights)

PROCESSOR = None


def cpu_init(model_path):
    global PROCESSOR
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    PROCESSOR = Qwen3VLVideoProcessor.from_pretrained(model_path, local_files_only=True)


def cpu_prepare(request):
    started = time.monotonic()
    assert file_sha(request['clip_path']) == request['clip_sha256']
    inputs, actual = official_inputs(request, PROCESSOR)
    return (inputs['pixel_values_videos'].numpy(), inputs['video_grid_thw'].numpy(),
            actual, time.monotonic()-started)


def ordered_inputs(rows, model_path):
    # 最多两项在途；每16片段重建CPU进程，避免解码库累积线程。
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'),
            initializer=cpu_init, initargs=(str(model_path),), max_tasks_per_child=16) as pool:
        pending = {}; submit = 0
        for index, row in enumerate(rows):
            while submit < min(len(rows), index+2):
                pending[submit] = pool.submit(cpu_prepare, rows[submit]['request']); submit += 1
            yield row, pending.pop(index).result()


def benchmark(args):
    """不占GPU；最短/最长及SHA顺序前4样本，两教师分别逐元素对比。"""
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    m = read(args.old_root/'manifest.json')
    ordered = sorted(m['rows'], key=lambda r:r['request']['clip_end_sec']-r['request']['clip_start_sec'])
    chosen = list({r['id']:r for r in [ordered[0], ordered[-1]]+m['rows'][:4]}.values())
    report = {}
    for teacher in MODELS:
        path = MODELS[teacher]['model']; processor = Qwen3VLVideoProcessor.from_pretrained(path, local_files_only=True)
        expected = {}; started = time.monotonic()
        for row in chosen:
            inp, actual = official_inputs(row['request'], processor)
            expected[row['id']] = (inp['pixel_values_videos'].numpy(), inp['video_grid_thw'].numpy(), actual)
        serial = time.monotonic()-started; started = time.monotonic()
        for row, (pixels, grid, actual, _) in ordered_inputs(chosen, path):
            ref = expected.pop(row['id'])
            assert np.array_equal(pixels, ref[0]) and np.array_equal(grid, ref[1]) and actual == ref[2]
        parallel = time.monotonic()-started
        report[teacher] = dict(bitwise_equal=True, samples=[r['id'] for r in chosen],
                               serial_seconds=serial, pipeline_seconds=parallel)
        print('CPU_EQUIVALENCE_PASS', teacher, serial, parallel, flush=True)
    save(args.report, dict(status='passed', code_commit=args.expected_head,
                          original_manifest_sha256=file_sha(args.old_root/'manifest.json'), teachers=report))


def reuse(args):
    clean_head(args.expected_head); old = args.old_root; root = args.root
    proof = read(old/'prepared.json'); m = read(old/'manifest.json'); audit = read(old/'input_audit.json')
    assert proof['manifest_sha256'] == file_sha(old/'manifest.json')
    assert proof['labels_sha256'] == file_sha(old/'source_labels.json')
    assert audit['manifest_sha256'] == proof['manifest_sha256']
    # 必须原科学模块逐字节一致，不通过“同配置”掩盖方法改动。
    originals = {}
    for name in ('rq4_local_evidence.py', 'run_rq4_local_evidence.py'):
        assert file_sha(CODE/'tools'/name) == file_sha(args.old_code/'tools'/name)
        originals[name] = file_sha(CODE/'tools'/name)
    bench = read(args.report)
    assert bench['status'] == 'passed'
    # 调度/退出修复不必重做相同CPU测试；逐函数AST绑定原测试代码。
    tested_source = subprocess.check_output(['git', 'show', bench['code_commit']+':tools/rq4_local_pipeline.py'], cwd=CODE, text=True)
    def cpu_ast(source):
        return {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name in ('cpu_init','cpu_prepare','ordered_inputs','benchmark')}
    assert cpu_ast(tested_source) == cpu_ast(Path(__file__).read_text())
    assert bench['original_manifest_sha256'] == proof['manifest_sha256']
    root.mkdir(parents=True, exist_ok=False)
    m['code_commit'] = args.expected_head; save(root/'manifest.json', m)
    shutil.copyfile(old/'source_labels.json', root/'source_labels.json')
    proof.update(manifest_sha256=file_sha(root/'manifest.json'), code_commit=args.expected_head)
    save(root/'prepared.json', proof)
    audit.update(manifest_sha256=proof['manifest_sha256'], code_commit=args.expected_head)
    save(root/'input_audit.json', audit)
    records = {}
    for teacher in MODELS:
        identity = old/teacher/'identity.json'
        completed = []
        if identity.exists():
            original_identity = read(identity)
            assert original_identity['manifest_sha256'] == file_sha(old/'manifest.json')
            assert original_identity['visual_tensor_sha256'] == audit['teachers'][teacher]['visual_tensor_sha256']
            for row in m['rows']:
                path = old/teacher/(row['id']+'.json'); feature = path.with_suffix('.npz')
                if not path.exists(): continue  # 中断中的孤立npz不充完成，也不覆盖。
                record = read(path)
                assert record['id'] == row['id'] and file_sha(feature) == record['sha256']
                with np.load(feature) as f:
                    assert np.isfinite(f['features']).all() and np.isfinite(f['times']).all()
                    assert len(f['times']) == f['features'].shape[0] == record['grid'][0]
                completed.append(record)
        records[teacher] = completed
    save(root/'reuse.json', dict(old_root=str(old), old_code=str(args.old_code),
         original_files=originals, records=records, original_manifest_sha256=file_sha(old/'manifest.json'),
         cpu_benchmark=str(args.report), cpu_benchmark_sha256=file_sha(args.report)))
    print('REUSE_VERIFIED', {t:len(r) for t,r in records.items()}, flush=True)


def extract(args):
    root = args.root; teacher = args.teacher; m = check(root, args.expected_head)
    audit = read(root/'input_audit.json'); reuse = read(root/'reuse.json')
    assert file_sha(reuse['cpu_benchmark']) == reuse['cpu_benchmark_sha256']
    out = root/teacher; out.mkdir(exist_ok=False); path = Path(MODELS[teacher]['model'])
    records = {r['id']:r for r in reuse['records'][teacher]}
    for key, record in records.items():
        for suffix in ('.npz', '.json'):
            shutil.copyfile(Path(reuse['old_root'])/teacher/(key+suffix), out/(key+suffix))
        assert file_sha(out/(key+'.npz')) == record['sha256']
    remaining = [r for r in m['rows'] if r['id'] not in records]
    # 复算三个已完成特征，证明完整GPU路径未改变；新教师则保留CPU等价证明。
    validation = [r for r in m['rows'] if r['id'] in records][:3]
    from tools.rq3_gpu import gpu_slot, idle_gpu_uuid
    with gpu_slot('3', BASE/'SADA_PLUS', wait=True) as card:
        os.environ['CUDA_VISIBLE_DEVICES'] = card['uuid']
        import torch
        model, cfg, sha = load_visual(path)
        assert sha == audit['teachers'][teacher]['visual_tensor_sha256']
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        processor = Qwen3VLVideoProcessor.from_pretrained(path, local_files_only=True)
        save(out/'identity.json', dict(code_commit=args.expected_head, teacher=teacher, device=card,
             weights=weights(path), visual_tensor_sha256=sha, visual_config=cfg.to_dict(), processor=processor.to_dict(),
             manifest_sha256=file_sha(root/'manifest.json'), reuse_sha256=file_sha(root/'reuse.json'),
             scope='unchanged official visual subtree; ordered CPU prefetch; batch1 bf16 sdpa'))
        assert idle_gpu_uuid(3) == card['uuid']
        torch.manual_seed(6); model = model.to('cuda'); start = time.monotonic(); timings = []
        for row, (pixels, grid, actual, prep_seconds) in ordered_inputs(validation+remaining, path):
            gpu_start = time.monotonic()
            with torch.inference_mode():
                tokens, _ = model(torch.from_numpy(pixels).to('cuda', dtype=torch.bfloat16),
                                  grid_thw=torch.from_numpy(grid).to('cuda'))
            feature, times = temporal_pool(tokens.float().cpu().numpy(), grid[0].tolist(), actual)
            times += row['request']['clip_start_sec']
            assert times[-1] <= row['request']['clip_end_sec']+.1
            if row['id'] in records:
                with np.load(out/(row['id']+'.npz')) as old:
                    assert np.array_equal(feature, old['features']) and np.array_equal(times, old['times'])
                print('GPU_BITWISE_REPLAY_PASS', teacher, row['id'], flush=True)
                continue
            target = out/(row['id']+'.npz'); np.savez_compressed(target, features=feature, times=times)
            record = dict(id=row['id'], sha256=file_sha(target), grid=grid[0].tolist(), frames=actual,
                          cpu_prepare_seconds=prep_seconds, gpu_and_save_seconds=time.monotonic()-gpu_start)
            save(out/(row['id']+'.json'), record); records[row['id']] = record
            timings.append(record)
            print('E1_PIPELINE', teacher, len(records), len(m['rows']),
                  'wall', time.monotonic()-start, 'cpu', prep_seconds, 'gpu_save', record['gpu_and_save_seconds'], flush=True)
        save(out/'features_complete.json', dict(code_commit=args.expected_head,
             identity_sha256=file_sha(out/'identity.json'), manifest_sha256=file_sha(root/'manifest.json'),
             records=[records[r['id']] for r in m['rows']], gpu_replay_ids=[r['id'] for r in validation],
             new_seconds=time.monotonic()-start, new_count=len(timings)))


def driver(args):
    active = []
    def stop(*_):
        # 不在正在Popen.wait的信号栈里再次wait，先展开栈再由except统一回收。
        raise RuntimeError('explicit stop; preserved artifacts')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def launch(mode, teacher):
        module = 'tools.rq4_local_pipeline' if mode == 'extract' else 'tools.run_rq4_local_evidence'
        log = (args.root/(teacher+'_'+mode+'.log')).open('x')
        env = dict(os.environ)
        if mode == 'evaluate': env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        child = subprocess.Popen([sys.executable, '-u', '-m', module, mode, '--root', str(args.root),
            '--expected-head', args.expected_head, '--teacher', teacher], cwd=CODE, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close(); active.append(child); print('PIPELINE_CHILD', mode, teacher, child.pid, flush=True)
        return child
    try:
        for teacher in MODELS:
            child = launch('extract', teacher)
            if child.wait(): raise RuntimeError(f'{teacher} extraction failed')
            for running in active:
                if running.poll() not in (None, 0): raise RuntimeError('prior scoring failed')
            launch('evaluate', teacher)  # CPU评分与下一教师GPU提取重叠。
        for child in active:
            if child.wait(): raise RuntimeError('evaluation failed')
        save(args.root/'complete.json', dict(status='complete', scope='E1-source-crossfit-only',
             results={t:dict(path=str(args.root/t/'result.json'), sha256=file_sha(args.root/t/'result.json')) for t in MODELS}))
    except BaseException:
        for child in active:
            if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        for child in active: child.wait()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['benchmark', 'reuse', 'extract', 'driver'])
    p.add_argument('--root', type=Path); p.add_argument('--old-root', type=Path)
    p.add_argument('--old-code', type=Path); p.add_argument('--report', type=Path)
    p.add_argument('--expected-head', required=True); p.add_argument('--teacher', choices=list(MODELS))
    args = p.parse_args(); clean_head(args.expected_head); globals()[args.mode](args)


if __name__ == '__main__': main()
