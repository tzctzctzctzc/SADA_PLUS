"""三条冷启动监督Oracle×六方向，四卡共享；不恢复任何旧任务。"""
import argparse
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha, code_identity, decision, validate_cuda_rng
from tools.prototype_cold_oracle import SPEC, SCHEMA, MODES, install, enable_controls
from tools.run_phase_endpoint_oracle import prepare_truth, REFERENCE, SHIFTS


def build(cfg):
    from tools import baseline_engine as base
    enable_controls()
    ctx = base.build(cfg)
    reference = Path(cfg['cold_oracle']['reference'])
    assert ctx.initial_state_sha == read(reference / 'termination.json')['initial_state_sha']
    assert ctx.initial_detector_sha == read(reference / 'detector_initialization.json')['sha256']
    path = cfg['cold_oracle']['truth']
    assert sha(path) == cfg['cold_oracle']['truth_sha256']
    install(ctx, cfg, read(path)['records'])
    from tools.cold_oracle_fastpath import install as install_fastpath
    install_fastpath(ctx.model)
    install_fastpath(ctx.model_ema.module)
    return ctx


def epoch_metrics(ctx, epoch, elapsed):
    values = ctx.train_metrics.result()
    return dict(epoch=epoch, seconds=elapsed, train_loss=float(values['t_final_loss']),
        source_cls_loss=float(values['t_cls_loss']), source_reg_loss=float(values['t_reg_loss']),
        oracle={k[len('t_cold_oracle_'):]:float(v) for k,v in values.items() if k.startswith('t_cold_oracle_')},
        source_Prototype={k[6:]:float(v) for k,v in values.items() if k.startswith('t_rq3_')})


def probe(cfg, entry, guard):
    import torch
    from tools import baseline_engine as base
    ctx = build(cfg)
    ctx.model.train()
    src = base.move_to_device(next(iter(ctx.train_loader_src)), ctx.model.device)
    tgt = base.move_to_device(next(iter(ctx.train_loader_tgt)), ctx.model.device)
    before = deepcopy(base.capture(ctx, cfg, 0, []))
    kwargs = dict(video_list_src=src, video_list_tgt=tgt, mode='train', curr_epoch=1,
        cls_static_anchors=None, mean_class_conf=None)
    expected = {k:v.detach().clone() for k,v in ctx.cold_oracle_original_forward(**kwargs)[0].items()}
    base.restore(ctx, before)
    ctx.cold_oracle_enabled = False
    output = ctx.model(**kwargs)[0]
    assert all(torch.equal(value, output[key].detach()) for key,value in expected.items())
    base.restore(ctx, before)
    ctx.cold_oracle_enabled = True
    del expected, output, src, tgt, before
    phases = {}
    for epoch in (1, 5):
        base.epoch_setup(ctx, epoch)
        loaders = ctx.train_loader_src, ctx.train_loader_tgt
        ctx.train_loader_src, ctx.train_loader_tgt = [base.StepLoader(x, 2) for x in loaders]
        old = {name:base.tensor_digest(module.state_dict()) for name,module in
            [('SGP',ctx.model.backbone_source),('cls',ctx.model.cls_head),('reg',ctx.model.reg_head)]}
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic(); ctx._train_epoch(epoch); torch.cuda.synchronize()
        value = epoch_metrics(ctx, epoch, time.monotonic()-started)
        value['updated'] = {name:old[name] != base.tensor_digest(module.state_dict()) for name,module in
            [('SGP',ctx.model.backbone_source),('cls',ctx.model.cls_head),('reg',ctx.model.reg_head)]}
        value['peak_mib'] = torch.cuda.max_memory_reserved()/2**20
        assert all(value['updated'].values())
        assert value['oracle']['positives'] > 0 and value['oracle']['regression_points'] > 0
        assert value['oracle']['target_sgp_gradient'] > 0
        mode = cfg['cold_oracle']['mode']
        assert (value['oracle']['target_classification_gradient'] > 0) == (mode != 'localization')
        assert (value['oracle']['target_regression_gradient'] > 0) == (mode != 'classification')
        phases[str(epoch)] = value
        ctx.train_loader_src, ctx.train_loader_tgt = loaders
    state = base.capture(ctx, cfg, 5, [])
    buffer = io.BytesIO(); torch.save(state, buffer); buffer.seek(0)
    restored = torch.load(buffer, map_location='cpu'); buffer.close()
    base.restore(ctx, restored)
    assert base.tensor_digest(base.capture(ctx, cfg, 5, [])) == base.tensor_digest(restored)
    def draws():
        import random
        import numpy as np
        return dict(py=random.random(), np=np.random.rand(3), cpu=torch.rand(3),
            cuda=torch.rand(3, device='cuda:0'), loaders={k:torch.randperm(8,generator=g) for k,g in ctx.generators.items()})
    base.restore(ctx, restored); draw = base.tensor_digest(draws())
    base.restore(ctx, restored); assert draw == base.tensor_digest(draws())
    save(entry/'result.json', dict(status='passed', schema=SCHEMA, mode=mode, shift=cfg['baseline']['shift'],
        cold_initial_state_sha=ctx.initial_state_sha, cold_initial_detector_sha=ctx.initial_detector_sha,
        mature_checkpoint_loaded=False, disabled_all_losses_exact=True, phases=phases,
        full_state_and_RNG_restore=True, peak_mib=max(x['peak_mib'] for x in phases.values()),
        guard=guard, guard_sha256=sha(guard), code=code_identity(), formal_training=False))


def train(cfg, entry):
    import torch
    from tools import baseline_engine as base
    ctx = build(cfg)
    ctx.cold_oracle_ledger = str(entry/'oracle_events.jsonl')
    save(entry/'runtime_config.json', ctx.cfg)
    save(entry/'seed_proof.json', ctx.cuda_seed_proof)
    save(entry/'cold_initialization.json', dict(initial_state_sha=ctx.initial_state_sha,
        initial_detector_sha=ctx.initial_detector_sha, mature_checkpoint_loaded=False,
        new_trainable_parameters=0, reference=cfg['cold_oracle']['reference']))
    history, rows, best = [], [], float('inf')
    for epoch in range(1, 51):
        base.epoch_setup(ctx, epoch)
        checked = ctx.checked_steps; started = time.monotonic()
        ctx._train_epoch(epoch); torch.cuda.synchronize()
        assert ctx.checked_steps-checked == cfg['baseline']['epoch_budget']['iterations_per_epoch']
        row = epoch_metrics(ctx, epoch, time.monotonic()-started)
        if epoch % 5 == 0:
            ctx.valid_metrics_src.reset()
            value = float(ctx._valid_epoch(epoch, ctx.val_loader_src, domain='src'))
            assert math.isfinite(value)
            history.append([epoch, value]); row['source_validation_loss'] = value
        rows.append(row)
        base.checkpoint(entry/'latest.pt', base.capture(ctx, cfg, epoch, history))
        if epoch % 5 == 0:
            os.link(entry/'latest.pt', entry/('epoch%03d.pt'%epoch))
            if value < best:
                best = value
                temp = entry/'best.pt.tmp'; os.link(entry/'latest.pt', temp); temp.replace(entry/'best.pt')
        save(entry/'metrics.json', rows)
        print('COLD_ORACLE_METRICS', json.dumps(row, allow_nan=False), flush=True)
        if epoch % 5 == 0 and decision(history)['reason'] != 'running':
            save(entry/'termination.json', dict(history=history, decision=decision(history),
                initial_state_sha=ctx.initial_state_sha, initialization='cold', mature_checkpoint_loaded=False))
            break
    assert (entry/'termination.json').exists()


def infer(cfg, entry, epochs, domains, prefix):
    import torch
    import numpy as np
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    from tools import baseline_engine as base
    fix_random_seed(123, include_cuda=True); torch.cuda.manual_seed(123)
    model = make_meta_arch('sada', deepcopy(cfg), device='cuda:0').cuda(); model.eval()
    from tools.cold_oracle_fastpath import install as install_fastpath
    install_fastpath(model)
    reports = []
    for epoch in epochs:
        path = entry/('epoch%03d.pt'%epoch)
        saved = torch.load(path, map_location='cpu')
        assert saved['epoch'] == epoch
        model.load_state_dict(saved['ema_state_dict'], strict=True)
        for key,value in saved['ema_controls'].items():
            if key != 'cold_oracle_state':
                current = getattr(model, key)
                if torch.is_tensor(current): current.data.copy_(value)
                elif isinstance(current, list):
                    for a,b in zip(current,value): a.data.copy_(b)
                else: setattr(model, key, deepcopy(value))
        for domain in domains:
            manifest = entry/(domain+'_val_unlabeled.json')
            assert all('annotations' not in x for x in read(manifest)['database'].values())
            ds = base.dataset(cfg, manifest, cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'], False, True)
            columns = {k:[] for k in ['video-id','t-start','t-end','label','score']}
            with torch.no_grad():
                for index in range(len(ds)):
                    video = base.move_to_device([ds[index]], 'cuda:0')
                    assert not video[0]['segments'].numel() and not video[0]['labels'].numel()
                    pred = model(video_list_src=video, video_list_tgt=video, mode='test', domain=domain,
                        curr_epoch=epoch)[0 if domain=='source' else 1][0]
                    columns['video-id'].extend([pred['video_id']]*len(pred['scores']))
                    for name,value in [('t-start',pred['segments'][:,0]),('t-end',pred['segments'][:,1]),('label',pred['labels']),('score',pred['scores'])]:
                        columns[name].extend(value.cpu().tolist())
            prediction = entry/('predictions/%s_e%03d.npz'%(domain,epoch)); prediction.parent.mkdir(exist_ok=True)
            np.savez_compressed(prediction, **{k:np.asarray(v) for k,v in columns.items()})
            reports.append(dict(epoch=epoch, domain=domain, prediction=str(prediction), prediction_sha256=sha(prediction), checkpoint=str(path), checkpoint_sha256=sha(path)))
    save(entry/prefix, dict(reports=reports, GT_loaded=False, mature_checkpoint_loaded=False, code=code_identity()))


def evaluate_cpu(cfg, entry, manifest, output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    from libs.evaluation import ANETdetection
    annotation = cfg['baseline']['annotation_for_offline_evaluation']
    assert sha(annotation) == cfg['baseline']['annotation_sha256']
    results = []
    for row in read(entry/manifest)['reports']:
        assert sha(row['prediction']) == row['prediction_sha256']
        with np.load(row['prediction'], allow_pickle=False) as source:
            pred = {k:source[k] for k in source.files}
        evaluator = ANETdetection(annotation, cfg['src_val_split'] if row['domain']=='source' else cfg['tgt_val_split'],
            num_workers=1, action_subsampling='10_majority_classes')
        ap,mean,*_ = evaluator.evaluate(pred, verbose=False)
        results.append(dict(row, map=float(mean), map_by_tiou=[float(x) for x in ap]))
    save(entry/output, dict(results=results, target_GT_scope='CPU_final_metrics_only'))


def audit(entry):
    import torch
    cfg = read(entry/'config.json'); complete = read(entry/'complete.json')
    assert complete['schema'] == SCHEMA and complete['code'] == code_identity()
    assert cfg['cold_oracle']['spec'] == SPEC
    original = read(Path(cfg['cold_oracle']['reference'])/'config.json')
    paired = deepcopy(cfg); paired.pop('cold_oracle')
    for key in ('code','tmpdir','schema'): paired['baseline'][key] = original['baseline'][key]
    assert paired == original
    for path,digest in complete['artifacts'].items(): assert sha(path) == digest, path
    terminal = read(entry/'termination.json'); selection = decision(terminal['history'])
    assert terminal['decision'] == selection and selection['reason'] != 'running'
    latest = torch.load(entry/'latest.pt', map_location='cpu'); best = torch.load(entry/'best.pt', map_location='cpu')
    assert latest['epoch'] == selection['epoch'] and best['epoch'] == selection['best_epoch']
    assert latest['history'] == terminal['history'] and len(latest['optimizers']) == len(latest['schedulers']) == 3
    validate_cuda_rng(latest['rng']['cuda'])
    identity = read(entry/'cold_initialization.json')
    assert identity['initial_state_sha'] == read(Path(cfg['cold_oracle']['reference'])/'termination.json')['initial_state_sha']
    assert not identity['mature_checkpoint_loaded']
    truth = read(cfg['cold_oracle']['truth']); members = set(truth['records'])
    assert sha(cfg['cold_oracle']['truth']) == cfg['cold_oracle']['truth_sha256'] and not members.intersection(truth['heldout_ids'])
    digest, steps = '0'*64, 0
    for line in (entry/'oracle_events.jsonl').open():
        event = json.loads(line); given = event.pop('digest'); steps += 1
        assert event['step'] == steps and all(x['video'] in members for x in event['observations'])
        digest = __import__('hashlib').sha256((digest+json.dumps(event, sort_keys=True)).encode()).hexdigest()
        assert digest == given
    state = latest['model_controls']['cold_oracle_state']
    assert steps == state['step'] == selection['epoch']*cfg['baseline']['epoch_budget']['iterations_per_epoch'] and state['digest'] == digest
    assert state['events'] > 0
    for domain in ('source','target'):
        assert all('annotations' not in x for x in read(entry/(domain+'_val_unlabeled.json'))['database'].values())
    for stage in ('train','inference'):
        receipt = read(entry/(stage+'_gpu.json')); guard = read(receipt['guard']); lease = read(guard['lease'])
        assert sha(receipt['guard']) == receipt['guard_sha256'] and sha(guard['lease']) == guard['lease_sha256']
        assert guard['owner_pid'] == lease['owner_pid'] and guard['observed']['uuid'] == lease['observed']['uuid']
        assert guard['observed']['free_mib'] >= lease['minimum_free_mib'] >= 12288
    final = read(entry/'evaluation/final.json')['results']
    primary = next(x for x in final if x['domain']=='target' and x['epoch']==selection['best_epoch'])
    signals = [read(entry/('signals/e%03d/evaluation.json'%epoch))['results'][0] for epoch,_ in terminal['history']]
    return dict(mode=cfg['cold_oracle']['mode'], shift=cfg['baseline']['shift'], map=primary['map'],
        best_epoch=selection['best_epoch'], stop_epoch=selection['epoch'], source_signals=signals,
        complete_sha256=sha(entry/'complete.json'), not_formal_UDA=True)


def worker(args):
    cfg = read(args.entry/'config.json'); assert cfg['baseline']['code'] == code_identity()
    if args.stage == 'evaluate_cpu':
        evaluate_cpu(cfg, args.entry, Path('inference_manifest.json'), Path('evaluation/final.json')); return
    if args.stage == 'signal_cpu':
        evaluate_cpu(cfg, args.entry, Path('signals/e%03d/manifest.json'%args.epoch), Path('signals/e%03d/evaluation.json'%args.epoch)); return
    from tools.phase_runner_gpu import first_cuda_guard
    guard = first_cuda_guard(); os.chdir(args.data_root)
    if args.stage == 'probe': probe(cfg, args.entry, guard)
    elif args.stage == 'train': train(cfg, args.entry)
    elif args.stage == 'signal': infer(cfg, args.entry, [args.epoch], ['source'], Path('signals/e%03d/manifest.json'%args.epoch))
    else:
        terminal = read(args.entry/'termination.json')['decision']
        # source已有每5轮旁路；最终只写target，避免并发覆盖同一source预测。
        infer(cfg, args.entry, sorted({terminal['best_epoch'],terminal['epoch']}), ['target'], Path('inference_manifest.json'))
    receipt = args.entry/('signal_%03d_gpu.json'%args.epoch if args.stage=='signal' else args.stage+'_gpu.json')
    save(receipt, dict(guard=guard, guard_sha256=sha(guard), code=code_identity(),
        GT_scope='target_train_supervised_Oracle_only' if args.stage in ('probe','train') else 'no_GT'))


def main(args):
    from tools.run_phase_boundary import reuse_reference_snapshot
    from tools.phase_runner_gpu import acquire
    from tools.baseline_gpu import snapshot
    root = args.output; root.mkdir(exist_ok=False)
    identity = code_identity(); references = reuse_reference_snapshot(root)
    tmp = Path('/vepfs/vlms/tmp')/('cold_oracle_'+identity['commit'][:7]); tmp.mkdir(exist_ok=False)
    configs = {}
    for shift in SHIFTS:
        original = read(REFERENCE/shift/'config.json'); truth = root/'oracle_train_truth'/(shift+'.json')
        prepare_truth(original, truth)
        for mode in MODES:
            cfg = deepcopy(original); cfg['baseline'].update(code=identity, tmpdir=str(tmp), schema=SCHEMA)
            cfg['cold_oracle'] = dict(spec=SPEC, mode=mode, reference=str(REFERENCE/shift), truth=str(truth), truth_sha256=sha(truth), target_train_GT_used=True)
            configs[mode,shift] = cfg
            for folder in ('probes','formal'):
                entry = root/folder/mode/shift; entry.mkdir(parents=True); save(entry/'config.json', cfg)
                if folder=='formal':
                    for domain in ('source','target'):
                        save(entry/(domain+'_val_unlabeled.json'), read(root/'reference_snapshot/formal'/shift/(domain+'_val_unlabeled.json')), sort_keys=False)
    save(root/'plan.json', dict(schema=SCHEMA, spec=SPEC, count=18, modes=list(MODES), shifts=list(SHIFTS), code=identity,
        A_reused_not_retrained=True, old_queues_remain_stopped=True, original_libs_and_training_loop_unmodified=True))
    save(root/'inputs.json', dict(references=references, configs={m+'/'+s:sha(root/'formal'/m/s/'config.json') for m,s in configs}))
    pending = [(m,s,'probe',None) for s in SHIFTS for m in MODES]
    active, reports, done, queued_signals = {}, {}, [], set()
    formal_started = False; minimum = 12288
    def spawn(mode, shift, stage, epoch, card):
        entry = root/('probes' if stage=='probe' else 'formal')/mode/shift
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE='1',
            PYTHONHASHSEED='123', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', CUBLAS_WORKSPACE_CONFIG=':4096:8', TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp))
        if card: env.update(CUDA_VISIBLE_DEVICES=card['uuid'], SADA_BASELINE_LEASE=card['lease'], SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
        command = [sys.executable,'-u','-B',__file__,'--output',str(root),'--entry',str(entry),'--stage',stage]
        if epoch is not None: command += ['--epoch',str(epoch)]
        name = stage+('_%03d'%epoch if epoch is not None else '')+'.log'
        with (entry/name).open('x') as stream:
            process = subprocess.Popen(command, cwd=CODE, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        return dict(process=process,mode=mode,shift=shift,stage=stage,epoch=epoch,gpu=card['index'] if card else None)
    try:
        while pending or active:
            for key,task in list(active.items()):
                rc = task['process'].poll()
                if rc is None: continue
                if task.get('context'): task['context'].__exit__(None,None,None)
                del active[key]
                assert rc == 0, (key,rc)
                m,s,stage,epoch = [task[k] for k in ('mode','shift','stage','epoch')]
                entry = root/'formal'/m/s
                if stage=='probe':
                    path = root/'probes'/m/s/'result.json'; result = read(path); assert result['status']=='passed'
                    reports[m+'/'+s] = dict(path=str(path),sha256=sha(path),report=result)
                elif stage=='train': pending.append((m,s,'inference',None))
                elif stage=='inference': pending.append((m,s,'evaluate_cpu',None))
                elif stage=='signal': pending.append((m,s,'signal_cpu',epoch))
            if len(reports)==18 and not formal_started:
                minimum = max(12288, math.ceil(max(x['report']['peak_mib'] for x in reports.values())+4096))
                save(root/'acceptance.json', dict(status='passed', reports=reports, minimum_free_mib=minimum))
                pending.extend((m,s,'train',None) for s in SHIFTS for m in MODES)
                formal_started = True
            if formal_started:
                for m,s in configs:
                    entry = root/'formal'/m/s
                    for path in sorted(entry.glob('epoch*.pt')):
                        epoch = int(path.stem[5:]); token = (m,s,epoch)
                        if token not in queued_signals:
                            queued_signals.add(token); pending.insert(0,(m,s,'signal',epoch))
                    if (entry/'evaluation/final.json').exists() and not (entry/'complete.json').exists():
                        busy = any(t['mode']==m and t['shift']==s for t in active.values()) or any(x[0]==m and x[1]==s for x in pending)
                        if not busy:
                            paths = [p for p in entry.rglob('*') if p.is_file() and p.name!='complete.json']
                            save(entry/'complete.json',dict(schema=SCHEMA,code=identity,artifacts={str(p):sha(p) for p in paths}))
                            done.append(audit(entry)); save(root/'progress.json',dict(count=len(done),results=done))
            priority = {'signal':0,'signal_cpu':0,'inference':1,'evaluate_cpu':1,'probe':2,'train':3}
            for m,s,stage,epoch in sorted(list(pending),key=lambda x:priority[x[2]]):
                card=context=None
                if stage in ('evaluate_cpu','signal_cpu'):
                    if sum(t['stage'] in ('evaluate_cpu','signal_cpu') for t in active.values())>=2: continue
                else:
                    is_signal = stage=='signal'
                    if is_signal and any(t['stage']=='signal' for t in active.values()): continue
                    if not is_signal and sum(t['gpu'] is not None and t['stage']!='signal' for t in active.values())>=8: continue
                    counts = {g:sum(t['gpu']==g for t in active.values()) for g in range(4)}
                    cards = [g for g in sorted(counts,key=counts.get) if counts[g]<(3 if is_signal else 2)]
                    cards = [g for g in cards if not (stage in ('probe','train') and any(t['gpu']==g and t['mode']==m and t['stage'] in ('probe','train') for t in active.values()))]
                    cards = [g for g in cards if snapshot(g)['free_mib']>=minimum*(1+counts[g])]
                    found = acquire(cards,args.data_root,root,minimum,identity['commit'])
                    if not found: continue
                    context,card = found
                task=spawn(m,s,stage,epoch,card)
                if context: task['context']=context
                active['%s/%s/%s/%s'%(m,s,stage,epoch)] = task
                pending.remove((m,s,stage,epoch))
            save(root/'state.json',dict(pid=os.getpid(),phase='formal' if formal_started else 'GPU_acceptance',
                formal_started=formal_started,pending=pending,complete_count=len(done),
                active={k:{'pid':t['process'].pid,**{x:t[x] for x in ('mode','shift','stage','epoch','gpu')}} for k,t in active.items()}))
            time.sleep(3)
        assert len(done)==18 and len({(x['mode'],x['shift']) for x in done})==18
        save(root/'final.json',dict(schema=SCHEMA,count=18,results=done,means={m:sum(x['map'] for x in done if x['mode']==m)/6 for m in MODES},scope=SPEC['scope']))
    except BaseException as error:
        save(root/'error.json',dict(type=type(error).__name__,message=str(error))); raise
    finally:
        for task in active.values():
            if task['process'].poll() is None: os.killpg(task['process'].pid,signal.SIGTERM)
        for task in active.values():
            try: task['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(task['process'].pid,signal.SIGKILL); task['process'].wait()
            if task.get('context'): task['context'].__exit__(None,None,None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--entry',type=Path);parser.add_argument('--stage',choices=['probe','train','inference','evaluate_cpu','signal','signal_cpu'])
    parser.add_argument('--epoch',type=int);parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    args=parser.parse_args()
    def stop(sig,frame): raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    if args.stage: worker(args)
    else: main(args)
