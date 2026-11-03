"""六方向SADA/Prototype从头对照，原版裁剪/排序/loader；不改活跃队列。"""
import argparse
from copy import deepcopy
import gc
import io
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,termination
from tools.release_data_protocol import SCHEMA,SPEC,manifests,configuration,assert_pair,stream_pair,release_decision

ORIGIN=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
DATA=Path('/vepfs/vlms/SADA_PLUS')
BLOCKERS=(Path('/vepfs/vlms/head_replacements12_764a29a_1104'),
          Path('/vepfs/vlms/trident_complete6_8fdd2ae_1104'))
TASKS=tuple(f'S{s}_{v}' for s in range(1,7) for v in ('sada','prototype'))


def shutdown(ctx):
    loaders=(ctx.train_loader_src,ctx.train_loader_tgt.loader,ctx.val_loader_src)
    for loader in loaders:
        iterator=getattr(loader,'_iterator',None)
        if iterator is not None:iterator._shutdown_workers();loader._iterator=None


def cpu_check(cfg,output):
    """每方向2源+2目标真实视频，与作者Dataset和truncate逐Tensor抽查。"""
    import hashlib
    import numpy as np
    import torch
    import libs.datasets.datasets as registry
    from libs.datasets import make_dataset
    from tools import baseline_engine as base
    official='a96a15b0d0c905a553c7a2b14f607c9485ebe758'
    def code(path):return subprocess.check_output(['git','show',official+':'+path],cwd=CODE,text=True)
    namespace={'__name__':'release_reference_utils'}
    exec(compile(code('libs/datasets/data_utils.py'),'author_data_utils','exec'),namespace)
    saved_class=registry.datasets['epic']
    reference={'__name__':'libs.datasets.release_reference','__package__':'libs.datasets'}
    try:exec(compile(code('libs/datasets/epic_kitchens.py'),'author_epic','exec'),reference)
    finally:registry.datasets['epic']=saved_class
    reference['truncate_feats']=namespace['truncate_feats']
    rows=[]
    for role,path,splits in [('source',cfg['dataset']['json_file'],cfg['src_train_split']),
                             ('target',cfg['baseline']['target_manifest'],cfg['tgt_train_split'])]:
        options=deepcopy(cfg['dataset']);options['json_file']=str(path)
        actual=make_dataset('epic',True,splits,**options)
        author=reference['EpicKitchensDataset'](True,splits,**options)
        assert len(actual)==len(author)
        ids=sorted(range(len(actual)),key=lambda i:hashlib.sha256(
            ('release-data-check|'+actual.data_list[i]['id']).encode()).hexdigest())[:2]
        for i in ids:
            def seed():random.seed(1104+i);np.random.seed(1104+i);torch.manual_seed(1104+i)
            seed();left=author[i];left_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
            seed();right=actual[i];right_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
            assert all(torch.equal(left[k],right[k]) for k in ('feats','segments','labels'))
            assert base.tensor_digest(left_rng)==base.tensor_digest(right_rng)
            rows.append(dict(role=role,video=right['metadata']['video_id'],
                original_crop_sort_tensors_exact=True,random_draws_exact=True,
                target_GT_count=right['labels'].numel() if role=='target' else None))
    save(output,dict(passed=True,checks=rows,scope='4 real videos per shift, not all video crops',
        retained_metadata_fix='keep_feat_num_frames_and_record_crop_start_length',reference_commit=official))


def objective(losses,cfg,epoch):
    if epoch<cfg['train_cfg']['warmup_adv']:return losses['downstream_loss']
    loss=cfg['model']['loss_weights']['weight_downstream']*losses['downstream_loss']
    if cfg['model'].get('use_adv_discriminator',True):
        for method,key in [('conditional_rev','loss_conditional_adv_rev'),('reverse','loss_adv_rev')]:
            if method in cfg['model']['adv_method']:loss=loss+cfg['model']['loss_weights']['weight_adv']*losses[key]
    return loss


def probe(cfg,entry,binding):
    import torch
    from tools import baseline_engine as base
    from libs.trainer.da_trainer import move_to_device
    ctx=base.build(cfg);model=ctx.model;initial=ctx.initial_state_sha
    assert ctx.initial_detector_sha==binding['historical_detector_initial_sha256']
    assert ctx.generators['source'] is ctx.generators['target'] is torch.default_generator
    assert ctx.train_loader_src.persistent_workers and ctx.train_loader_tgt.loader.persistent_workers
    reports={}
    for epoch in (1,cfg['train_cfg']['warmup_adv']):
        base.epoch_setup(ctx,epoch);loaders=ctx.train_loader_src,ctx.train_loader_tgt
        steps=min(8,cfg['baseline']['epoch_budget']['iterations_per_epoch'])
        ctx.train_loader_src,ctx.train_loader_tgt=[base.StepLoader(x,steps) for x in loaders]
        before=base.tensor_digest(model.state_dict())
        torch.cuda.synchronize();started=time.monotonic();ctx._train_epoch(epoch);torch.cuda.synchronize()
        seconds=time.monotonic()-started;metrics=ctx.train_metrics.result()
        assert before!=base.tensor_digest(model.state_dict()) and metrics['t_final_loss']>0
        reports[str(epoch)]=dict(seconds=seconds,steps=steps,loss=float(metrics['t_final_loss']),
            projected_epoch_seconds=seconds/steps*cfg['baseline']['epoch_budget']['iterations_per_epoch'],
            auxiliary={k:float(v) for k,v in metrics.items() if k.startswith('t_rq3_')})
        ctx.train_loader_src,ctx.train_loader_tgt=loaders
    source=move_to_device(next(iter(ctx.train_loader_src)),model.device)
    target=move_to_device(next(iter(ctx.train_loader_tgt)),model.device)
    epoch=cfg['train_cfg']['warmup_adv']
    # 严格固定同一批输入：删除目标标签不会改变实际优化目标和全部参数梯度。
    buffer=io.BytesIO();torch.save(base.capture(ctx,cfg,epoch,[]),buffer);buffer.seek(0)
    saved=torch.load(buffer,map_location='cpu');buffer.close()
    base.checkpoint(entry/'probe_state.pt',saved)
    def backward(batch):
        for opt in base.optimizers(ctx):opt.zero_grad(set_to_none=True)
        model.train();loss=objective(model(video_list_src=source,video_list_tgt=batch,mode='train',curr_epoch=epoch)[0],cfg,epoch)
        loss.backward()
        grads={n:p.grad for n,p in model.named_parameters() if p.grad is not None}
        assert grads and all(torch.isfinite(g).all() for g in grads.values())
        return base.tensor_digest(dict(loss=loss.detach(),grads=grads))
    base.restore(ctx,saved);labeled=backward(target)
    empty=[dict(v,segments=v['segments'].new_empty((0,2)),labels=v['labels'].new_empty((0,))) for v in target]
    base.restore(ctx,saved);assert labeled==backward(empty)
    def update():
        proof=backward(target)
        clip=cfg['model']['weight_clip_value']
        if clip>0:
            for module in list(model.disc_dom_adv)+list(model.disc_dom_conditional_adv):
                for p in module.parameters():p.data.clamp_(-clip,clip)
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train_cfg']['clip_grad_l2norm'])
        ctx.optimizer_disc.step();ctx.optimizer_gen_source_alignment.step();ctx.model_ema.update(model)
        for scheduler in base.schedulers(ctx):scheduler.step()
        return proof,base.tensor_digest(base.capture(ctx,cfg,epoch,[]))
    base.restore(ctx,saved);expected=update();base.restore(ctx,saved);assert expected==update()
    save(entry/'probe.json',dict(passed=True,initial_state_sha256=initial,
        detector_initial_sha256=ctx.initial_detector_sha,phases=reports,
        target_labels_removed_same_objective_and_all_gradients=True,
        fixed_batch_full_next_update_exact=True,replay_sha256=expected[1],
        probe_state_sha256=sha(entry/'probe_state.pt'),shared_default_generator=True,persistent_workers=True,
        loader_restart_equivalence_claimed=False,probe_parameters_transferred=False))
    shutdown(ctx);del ctx,model,saved;gc.collect();torch.cuda.empty_cache()
    return initial


def worker(entry,stage):
    from tools.baseline_gpu import first_cuda_guard
    from tools import baseline_engine as base
    from tools.native_ap_fast import install
    efficiency=read(entry.parent/'efficiency_handoff.json')
    proof=read(efficiency['optimization_proof'])
    assert proof['passed'] and proof['fast_code_sha256']==sha(CODE/'tools/native_ap_fast.py')
    install()
    cfg=read(entry/'config.json');binding=read(entry/'origin_binding.json')
    assert cfg['baseline']['code']==code_identity()
    for path,digest in binding['inputs'].items():assert sha(path)==digest
    os.chdir(DATA);guard=first_cuda_guard();started=time.monotonic()
    if stage=='train':
        initial=probe(cfg,entry,binding);original=base.build
        def fresh(config):
            ctx=original(config);assert ctx.initial_state_sha==initial
            assert ctx.initial_detector_sha==binding['historical_detector_initial_sha256']
            save(entry/'formal_initialization.json',dict(passed=True,initial_state_sha256=initial,
                detector_initial_sha256=ctx.initial_detector_sha,loaded_trained_weights=False))
            return ctx
        base.build=fresh
        from tools.release_training import train
        try:train(cfg,entry)
        finally:base.build=original
        assert release_decision(read(entry/'termination.json')['history'])['reason']!='running'
    else:base.evaluate(cfg,entry,cfg['baseline']['annotation_for_offline_evaluation'])
    save(entry/(stage+'_complete.json'),dict(passed=True,seconds=time.monotonic()-started,
        guard=guard,guard_sha256=sha(guard),config_sha256=sha(entry/'config.json')))


def prepare(root):
    identity=code_identity();root.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(code=identity,specification=SPEC,tasks=TASKS,
        changes='restore target GT crop, interval column sort, original loader and native Trainer validation/stopping; all AP uses full matching GT',
        comparison='both new arms share release data; old corrected results remain separate',
        prototype='unchanged control branch and frozen 40/200 teacher evidence',
        target_GT='available to original cropping/bookkeeping; no new target detection supervision',
        source_validation='native BaseTrainerDA.train every2 first_non_improvement max50; target_train validation precedes source_val',
        evaluation_scope='full prediction videos and full corresponding GT; never first subset only',
        efficiency='persistent original workers and non-mutating CPU cycle cache; no batch/precision/budget changes',
        restore='no automatic restart; persistent worker RNG cannot be reconstructed from standard checkpoints',
        blockers=[str(x) for x in BLOCKERS]))
    for s in range(1,7):
        shift=f'S{s}';origin=ORIGIN/shift;old=read(origin/'config.json')
        annotation=old['baseline']['annotation_for_offline_evaluation'];assert sha(annotation)==old['baseline']['annotation_sha256']
        db=read(annotation)['database'];source,target=manifests(db,old)
        sp,tp=root/'inputs'/shift/'source.json',root/'inputs'/shift/'target_release_GT.json'
        save(sp,source,sort_keys=False);save(tp,target,sort_keys=False)
        # JSON排序规则可不同；视频次序/内容必须与旧源manifest逐项一致。
        assert list(source['database'].items())==list(read(old['dataset']['json_file'])['database'].items())
        assert list(target['database'])==list(read(old['baseline']['target_manifest'])['database'])
        digest=dict(source=sha(sp),target=sha(tp))
        configs={v:configuration(old,v,identity,sp,tp,root/'tmp',digest) for v in ('sada','prototype')}
        assert_pair(configs['sada'],configs['prototype'])
        cache=old['prototype_capacity']['prepared_cache'];assert sha(cache)==old['prototype_capacity']['prepared_cache_sha256']
        historical=read(origin/'detector_initialization.json')['sha256']
        binding=dict(origin=str(origin),original_config_sha256=sha(origin/'config.json'),
            historical_detector_initial_sha256=historical,
            inputs={str(p):sha(p) for p in (sp,tp,Path(annotation),Path(cache))})
        cpu_check(configs['prototype'],root/'inputs'/shift/'CPU_release_reference.json')
        for v,cfg in configs.items():
            entry=root/(shift+'_'+v);entry.mkdir();(entry/'evaluation').mkdir()
            save(entry/'config.json',cfg);save(entry/'origin_binding.json',binding)
    (root/'tmp').mkdir();save(root/'prepare_complete.json',dict(passed=True,tasks=TASKS,
        CPU_checks={f'S{s}':sha(root/'inputs'/f'S{s}'/'CPU_release_reference.json') for s in range(1,7)}))


def blocker_slots():
    used=[];states={}
    for root in BLOCKERS:
        state=read(root/'state.json');states[str(root)]=state
        if state['pending'] or state.get('waiting_evaluate'):return list(range(4)),states
        used.extend(x['gpu'] for x in state['active'].values())
    return used,states


def run(root,adoption_proof=None):
    import fcntl
    from tools.baseline_gpu import snapshot
    identity=code_identity();handoff=None;audit_process=None
    if adoption_proof:
        from tools.native_queue_handoff import adopt
        pending,waiting,active,done,failed,handoff=adopt(root,identity,adoption_proof,CODE)
    else:
        prepare(root);pending=list(TASKS);waiting=[];active={};done=[];failed={}
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for a in active.values():
            if a['p'].poll() is None:os.killpg(a['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def start(task,stage,gpu,lease,lock):
        worker_code=Path(handoff['versions'][task]) if handoff else CODE
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['observed']['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),
            PYTHONHASHSEED='123',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            SADA_BASELINE_LEASE=lease['path'],SADA_BASELINE_LEASE_SHA256=sha(lease['path']),
            TMPDIR=str(root/'tmp'),TMP=str(root/'tmp'),TEMP=str(root/'tmp'))
        env['PYTHONPATH']=str(worker_code)
        with (root/task/(stage+'.log')).open('x') as out:
            p=subprocess.Popen([sys.executable,'-u','-B',str(worker_code/'tools/run_release_data_reproduction.py'),'--entry',str(root/task),'--stage',stage],
                cwd=worker_code,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        active[task]=dict(p=p,stage=stage,gpu=gpu,lease=lease,lock=lock)
    def admission(gpu,task):
        observed=snapshot(gpu)
        if observed['free_mib']<16384:return None
        lockpath=root/f'private_fast_gpu_{gpu}.lock';lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'leases'/(str(gpu)+'_'+str(time.time_ns())+'.json')
        cfg=read(root/task/'config.json')
        lease=dict(owner_pid=os.getpid(),observed=observed,lock=str(lockpath),minimum_free_mib=16384,
                   commit=cfg['baseline']['code']['commit'])
        save(path,lease);lease['path']=str(path)
        return lease,lock
    try:
        while pending or waiting or active:
            for task,a in list(active.items()):
                if a['p'].poll() is None:continue
                del active[task]
                if a['p'].returncode:
                    failed[task]=dict(stage=a['stage'],exit_code=a['p'].returncode);a['lock'].close();continue
                if a['stage']=='train':waiting.append((task,a));continue
                entry=root/task
                files=['config.json','origin_binding.json','probe.json','probe_state.pt','formal_initialization.json',
                    'train_complete.json','evaluate_complete.json','runtime_config.json','seed_proof.json','release_runtime.json',
                    'metrics.jsonl','termination.json','best.pt','latest.pt','evaluation/source_val.json',
                    'evaluation/target_val.json','evaluation/source_predictions.pt','evaluation/target_predictions.pt']
                files.extend(str(p.relative_to(entry)) for p in sorted((entry/'streams').glob('epoch*.json')))
                files.extend(str(p.relative_to(entry)) for p in sorted((entry/'debug_evaluation').rglob('*')) if p.is_file())
                if (entry/'phase_timings.jsonl').exists():files.append('phase_timings.jsonl')
                save(entry/'complete.json',dict(passed=True,artifacts={f:sha(entry/f) for f in files}))
                done.append(task);a['lock'].close()
            while waiting and sum(a['stage']=='evaluate' for a in active.values())<2:
                task,a=waiting[0]
                if a.get('inherited'):
                    accepted=admission(a['gpu'],task)
                    if accepted is None:break
                    a['lease'],a['lock']=accepted
                waiting.pop(0);start(task,'evaluate',a['gpu'],a['lease'],a['lock'])
            if handoff and not (root/'previous_dispatcher_retired.json').exists() and not any(a.get('inherited') for a in active.values()):
                from tools.native_queue_handoff import retire_previous
                retire_previous(root,handoff)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                with (root/'independent_fast_audit.log').open('x') as out:
                    audit_process=subprocess.Popen([sys.executable,'-u','-B',str(CODE/'tools/audit_release_reproduction.py'),str(root)],
                        cwd=CODE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                save(root/'independent_fast_audit_launch.json',dict(pid=audit_process.pid,commit=identity['commit'],code=str(CODE)))
            blocked,states=blocker_slots()
            used=blocked+[a['gpu'] for a in active.values()]+[a['gpu'] for _,a in waiting]
            for gpu in range(4):
                if not pending or gpu in used:continue
                accepted=admission(gpu,pending[0])
                if accepted is None:continue
                lease,lock=accepted;start(pending.pop(0),'train',gpu,lease,lock)
            save(root/'state.json',dict(owner_pid=os.getpid(),pending=pending,waiting_evaluate=[t for t,_ in waiting],
                active={t:dict(pid=a['p'].pid,gpu=a['gpu'],stage=a['stage']) for t,a in active.items()},
                completed=done,failed=failed,blocked_by_existing_authorized_queues=bool(blocked),observed_unix=time.time()))
            time.sleep(10)
        results={t:dict(target=read(root/t/'evaluation/target_val.json'),source=read(root/t/'evaluation/source_val.json'),
            complete_sha256=sha(root/t/'complete.json')) for t in done}
        streams={}
        for s in range(1,7):
            pair=[root/f'S{s}_{v}' for v in ('sada','prototype')]
            if all((e/'complete.json').exists() for e in pair):
                streams[f'S{s}']=stream_pair(*[[json.loads(x) for x in (e/'metrics.jsonl').read_text().splitlines()] for e in pair])
        save(root/'final.json',dict(passed=len(done)==12,completed=done,failed=failed,results=results,paired_streams=streams))
    finally:
        for a in active.values():
            if a['p'].poll() is None and not a.get('inherited'):os.killpg(a['p'].pid,signal.SIGTERM)
        for a in list(active.values())+[a for _,a in waiting]:a['lock'].close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path)
    p.add_argument('--adoption-proof',type=Path)
    p.add_argument('--stage',choices=['train','evaluate']);a=p.parse_args()
    worker(a.entry,a.stage) if a.entry else run(a.root,a.adoption_proof)
