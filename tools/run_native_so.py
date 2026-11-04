"""六方向新协议SO，从头训练；复用已终审UDA，避开活动Trident队列。"""
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
from tools.release_data_protocol import SCHEMA,SPEC,release_decision
from tools.native_so_protocol import source_only_configuration

DATA=Path('/vepfs/vlms/SADA_PLUS')
BLOCKERS=(Path('/vepfs/vlms/trident_repaired6_820b9e3_1104'),)
NATIVE=Path('/vepfs/vlms/sada_native_fullGT12_47f6118_1104')
EXPECTED_NATIVE_AUDIT='6cd04b4721a2a56277cc9df6d64f634d2c2efa3055aaf9511466d5ed57dde119'
CPU_PROOF=Path('/vepfs/vlms/native_efficiency_caaa3e0_1104/real_prediction_benchmark.json')
TASKS=tuple(f'S{s}_so' for s in range(1,7))


def shutdown(ctx):
    loaders=(ctx.train_loader_src,ctx.train_loader_tgt.loader,ctx.val_loader_src)
    for loader in loaders:
        iterator=getattr(loader,'_iterator',None)
        if iterator is not None:iterator._shutdown_workers();loader._iterator=None


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
    assert ctx.cfg["model"]["adv_method"]==["None"]
    source_ids={v["id"] for v in ctx.train_loader_src.dataset.data_list}
    second_ids={v["id"] for v in ctx.train_loader_tgt.loader.dataset.data_list}
    assert source_ids==second_ids
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
        assert not ctx.optimizer_disc.state, 'SO判别器不应有Adam更新状态'
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
        model.train();loss=objective(model(video_list_src=source,video_list_tgt=batch,mode='train',curr_epoch=epoch)[0],ctx.cfg,epoch)
        loss.backward()
        grads={n:p.grad for n,p in model.named_parameters() if p.grad is not None}
        assert grads and all(torch.isfinite(g).all() for g in grads.values())
        assert not any(n.startswith(('disc_dom_adv.','disc_dom_conditional_adv.','learnable_embeddings.')) for n in grads)
        return base.tensor_digest(dict(loss=loss.detach(),grads=grads))
    base.restore(ctx,saved);labeled=backward(target)
    empty=[dict(v,segments=v['segments'].new_empty((0,2)),labels=v['labels'].new_empty((0,))) for v in target]
    base.restore(ctx,saved);assert labeled==backward(empty)
    zero=[dict(v,feats=torch.zeros_like(v['feats'])) for v in empty]
    base.restore(ctx,saved);assert labeled==backward(zero), 'SO梯度不能依赖第二支输入内容'
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
        loader_restart_equivalence_claimed=False,probe_parameters_transferred=False,
        source_only_fit_members=sorted(source_ids),second_slot_source_members_exact=True,
        no_discriminator_gradients_or_Adam_updates=True,second_slot_zero_features_same_loss_and_gradients=True,
        native_UDA_complete_sha256=binding['native_complete_sha256']))
    shutdown(ctx);del ctx,model,saved;gc.collect();torch.cuda.empty_cache()
    return initial


def worker(entry,stage):
    from tools.baseline_gpu import first_cuda_guard
    from tools import baseline_engine as base
    from tools.native_ap_fast import install
    proof=read(CPU_PROOF)
    assert sha(CPU_PROOF)=='f60cd523b37af8f5850d6dfa09424e22840d43d000b52412bf2fbadeff44ceac'
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
    import shutil
    identity=code_identity()
    assert sha(NATIVE/'independent_final_audit.json')==EXPECTED_NATIVE_AUDIT
    assert read(NATIVE/'independent_final_audit.json')['passed']
    root.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(code=identity,tasks=TASKS,
        paired_native_root=str(NATIVE),paired_native_audit_sha256=EXPECTED_NATIVE_AUDIT,
        regime='source_only_fit_no_Prototype',source_budget='native_UDA_iterations_per_epoch_unchanged',
        evaluation_scope='all_prediction_videos_and_all_corresponding_GT',
        target_use='no_gradient_native_validation_debug_AP_only_never_selection',
        attribution='SO_vs_complete_UDA_pipeline; different_second_loader_changes_source_random_trajectory',
        initialization='fresh_seed123_no_trained_weights',
        resume='disabled_persistent_worker_RNG_not_reconstructible',
        blockers=[str(x) for x in BLOCKERS]))
    checks={}
    for s in range(1,7):
        shift=f'S{s}';native=NATIVE/(shift+'_sada')
        independent=read(NATIVE/'independent_results'/(shift+'_sada.json'))
        assert independent['passed'] and sha(native/'complete.json')==independent['complete_sha256']
        old=read(native/'config.json');cfg=source_only_configuration(old,identity)
        cfg['baseline']['tmpdir']=str(root/'tmp')
        entry=root/(shift+'_so');entry.mkdir();(entry/'evaluation').mkdir()
        binding=read(native/'origin_binding.json')
        binding.update(native_uda_entry=str(native),native_complete_sha256=sha(native/'complete.json'),
            native_config_sha256=sha(native/'config.json'),native_independent_sha256=sha(NATIVE/'independent_results'/(shift+'_sada.json')))
        for path,digest in binding['inputs'].items():assert sha(path)==digest
        save(entry/'config.json',cfg);save(entry/'origin_binding.json',binding)
        source=NATIVE/'inputs'/shift/'CPU_release_reference.json'
        output=root/'inputs'/shift/'CPU_release_reference.json';output.parent.mkdir(parents=True)
        shutil.copyfile(source,output);assert sha(output)==sha(source) and read(output)['passed']
        checks[shift]=sha(output)
    (root/'tmp').mkdir()
    save(root/'prepare_complete.json',dict(passed=True,tasks=TASKS,CPU_checks=checks,
        source_dataset_reference='reuse_unchanged_audited_native_four_video_checks_per_shift_not_new_full_check'))


def blocker_slots():
    used=[];states={}
    for root in BLOCKERS:
        state=read(root/'state.json');states[str(root)]=state
        if state['pending'] or state.get('waiting_evaluate'):return list(range(4)),states
        used.extend(x['gpu'] for x in state['active'].values())
    return used,states


def run(root):
    import fcntl
    from tools.baseline_gpu import snapshot
    identity=code_identity();audit_process=None
    prepare(root);pending=list(TASKS);waiting=[];active={};done=[];failed={}
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for a in active.values():
            if a['p'].poll() is None:os.killpg(a['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def start(task,stage,gpu,lease,lock):
        worker_code=CODE
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['observed']['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),
            PYTHONHASHSEED='123',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            SADA_BASELINE_LEASE=lease['path'],SADA_BASELINE_LEASE_SHA256=sha(lease['path']),
            TMPDIR=str(root/'tmp'),TMP=str(root/'tmp'),TEMP=str(root/'tmp'))
        env['PYTHONPATH']=str(worker_code)
        with (root/task/(stage+'.log')).open('x') as out:
            p=subprocess.Popen([sys.executable,'-u','-B',str(worker_code/'tools/run_native_so.py'),'--entry',str(root/task),'--stage',stage],
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
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (root/'independent_audit.log').open('x') as out:
            audit_process=subprocess.Popen([sys.executable,'-u','-B',str(CODE/'tools/audit_native_so.py'),str(root)],cwd=CODE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'independent_audit_launch.json',dict(pid=audit_process.pid,code=str(CODE),commit=identity['commit']))
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
                waiting.pop(0);start(task,'evaluate',a['gpu'],a['lease'],a['lock'])
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
        save(root/'final.json',dict(passed=len(done)==6,completed=done,failed=failed,results=results,paired_native_audit_sha256=EXPECTED_NATIVE_AUDIT))
    finally:
        for a in active.values():
            if a['p'].poll() is None:os.killpg(a['p'].pid,signal.SIGTERM)
        for a in list(active.values())+[a for _,a in waiting]:a['lock'].close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path)
    p.add_argument('--stage',choices=['train','evaluate']);a=p.parse_args()
    worker(a.entry,a.stage) if a.entry else run(a.root)
