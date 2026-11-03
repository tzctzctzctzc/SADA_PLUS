"""六方向×两个局部头，四卡共享队列；原SADA/Prototype与源早停完整重训。"""
import argparse
from copy import deepcopy
import gc
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,termination
from tools.head_replacements import SPEC

ORIGIN=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
DATA=Path('/vepfs/vlms/SADA_PLUS')
VARIANTS=('trident',)
TASKS=tuple(f'S{s}_{v}' for s in range(1,7) for v in VARIANTS)
INTERACTION_ROOT=Path('/vepfs/vlms/head_replacements12_764a29a_1104')


def configuration(original,variant,identity,tmpdir):
    cfg=deepcopy(original)
    assert not cfg['model']['use_trident_head']
    cfg['baseline'].update(code=identity,tmpdir=str(tmpdir))
    cfg['head_replacement']=dict(variant=variant,specification=SPEC)
    return cfg


def probe(cfg,entry,binding):
    import torch
    from tools import baseline_engine as base
    from libs.trainer.da_trainer import move_to_device
    ctx=base.build(cfg)
    model=ctx.model
    assert model.head_replacement_base_sha==binding['historical_initial_state_sha']
    initial=ctx.initial_state_sha
    reports={}
    # 实测原数据流的早期/对抗阶段；probe不转移到正式训练。
    for epoch in (1,cfg['train_cfg']['warmup_adv']):
        base.epoch_setup(ctx,epoch)
        loaders=ctx.train_loader_src,ctx.train_loader_tgt
        count=min(8,cfg['baseline']['epoch_budget']['iterations_per_epoch'])
        ctx.train_loader_src,ctx.train_loader_tgt=[base.StepLoader(loader,count) for loader in loaders]
        before=base.tensor_digest(model.state_dict())
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        ctx._train_epoch(epoch);torch.cuda.synchronize()
        seconds=time.monotonic()-started
        metrics=ctx.train_metrics.result()
        assert before!=base.tensor_digest(model.state_dict()) and metrics['t_final_loss']>0
        reports[str(epoch)]=dict(seconds=seconds,steps=count,loss=float(metrics['t_final_loss']),
            projected_epoch_seconds=seconds/count*cfg['baseline']['epoch_budget']['iterations_per_epoch'],
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        ctx.train_loader_src,ctx.train_loader_tgt=loaders
    source=move_to_device(next(iter(ctx.train_loader_src)),model.device)
    target=move_to_device(next(iter(ctx.train_loader_tgt)),model.device)
    epoch=cfg['train_cfg']['warmup_adv']
    def step():
        for opt in base.optimizers(ctx):opt.zero_grad(set_to_none=True)
        model.train()
        losses=model(video_list_src=source,video_list_tgt=target,mode='train',curr_epoch=epoch)[0]
        weight=cfg['model']['loss_weights']['weight_adv']
        loss=losses['downstream_loss']*cfg['model']['loss_weights']['weight_downstream']
        if cfg['model'].get('use_adv_discriminator',True):
            for method,name in (('conditional_rev','loss_conditional_adv_rev'),('reverse','loss_adv_rev')):
                if method in cfg['model']['adv_method']:loss=loss+weight*losses[name]
        loss.backward()
        names=(['start_head.cls_head.conv.weight','end_head.cls_head.conv.weight','reg_head.offset_head.conv.weight']
               if cfg['head_replacement']['variant']=='trident' else
               ['task_interaction.cls_output.weight','task_interaction.reg_output.weight','task_interaction.cls_gate.weight','task_interaction.reg_gate.weight'])
        params=dict(model.named_parameters())
        gradients={name:params[name].grad.detach().clone() for name in names}
        assert all(torch.isfinite(g).all() and g.norm()>0 for g in gradients.values())
        clip=cfg['model']['weight_clip_value']
        if clip>0:
            for module in list(model.disc_dom_adv)+list(model.disc_dom_conditional_adv):
                for parameter in module.parameters():parameter.data.clamp_(-clip,clip)
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train_cfg']['clip_grad_l2norm'])
        ctx.optimizer_disc.step();ctx.optimizer_gen_source_alignment.step();ctx.model_ema.update(model)
        for scheduler in base.schedulers(ctx):scheduler.step()
        return base.tensor_digest(dict(loss=loss.detach(),gradients=gradients))
    buffer=io.BytesIO();torch.save(base.capture(ctx,cfg,epoch,[]),buffer);buffer.seek(0)
    saved=torch.load(buffer,map_location='cpu');buffer.close()
    base.checkpoint(entry/'probe_state.pt',saved)
    expected_step=step();expected_state=base.tensor_digest(base.capture(ctx,cfg,epoch,[]))
    base.restore(ctx,saved)
    assert expected_step==step()
    assert expected_state==base.tensor_digest(base.capture(ctx,cfg,epoch,[]))
    save(entry/'probe.json',dict(passed=True,initial_state_sha=initial,
        historical_base_initial_sha=model.head_replacement_base_sha,
        shared_initialization_matches_historical=True,full_next_update_replayed=True,
        replay_sha256=expected_state,probe_state_sha256=sha(entry/'probe_state.pt'),phases=reports,
        target_GT=False,probe_parameters_transferred=False))
    del ctx,model,saved;gc.collect();torch.cuda.empty_cache()
    return initial


def worker(entry,stage):
    from tools.baseline_gpu import first_cuda_guard
    cfg,binding=read(entry/'config.json'),read(entry/'origin_binding.json')
    assert cfg['baseline']['code']==code_identity()
    for path,digest in binding['inputs'].items():assert sha(path)==digest
    os.chdir(DATA);guard=first_cuda_guard();started=time.monotonic()
    from tools import baseline_engine as base
    if stage=='train':
        initial=probe(cfg,entry,binding)
        original_build=base.build
        def checked(config):
            ctx=original_build(config)
            assert ctx.initial_state_sha==initial
            assert ctx.model.head_replacement_base_sha==binding['historical_initial_state_sha']
            save(entry/'formal_initialization.json',dict(passed=True,initial_state_sha=initial,
                base_initial_sha=ctx.model.head_replacement_base_sha,loaded_trained_checkpoint=False))
            return ctx
        base.build=checked
        try:base.train(cfg,entry)
        finally:base.build=original_build
        termination(read(entry/'termination.json')['history'])
    else:base.evaluate(cfg,entry,cfg['baseline']['annotation_for_offline_evaluation'])
    save(entry/(stage+'_complete.json'),dict(passed=True,seconds=time.monotonic()-started,
         guard=guard,guard_sha256=sha(guard),config_sha256=sha(entry/'config.json')))


def run(root):
    import fcntl
    from tools.baseline_gpu import snapshot
    identity=code_identity();root.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(code=identity,specification=SPEC,tasks=TASKS,
        initialization='original seed123 fresh random, unchanged base hash, fork_rng for new heads',
        protocol='original batch2/3optimizers/schedules/EMA/source every5 patience15 max50',
        selection='source cls+reg validation only',target_GT_training=False,
        comparison='two separate changes vs existing corresponding SADA+Prototype, no VAE stacking',
        note='Trident preserves original focal; interaction is head-only TOOD-inspired adaptation, not full TOOD',
        resource='four private slots, min16384MiB, external tasks untouched, no automatic retry of failed runs'))
    for s in range(1,7):
        shift=f'S{s}';old=read(ORIGIN/shift/'config.json');term=read(ORIGIN/shift/'termination.json')
        inputs={p:sha(p) for p in (old['dataset']['json_file'],old['baseline']['target_manifest'],old['prototype_capacity']['prepared_cache'])}
        for path,key in ((old['dataset']['json_file'],'source_manifest_sha256'),(old['baseline']['target_manifest'],'target_manifest_sha256')):
            assert inputs[path]==old['baseline'][key]
        assert inputs[old['prototype_capacity']['prepared_cache']]==old['prototype_capacity']['prepared_cache_sha256']
        for variant in VARIANTS:
            entry=root/(shift+'_'+variant);entry.mkdir();(entry/'evaluation').mkdir()
            save(entry/'config.json',configuration(old,variant,identity,root/'tmp'))
            save(entry/'origin_binding.json',dict(origin=str(ORIGIN/shift),inputs=inputs,
                historical_initial_state_sha=term['initial_state_sha'],original_config_sha256=sha(ORIGIN/shift/'config.json'),
                historical_target_map=read(ORIGIN/shift/'evaluation/target_val.json')['map_0.1_to_0.5']))
    (root/'tmp').mkdir();pending=list(TASKS);waiting=[];active={};completed=[];failed={}
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for item in active.values():
            if item['p'].poll() is None:os.killpg(item['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def start(task,stage,gpu,lease,lock):
        entry=root/task
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['observed']['uuid'],PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=str(CODE),PYTHONHASHSEED='123',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',SADA_BASELINE_LEASE=lease['path'],
            SADA_BASELINE_LEASE_SHA256=sha(lease['path']),TMPDIR=str(root/'tmp'),TMP=str(root/'tmp'),TEMP=str(root/'tmp'))
        with (entry/(stage+'.log')).open('x') as output:
            process=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(entry),'--stage',stage],
                cwd=CODE,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        active[task]=dict(p=process,stage=stage,gpu=gpu,lease=lease,lock=lock)
    try:
        while pending or waiting or active:
            for task,item in list(active.items()):
                if item['p'].poll() is None:continue
                del active[task]
                if item['p'].returncode:
                    failed[task]=dict(stage=item['stage'],exit_code=item['p'].returncode);item['lock'].close();continue
                if item['stage']=='train':waiting.append((task,item));continue
                files=['config.json','origin_binding.json','probe.json','formal_initialization.json','train_complete.json',
                    'evaluate_complete.json','metrics.jsonl','termination.json','best.pt','latest.pt',
                    'evaluation/source_val.json','evaluation/target_val.json','evaluation/source_predictions.pt','evaluation/target_predictions.pt']
                save(root/task/'complete.json',dict(passed=True,artifacts={f:sha(root/task/f) for f in files}))
                completed.append(task);item['lock'].close()
            while waiting and sum(v['stage']=='evaluate' for v in active.values())<2:
                task,item=waiting.pop(0);start(task,'evaluate',item['gpu'],item['lease'],item['lock'])
            used=[v['gpu'] for v in active.values()]+[v['gpu'] for _,v in waiting]
            # 已有交互头不重训、不打断。两父进程不争用同一自有GPU槽。
            other=read(INTERACTION_ROOT/'state.json')
            if other['pending']:
                used=list(range(4))
            else:
                used += [x['gpu'] for x in other['active'].values()]+[
                    read(INTERACTION_ROOT/'state.json')['active'][t]['gpu']
                    for t in other.get('waiting_evaluate',[]) if t in other['active']]
            for gpu in range(4):
                if not pending or gpu in used:continue
                observed=snapshot(gpu)
                if observed['free_mib']<16384:continue
                path=root/f'private_gpu_{gpu}.lock';lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                leasepath=root/'leases'/(str(gpu)+'_'+str(time.time_ns())+'.json')
                lease=dict(owner_pid=os.getpid(),observed=observed,lock=str(path),minimum_free_mib=16384,commit=identity['commit'])
                save(leasepath,lease);lease['path']=str(leasepath);start(pending.pop(0),'train',gpu,lease,lock)
            save(root/'state.json',dict(owner_pid=os.getpid(),pending=pending,waiting_evaluate=[t for t,_ in waiting],
                active={t:dict(pid=x['p'].pid,stage=x['stage'],gpu=x['gpu']) for t,x in active.items()},completed=completed,failed=failed))
            time.sleep(5)
        results={task:dict(source=read(root/task/'evaluation/source_val.json'),target=read(root/task/'evaluation/target_val.json'),
                  complete_sha256=sha(root/task/'complete.json')) for task in completed}
        save(root/'final.json',dict(passed=len(completed)==len(TASKS),completed=completed,failed=failed,results=results))
    finally:
        for item in active.values():
            if item['p'].poll() is None:os.killpg(item['p'].pid,signal.SIGTERM)
        for item in list(active.values())+[x for _,x in waiting]:item['lock'].close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path)
    p.add_argument('--stage',choices=['train','evaluate']);args=p.parse_args()
    worker(args.entry,args.stage) if args.entry else run(args.root)
