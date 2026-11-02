"""真实已保存状态的native/fast配对测速；不启动或修改正式训练。"""
import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity


def case(entry, checkpoint, output, steps):
    import torch
    from tools import baseline_engine as base
    from tools.run_prototype_cold_oracle import build
    cfg=read(entry/'config.json');ctx=build(cfg)
    saved=torch.load(checkpoint,map_location='cpu');assert saved['config']==cfg
    epoch=saved['epoch']+1;history=saved['history']
    blob=io.BytesIO();torch.save(saved,blob);serialized=blob.getvalue();del saved,blob
    loaders=ctx.train_loader_src,ctx.train_loader_tgt
    def reset(enabled,n):
        state=torch.load(io.BytesIO(serialized),map_location='cpu');base.restore(ctx,state)
        assert base.tensor_digest(base.capture(ctx,cfg,state['epoch'],history))==base.tensor_digest(state)
        ctx.model.equivalent_fast_enabled=ctx.model_ema.module.equivalent_fast_enabled=enabled
        from tools.cold_metric_readback import configure
        configure(ctx,enabled)
        ctx.cold_oracle_gradient_epoch=None;ctx.cold_oracle_ledger=None
        ctx.train_loader_src,ctx.train_loader_tgt=[base.StepLoader(x,n) for x in loaders]
        base.epoch_setup(ctx,epoch)
    # native预热不计入速度；正式比较两路径均从同一个完整保存点与RNG开始。
    reset(False,2);ctx._train_epoch(epoch);torch.cuda.synchronize()
    reports={}
    for enabled in [False,True]:
        reset(enabled,steps);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.perf_counter();ctx._train_epoch(epoch);torch.cuda.synchronize();elapsed=time.perf_counter()-start
        state=base.capture(ctx,cfg,epoch,history)
        reports['fast' if enabled else 'native']=dict(seconds=elapsed,steps=steps,
            state_sha256=base.tensor_digest(state),metrics=ctx.train_metrics.result(),
            metrics_sha256=base.tensor_digest(ctx.train_metrics.result()),peak_mib=torch.cuda.max_memory_reserved()/2**20,
            oracle_step=state['model_controls']['cold_oracle_state']['step'])
        del state
    exact=reports['native']['state_sha256']==reports['fast']['state_sha256'] and reports['native']['metrics_sha256']==reports['fast']['metrics_sha256']
    # 新建完整训练上下文，真实检查进程重建语义的恢复helper及接续结果。
    from tools.cold_oracle_resume import restore_training,validate_prefix
    recovery=output.parent/(output.stem+'_restore');recovery.mkdir()
    old=torch.load(io.BytesIO(serialized),map_location='cpu')
    rows=[x for x in read(entry/'metrics.json') if x['epoch']<=old['epoch']]
    with (entry/'oracle_events.jsonl').open('rb') as stream:
        ledger=validate_prefix(old,rows,cfg['baseline']['epoch_budget']['iterations_per_epoch'],stream)
    os.link(checkpoint,recovery/'latest.pt');save(recovery/'metrics.json',rows)
    (recovery/'oracle_events.jsonl').write_bytes(ledger)
    save(recovery/'resume.json',dict(checkpoint_sha256=sha(checkpoint),pause_receipt_sha256='GPU_test_only'))
    del old,ctx
    import gc
    gc.collect();torch.cuda.empty_cache()
    ctx=build(cfg);loaders=ctx.train_loader_src,ctx.train_loader_tgt
    recovered_history,recovered_rows,start_epoch=restore_training(ctx,cfg,recovery)
    assert start_epoch==epoch and recovered_history==history and recovered_rows==rows
    ctx.cold_oracle_gradient_epoch=None;ctx.cold_oracle_ledger=None
    ctx.train_loader_src,ctx.train_loader_tgt=[base.StepLoader(x,steps) for x in loaders]
    base.epoch_setup(ctx,epoch);ctx._train_epoch(epoch);torch.cuda.synchronize()
    restarted=dict(state_sha256=base.tensor_digest(base.capture(ctx,cfg,epoch,history)),
        metrics_sha256=base.tensor_digest(ctx.train_metrics.result()))
    restart_exact=all(restarted[k]==reports['fast'][k] for k in restarted)
    exact=exact and restart_exact
    # 真正无GT源验证视频的完整forward/NMS/voting也逐tensor核验。
    ctx.train_loader_src,ctx.train_loader_tgt=loaders
    model=ctx.model_ema.module;model.eval()
    ds=base.dataset(cfg,entry/'source_val_unlabeled.json',cfg['src_val_split'],False,True)
    video=base.move_to_device([ds[0]],'cuda:0');assert not video[0]['segments'].numel()
    rng=torch.cuda.get_rng_state(0);evaluation={}
    for enabled in [False,True]:
        model.equivalent_fast_enabled=enabled;torch.cuda.set_rng_state(rng,device=0)
        with torch.no_grad():
            model(video_list_src=video,video_list_tgt=video,mode='test',domain='src',curr_epoch=epoch)
            torch.cuda.synchronize();start=time.perf_counter()
            result=model(video_list_src=video,video_list_tgt=video,mode='test',domain='src',curr_epoch=epoch)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
        evaluation['fast' if enabled else 'native']=dict(seconds=elapsed,result_sha256=base.tensor_digest(result),rng_sha256=base.tensor_digest(torch.cuda.get_rng_state(0)))
    exact_eval=evaluation['native']['result_sha256']==evaluation['fast']['result_sha256'] and evaluation['native']['rng_sha256']==evaluation['fast']['rng_sha256']
    report=dict(status='passed' if exact and exact_eval else 'rejected',code=code_identity(),entry=str(entry),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        mode=cfg['cold_oracle']['mode'],shift=cfg['baseline']['shift'],epoch=epoch,training=reports,evaluation=evaluation,
        train_speedup=reports['native']['seconds']/reports['fast']['seconds'],eval_speedup=evaluation['native']['seconds']/evaluation['fast']['seconds'],
        full_state_model_EMA_three_optimizers_schedulers_controls_all_RNG_exact=exact,source_eval_all_outputs_and_RNG_exact=exact_eval,
        rebuilt_context_resume_exact=restart_exact,resume_proof=str(recovery/'restore_proof.json'),
        source_eval_video=video[0]['metadata']['video_id'],formal_training=False)
    save(output,report);print(json.dumps(report),flush=True)


def main(args):
    from tools.phase_runner_gpu import acquire
    root=args.output;root.mkdir(exist_ok=False);identity=code_identity()
    cases=[('joint','S6',5),('classification','S4',20),('localization','S5',5)]
    save(root/'plan.json',dict(code=identity,cases=cases,steps=args.steps,new_training=False,
        changes='discarded PL RNG replay; conditional loss host-sync removal; batch scalar readback; rebuilt-context exact resume'))
    for mode,shift,epoch in cases:
        entry=args.original/'formal'/mode/shift;checkpoint=entry/('epoch%03d.pt'%epoch)
        assert checkpoint.exists()
        found=None
        while not found:
            found=acquire([2,0,3],args.data_root,root,12288,identity['commit'])
            if not found:time.sleep(3)
        context,card=found
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=card['uuid'],PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',
            PYTHONHASHSEED='123',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
        command=[sys.executable,'-u','-B',__file__,'--output',str(root),'--entry',str(entry),'--checkpoint',str(checkpoint),'--steps',str(args.steps)]
        with (root/(mode+'_'+shift+'.log')).open('x') as stream:
            process=subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'state.json',dict(parent=os.getpid(),child=process.pid,mode=mode,shift=shift,GPU=card['index']))
        try:rc=process.wait()
        finally:context.__exit__(None,None,None)
        assert rc==0,(mode,shift,rc)
        result=read(root/(mode+'_'+shift+'.json'));assert result['status']=='passed'
    reports=[read(root/(m+'_'+s+'.json')) for m,s,e in cases]
    save(root/'final.json',dict(status='passed',count=3,results=reports,all_full_states_and_source_outputs_exact=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,default=Path('/vepfs/vlms/prototype_cold_oracle18_87c165a_1102'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--entry',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--steps',type=int,default=8);p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    args=p.parse_args()
    if args.entry:
        from tools.phase_runner_gpu import first_cuda_guard
        guard=first_cuda_guard();os.chdir(args.data_root)
        case(args.entry,args.checkpoint,args.output/(read(args.entry/'config.json')['cold_oracle']['mode']+'_'+read(args.entry/'config.json')['baseline']['shift']+'.json'),args.steps)
        save(args.output/(read(args.entry/'config.json')['cold_oracle']['mode']+'_'+read(args.entry/'config.json')['baseline']['shift']+'_gpu.json'),dict(guard=guard,guard_sha256=sha(guard)))
    else:main(args)
