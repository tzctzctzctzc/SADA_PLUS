"""修订版真实数据短验收：每方向40次更新，无目标GT、无最终mAP选参。"""
import argparse,copy,json,time,sys,itertools
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))


def main():
    import torch
    from tools import baseline_engine as base
    from tools.run_uda_two_routes import save
    from libs.trainer.da_trainer import move_to_device
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--entry',required=True)
    p.add_argument('--prior-entry')
    args=p.parse_args();entry=Path(args.entry);entry.mkdir(parents=True,exist_ok=True)
    with (entry/'short_claim.json').open('x') as stream:json.dump(dict(max_updates=40,formal_runs=0),stream)
    cfg=json.loads(Path(args.config).read_text());settings=copy.deepcopy(cfg);settings['loader']['num_workers']=0
    ctx=base.build(settings);model=ctx.model;branch=model.adapt_route;route=cfg['uda_route']['kind']
    binding=json.loads((Path(args.config).parent/'binding.json').read_text())
    assert ctx.route_native_initial_sha==binding['initial_state_sha'],'基线初始权重不一致'
    prior_steps=0;restored_probe_step=None;prior_natural_change=False
    if args.prior_entry:
        prior=Path(args.prior_entry);prior_steps=json.loads((prior/'progress.json').read_text())['step']
        state=torch.load(prior/'resume.pt',map_location='cpu')
        assert state['config']['uda_route']==settings['uda_route']
        if route=='mda':prior_natural_change=int(state['state_dict']['adapt_route.aggregations'])>0 and float(state['state_dict']['adapt_route.last_parameter_change'])>0
        base.restore(ctx,state);restored_probe_step=20;del state
    updates=40-prior_steps
    assert 0<updates<=40
    # 与原训练相同，短数据集循环复用当前epoch的CPU batch。
    src=itertools.cycle(iter(ctx.train_loader_src));tgt=itertools.cycle(iter(ctx.train_loader_tgt))
    captures=[];outputs=[]
    h=model.neck_target.register_forward_hook(lambda m,i,o:captures.append(o[0]))
    h2=branch.register_forward_hook(lambda m,i,o:outputs.append(o)) if route=='time_frequency' else None
    base.epoch_setup(ctx,5);model.train();rows=[]
    for step in range(1,updates+1):
        captures.clear();outputs.clear()
        source=move_to_device(next(src),model.device);target=move_to_device(next(tgt),model.device)
        assert all(v['segments'].numel()==v['labels'].numel()==0 for v in target)
        for optimizer in base.optimizers(ctx):optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();started=time.perf_counter()
        losses=model(video_list_src=source,video_list_tgt=target,mode='train',curr_epoch=5)[0]
        target_norm=None
        if route=='time_frequency' and step in (1,max(1,updates//2),updates):
            gs=torch.autograd.grad(outputs[-1][0],captures[1],retain_graph=True,allow_unused=True)
            target_norm=sum(float(g.detach().norm()) for g in gs if g is not None)
            assert target_norm>0 and outputs[-1][1]['aligned_groups']>0
        total=cfg['model']['loss_weights']['weight_downstream']*losses['downstream_loss']+cfg['model']['loss_weights']['weight_adv']*losses.get('loss_conditional_adv_rev',0)
        assert torch.isfinite(total);total.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train_cfg']['clip_grad_l2norm'])
        ctx.optimizer_disc.step();ctx.optimizer_gen_source_alignment.step()
        if route=='mda':branch.after_step(model.disc_dom_conditional_adv,5)
        for scheduler in base.schedulers(ctx):scheduler.step()
        ctx.model_ema.update(model);torch.cuda.synchronize()
        rows.append(dict(step=step,loss=float(total.detach()),seconds=time.perf_counter()-started,target_gradient=target_norm,
            metrics={k:float(v) for k,v in losses.items() if k.startswith('route_')},
            aggregations=int(branch.aggregations) if route=='mda' else None,
            parameter_change=float(branch.last_parameter_change) if route=='mda' else None))
        save(entry/'progress.json',dict(step=prior_steps+step,last=rows[-1]))
        if step==max(1,updates//2):
            base.checkpoint(entry/'resume.pt',base.capture(ctx,settings,5,[]))
            state=torch.load(entry/'resume.pt',map_location='cpu');base.restore(ctx,state);del state
    h.remove()
    if h2:h2.remove()
    if route=='mda':assert branch.aggregations>=1 and (prior_natural_change or any(r['parameter_change']>0 for r in rows))
    else:assert all(r['metrics']['route_target_batch_nonempty']==1 for r in rows)
    result=dict(status='passed',route=route,shift=cfg['baseline']['shift'],updates=prior_steps+updates,updates_this_attempt=updates,prior_probe_updates=prior_steps,
        restored_probe_step=restored_probe_step,exact_formal_resume_claimed=False,formal_training_runs=0,
        target_train_gt=False,native_initial_sha=ctx.route_native_initial_sha,strict_bitwise_required=False,
        restored_complete_state_and_subsequent_finite_updates=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        natural_aggregations=int(branch.aggregations) if route=='mda' else None,
        target_batch_nonempty_fraction=1. if route=='time_frequency' else None,
        mean_step_seconds=sum(r['seconds'] for r in rows)/updates,rows=rows)
    save(entry/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
