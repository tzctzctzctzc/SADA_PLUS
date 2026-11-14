"""旧Prototype+域对抗为唯一基线，TF/MDA各六向；无Trident，无额外对照训练。"""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
BASELINE=[62.421912561323424,50.93514017784955,58.482486547746625,46.79090427905215,63.55325748533316,46.20811131704408]


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)


def prepare(reference,root):
    from libs.models.SADA.uda_routes import TF_V2_SPEC as TF_SPEC,MDA_V2_SPEC as MDA_SPEC
    from tools.baseline_protocol import sha,read
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    for i,expected in enumerate(BASELINE,1):
        original=Path(reference)/('S%d'%i);cfg=read(original/'config.json')
        assert not cfg.get('head_replacement') and not cfg['model']['use_trident_head'] and cfg['seed']==123
        actual=100*read(original/'evaluation/target_val.json')['map_0.1_to_0.5'];assert abs(actual-expected)<1e-9
        cached=read(Path(root)/'previous_baseline_binding.json')['rows']['S%d'%i]
        assert cached['reference']==str(original)
        binding=dict(reference=str(original),initial_state_sha=read(original/'termination.json')['initial_state_sha'],
            baseline_target_percent=actual,config_sha256=sha(original/'config.json'),checkpoint_sha256=cached['checkpoint_sha256'],
            checkpoint_hash_reused_from_prior_audit=True,
            inputs={str(p):sha(p) for p in [cfg['dataset']['json_file'],cfg['baseline']['target_manifest'],cfg['prototype_capacity']['prepared_cache']]})
        for route,spec in [('time_frequency',TF_SPEC),('mda',MDA_SPEC)]:
            new=copy.deepcopy(cfg);new['uda_route']=dict(spec)
            new['baseline']['tmpdir']=str(Path(root)/'tmp')
            new['baseline']['code']=dict(commit=head,base_config=str(original/'config.json'))
            if route=='time_frequency':new['model'].update(adv_method=['None'],use_adv_discriminator=False)
            # S2旧配置为共享D；跨尺度聚合必须让六个D各接收本尺度梯度。
            else:new['model']['use_level_disc']=True
            entry=Path(root)/route/('S%d'%i);new['baseline']['route_entry']=str(entry)
            save(entry/'config.json',new);save(entry/'binding.json',binding)
    save(Path(root)/'baseline_binding.json',dict(reference=str(reference),mean_percent=sum(BASELINE)/6,per_shift=BASELINE,reused_training_runs=6,new_baseline_runs=0))


def probe(cfg,entry):
    import torch
    from tools import baseline_engine as base
    from libs.trainer.da_trainer import move_to_device
    entry=Path(entry);settings=copy.deepcopy(cfg);settings['loader']['num_workers']=0
    ctx=base.build(settings);model=ctx.model;branch=model.adapt_route;route=cfg['uda_route']['kind']
    binding=json.loads((Path(cfg['baseline']['route_entry'])/'binding.json').read_text())
    assert ctx.route_native_initial_sha==binding['initial_state_sha'],'旧Prototype配对初值不一致'
    source=move_to_device(next(iter(ctx.train_loader_src)),model.device)
    target=move_to_device(next(iter(ctx.train_loader_tgt)),model.device)
    assert all(not v['segments'].numel() and not v['labels'].numel() for v in target)
    captures=[];outputs=[]
    hook=model.neck_target.register_forward_hook(lambda m,i,o:captures.append(o[0]))
    handle=branch.register_forward_hook(lambda m,i,o:outputs.append(o)) if route=='time_frequency' else None
    rows=[]
    for epoch in (1,5,6):
        captures.clear();outputs.clear();base.epoch_setup(ctx,epoch);model.train()
        for optimizer in base.optimizers(ctx):optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();started=time.perf_counter()
        losses=model(video_list_src=source,video_list_tgt=target,mode='train',curr_epoch=epoch)[0]
        pure_target_gradient=0.
        if route=='time_frequency' and epoch>=5:
            gradients=torch.autograd.grad(outputs[-1][0],captures[1],retain_graph=True,allow_unused=True)
            pure_target_gradient=sum(float(g.detach().norm()) for g in gradients if g is not None)
        total=losses['downstream_loss']
        if epoch>=5:total=cfg['model']['loss_weights']['weight_downstream']*total+cfg['model']['loss_weights']['weight_adv']*losses.get('loss_conditional_adv_rev',0)
        assert torch.isfinite(total);total.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        extra_gradient=sum(float(p.grad.abs().sum()) for p in branch.parameters() if p.grad is not None)
        if route=='time_frequency' and epoch>=5:assert extra_gradient>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train_cfg']['clip_grad_l2norm'])
        if epoch<5:ctx.optimizer_gen_source.step()
        else:ctx.optimizer_disc.step();ctx.optimizer_gen_source_alignment.step()
        if route=='mda':branch.after_step(model.disc_dom_conditional_adv,epoch)
        for scheduler in base.schedulers(ctx):scheduler.step()
        ctx.model_ema.update(model);torch.cuda.synchronize()
        rows.append(dict(epoch=epoch,loss=float(total.detach()),seconds=time.perf_counter()-started,
            new_parameter_gradient_l1=extra_gradient,pure_target_gradient_norm=pure_target_gradient,
            metrics={k:float(v) for k,v in losses.items() if k.startswith('route_')}))
        save(entry/'probe_updates.json',rows)
    hook.remove()
    if handle:handle.remove()
    forced=False
    if route=='mda':
        # 强制触发仅验功能；正式阈值仍0.5，探针权重不转正式训练。
        from libs.models.SADA.uda_routes import MultiscaleDiscriminatorAggregation
        test=MultiscaleDiscriminatorAggregation(model.n_levels,dict(cfg['uda_route'],accuracy_trigger=1.01)).to(model.device)
        for level in range(model.n_levels):
            dimension=next(p for p in model.disc_dom_conditional_adv[level].parameters() if p.ndim==2).shape[1]
            x=torch.linspace(-1,1,dimension,device=model.device)[None].repeat(8,1)
            test.observe(level,x,0,1.);test.observe(level,-x,1,1.)
        test.after_step(model.disc_dom_conditional_adv,6)
        assert test.aggregations==1 and test.last_parameter_change>0;forced=True
    base.checkpoint(entry/'resume.pt',base.capture(ctx,settings,6,[]))
    saved=torch.load(entry/'resume.pt',map_location='cpu');base.restore(ctx,saved)
    for optimizer in base.optimizers(ctx):optimizer.zero_grad(set_to_none=True)
    losses=model(video_list_src=source,video_list_tgt=target,mode='train',curr_epoch=6)[0]
    total=losses['downstream_loss']+cfg['model']['loss_weights']['weight_adv']*losses.get('loss_conditional_adv_rev',0)
    assert torch.isfinite(total);total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train_cfg']['clip_grad_l2norm'])
    ctx.optimizer_disc.step();ctx.optimizer_gen_source_alignment.step()
    if route=='mda':branch.after_step(model.disc_dom_conditional_adv,6)
    model.eval()
    with torch.no_grad():
        a=model([source[0]],[source[0]],mode='test',curr_epoch=6)[0][0]
        model.adapt_route=None
        b=model([source[0]],[source[0]],mode='test',curr_epoch=6)[0][0]
        model.adapt_route=branch
    assert torch.allclose(a['scores'],b['scores'],atol=1e-6,rtol=1e-5) and torch.allclose(a['segments'],b['segments'],atol=1e-6,rtol=1e-5)
    report=dict(status='passed',route=route,shift=cfg['baseline']['shift'],native_initial_state_sha=ctx.route_native_initial_sha,
        no_trident=True,target_train_GT=False,restored_next_update_usable=True,native_predictions_on_off_close=True,
        strict_bitwise_required=False,updates=rows,forced_mda_functional_probe=forced,
        natural_mda_aggregations=int(branch.aggregations) if route=='mda' else None,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,formal_training_runs=0)
    save(entry/'result.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','probe','train','evaluate','audit'])
    p.add_argument('--reference');p.add_argument('--root');p.add_argument('--config');p.add_argument('--entry');p.add_argument('--resume')
    args=p.parse_args()
    if args.mode=='prepare':return prepare(args.reference,args.root)
    if args.mode=='audit':
        from tools.audit_uda_two_routes import audit
        return audit(Path(args.entry))
    cfg=json.loads(Path(args.config).read_text());entry=Path(args.entry);entry.mkdir(parents=True,exist_ok=True)
    if args.mode=='probe':return probe(cfg,entry)
    from tools import baseline_engine as base
    if args.mode=='train':
        from tools.baseline_protocol import read,sha
        root=entry.parents[1];auth=read(root/'formal_authorization.json')
        assert auth['budget']==12 and auth['seed']==cfg['seed']==123
        assert cfg['baseline']['shift'] in auth['shifts'] and cfg['uda_route']['kind'] in auth['routes']
        assert cfg['uda_route'].get('revision')==2 and auth['implementation_commit']==cfg['baseline']['code']['commit']
        acceptance=read(root/'acceptance.json');assert acceptance['status']=='revision_implementation_and_all6_short_acceptance_complete'
        if not args.resume:
            with (entry/'train_claim.json').open('x') as stream:json.dump(dict(seed=123,cold_start=True,
                config_sha256=sha(args.config),authorization_sha256=sha(root/'formal_authorization.json'),acceptance_sha256=sha(root/'acceptance.json')),stream)
        build=base.build
        def paired_build(settings):
            ctx=build(settings);binding=read(entry/'binding.json')
            assert ctx.route_native_initial_sha==binding['initial_state_sha'],'正式初值与旧Prototype基线不配对'
            save(entry/'formal_initialization.json',dict(native_sha=ctx.route_native_initial_sha,full_sha=ctx.initial_state_sha,
                reference=binding['reference'],cold_start=not bool(args.resume),short_probe_weights_loaded=False))
            return ctx
        base.build=paired_build
        save(entry/'config.json',cfg);return base.train(cfg,entry,args.resume)
    from tools.native_ap_fast import install
    install();return base.evaluate(cfg,entry,cfg['baseline']['annotation_for_offline_evaluation'])


if __name__=='__main__':main()
