"""源锚定类条件时频六向冷启动；短训mAP不是派发门槛。"""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha


def specification():
    from libs.models.SADA.class_path_alignment import CLASS_PATH_SPEC
    return dict(CLASS_PATH_SPEC,source_anchor=True)


def prepare(reference,root):
    from tools.run_uda_two_routes import BASELINE
    root=Path(root);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    auth=read(root/'formal_authorization.json')
    assert auth['budget']==6 and auth['implementation_commit']==head
    for i,expected in enumerate(BASELINE,1):
        old=Path(reference)/('S%d'%i);cfg=read(old/'config.json')
        assert cfg['seed']==123 and not cfg['model']['use_trident_head']
        assert 'conditional_rev' in cfg['model']['adv_method'] and cfg['model'].get('use_adv_discriminator',True)
        baseline=100*read(old/'evaluation/target_val.json')['map_0.1_to_0.5'];assert abs(baseline-expected)<1e-9
        entry=root/('S%d'%i);new=copy.deepcopy(cfg);new['uda_route']=specification()
        new['baseline']['tmpdir']=str(root/'tmp');new['baseline']['route_entry']=str(entry)
        new['baseline']['code']=dict(commit=head,base_config=str(old/'config.json'))
        binding=dict(reference=str(old),initial_state_sha=read(old/'termination.json')['initial_state_sha'],
            baseline_target_percent=baseline,config_sha256=sha(old/'config.json'),
            inputs={str(p):sha(p) for p in [cfg['dataset']['json_file'],cfg['baseline']['target_manifest'],cfg['prototype_capacity']['prepared_cache']]})
        save(entry/'config.json',new);save(entry/'binding.json',binding)
    save(root/'baseline_binding.json',dict(reference=str(reference),mean_percent=sum(BASELINE)/6,per_shift=BASELINE,new_baseline_runs=0))


def probe(cfg,entry):
    import torch
    from tools import baseline_engine as base
    from libs.trainer.da_trainer import move_to_device
    with (entry/'probe_claim.json').open('x') as f:json.dump(dict(updates=2,formal_runs=0),f)
    settings=copy.deepcopy(cfg);settings['loader']['num_workers']=0
    ctx=base.build(settings);model=ctx.model;binding=read(entry.parent/'binding.json')
    assert ctx.route_native_initial_sha==binding['initial_state_sha']
    source=move_to_device(next(iter(ctx.train_loader_src)),model.device)
    target=move_to_device(next(iter(ctx.train_loader_tgt)),model.device)
    assert all(not v['segments'].numel() and not v['labels'].numel() for v in target)
    captures=[]
    handle=model.adapt_route.register_forward_hook(lambda m,i,o:captures.append((i,o)))
    rows=[]
    for step in range(2):
        captures.clear();base.epoch_setup(ctx,5);model.train()
        for opt in base.optimizers(ctx):opt.zero_grad(set_to_none=True)
        started=time.perf_counter()
        losses=model(video_list_src=source,video_list_tgt=target,mode='train',curr_epoch=5)[0]
        inputs,output=captures[-1];sf=tuple(inputs[3]);tf=tuple(inputs[4])
        grads=torch.autograd.grad(output[0],sf+tf,retain_graph=True,allow_unused=True)
        norm=lambda gs:sum(float(g.square().sum()) for g in gs if g is not None)**.5
        sn=norm(grads[:len(sf)]);tn=norm(grads[len(sf):])
        assert sn==0 and tn>0
        total=cfg['model']['loss_weights']['weight_downstream']*losses['downstream_loss']+cfg['model']['loss_weights']['weight_adv']*losses['loss_conditional_adv_rev']
        assert torch.isfinite(total);total.backward()
        assert all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)
        assert any(p.grad is not None and p.grad.norm()>0 for p in model.disc_dom_conditional_adv.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train_cfg']['clip_grad_l2norm'])
        ctx.optimizer_disc.step();ctx.optimizer_gen_source_alignment.step()
        for scheduler in base.schedulers(ctx):scheduler.step()
        ctx.model_ema.update(model);torch.cuda.synchronize()
        rows.append(dict(step=step+1,loss=float(total.detach()),alignment_source_grad=sn,alignment_target_grad=tn,seconds=time.perf_counter()-started))
        if step==0:
            base.checkpoint(entry/'resume.pt',base.capture(ctx,settings,5,[]))
            saved=torch.load(entry/'resume.pt',map_location='cpu');base.restore(ctx,saved);del saved
    handle.remove()
    save(entry/'result.json',dict(status='passed',updates=2,native_sha=ctx.route_native_initial_sha,rows=rows,
        target_train_gt=False,discriminator_active=True,complete_state_restore_usable=True,
        strict_bitwise_claimed=False,formal_weights_reused=False,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','probe','train','evaluate','audit'])
    p.add_argument('--root');p.add_argument('--reference');p.add_argument('--entry');p.add_argument('--config')
    args=p.parse_args()
    if args.mode=='prepare':return prepare(args.reference,args.root)
    entry=Path(args.entry);entry.mkdir(parents=True,exist_ok=True);cfg=read(args.config)
    assert cfg['uda_route']==specification()
    if args.mode=='audit':
        from tools.audit_class_path_formal import audit
        return audit(entry)
    if args.mode=='probe':return probe(cfg,entry)
    from tools import baseline_engine as base
    if args.mode=='train':
        root=entry.parent;auth=read(root/'formal_authorization.json');gate=read(root/'acceptance.json')
        assert auth['budget']==6 and auth['seed']==123 and gate['status']=='passed'
        assert auth['implementation_commit']==cfg['baseline']['code']['commit']
        assert cfg['baseline']['shift'] in auth['shifts']
        with (entry/'train_claim.json').open('x') as f:json.dump(dict(cold_start=True,seed=123,config_sha256=sha(args.config)),f)
        build=base.build
        def paired(settings):
            ctx=build(settings);binding=read(entry/'binding.json')
            assert ctx.route_native_initial_sha==binding['initial_state_sha']==read(entry/'probe/result.json')['native_sha']
            save(entry/'formal_initialization.json',dict(native_sha=ctx.route_native_initial_sha,cold_start=True,probe_weights_loaded=False))
            return ctx
        base.build=paired
        return base.train(cfg,entry)
    from tools.native_ap_fast import install
    install();return base.evaluate(cfg,entry,cfg['baseline']['annotation_for_offline_evaluation'])


if __name__=='__main__':main()
