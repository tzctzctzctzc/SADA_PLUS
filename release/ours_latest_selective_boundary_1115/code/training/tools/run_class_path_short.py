"""有界热启动配对试验；固定100步末态评估，不冒称冷启动六向成绩。"""
import argparse,copy,json,time,sys,itertools,hashlib,traceback
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))

def main():
    import torch,numpy as np
    from tools import baseline_engine as base
    from tools.baseline_protocol import save,sha
    from libs.trainer.da_trainer import move_to_device
    from libs.models.SADA.class_path_alignment import CLASS_PATH_SPEC
    from libs.evaluation import ANETdetection
    from tools.native_ap_fast import install
    install();torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--out',required=True)
    p.add_argument('--variant',choices=['control','class_path','source_anchor'],required=True)
    args=p.parse_args();entry=Path(args.out);entry.mkdir(parents=True,exist_ok=True)
    with (entry/'train_claim.json').open('x') as f:json.dump(dict(max_updates=100,variant=args.variant),f)
    reference=Path(args.reference);cfg=json.loads((reference/'config.json').read_text())
    cfg.pop('uda_route',None)
    if args.variant!='control':cfg['uda_route']=dict(CLASS_PATH_SPEC,source_anchor=args.variant=='source_anchor')
    cfg['loader']['num_workers']=0;cfg['baseline']['tmpdir']=str(entry/'tmp')
    cfg['baseline']['code']['path']=str(CODE)
    save(entry/'config.json',cfg)
    ctx=base.build(cfg);saved=torch.load(reference/'best.pt',map_location='cpu')
    initial_epoch=saved['epoch'];epoch=initial_epoch+1
    # 新模块仅有固定buffer，无新增参数，因此三个优化器可原样恢复。
    for field,model in [('state_dict',ctx.model),('ema_state_dict',ctx.model_ema.module)]:
        extras={k:v for k,v in model.state_dict().items() if k.startswith('adapt_route.')}
        saved[field].update(extras)
    base.restore(ctx,saved)
    native=lambda state:{k:v for k,v in state.items() if not k.startswith('adapt_route.')}
    initial_sha=base.tensor_digest(native(ctx.model.state_dict()))
    assert initial_sha==base.tensor_digest(native(saved['state_dict']))
    assert cfg['model'].get('use_adv_discriminator',True) and 'conditional_rev' in cfg['model']['adv_method']
    assert all(x.requires_grad for x in ctx.model.disc_dom_conditional_adv.parameters())
    save(entry/'initialization.json',dict(reference=str(reference/'best.pt'),reference_sha256=sha(reference/'best.pt'),
         reference_epoch=initial_epoch,native_sha=initial_sha,variant=args.variant,
         optimizer_state_restored=True,source_selected_initial_checkpoint=True,
         warm_start_development_only=True,steps=100,target_train_gt=False))
    del saved
    ctx.model.train();base.epoch_setup(ctx,epoch)
    src=itertools.cycle(iter(ctx.train_loader_src));tgt=itertools.cycle(iter(ctx.train_loader_tgt))
    trace=[];rows=[];branch_output=[];branch_inputs=[]
    if ctx.model.adapt_route is not None:
        def hook(module,inputs,output):branch_inputs.append(inputs);branch_output.append(output)
        handle=ctx.model.adapt_route.register_forward_hook(hook)
    began=time.perf_counter()
    for step in range(100):
        source=move_to_device(next(src),ctx.model.device);target=move_to_device(next(tgt),ctx.model.device)
        assert all(v['segments'].numel()==v['labels'].numel()==0 for v in target)
        trace.append([[str(v['metadata']['video_id']),list(v['feats'].shape)] for v in source+target])
        branch_inputs.clear();branch_output.clear()
        for opt in base.optimizers(ctx):opt.zero_grad(set_to_none=True)
        losses=ctx.model(video_list_src=source,video_list_tgt=target,mode='train',curr_epoch=epoch)[0]
        norms=None
        if branch_output and step in (0,99):
            features=tuple(branch_inputs[0][3])+tuple(branch_inputs[0][4])
            grad=torch.autograd.grad(branch_output[0][0],features,retain_graph=True,allow_unused=True)
            ns=len(branch_inputs[0][3])
            norms=dict(source=sum(float(g.square().sum()) for g in grad[:ns] if g is not None)**.5,
                       target=sum(float(g.square().sum()) for g in grad[ns:] if g is not None)**.5)
        total=cfg['model']['loss_weights']['weight_downstream']*losses['downstream_loss']
        total=total+cfg['model']['loss_weights']['weight_adv']*losses['loss_conditional_adv_rev']
        assert torch.isfinite(total);total.backward()
        assert all(p.grad.isfinite().all() for p in ctx.model.parameters() if p.grad is not None)
        clip=cfg['train_cfg']['clip_grad_l2norm']
        if clip>0:torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),clip)
        weight_clip=cfg['model']['weight_clip_value']
        if weight_clip>0:
            for modules in (ctx.model.disc_dom_adv,ctx.model.disc_dom_conditional_adv):
                for parameter in modules.parameters():parameter.data.clamp_(-weight_clip,weight_clip)
        ctx.optimizer_disc.step();ctx.optimizer_gen_source_alignment.step()
        virtual_step=(epoch-1)*cfg['baseline']['epoch_budget']['iterations_per_epoch']+step
        if cfg['train_cfg']['ema_active'] and step%cfg['train_cfg']['update_ema_every']==0 and virtual_step>cfg['train_cfg']['step_start_ema']:
            ctx.model_ema.update(ctx.model)
        for scheduler in base.schedulers(ctx):scheduler.step()
        row=dict(step=step+1,loss=float(total.detach()),alignment_grad=norms,
                 route={k:float(v) for k,v in losses.items() if k.startswith('route_')})
        rows.append(row)
        save(entry/'progress.json',dict(stage='training',step=step+1,last=row))
    torch.cuda.synchronize();elapsed=time.perf_counter()-began
    if ctx.model.adapt_route is not None:handle.remove()
    state=base.capture(ctx,cfg,initial_epoch,[]);state['short_trial']=dict(steps=100,virtual_epoch=epoch,
        selection='fixed_final_step',formal_training=False,loader_position_resume_verified=False)
    base.checkpoint(entry/'final.pt',state);del state
    save(entry/'train_metrics.json',dict(rows=rows,elapsed_seconds=elapsed,seconds_per_step=elapsed/100,
        trace_sha256=hashlib.sha256(json.dumps(trace).encode()).hexdigest(),sample_trace=trace,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    save(entry/'progress.json',dict(stage='evaluating',step=100))
    model=ctx.model_ema.module;model.eval();annotation=cfg['baseline']['annotation_for_offline_evaluation']
    # 标注仅由离线评价使用；训练只加载无标签target manifest。
    ds=base.dataset(cfg,annotation,cfg['tgt_val_split'],False)
    dl=base.loader(ds,False,torch.Generator().manual_seed(cfg['seed']),1,0)
    cols={'video-id':[],'t-start':[],'t-end':[],'label':[],'score':[]}
    with torch.no_grad():
        for video in dl:
            video=move_to_device(video,model.device)
            predictions=model(video_list_src=video,video_list_tgt=video,mode='test',domain='target',curr_epoch=epoch)[1]
            for pred in predictions:
                cols['video-id'].extend([pred['video_id']]*len(pred['scores']))
                for key,value in [('t-start',pred['segments'][:,0]),('t-end',pred['segments'][:,1]),('label',pred['labels']),('score',pred['scores'])]:
                    cols[key].extend(value.cpu().tolist())
    base.checkpoint(entry/'predictions.pt',cols)
    evaluator=ANETdetection(str(annotation),cfg['tgt_val_split'],num_workers=2,action_subsampling='10_majority_classes')
    ap,mean,*_=evaluator.evaluate({k:np.asarray(v) for k,v in cols.items()},verbose=True)
    save(entry/'result.json',dict(status='complete',variant=args.variant,shift=cfg['baseline']['shift'],steps=100,
        map_0_1_to_0_5=float(mean),map_by_tiou=[float(x) for x in ap],videos=len(ds),
        reference_epoch=initial_epoch,selection='fixed_final_step',state='EMA',target_train_gt=False,
        warm_start_development_only=True,seconds_per_step=elapsed/100))

if __name__=='__main__':
    try:main()
    except Exception:
        path=Path(sys.argv[sys.argv.index('--out')+1]);path.mkdir(parents=True,exist_ok=True)
        (path/'error.json').write_text(json.dumps(dict(error=traceback.format_exc())));raise
