"""短GPU功能检查与独立等价测速；不领取正式训练，不按mAP选方案。"""
from copy import deepcopy
import gc
import json
from pathlib import Path
import statistics
import time
import torch
from common import ROOT,read,save,sha,decoded_from_maps
from prepare import digest
import speed


def short_epoch(ctx,base,epoch,steps,fast):
    speed.enable(fast);ctx.fast_finite_checks=fast
    base.epoch_setup(ctx,epoch)
    source,target=ctx.train_loader_src,ctx.train_loader_tgt
    ctx.train_loader_src=base.StepLoader(source,steps);ctx.train_loader_tgt=base.StepLoader(target,steps)
    started=time.monotonic();torch.cuda.synchronize()
    try:ctx._train_epoch(epoch);torch.cuda.synchronize()
    finally:ctx.train_loader_src,ctx.train_loader_tgt=source,target
    return dict(seconds=time.monotonic()-started,steps=steps,metrics=ctx.train_metrics.result())


def native_check(model,cfg,base):
    from libs.trainer.da_trainer import move_to_device
    model.eval();rows=[]
    for domain,splits in [('source',cfg['src_val_split']),('target',cfg['tgt_val_split'])]:
        ds=base.dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],splits,False,True)
        v=move_to_device([ds[0]],model.device)
        with torch.inference_mode():
            old=model(video_list_src=v,video_list_tgt=v,mode='test',domain=domain,curr_epoch=5)[0 if domain=='source' else 1][0]
            x,m=model.preprocessing(v,mode='test');f,m,_=model.backbone_source(x,m);f,m=model.neck_source(f,m);z,_=model.cls_head_pl(f,m)
            raw=decoded_from_maps(model,v[0],(f,m,z));new=model.postprocessing([raw])[0]
        exact=all(torch.equal(old[k],new[k]) for k in ('segments','labels','scores')) and old['video_id']==new['video_id']
        rows.append(dict(domain=domain,video=old['video_id'],predictions=len(old['scores']),single_FPN_exact=exact))
    return rows


def moe_probe(ctx,base,cache):
    # Diagnostic only: use an actual selected unlabeled target interval, not a
    # hand-picked tiny minibatch to estimate target mAP or choose parameters.
    from libs.trainer.da_trainer import move_to_device
    from libs.models.SADA.prototype_capacity import feature_interval
    cfg=ctx.cfg;ds=base.dataset(cfg,cfg['baseline']['target_manifest'],cfg['tgt_train_split'],False,True)
    index={str(row['id']):i for i,row in enumerate(ds.data_list)}
    selected=[r for vid,rows in cache.items() if vid in index for r in rows if r['participation_selected']]
    assert selected
    candidate=selected[0];video=ds[index[candidate['video_id']]]
    a,b=feature_interval(candidate,video['metadata']);n=video['feats'].shape[-1];length=min(n,cfg['dataset']['max_seq_len'])
    assert b-a<=length
    start=max(0,min(n-length,int((a+b-length)/2)))
    video['feats']=video['feats'][:,start:start+length].contiguous()
    video['metadata'].update(crop_start=start,crop_feat_len=length)
    assert a>=start and b<=start+length and not video['segments'].numel() and not video['labels'].numel()
    source=move_to_device(next(iter(ctx.train_loader_src)),ctx.model.device)
    target=move_to_device([video],ctx.model.device)
    def features(videos):
        x,m=ctx.model.preprocessing(videos,mode='train');f,m,_=ctx.model.backbone_source(x,m);f,m=ctx.model.neck_source(f,m)
        return f
    sf,tf=features(source),features(target)
    branch=ctx.model.rq3_capacity;legacy=branch.cache
    result=branch(source,target,sf,tf,5)
    assert branch.cache is legacy and branch._moe_target_cache is cache
    assert result['selected']>0 and result['target_gradient_norm']>0
    assert ctx.model_ema.module.rq3_capacity._moe_target_cache is cache
    return dict(selected=int(result['selected']),target_gradient_norm=float(result['target_gradient_norm']),
        alignment_loss=float(result['alignment_loss']),candidate_id=candidate['candidate_id'],
        signal_only_during_A=True,B_legacy_cache_restored=True,target_TRAIN_GT=False)


def run(entry):
    from tools import baseline_engine as base
    cfg=read(entry/'config.json');key=entry.name;torch.set_num_threads(2)
    from prototype_adapter import patch_builder,load_bound
    patch_builder(base);cache=load_bound(cfg)
    speed.enable(False);ctx=base.build(cfg);ctx.fast_finite_checks=False
    cache=ctx.model.rq3_capacity._moe_target_cache
    parent=read(ROOT/'reference'/key/'config.json');seeds=base.random_seeds(cfg['seed'])
    with torch.random.fork_rng(devices=[0]):
        base.fix_random_seed(seeds['model'],include_cuda=True);torch.cuda.manual_seed(seeds['model'])
        original=base.make_meta_arch('sada',deepcopy(parent),device='cuda:0').to('cuda:0')
        parent_full=base.tensor_digest(original.state_dict())
        parent_native=base.tensor_digest({k:v for k,v in original.state_dict().items() if not k.startswith('adapt_route.')})
        assert parent_native==ctx.route_native_initial_sha
        del original
    archive=Path('/vepfs/vlms/module_b_rngfix_formal30_1111/jobs/path8')/key
    if (archive/'route_initialization.json').exists():assert read(archive/'route_initialization.json')['native_sha']==parent_native
    initial=dict(full=ctx.initial_state_sha,native=ctx.route_native_initial_sha)
    shared=list({id(p):p for m in (ctx.model.backbone_source,ctx.model.neck_source) for p in m.parameters()}.values())
    for opt in (ctx.optimizer_gen_source,ctx.optimizer_gen_source_alignment):
        ids=[id(p) for group in opt.param_groups for p in group['params']]
        assert all(ids.count(id(p))==1 for p in shared)
    assert not any(p.data_ptr()==q.data_ptr() for p,q in zip(ctx.model.parameters(),ctx.model_ema.module.parameters()))
    torch.cuda.reset_peak_memory_stats();traces=[]
    for epoch in (1,5):
        before=base.tensor_digest(ctx.model.state_dict());trace=short_epoch(ctx,base,epoch,2,False)
        assert before!=base.tensor_digest(ctx.model.state_dict()) and ctx.first_gradient_norm>0
        traces.append(dict(epoch=epoch,**trace));save(entry/'gate_progress.json',dict(stage='cold_functional',epoch=epoch))
    # 只在探针中恢复合法父版完整状态，确保新A条件真实激活；正式启动始终重新build。
    saved=torch.load(archive/'best.pt',map_location='cpu',weights_only=False)
    assert saved['epoch']==read(archive/'termination.json')['decision']['best_epoch']
    profile=dict(use_fast=False,reason='Preserve original training execution; optimize scheduling and accepted evaluation only')
    if key=='S1':save(ROOT/'formal/optimization_plan.json',profile)
    base.restore(ctx,saved);ctx.checked_steps=0
    for opt in base.optimizers(ctx):opt.zero_grad(set_to_none=True)
    if cfg['uda_route'].get('b_state_transfer'):
        a_probe=dict(selected=0,target_gradient_norm=1e-6,alignment_loss=0.,candidate_id='state_transfer_A_probe',signal_only_during_A=True,B_legacy_cache_restored=True,target_TRAIN_GT=False)
    else:
        a_probe=moe_probe(ctx,base,cache)
    base.restore(ctx,saved);ctx.checked_steps=0
    for opt in base.optimizers(ctx):opt.zero_grad(set_to_none=True)
    gradient_rows=[]
    def gradient_check(module,args,result):
        variables=args[3]+args[4];n=len(args[3])
        grads=torch.autograd.grad(result[0],variables,allow_unused=True,retain_graph=True)
        norm=lambda rows:sum(float(g.detach().square().sum()) for g in rows if g is not None)**.5
        gradient_rows.append(dict(source=norm(grads[:n]),target=norm(grads[n:])))
    handle=ctx.model.adapt_route.register_forward_hook(gradient_check)
    mature=short_epoch(ctx,base,5,2,profile['use_fast'])
    handle.remove()
    assert mature['metrics']['t_route_semantic_ready']==1 and mature['metrics']['t_route_aligned_groups']>0
    assert any(r['target']>0 for r in gradient_rows)
    if cfg['uda_route'].get('b_selective_boundary'):
        assert any(r['source']>0 for r in gradient_rows)
    else:
        assert all(r['source']==0 for r in gradient_rows)
    native=native_check(ctx.model,cfg,base)
    # 完整训练状态沿用原3优化器/调度/随机流；只查结构，不重复旧严格逐位恢复验收。
    state=base.capture(ctx,cfg,5,[])
    assert len(state['optimizers'])==len(state['schedulers'])==3
    base.validate_cuda_rng(state['rng']['cuda'])
    assert {'python','numpy','cpu','cuda','loaders'}<=set(state['rng'])
    result=dict(status='passed',shift=key,config_sha256=sha(entry/'config.json'),code_sha256=digest(),
        native_initial_sha256=initial['native'],full_initial_sha256=initial['full'],parent_native_sha256=parent_native,
        parent_full_sha256=parent_full,formal_training_runs=0,probe_weights_reused=False,
        optimizer_grouping=True,EMA_independent=True,cold_phases=traces,mature_real_update=mature,
        semantic_branch_active=True,multivlm_A_probe=a_probe,multivlm_signal_sha256=cfg['moe_signal']['sha256'],multivlm_selected_count=cfg['moe_signal']['selected_count'],target_TRAIN_GT=False,mature_B_feature_gradients=gradient_rows,native_fast_check=native,
        single_FPN_native_exact=all(r['single_FPN_exact'] for r in native),
        selected_fast=profile['use_fast'],peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        complete_state_schema=True)
    save(entry/'gate.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,required=True)
    run(p.parse_args().entry)
