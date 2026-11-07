"""原Prototype跨完整检测器的同协议训练/评估，原作者结构结果只读复用。"""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity

REFERENCE=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
MODES=('so','sada','sada_prototype')


def prepare(root):
    identity=code_identity()
    for detector in ('actionformer','tridet'):
        for shift in ('S'+str(i) for i in range(1,7)):
            original=read(REFERENCE/shift/'config.json')
            for condition in MODES:
                cfg=deepcopy(original)
                cfg['baseline'].update(code=identity,tmpdir=str(root/'tmp'),
                    regime='so' if condition=='so' else 'uda')
                cfg['generality']=dict(detector=detector,condition=condition,
                    source='official complete detector, native classification/localization losses',
                    original_prototype_commit='5f9a157e17e4a605ff4a6ed8dac4bd725b44d7b4')
                if condition!='sada_prototype':cfg.pop('prototype_capacity')
                cfg['model']['adv_method']=['None'] if condition=='so' else ['conditional_rev']
                entry=root/detector/condition/shift
                save(entry/'config.json',cfg)
                save(entry/'binding.json',dict(reference=str(REFERENCE/shift),
                    reference_config_sha256=sha(REFERENCE/shift/'config.json'),
                    source_manifest_sha256=cfg['baseline']['source_manifest_sha256'],
                    target_manifest_sha256=cfg['baseline']['target_manifest_sha256'],
                    target_GT_training=False,new_VLM_queries=0))
    save(root/'plan.json',dict(code=identity,total_table_cells=54,reused_original_sada_cells=18,
        max_new_formal_runs=36,seed=123,source_selection='cls+reg loss every5/patience15/max50',
        original_prototype_only=True,old_sada_prototype_trident_hybrid=False,
        native_tridet_is_external_detector=True,posthoc_readout=False,
        native_sources=read(CODE/'tools/native_sources/provenance.json')))


def probe(cfg,entry):
    import torch
    from tools.generalization_models import install_runtime,native_videos
    from libs.trainer.da_trainer import move_to_device
    base=install_runtime();ctx=base.build(cfg);model=ctx.model
    source=move_to_device(next(iter(ctx.train_loader_src)),model.device)
    target=move_to_device(next(iter(ctx.train_loader_tgt)),model.device)
    model.train();normalizer=model.loss_normalizer
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state(0)
    original=model.native(native_videos(source))
    params=list(model.native.parameters())
    expected=torch.autograd.grad(original['final_loss'],params,allow_unused=True)
    expected=[None if g is None else g.detach().cpu() for g in expected]
    expected_losses={k:float(v.detach()) for k,v in original.items()}
    del original
    model.loss_normalizer=normalizer;torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
    actual,*_=model.source_dense(source)
    gradients=torch.autograd.grad(actual['final_loss'],params,allow_unused=True)
    max_error=0.
    for want,got in zip(expected,gradients):
        assert (want is None)==(got is None)
        if want is not None:
            got=got.detach().cpu();torch.testing.assert_close(got,want,atol=1e-6,rtol=1e-4)
            max_error=max(max_error,float((got-want).abs().max()))
    for key,value in expected_losses.items():
        assert abs(float(actual[key].detach())-value)<=1e-6+1e-4*abs(value)
    del actual,gradients,expected;gc.collect()
    rows=[]
    for epoch in (1,cfg['train_cfg']['warmup_adv']):
        base.epoch_setup(ctx,epoch)
        for optimizer in base.optimizers(ctx):optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();start=time.monotonic()
        result=model(video_list_src=source,video_list_tgt=target,mode='train',curr_epoch=epoch)[0]
        value=result['downstream_loss']
        if epoch>=cfg['train_cfg']['warmup_adv']:
            value=value*cfg['model']['loss_weights']['weight_downstream']
            if cfg['generality']['condition']!='so':
                value=value+cfg['model']['loss_weights']['weight_adv']*result['loss_conditional_adv_rev']
        value.backward()
        native_grad=sum(float(p.grad.detach().abs().sum()) for p in model.native.parameters() if p.grad is not None)
        assert native_grad>0 and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if cfg.get('prototype_capacity'):
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.rq3_capacity.parameters())
        if epoch>=cfg['train_cfg']['warmup_adv'] and cfg['generality']['condition']!='so':
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.disc_dom_conditional_adv.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train_cfg']['clip_grad_l2norm'])
        ctx.optimizer_gen_source.step() if epoch<cfg['train_cfg']['warmup_adv'] else ctx.optimizer_gen_source_alignment.step()
        if epoch>=cfg['train_cfg']['warmup_adv']:ctx.optimizer_disc.step()
        for scheduler in base.schedulers(ctx):scheduler.step()
        ctx.model_ema.update(model);torch.cuda.synchronize()
        rows.append(dict(epoch=epoch,loss=float(value.detach()),native_gradient_l1=native_grad,
            seconds=time.monotonic()-start,metrics={k:float(v.detach()) for k,v in result.items()}))
    # 全部状态真实落盘再恢复；新实验不使用逐位门槛淘汰科学方案。
    saved=base.capture(ctx,cfg,5,[]);base.checkpoint(entry/'probe_state.pt',saved)
    state=torch.load(entry/'probe_state.pt',map_location='cpu');base.restore(ctx,state)
    assert base.tensor_digest(base.capture(ctx,cfg,5,[]))==base.tensor_digest(state)
    model.eval()
    with torch.no_grad():
        check=move_to_device(next(iter(ctx.val_loader_src)),model.device)
        outputs=model(video_list_src=check,video_list_tgt=check,mode='test',domain='src')[0]
        assert len(outputs)==len(check) and all(torch.isfinite(p['scores']).all() for p in outputs)
    save(entry/'probe.json',dict(passed=True,native_source_forward_gradient_reference_passed=True,
        max_native_gradient_absolute_error=max_error,full_disk_restore=True,
        original_prototype_file_sha256=sha(CODE/'libs/models/SADA/prototype_capacity.py'),
        target_GT_training=False,new_VLM_queries=0,probe_weights_transferred=False,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,updates=rows))


def audit(cfg,entry):
    from tools.baseline_protocol import termination
    results=read(entry/'evaluation/manifest.json')
    best=termination(read(entry/'termination.json')['history'])['best_epoch']
    assert results['checkpoint_sha256']==sha(entry/'best.pt')
    assert all(v['selected_epoch']==best for v in results['results'].values())
    save(entry/'audit.json',dict(passed=True,source_selected=True,selected_epoch=best,
        detector=cfg['generality']['detector'],condition=cfg['generality']['condition'],
        shift=cfg['baseline']['shift'],target_GT_training=False,seed=123,
        results=results['results'],checkpoint_sha256=sha(entry/'best.pt'),
        complete_states=['model','EMA','3 optimizers','3 schedulers','all RNG and loaders']))


def worker(mode,cfg,entry):
    if mode!='audit':
        from tools.baseline_gpu import first_cuda_guard
        first_cuda_guard()
    from tools.generalization_models import install_runtime
    base=install_runtime()
    os.chdir('/vepfs/vlms/SADA_PLUS')
    if mode=='probe':probe(cfg,entry)
    elif mode=='train':base.train(cfg,entry)
    elif mode=='evaluate':base.evaluate(cfg,entry,cfg['baseline']['annotation_for_offline_evaluation'])
    else:audit(cfg,entry)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','probe','train','evaluate','audit'])
    p.add_argument('--root');p.add_argument('--entry');a=p.parse_args()
    if a.mode=='prepare':prepare(Path(a.root))
    else:
        e=Path(a.entry);worker(a.mode,read(e/'config.json'),e)
