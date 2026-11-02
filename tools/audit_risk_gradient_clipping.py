"""独立只读GPU：原完整e5状态/同一真实batch，测detach头对全局裁剪的耦合。"""
import argparse
from copy import deepcopy
import json,os,subprocess,sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))


def run(entry,output):
    from tools.baseline_protocol import read,save,sha
    output.mkdir(exist_ok=False,parents=True)
    uuid=os.environ['CUDA_VISIBLE_DEVICES']
    row=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert row[0].strip()==uuid and int(row[1])>=12288
    save(output/'guard.json',dict(uuid=uuid,free_mib=int(row[1]),child=os.getpid(),parent=os.getppid(),purpose='read_only_no_optimizer_steps'))
    import torch
    from tools import baseline_engine as base
    from tools.run_background_risk import build
    cfg=read(entry/'config.json');saved=torch.load(entry/'epoch005.pt',map_location='cpu')
    assert saved['epoch']==5 and saved['config']==cfg
    ctx=build(cfg);base.restore(ctx,saved);base.epoch_setup(ctx,6);ctx.model.train()
    source=base.move_to_device(next(iter(ctx.train_loader_src)),'cuda:0')
    target=base.move_to_device(next(iter(ctx.train_loader_tgt)),'cuda:0')
    frozen=deepcopy(base.capture(ctx,cfg,5,saved['history']))
    results={}
    for enabled in (False,True):
        base.restore(ctx,frozen);ctx.risk_enabled=enabled
        for p in ctx.model.parameters():p.grad=None
        loss=ctx.model(video_list_src=source,video_list_tgt=target,mode='train',curr_epoch=6)[0]
        final=cfg['model']['loss_weights']['weight_downstream']*loss['downstream_loss']+cfg['model']['loss_weights']['weight_adv']*loss['loss_conditional_adv_rev']
        final.backward()
        def norm(parameters):
            return float(torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in parameters if p.grad is not None])))
        before=norm(ctx.model.backbone_source.parameters())
        grad_digest=base.tensor_digest({k:p.grad for k,p in ctx.model.backbone_source.named_parameters()})
        head=norm(ctx.model.supervision_risk.parameters()) if enabled else 0.
        total=float(torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),cfg['train_cfg']['clip_grad_l2norm']))
        after=norm(ctx.model.backbone_source.parameters())
        results[str(enabled)]=dict(SGP_gradient_before_clip=before,SGP_gradient_after_clip=after,
            preclip_SGP_digest=grad_digest,risk_head_gradient_norm=head,total_gradient_before_clip=total,
            effective_clip_factor=after/before,loss=float(final.detach()))
    base.restore(ctx,frozen)
    assert base.tensor_digest(base.capture(ctx,cfg,5,saved['history']))==base.tensor_digest(frozen)
    result=dict(rows=results,same_SGP_gradient_before_clip=results['True']['preclip_SGP_digest']==results['False']['preclip_SGP_digest'],
        relative_SGP_clip_factor=results['True']['effective_clip_factor']/results['False']['effective_clip_factor'],
        checkpoint_sha256=sha(entry/'epoch005.pt'),source_videos=[v['metadata']['video_id'] for v in source],
        no_optimizer_steps=True,full_state_restored=True,target_new_loss_disabled_epoch6=True,one_batch_not_trajectory_causality=True)
    save(output/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('entry',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.entry,a.output)
