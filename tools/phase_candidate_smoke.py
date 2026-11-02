"""完整源课程前的单方向真实GPU内核检查；不写正式训练权重。"""
import argparse
from copy import deepcopy
import io
import os
from pathlib import Path
import sys
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity


def main(args):
    from tools.phase_runner_gpu import first_cuda_guard
    guard=first_cuda_guard()
    import torch
    from tools import baseline_engine as base
    from tools.phase_boundary import direct,strong_view,map_coordinates,source_loss
    from tools.phase_candidate_components import write_source_cache,_cpu_tree,_device_tree,paired_source_error_loss
    cfg=read(args.entry/'config.json');assert cfg['baseline']['code']==code_identity()
    os.chdir('/vepfs/vlms/SADA_PLUS');ctx=base.build(cfg);model=ctx.model;head=model.phase_boundary
    videos=base.move_to_device(next(iter(ctx.train_loader_src)),model.device);model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        weak=direct(model,videos);strong_videos=[];views=[]
        for i,v in enumerate(videos):
            x,view=strong_view(v,1,i);view['original_length']=v['feats'].shape[-1]
            x['segments']=map_coordinates(v['segments'],view);x['labels']=v['labels']
            strong_videos.append(x);views.append(view)
        strong=direct(model,strong_videos)
    data=_device_tree(_cpu_tree(dict(weak=weak,strong=strong,videos=videos,strong_videos=strong_videos,views=views)),model.device)
    cache_path=args.entry/'source_cache_probe'/'train'/'0000.pt';write_source_cache(cache_path,data)
    assert base.tensor_digest(torch.load(cache_path,map_location='cpu'))==base.tensor_digest(_cpu_tree(data))
    opt=torch.optim.AdamW(head.parameters(),lr=.0003);opt.zero_grad()
    f,m,c,r=data['weak'];loss,stats=source_loss(model,head,f,m,r,data['videos'],c)
    paired,pair_stats=paired_source_error_loss(model,head,data['weak'],data['strong'],data['views'],data['videos'])
    loss=loss+paired;assert torch.isfinite(loss);loss.backward()
    gradients={k:float(p.grad.norm()) for k,p in [('foreground',head.foreground.weight),('candidate_error',head.error_model[-1].weight)]}
    assert all(v>0 for v in gradients.values());opt.step();del data,weak,strong,opt
    before=base.tensor_digest(model.backbone_source.state_dict());reg_before=base.tensor_digest(model.reg_head.state_dict())
    base.epoch_setup(ctx,1);loaders=ctx.train_loader_src,ctx.train_loader_tgt
    ctx.train_loader_src,ctx.train_loader_tgt=[base.StepLoader(x,2) for x in loaders]
    ctx._train_epoch(1);torch.cuda.synchronize();metrics=ctx.train_metrics.result()
    assert before!=base.tensor_digest(model.backbone_source.state_dict()) and reg_before!=base.tensor_digest(model.reg_head.state_dict())
    ctx.train_loader_src,ctx.train_loader_tgt=loaders
    captured=base.capture(ctx,cfg,1,[]);stream=io.BytesIO();torch.save(captured,stream);stream.seek(0)
    expected=torch.load(stream,map_location='cpu');base.restore(ctx,expected)
    assert base.tensor_digest(base.capture(ctx,cfg,1,[]))==base.tensor_digest(expected)
    result=dict(status='passed',scope='one_shift_source_candidate_kernel_and_two_original_optimizer_steps; no formal weights',
        source_videos=[v['metadata']['video_id'] for v in videos],source_loss=float(loss.detach()),source_stats=stats,
        paired_source_stats=pair_stats,source_head_gradients=gradients,backbone_updated=True,regression_updated=True,
        source_cache_disk_roundtrip='passed',
        original_optimizer_steps=2,full_state_roundtrip='passed',new_target_channel_tested=False,
        metrics={k:float(v) for k,v in metrics.items() if k.startswith('t_phase_')},
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,guard=guard,guard_sha256=sha(guard),code=code_identity(),formal_training=False)
    save(args.entry/'result.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,required=True);main(p.parse_args())
