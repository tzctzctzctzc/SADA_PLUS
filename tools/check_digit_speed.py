"""不改在训代码的真实保存点ABBA测速；仅合并finite/日志的GPU→CPU同步。

原官方CUDA算子已知非逐位确定，所以同时运行原版自身重放作为噪声对照；
batch、FP32、损失、梯度裁剪、优化器、EMA、scheduler、数据和RNG均不改。
"""
import argparse,os,sys,time,statistics,json
from copy import deepcopy
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))


def batched_step(ctx,batch,measure=False):
    import torch
    from tools.digit_engine import device_batch,group_parameters
    m=ctx['model'];m.train();ctx['criterion'].train();ctx['opt'].zero_grad(set_to_none=True)
    x,info=device_batch(batch,ctx['device']);out=m(x,info);parts=ctx['criterion'](out,info);weights=ctx['criterion'].weight_dict
    loss=sum(parts[k]*weights[k] for k in weights)
    finite=torch.stack([torch.isfinite(loss).all()]+[torch.isfinite(parts[k]).all() for k in weights]).all()
    assert bool(finite)
    loss.backward();norms={}
    if measure:
        for name,params in group_parameters(m).items():
            gs=[p.grad.detach().norm() for p in params.values() if p.grad is not None]
            norms[name]=float(torch.linalg.vector_norm(torch.stack(gs))) if gs else 0.
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),.1);assert torch.isfinite(norm)
    ctx['opt'].step();ctx['ema'].update_parameters(m)
    values=torch.stack([loss.detach(),parts['loss_ce'].detach(),parts['loss_iou'].detach()]).cpu().tolist()
    return dict(zip(('final_loss','cls_loss','reg_loss'),values)),norms


def main(args):
    import torch,itertools
    from tools import digit_engine as e
    from tools.digit_protocol import read,save,sha,identity
    from tools.digit_gpu import first_cuda_guard
    guard=first_cuda_guard();identity=identity();out=args.output.resolve();out.mkdir(exist_ok=False)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=saved['config']
    source_only=cfg['benchmark']['spec']['target_training'] is False
    assert source_only
    os.chdir('/vepfs/vlms/SADA_PLUS');torch.set_num_threads(8)
    ctx=e.build(cfg);batches=list(itertools.islice(ctx['train'],args.steps));assert len(batches)==args.steps
    # AdamW的CPU step标量可能被load_state_dict直接引用，重放必须深拷贝，
    # 否则测速本身会递增保存点里的step计数；正式训练路径完全未修改。
    saved_digest=e.digest(saved)
    e.restore(ctx,deepcopy(saved))
    for batch in batches:e.step(ctx,batch)
    torch.cuda.synchronize();assert e.digest(saved)==saved_digest
    reference=None;timings={};comparisons={};losses={}
    for label,fn in [('original1',e.step),('batched1',batched_step),('batched2',batched_step),('original2',e.step)]:
        e.restore(ctx,deepcopy(saved));torch.cuda.synchronize();start=time.perf_counter();trace=[]
        for i,batch in enumerate(batches):
            value,_=fn(ctx,batch,measure=i==0);trace.append(value)
        torch.cuda.synchronize();timings[label]=time.perf_counter()-start
        after=e.cpu_copy(e.capture(ctx,cfg,saved['epoch'],saved['history'],saved['metrics']))
        if reference is None:reference=after
        else:
            comparisons[label]=e.repeat_difference(reference,after)
            comparisons[label]['rng_exact']=e.digest(reference['rng'])==e.digest(after['rng'])
            comparisons[label]['loss_max_absolute_difference']=max(abs(x[k]-y[k]) for x,y in zip(losses['original1'],trace) for k in x)
        losses[label]=trace
        assert e.digest(saved)==saved_digest,'benchmark mutated reference checkpoint'
        if after is not reference:del after
        print('BENCH',label,timings[label],comparisons.get(label),flush=True)
    original=statistics.median([timings['original1'],timings['original2']]);batched=statistics.median([timings['batched1'],timings['batched2']])
    passed=all(x['allclose'] and x['rng_exact'] for x in comparisons.values())
    save(out/'complete.json',dict(status='passed-numerical-equivalence' if passed else 'rejected-numerical-equivalence',
        code=identity,guard=guard,guard_sha256=sha(guard),checkpoint=str(args.checkpoint),epoch=saved['epoch'],
        config_sha256=e.digest(cfg),steps=args.steps,timings=timings,comparisons=comparisons,
        median_speedup=original/batched,source_only=True,target_GT_read=False,active_training_modified=False,
        batch_changed=False,precision_changed=False,mathematics_changed=False,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        criterion='atol1e-6-rtol1e-5-same-as-official-operator-replay-gate',
        immutable_checkpoint_replay=True,all_measured_shapes_warmed=True,
        recommended_adoption=passed and original/batched>=1.1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=12)
    p.add_argument('--extension',required=True);a=p.parse_args();sys.path.insert(0,a.extension);main(a)
