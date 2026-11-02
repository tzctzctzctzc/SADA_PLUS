"""只读源缓存，克隆新头比较逐端点/合批耗时、梯度及AdamW更新。"""
import argparse,os,sys,time,json
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))

def main(a):
    from tools.phase_runner_gpu import first_cuda_guard
    guard=first_cuda_guard()
    import torch
    from tools.baseline_protocol import read,save,sha,code_identity
    from tools.phase_boundary import PhaseBoundary,source_loss,SPEC
    from tools.phase_candidate_components import _device_tree,paired_source_error_loss
    cfg=read(a.source/'config.json');idx=read(a.source/'source_candidate_cache_index.json')
    start=time.perf_counter();data=_device_tree(torch.load(idx['roles']['train'][0]['path'],map_location='cpu'),'cuda:0');torch.cuda.synchronize()
    load=time.perf_counter()-start
    state=torch.load(a.snapshot,map_location='cpu');h=PhaseBoundary(torch.load(SPEC['text_cache'],map_location='cpu')['embeddings']).cuda();h.load_state_dict(state['state_dict'])
    model=SimpleNamespace(fpn_strides=[cfg['model']['scale_factor']**i for i in range(cfg['model']['backbone_arch'][-1]+1)])
    out=[];models=[];optimizers=[];grads=[]
    for batch in [False,True]:
        head=deepcopy(h);head.batch_candidate_errors=batch;opt=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.05);opt.load_state_dict(deepcopy(state['optimizer']))
        torch.cuda.reset_peak_memory_stats();rows=[]
        for step in range(3):
            opt.zero_grad();torch.cuda.synchronize();t=time.perf_counter();losses=[]
            for key,videos in [('weak','videos'),('strong','strong_videos')]:
                f,m,c,r=data[key];loss,stats=source_loss(model,head,f,m,r,data[videos],c);losses.append(loss)
            paired,_=paired_source_error_loss(model,head,data['weak'],data['strong'],data['views'],data['videos'])
            loss=torch.stack(losses).mean()+paired;loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);torch.cuda.synchronize()
            rows.append(dict(step=step,seconds=time.perf_counter()-t,loss=float(loss.detach()),stats=stats))
            opt.step()
        out.append(dict(batch=batch,steps=rows,peak_mib=torch.cuda.max_memory_reserved()/2**20))
        grads.append({k:p.grad.detach().cpu() for k,p in head.named_parameters() if p.grad is not None});models.append({k:v.cpu() for k,v in head.state_dict().items()});optimizers.append(deepcopy(opt.state_dict()))
        del head,opt;torch.cuda.empty_cache()
    delta=max(float((models[0][k]-models[1][k]).abs().max()) for k in models[0]);gd=max(float((grads[0][k]-grads[1][k]).abs().max()) for k in grads[0])
    ld=max(abs(x['loss']-y['loss']) for x,y in zip(out[0]['steps'],out[1]['steps']))
    assert all(x['stats']==y['stats'] for x,y in zip(out[0]['steps'],out[1]['steps']))
    passed=delta<5e-5 and gd<2e-5 and ld<1e-5
    result=dict(status='passed' if passed else 'numerical_gate_failed',code=code_identity(),guard=guard,guard_sha256=sha(guard),source_only=True,formal_training=False,
                snapshot_sha256=sha(a.snapshot),snapshot_epoch=state['epoch'],cache_load_and_transfer_seconds=load,runs=out,
                max_parameter_difference=delta,max_gradient_difference=gd,max_loss_difference=ld,
                steady_step_speedup=sum(x['seconds'] for x in out[0]['steps'][1:])/sum(x['seconds'] for x in out[1]['steps'][1:]))
    save(a.output,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
