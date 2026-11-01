"""持公共锁运行的真实五步二阶等价/测速门禁；不读目标数据、不写正式训练。"""
import argparse, json, os, random, sys, time
from pathlib import Path
from copy import deepcopy
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))


def main(args):
    import numpy as np
    import torch
    from tools.baseline_protocol import read,save,sha,code_identity
    from tools.baseline_gpu import first_cuda_guard
    from tools.baseline_engine import tensor_digest
    from tools.run_meta_ttt import load,data
    from tools.meta_ttt import SPEC,episode
    from tools.meta_ttt_speed import BACKEND,install,restore_training,check_resume
    guard=first_cuda_guard();torch.set_num_threads(8)
    job=read(args.job);os.chdir(job['data_root']);output=args.output;output.mkdir(exist_ok=False,parents=True)
    reports={}
    for arm in ['meta','joint']:
        packets={};timings={};peaks={}
        for backend in ['native',BACKEND]:
            j=dict(job,arm=arm);model,ref,enc,dec,cfg,initial,head_initial=load(j,'cuda:0')
            names=install(enc) if backend==BACKEND else []
            model.cls_head_target.requires_grad_(True);model.reg_head_target.requires_grad_(True)
            params=list(enc.parameters())+list(model.cls_head_target.parameters())+list(model.reg_head_target.parameters())+list(dec.parameters())
            ds=data(j,cfg,cfg['src_train_split'],True)
            random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.cuda.manual_seed(123)
            batch=[ds[0],ds[1]]
            tokens=[v['feats'].shape[-1] for v in batch]
            torch.cuda.reset_peak_memory_stats();seconds=[];packet=None
            for repeat in range(3):
                for p in params:p.grad=None
                torch.cuda.synchronize();start=time.perf_counter();loss_sum=0.
                for source in batch:
                    loss=episode(model,ref,enc,dec,source,arm,0)/len(batch)
                    loss.backward();loss_sum+=float(loss.detach())
                torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
                if repeat==0:
                    packet=dict(loss=loss_sum,none=[p.grad is None for p in params],
                        grads=[p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu') for p in params])
            timings[backend]=seconds[1:];peaks[backend]=torch.cuda.max_memory_reserved()/2**20
            assert tensor_digest(model.state_dict())==initial and tensor_digest(ref.state_dict())==initial
            packets[backend]=packet
            del model,ref,enc,dec,params,ds,batch,loss
            torch.cuda.empty_cache()
        a,b=packets['native'],packets[BACKEND]
        assert a['none']==b['none']
        av=torch.cat([v.flatten() for v in a['grads']]);bv=torch.cat([v.flatten() for v in b['grads']])
        relative=float(torch.linalg.vector_norm(av-bv)/torch.linalg.vector_norm(av))
        cosine=float(torch.nn.functional.cosine_similarity(av[None],bv[None]))
        max_absolute=max(float((x-y).abs().max()) for x,y in zip(a['grads'],b['grads']))
        assert abs(a['loss']-b['loss'])<1e-5 and relative<1e-3 and cosine>.99999,(arm,a['loss'],b['loss'],relative,cosine)
        old=sum(timings['native'])/2;new=sum(timings[BACKEND])/2
        reports[arm]=dict(status='passed',batch=2,inner_steps=5,second_order=True,
            source_tokens=tokens,
            original_loss=a['loss'],optimized_loss=b['loss'],gradient_relative_l2=relative,gradient_cosine=cosine,
            maximum_gradient_absolute_error=max_absolute,seconds=timings,speedup=old/new,peak_reserved_mib=peaks,
            patched_layers=names,all_parameters_and_reference_unchanged=True,target_GT_used=False)
        save(output/('comparison_'+arm+'.json'),reports[arm]);print('BENCHMARK',arm,json.dumps(reports[arm]),flush=True)
        del packets,av,bv,a,b
    # 原已训练保存点的完整state/AdamW/RNG恢复与原生精度下下一批次一致。
    if args.resume_job:
        j=read(args.resume_job);j['execution_backend']=BACKEND
        model,ref,enc,dec,cfg,initial,head_initial=load(j,'cuda:0')
        model.cls_head_target.requires_grad_(True);model.reg_head_target.requires_grad_(True)
        base=list(enc.parameters())+list(model.cls_head_target.parameters())+list(model.reg_head_target.parameters())
        optimizer=torch.optim.AdamW([dict(params=base,lr=SPEC['outer_lr']),dict(params=dec.parameters(),lr=SPEC['decoder_lr'])],weight_decay=SPEC['weight_decay'])
        source=Path(j['output']);saved=torch.load(source/'latest.pt',map_location='cpu')
        epoch=check_resume(saved,j,cfg,initial,head_initial)
        from tools.baseline_engine import loader
        generator=torch.Generator();ds=data(j,cfg,cfg['src_train_split'],True)
        draws=[]
        for _ in range(2):
            restore_training(saved,model,dec,optimizer,generator)
            dl=loader(ds,True,generator,SPEC['source_batch'],0);batch=next(iter(dl))
            draws.append(tensor_digest(dict(batch=batch,optimizer=optimizer.state_dict(),rng=[random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state(0),generator.get_state()])))
        assert draws[0]==draws[1]
        resume=dict(status='passed',source_checkpoint=str(source/'latest.pt'),source_checkpoint_sha256=sha(source/'latest.pt'),
            epoch=epoch,next_epoch=epoch+1,all_rng_optimizer_and_next_source_crop_exact=True,draw_sha256=draws[0])
        save(output/'resume_check.json',resume)
    result=dict(status='passed',code=code_identity(),guard=str(guard),reports=reports,
        mathematical_protocol=SPEC,precision='unchanged-fp32',target_GT_used=False,
        resume=read(output/'resume_check.json') if (output/'resume_check.json').exists() else None)
    save(output/'complete.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-job',type=Path);main(p.parse_args())
