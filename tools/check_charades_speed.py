"""真实同批数据、同保存点的原版/优化版逐步状态与吞吐门禁，无新增教师查询。"""
import argparse,importlib.util,io,itertools,json,os,sys,time
from pathlib import Path
from types import FunctionType,MethodType
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))


class FixedLoader:
    def __init__(self,batches,budget):self.batches,self.budget=batches,budget
    def __iter__(self):return iter(self.batches)
    def __len__(self):return self.budget


def limited_epoch(ctx,steps):
    f=ctx._train_epoch.__func__
    env=dict(f.__globals__,tqdm=lambda rows:itertools.islice(rows,steps))
    ctx._train_epoch=MethodType(FunctionType(f.__code__,env,f.__name__,f.__defaults__,f.__closure__),ctx)


def cpu_copy(value):
    import torch
    stream=io.BytesIO();torch.save(value,stream);stream.seek(0)
    return torch.load(stream,map_location='cpu')


def main(args):
    import torch
    from tools import baseline_engine as fast
    from tools.baseline_protocol import read,save,sha,code_identity
    from tools.baseline_gpu import first_cuda_guard
    proof=first_cuda_guard();torch.set_num_threads(8);os.chdir('/vepfs/vlms/SADA_PLUS')
    spec=importlib.util.spec_from_file_location('original_charades_engine',args.original_code/'tools/baseline_engine.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    args.output.mkdir(parents=True,exist_ok=False);reports={}
    for arm in ['adversarial','no_adversarial']:
        entry=args.previous_root/'formal'/arm;cfg=read(entry/'config.json');saved=torch.load(entry/'latest.pt',map_location='cpu')
        packets={};seconds={};fixed=None;peaks={}
        for name,engine in [('native',original),('optimized',fast)]:
            ctx=engine.build(cfg);engine.restore(ctx,saved);epoch=saved['epoch']+1;budget=cfg['baseline']['epoch_budget']['iterations_per_epoch']
            if fixed is None:
                source=list(itertools.islice(iter(ctx.train_loader_src),args.steps))
                target=list(itertools.islice(iter(ctx.train_loader_tgt),args.steps))
                # 诊断明确覆盖固定缓存中的正例，不改变正式训练抽样。
                cache=read(args.previous_root/'prepared.json');vid=next(v for v,rows in cache['by_video'].items() if any(x['participation_selected'] for x in rows))
                ds=ctx.train_loader_tgt.loader.dataset;index=next(i for i,row in enumerate(ds.data_list) if row['id']==vid)
                target[0][0]=ds[index];fixed=(source,target)
            ctx.train_loader_src=FixedLoader(fixed[0],budget);ctx.train_loader_tgt=FixedLoader(fixed[1],budget)
            # 先热身，再完整恢复；epoch内EMA时钟仍使用原1542步预算。
            limited_epoch(ctx,2);engine.epoch_setup(ctx,epoch);ctx._train_epoch(epoch)
            engine.restore(ctx,saved);engine.epoch_setup(ctx,epoch)
            # 重绑定原函数，避免限2步的包装嵌套；数值计算没有变化。
            if name=='native' or not ctx.charades_speed_enabled:ctx._train_epoch=MethodType(original.Trainer._train_epoch,ctx)
            else:
                from tools.charades_speed import epoch_function
                ctx._train_epoch=MethodType(epoch_function(),ctx)
            limited_epoch(ctx,args.steps);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            ctx._train_epoch(epoch);torch.cuda.synchronize();seconds[name]=time.perf_counter()-start
            packets[name]=dict(state=cpu_copy(engine.capture(ctx,cfg,epoch,saved['history'])),metrics=ctx.train_metrics.result())
            peaks[name]=torch.cuda.max_memory_reserved()/2**20
            del ctx;torch.cuda.empty_cache()
        # 参数/EMA/全部optimizer/scheduler/normalizer/RNG必须逐位完全相同。
        a=fast.tensor_digest(packets['native']['state']);b=fast.tensor_digest(packets['optimized']['state'])
        assert a==b,('full state changed',arm,a,b)
        assert packets['native']['metrics']==packets['optimized']['metrics'],('metrics changed',arm)
        aux=packets['optimized']['metrics'];assert aux['t_rq3_selected']>0 and aux['t_rq3_target_gradient_norm']>0
        reports[arm]=dict(status='passed',epoch=epoch,steps=args.steps,original_budget=budget,batch=2,
            seconds=seconds,speedup=seconds['native']/seconds['optimized'],peak_reserved_mib=peaks,
            full_model_EMA_optimizers_schedulers_controls_RNG_exact=True,metrics_exact=True,full_state_sha256=a,
            prototype_target_gradient=aux['t_rq3_target_gradient_norm'],target_GT_used=False,
            resume_checkpoint_sha256=sha(entry/'latest.pt'))
        reports[arm]['effective_path']='batched-metrics-alias-aware-EMA' if arm=='no_adversarial' else 'original-adversarial-preserved'
        save(args.output/(arm+'.json'),reports[arm]);print('BENCHMARK',arm,json.dumps(reports[arm]),flush=True)
        del saved,packets,fixed
    save(args.output/'complete.json',dict(status='passed',code=code_identity(),reports=reports,guard=str(proof),
        mathematical_changes=False,teacher_queries=0,cache_sha256=sha(args.previous_root/'prepared.json'),
        original_engine_sha256=sha(args.original_code/'tools/baseline_engine.py')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous-root',type=Path,required=True);p.add_argument('--original-code',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=12);main(p.parse_args())
