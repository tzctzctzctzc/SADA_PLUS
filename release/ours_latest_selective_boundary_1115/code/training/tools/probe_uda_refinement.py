"""六向CPU真实batch接线检查；0更新，不替代正式冷启动训练验收。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import sys,json,argparse,time,gc,random,traceback
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))

def main():
    import torch,numpy as np
    from copy import deepcopy
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.uda_refinement import TF_V3_SPEC,MDA_V3_SPEC
    from tools.baseline_engine import dataset,loader
    from tools.baseline_protocol import random_seeds
    from tools.run_uda_two_routes import save
    p=argparse.ArgumentParser();p.add_argument('--reference-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--reuse')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),optimizer_updates=0),f)
    torch.set_num_threads(2);reports={}
    for shift in ['S%d'%i for i in range(1,7)]:
        original=Path(a.reference_root)/shift
        for route,spec in [('time_frequency',TF_V3_SPEC),('mda',MDA_V3_SPEC)]:
            key=route+'/'+shift;save(out/'progress.json',dict(job=key,checked=time.time()))
            if a.reuse and (Path(a.reuse)/route/shift/'result.json').is_file():
                report=json.loads((Path(a.reuse)/route/shift/'result.json').read_text())
                assert report['status']=='passed' and report['optimizer_updates']==0
                reports[key]=dict(report,reused_from=str(Path(a.reuse)/route/shift/'result.json'))
                save(out/route/shift/'result.json',reports[key]);continue
            cfg=json.loads((original/'config.json').read_text());cfg['uda_route']=dict(spec)
            if route=='time_frequency':cfg['model'].update(adv_method=['None'],use_adv_discriminator=False)
            else:cfg['model']['use_level_disc']=True
            cfg['baseline']['tmpdir']=str(out/'tmp');torch.manual_seed(123);np.random.seed(123);random.seed(123)
            model=make_meta_arch('sada',deepcopy(cfg),device='cpu').cpu()
            saved=torch.load(original/'best.pt',map_location='cpu')
            incompatible=model.load_state_dict(saved['state_dict'],strict=False)
            assert not incompatible.unexpected_keys and all(k.startswith('adapt_route.') for k in incompatible.missing_keys)
            for k,v in saved['model_controls'].items():
                cur=getattr(model,k)
                if torch.is_tensor(cur):cur.data.copy_(v)
                elif isinstance(cur,list):
                    for x,y in zip(cur,v):x.data.copy_(y)
                else:setattr(model,k,v)
            del saved;gc.collect();model.eval();model.requires_grad_(False)
            model.adapt_route.requires_grad_(True)
            if route=='mda':
                model.disc_dom_conditional_adv.requires_grad_(True)
                model.adapt_route.calls.fill_(19) # 仅检查第20次的功能路径，不声称自然训练触发。
            seeds=random_seeds(cfg['seed'])
            src=iter(loader(dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],True),True,torch.Generator().manual_seed(seeds['source']),cfg['loader']['batch_size'],0))
            tgt=iter(loader(dataset(cfg,cfg['baseline']['target_manifest'],cfg['tgt_train_split'],True,True),True,torch.Generator().manual_seed(seeds['target']),cfg['loader']['batch_size'],0))
            source=next(src);target=next(tgt)
            assert all(v['segments'].numel()==v['labels'].numel()==0 for v in target)
            captures=[]
            def hook(m,i,o):
                fs=tuple(x.detach().requires_grad_(True) for x in o[0]);captures.append(fs);return fs,o[1]
            h=model.neck_source.register_forward_hook(hook)
            started=time.perf_counter();losses=model(video_list_src=source,video_list_tgt=target,mode='train',curr_epoch=10)[0]
            total=cfg['model']['loss_weights']['weight_downstream']*losses['downstream_loss']+cfg['model']['loss_weights']['weight_adv']*losses.get('loss_conditional_adv_rev',0)
            assert torch.isfinite(total);total.backward()
            grads=[x.grad for x in model.parameters() if x.grad is not None]
            assert grads and all(x.isfinite().all() for x in grads)
            target_grad=sum(float(x.grad.square().sum()) for x in captures[1] if x.grad is not None)**.5
            assert target_grad>0
            report=dict(status='passed',optimizer_updates=0,reference=str(original/'best.pt'),mature_reference_weights_only=True,
                model_mode='eval_dropout_off',target_GT=False,finite_gradient=True,target_feature_grad_norm=target_grad,
                metrics={k:float(v) for k,v in losses.items() if k.startswith('route_')},seconds=time.perf_counter()-started,
                mda_schedule_functionally_positioned_at_20=route=='mda')
            reports[key]=report;save(out/route/shift/'result.json',report)
            h.remove();del model,src,tgt,source,target,captures,losses,total;gc.collect()
    save(out/'result.json',dict(status='passed',jobs=reports,optimizer_updates=0,formal_training_runs=0,cpu_threads=2))

if __name__=='__main__':
    try:main()
    except Exception:
        out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
        (out/'error.json').write_text(json.dumps(dict(error=traceback.format_exc())));raise
