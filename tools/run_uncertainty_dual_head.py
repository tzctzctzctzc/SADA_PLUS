"""四卡公共锁队列：路线二六方向，独立端点专家和完整离线评估。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.uncertainty_dual_head import SCHEMA,SPEC
REFERENCE=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029')

def gpu(args):
    from tools.baseline_gpu import first_cuda_guard
    guard=first_cuda_guard()
    import torch
    import numpy as np
    from libs.models.SADA import make_meta_arch
    from libs.trainer.da_trainer import move_to_device
    from libs.utils import fix_random_seed
    from tools import baseline_engine as base
    from tools.prototype_target_cache import detector_features,assert_same_prediction
    from tools.uncertainty_dual_head import EndpointHead,point_context,objective,offset_iou,fuse
    e=args.entry;cfg=read(e/'config.json');bound=read(e/'inputs.json')
    assert code_identity()==read(args.output/'plan.json')['code']
    old=Path(bound['reference']);assert sha(old/'best.pt')==bound['checkpoint_sha256']
    os.chdir(args.data_root);fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda()
    saved=torch.load(old/'best.pt',map_location='cpu');model.load_state_dict(saved['ema_state_dict'],strict=True);del saved
    model.eval().requires_grad_(False);initial=base.tensor_digest(model.state_dict())
    head=EndpointHead().cuda();generator=torch.Generator().manual_seed(123)
    head.initialize(model.reg_head.offset_head.conv.weight,model.reg_head.offset_head.conv.bias,
                    torch.cat([x.scale.reshape(-1) for x in model.reg_head.scale]))
    assert not {p.data_ptr() for p in head.parameters()} & {p.data_ptr() for p in model.reg_head.parameters()}
    def dataset(role):
        return base.dataset(cfg,e/(role+'_gt.json'),cfg['src_train_split'] if role=='source_train' else cfg['src_val_split'],False,False)
    def collect(role):
        ds=dataset(role);result={k:[] for k in ['x','gt','a','levels','video_ids']}
        with torch.no_grad():
            for i in range(len(ds)):
                v=move_to_device([ds[i]],'cuda:0')[0]
                f,m=detector_features(model,v);points=model.point_generator(f);offsets,_=model.reg_head(f,m)
                x,valid,levels=point_context(f,m,model.reg_head);cls,gt=model.label_points(points,[v['segments']],[v['labels']])
                ids=(valid & (cls[0].sum(1)>0)).nonzero(as_tuple=True)[0]
                if len(ids)>2048:ids=ids[torch.randperm(len(ids),generator=generator)[:2048].cuda()]
                a=torch.cat([z[0].T for z in offsets])
                for key,value in [('x',x),('gt',gt[0]),('a',a),('levels',levels)]:result[key].append(value[ids].cpu())
                result['video_ids'].extend([v['metadata']['video_id']]*len(ids))
                print('DENSE_SOURCE_POINTS',role,i,len(ds),len(ids),flush=True)
        for k in ['x','gt','a','levels']:result[k]=torch.cat(result[k])
        if not len(result['x']):raise ValueError('无有效源正样本')
        torch.save(result,e/(role+'_points.pt'));return result
    train=collect('source_train');val=collect('source_val')
    assert set(train['video_ids']).isdisjoint(val['video_ids'])
    ids=np.asarray(val['video_ids']);counts={v:int((ids==v).sum()) for v in set(ids)}
    weights=torch.tensor([1/counts[v] for v in ids],device='cuda:0')
    opt=torch.optim.AdamW(head.parameters(),lr=.001);best=float('inf');best_epoch=0;history=[];accepted=False
    def validate():
        head.eval();terms=[];bp=[];sp=[]
        with torch.no_grad():
            for start in range(0,len(val['x']),512):
                loss,b,s=objective(head,val['x'][start:start+512].cuda(),val['gt'][start:start+512].cuda())
                terms.append(loss);bp.append(b);sp.append(s)
        return float((torch.cat(terms)*weights).sum()/weights.sum()),torch.cat(bp),torch.cat(sp)
    initial_value,initial_prediction,_=validate()
    difference=float((initial_prediction-val['a'].cuda()).abs().max())
    if not torch.allclose(initial_prediction,val['a'].cuda(),rtol=1e-4,atol=1e-5):
        raise ValueError('源预训练回归初始化不匹配')
    best=initial_value;best_epoch=0;history.append(dict(epoch=0,source_validation=best,initialization=True))
    torch.save(dict(state_dict=head.state_dict(),epoch=0,specification=SPEC),e/'best_auxiliary.pt')
    save(e/'initialization_proof.json',dict(status='passed',maximum_absolute_error=difference,
        original_regression_initialization=True,epoch_zero_source_loss=best,original_detector_not_modified=True))
    for epoch in range(1,51):
        head.train()
        for idx in torch.randperm(len(train['x']),generator=generator).split(512):
            before=base.tensor_digest(head.state_dict()) if not accepted else None
            loss,_,_=objective(head,train['x'][idx].cuda(),train['gt'][idx].cuda());loss=loss.mean()
            opt.zero_grad();loss.backward();grad=sum(float(p.grad.square().sum()) for p in head.parameters() if p.grad is not None)**.5
            if not torch.isfinite(loss) or not grad>0:raise ValueError('独立端点分支损失/梯度无效')
            opt.step()
            if not accepted:
                assert before!=base.tensor_digest(head.state_dict())
                save(e/'acceptance.json',dict(status='passed',source_loss=float(loss),gradient_norm=grad,
                    branch_parameters_updated=True,independent_parameter_storage=True,target_gt_used=False,
                    shared_frozen_regression_tower=True,independent_trainable_final_layer=True,
                    guard=guard,guard_sha256=sha(guard)));accepted=True
        value,_,_=validate();history.append(dict(epoch=epoch,source_validation=value))
        if value<best:
            best=value;best_epoch=epoch;torch.save(dict(state_dict=head.state_dict(),epoch=epoch,specification=SPEC),e/'best_auxiliary.pt')
        torch.save(dict(state_dict=head.state_dict(),optimizer=opt.state_dict(),history=history,
            generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0)),e/'latest_auxiliary.pt')
        save(e/'auxiliary_history.json',history);print('ENDPOINT_HEAD_EPOCH',epoch,value,flush=True)
        if epoch-best_epoch>=15:break
    head.load_state_dict(torch.load(e/'best_auxiliary.pt',map_location='cpu')['state_dict'],strict=True);head.eval()
    value,b,scale=validate();a=val['a'].cuda();gt=val['gt'].cuda();levels=val['levels'].cuda()
    aerr=(a-gt).abs();berr=(b-gt).abs();a_scales=[];ratios=[]
    # 每个源验证视频等权拟合尺度；属于源校准，不把同集效果冒充独立泛化证据。
    def average(values,take):
        w=weights*take
        return (values*w[:,None]).sum(0)/w.sum().clamp_min(1e-6)
    for level in range(6):
        take=levels==level
        if int(take.sum())<20:take=torch.ones_like(take)
        a_scales.append(average(aerr,take).clamp_min(.01))
        ratios.append((average(berr,take)/average(scale,take).clamp_min(.01)).clamp_min(.01))
    calibration=dict(a_scale=torch.stack(a_scales),b_ratio=torch.stack(ratios))
    torch.save({k:v.cpu() for k,v in calibration.items()},e/'source_calibration.pt')
    save(e/'source_mechanism.json',dict(source_only=True,calibration_fitted_on_source_validation=True,
        independent_generalization_established=False,samples=len(a),videos=len(set(ids)),
        original_point_iou=float(offset_iou(a,gt).mean()),auxiliary_point_iou=float(offset_iou(b,gt).mean()),
        mean_endpoint_disagreement=((a-b).abs().mean(0)).cpu().tolist()))
    for domain in ['source','target']:
        ds=base.dataset(cfg,e/(domain+'_val_unlabeled.json'),cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'],False,True)
        columns={mode:{k:[] for k in ['video-id','t-start','t-end','label','score']} for mode in ['control','refined']}
        with torch.no_grad():
            for index in range(len(ds)):
                v=move_to_device([ds[index]],'cuda:0')[0]
                assert v['segments'].numel()==0 and v['labels'].numel()==0
                f,m=detector_features(model,v);points=model.point_generator(f);logits,_=model.cls_head(f,m);offsets,_=model.reg_head(f,m)
                x,valid,levels=point_context(f,m,model.reg_head);original=torch.cat([z[0].T for z in offsets]);prob=torch.cat([z[0].T.sigmoid().max(1).values for z in logits])*valid
                fused=original.clone()
                for start in range(0,len(x),512):
                    b,s=head(x[start:start+512]);fused[start:start+512],_=fuse(original[start:start+512],b,s,levels[start:start+512],calibration,prob[start:start+512])
                sizes=[z.shape[-1] for z in offsets];new_offsets=[z[None] for z in fused.split(sizes)]
                def infer(outs):
                    return model.inference([v],points,[z.squeeze(1) for z in m],[z.permute(0,2,1) for z in logits],outs,[None]*len(f),[None]*len(f))[0]
                pred=infer([z.permute(0,2,1) for z in offsets]);new=infer(new_offsets)
                if index==0:
                    legacy=model(video_list_src=[v],video_list_tgt=[v],mode='test',domain=domain,curr_epoch=0)
                    assert_same_prediction(pred,legacy[0][0])
                for mode,p in [('control',pred),('refined',new)]:
                    c=columns[mode];c['video-id'].extend([v['metadata']['video_id']]*len(p['scores']))
                    for k,z in [('t-start',p['segments'][:,0]),('t-end',p['segments'][:,1]),('label',p['labels']),('score',p['scores'])]:c[k].extend(z.cpu().tolist())
                print('DUAL_HEAD_INFERENCE',domain,index,len(ds),flush=True)
        for mode,c in columns.items():np.savez(e/(domain+'_'+mode+'.npz'),**{k:np.asarray(v) for k,v in c.items()})
    assert initial==base.tensor_digest(model.state_dict())
    save(e/'gpu_complete.json',dict(status='passed',frozen_detector_unchanged=True,best_epoch=best_epoch,stop_epoch=epoch,
        independent_head_trained=True,guard=guard,guard_sha256=sha(guard),code=code_identity(),target_gt_used=False))


def evaluate(args):
    import numpy as np
    from libs.evaluation import ANETdetection
    e=args.entry;cfg=read(e/'config.json');results={}
    os.chdir(args.data_root)
    annotation=cfg['baseline']['annotation_for_offline_evaluation']
    assert sha(annotation)==cfg['baseline']['annotation_sha256']
    for domain in ['source','target']:
        results[domain]={}
        for mode in ['control','refined']:
            p=e/(domain+'_'+mode+'.npz')
            with np.load(p,allow_pickle=False) as f:pred={k:f[k] for k in f.files}
            ev=ANETdetection(annotation,cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'],num_workers=2,action_subsampling='10_majority_classes')
            ap,mean,*_=ev.evaluate(pred,verbose=True)
            result=dict(map=float(mean),map_by_tiou=ap.tolist(),prediction_sha256=sha(p))
            save(e/'evaluation'/(domain+'_'+mode+'.json'),result);results[domain][mode]=float(mean)
        expected=read(Path(read(e/'inputs.json')['reference'])/'evaluation'/(domain+'_val.json'))['map_0.1_to_0.5']
        assert abs(results[domain]['control']-expected)<1e-7
    artifacts=[p for p in e.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.pt'] and p.name!='complete.json']
    save(e/'complete.json',dict(status='passed',schema=SCHEMA,shift=e.name,results=results,
        delta_pp=100*(results['target']['refined']-results['target']['control']),code=code_identity(),artifacts={str(p):sha(p) for p in artifacts}))

def main(args):
    from tools.run_prototype_target_cache import prepare,INVENTORY,SHIFTS
    from tools.run_sada_baseline import acquire
    identity=code_identity();args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'plan.json',dict(schema=SCHEMA,code=identity,specification=SPEC,route=2,revision=1,
        maximum_revisions=2,gpus=[0,1,2,3],reference=str(REFERENCE),target_parameter_search=False))
    prepare(args.output,REFERENCE,INVENTORY,identity)
    for shift in SHIFTS:
        e=args.output/'formal'/shift;cfg=read(e/'config.json')
        db=read(cfg['baseline']['annotation_for_offline_evaluation'])['database']
        for role,splits in [('source_train',cfg['src_train_split']),('source_val',cfg['src_val_split'])]:
            wanted={s.lower() for s in splits};save(e/(role+'_gt.json'),dict(database={k:v for k,v in db.items() if v['subset'].lower() in wanted}),sort_keys=False)
    if args.previous:raise ValueError('路线二未实现部分恢复入口，禁止重送计划')
    active={};pending=list(SHIFTS);evals=[];complete=[]
    def spawn(shift,stage,lease=None):
        e=args.output/'formal'/shift;env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        if lease:env.update(CUDA_VISIBLE_DEVICES=lease['uuid'],SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
        with (e/(stage+'.log')).open('x') as log:return subprocess.Popen([sys.executable,'-u','-B',__file__,'--output',str(args.output),'--entry',str(e),'--stage',stage],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        while pending or active or evals:
            for key,t in list(active.items()):
                code=t['p'].poll()
                if code is None:continue
                if t.get('ctx'):t['ctx'].__exit__(None,None,None)
                del active[key]
                if code:raise RuntimeError('阶段失败: '+key+' exit='+str(code))
                if t['stage']=='gpu':evals.append(t['shift'])
                else:
                    e=args.output/'formal'/t['shift'];r=read(e/'complete.json')
                    from tools.uncertainty_dual_head import audit_entry
                    audit_entry(e)
                    complete.append(dict(shift=t['shift'],complete=str(e/'complete.json'),sha256=sha(e/'complete.json'),delta_pp=r['delta_pp'],results=r['results']))
            while evals and sum(t['stage']=='evaluate' for t in active.values())<2:
                s=evals.pop(0);active[s+'/evaluate']=dict(p=spawn(s,'evaluate'),shift=s,stage='evaluate',gpu=None)
            for s in list(pending):
                lease=acquire([0,1,2,3],args.data_root,args.output,12288,identity['commit'])
                if not lease:break
                ctx,card=lease
                try:p=spawn(s,'gpu',card)
                except BaseException:ctx.__exit__(None,None,None);raise
                pending.remove(s);active[s+'/gpu']=dict(p=p,shift=s,stage='gpu',gpu=card['index'],ctx=ctx)
            save(args.output/'state.json',dict(phase='running',pid=os.getpid(),pending=pending,completed=complete,
                active={k:dict(pid=t['p'].pid,shift=t['shift'],stage=t['stage'],gpu=t['gpu']) for k,t in active.items()}))
            time.sleep(3)
        assert len(complete)==6 and {r['shift'] for r in complete}==set(SHIFTS)
        save(args.output/'final_audit.json',dict(status='passed',count=6,results=complete,mean_delta_pp=sum(r['delta_pp'] for r in complete)/6))
        save(args.output/'state.json',dict(phase='complete',completed=complete,active={}))
    except BaseException as err:
        save(args.output/'error.json',dict(error=str(err)));raise
    finally:
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        for t in active.values():
            try:t['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['p'].pid,signal.SIGKILL);t['p'].wait()
            if t.get('ctx'):t['ctx'].__exit__(None,None,None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--entry',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--stage',choices=['gpu','evaluate'])
    p.add_argument('--previous',type=Path)
    a=p.parse_args();a.output=a.output.resolve()
    if a.stage=='gpu':gpu(a)
    elif a.stage=='evaluate':evaluate(a)
    else:
        def stop(sig,frame):raise SystemExit(128+sig)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main(a)
