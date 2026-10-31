"""四卡公共锁队列：路线一六方向，源监督精修和完整离线评估。"""
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
from tools.uncertainty_refiner import SCHEMA,SPEC
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
    from tools.prototype_target_cache import detector_forward,duration_levels,assert_same_prediction
    from tools.uncertainty_refiner import Refiner,context,objective,iou,aligned_iou
    e=args.entry;cfg=read(e/'config.json');bound=read(e/'inputs.json')
    assert code_identity()==read(args.output/'plan.json')['code']
    old=Path(bound['reference']);assert sha(old/'best.pt')==bound['checkpoint_sha256']
    os.chdir(args.data_root);fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda()
    saved=torch.load(old/'best.pt',map_location='cpu');model.load_state_dict(saved['ema_state_dict'],strict=True)
    model.eval().requires_grad_(False);del saved
    initial=base.tensor_digest(model.state_dict());head=Refiner().cuda()
    generator=torch.Generator().manual_seed(123)
    @torch.no_grad()
    def forward_raw(v):
        from tools.prototype_target_cache import detector_features
        f,m=detector_features(model,v);logits,_=model.cls_head(f,m);offsets,_=model.reg_head(f,m)
        points=model.point_generator(f);parts=[];levels=[]
        for l in range(len(f)):
            p=model.inference_single_video([points[l]],[m[l][0,0]],[logits[l][0].T],[offsets[l][0].T],[None],[None])
            parts.append(p);levels.append(torch.full((len(p['scores']),),l,device='cuda:0',dtype=torch.long))
        raw={k:torch.cat([p[k] for p in parts]) for k in ['segments','scores','labels']}
        meta=v['metadata'];raw.update({k:meta[k] for k in ['video_id','fps','duration','feat_stride','feat_num_frames','use_temporal_grid_feats','video_length']})
        return f,m,raw,model.postprocessing([deepcopy(raw)])[0],torch.cat(levels)
    def to_grid(box,v):
        m=v['metadata'];return (box*float(m['fps'])-.5*float(m['feat_num_frames']))/float(m['feat_stride'])
    def source_data(role):
        if (e/(role+'_examples.pt')).exists():
            return torch.load(e/(role+'_examples.pt'),map_location='cpu')
        ds=base.dataset(cfg,e/(role+'_gt.json'),cfg['src_train_split'] if role=='source_train' else cfg['src_val_split'],False,False)
        xs=[];bs=[];gs=[];ids=[]
        for index in range(len(ds)):
            v=move_to_device([ds[index]],'cuda:0')[0];features,masks,raw,pred,_=forward_raw(v)
            gt=v['segments'];labels=v['labels']
            if not len(gt):continue
            duration=(gt[:,1]-gt[:,0]).clamp_min(1)
            jitter=(torch.rand((len(gt),4,2),generator=generator)*.4-.2).cuda()*duration[:,None,None]
            boxes=(gt[:,None]+jitter).reshape(-1,2);truth=gt[:,None].expand(-1,4,-1).reshape(-1,2)
            # 加入真实检测器错误；匹配相同类别实例，禁止跨类别GT监督。
            n=min(200,len(pred['scores']));order=pred['scores'].argsort(descending=True)[:n]
            real=to_grid(pred['segments'][order].cuda(),v);rl=pred['labels'][order].cuda()
            matches=iou(real,gt);matches.masked_fill_(rl[:,None]!=labels[None,:],-1)
            quality,match=matches.max(1);keep=quality>=.1
            boxes=torch.cat([boxes,real[keep]]);truth=torch.cat([truth,gt[match[keep]]])
            levels=duration_levels(boxes,model.reg_range)
            x,valid=context(features,masks,boxes,levels,model.fpn_strides)
            valid &= (boxes[:,0]>=0)&(boxes[:,1]<=v['feats'].shape[-1])&(boxes[:,1]>boxes[:,0])
            xs.append(x[valid].cpu());bs.append(boxes[valid].cpu());gs.append(truth[valid].cpu())
            ids.extend([v['metadata']['video_id']]*int(valid.sum()))
            print('SOURCE_FEATURES',role,index,len(ds),int(valid.sum()),flush=True)
        result=dict(x=torch.cat(xs),boxes=torch.cat(bs),gt=torch.cat(gs),video_ids=ids)
        torch.save(result,e/(role+'_examples.pt'));return result
    train=source_data('source_train');val=source_data('source_val')
    assert set(train['video_ids']).isdisjoint(val['video_ids'])
    opt=torch.optim.AdamW(head.parameters(),lr=.001)
    def validation():
        losses=[];before=[];after=[];errors=[];scales=[]
        head.eval()
        with torch.no_grad():
            for start in range(0,len(val['boxes']),256):
                x,b,g=[val[k][start:start+256].cuda() for k in ['x','boxes','gt']]
                loss,p,s=objective(head,x,b,g);losses.append((float(loss),len(b)))
                before.extend(aligned_iou(b,g).cpu().tolist());after.extend(aligned_iou(p,g).cpu().tolist())
                errors.extend(((p-g).abs()/(b[:,1]-b[:,0])[:,None]).cpu().tolist());scales.extend(s.cpu().tolist())
        return dict(loss=sum(a*n for a,n in losses)/sum(n for _,n in losses),
            mean_iou_before=float(np.mean(before)),mean_iou_after=float(np.mean(after)),
            endpoint_error=errors,endpoint_scale=scales)
    history=[];best=float('inf');best_epoch=0;accepted=False;skip_training=False
    if (e/'recovery.json').exists():
        history=read(e/'training_history.json');saved_head=torch.load(e/'best_refiner.pt',map_location='cpu')
        best_epoch=saved_head['epoch'];epoch=history[-1]['epoch']
        assert epoch==50 or epoch-best_epoch>=15
        head.load_state_dict(saved_head['state_dict'],strict=True);skip_training=True
    for epoch in ([] if skip_training else range(1,51)):
        head.train();order=torch.randperm(len(train['boxes']),generator=generator);total=0.
        for indices in order.split(256):
            x,b,g=[train[k][indices].cuda() for k in ['x','boxes','gt']]
            before=base.tensor_digest(head.state_dict()) if not accepted else None
            loss,_,_=objective(head,x,b,g);opt.zero_grad();loss.backward()
            grad=sum(float(p.grad.square().sum()) for p in head.parameters() if p.grad is not None)**.5
            if not torch.isfinite(loss) or not grad>0:raise ValueError('精修损失/梯度无效')
            opt.step();total+=float(loss)
            if not accepted:
                assert before!=base.tensor_digest(head.state_dict())
                save(e/'acceptance.json',dict(status='passed',source_loss=float(loss),gradient_norm=grad,
                    trainable_branch_updated=True,target_gt_used=False,guard=guard,guard_sha256=sha(guard)))
                accepted=True
        report=validation();history.append(dict(epoch=epoch,source_validation=report['loss'],train_loss_sum=total))
        if report['loss']<best:
            best=report['loss'];best_epoch=epoch
            torch.save(dict(state_dict=head.state_dict(),epoch=epoch,specification=SPEC),e/'best_refiner.pt')
        torch.save(dict(state_dict=head.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=history,
                        generator=generator.get_state(),cuda_rng=torch.cuda.get_rng_state(0)),e/'latest_refiner.pt')
        save(e/'training_history.json',history);print('REFINER_EPOCH',epoch,report['loss'],report['mean_iou_before'],report['mean_iou_after'],flush=True)
        if epoch-best_epoch>=15:break
    head.load_state_dict(torch.load(e/'best_refiner.pt',map_location='cpu')['state_dict'],strict=True);head.eval()
    save(e/'source_mechanism.json',validation())
    for domain in ['source','target']:
        ds=base.dataset(cfg,e/(domain+'_val_unlabeled.json'),cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'],False,True)
        columns={mode:{k:[] for k in ['video-id','t-start','t-end','label','score']} for mode in ['control','refined']}
        for index in range(len(ds)):
            v=move_to_device([ds[index]],'cuda:0')[0]
            assert v['segments'].numel()==0 and v['labels'].numel()==0
            features,masks,raw,pred,levels=forward_raw(v)
            if index==0:
                original=model(video_list_src=[v],video_list_tgt=[v],mode='test',domain=domain,curr_epoch=0)
                assert_same_prediction(pred,original[0][0])
            # 使用尚未裁边、尚未NMS的真实候选和原始检测层，避免二次NMS/提前裁边混淆。
            boxes=raw['segments'];refined=boxes.clone()
            with torch.no_grad():
                for start in range(0,len(boxes),128):
                    b=boxes[start:start+128];x,valid=context(features,masks,b,levels[start:start+128],model.fpn_strides)
                    p,_=head(x,b);refined[start:start+128]=torch.where(valid[:,None],p,b)
            meta=v['metadata'];raw['segments']=refined
            new=model.postprocessing([raw])[0]
            for mode,p in [('control',pred),('refined',new)]:
                count=len(p['scores']);c=columns[mode];c['video-id'].extend([meta['video_id']]*count)
                for key,values in [('t-start',p['segments'][:,0]),('t-end',p['segments'][:,1]),('label',p['labels']),('score',p['scores'])]:c[key].extend(values.cpu().tolist())
            print('EVALUATING',domain,index,len(ds),flush=True)
        for mode,c in columns.items():np.savez(e/(domain+'_'+mode+'.npz'),**{k:np.asarray(v) for k,v in c.items()})
    assert initial==base.tensor_digest(model.state_dict())
    save(e/'gpu_complete.json',dict(status='passed',frozen_detector_unchanged=True,best_epoch=best_epoch,stop_epoch=epoch,
        guard=guard,guard_sha256=sha(guard),code=code_identity(),target_gt_used=False))

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
    save(args.output/'plan.json',dict(schema=SCHEMA,code=identity,specification=SPEC,route=1,revision=0,
        maximum_revisions=2,gpus=[0,1,2,3],reference=str(REFERENCE),target_parameter_search=False))
    prepare(args.output,REFERENCE,INVENTORY,identity)
    for shift in SHIFTS:
        e=args.output/'formal'/shift;cfg=read(e/'config.json')
        db=read(cfg['baseline']['annotation_for_offline_evaluation'])['database']
        for role,splits in [('source_train',cfg['src_train_split']),('source_val',cfg['src_val_split'])]:
            wanted={s.lower() for s in splits};save(e/(role+'_gt.json'),dict(database={k:v for k,v in db.items() if v['subset'].lower() in wanted}),sort_keys=False)
    if args.previous:
        import shutil
        for old in args.previous.glob('formal/*'):
            if not (old/'best_refiner.pt').exists():continue
            if (old/'complete.json').exists():raise ValueError('完整项需独立复用接口，拒绝重复评估')
            hist=read(old/'training_history.json');saved=__import__('torch').load(old/'best_refiner.pt',map_location='cpu')
            if hist[-1]['epoch']!=50 and hist[-1]['epoch']-saved['epoch']<15:
                raise ValueError('部分头训练不允许从头重跑')
            new=args.output/'formal'/old.name;copied={}
            for name in ['best_refiner.pt','latest_refiner.pt','training_history.json','source_train_examples.pt','source_val_examples.pt','acceptance.json']:
                p=old/name;assert p.exists();shutil.copyfile(p,new/name);copied[str(p)]=sha(p)
            save(new/'recovery.json',dict(previous=str(old),previous_code=read(args.previous/'plan.json')['code'],artifacts=copied,training_reused=True,only_evaluation_restarted=True))
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
                    for p,digest in r['artifacts'].items():assert sha(p)==digest
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
