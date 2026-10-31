"""路线四：四卡公共锁，源监督联合区间能量和完整离线评估。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.uncertainty_interval_energy import SCHEMA,SPEC
REFERENCE=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029')
SOURCE_EXAMPLES=Path('/vepfs/vlms/uncertainty_refiner6_c51d816_1101/formal')


def gpu(args):
    from tools.baseline_gpu import first_cuda_guard
    guard=first_cuda_guard()
    import torch
    import numpy as np
    from libs.models.SADA import make_meta_arch
    from libs.trainer.da_trainer import move_to_device
    from libs.utils import fix_random_seed
    from tools import baseline_engine as base
    from tools.prototype_target_cache import detector_features,duration_levels,assert_same_prediction
    from tools.uncertainty_refiner import context,iou,aligned_iou
    from tools.uncertainty_interval_energy import IntervalEnergy,candidates,select_interval,objective
    e=args.entry;cfg=read(e/'config.json');bound=read(e/'inputs.json')
    assert code_identity()==read(args.output/'plan.json')['code']
    reference=Path(bound['reference']);assert sha(reference/'best.pt')==bound['checkpoint_sha256']
    os.chdir(args.data_root);fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda()
    saved=torch.load(reference/'best.pt',map_location='cpu');model.load_state_dict(saved['ema_state_dict'],strict=True);del saved
    model.eval().requires_grad_(False);initial=base.tensor_digest(model.state_dict())
    generator=torch.Generator().manual_seed(123);original=SOURCE_EXAMPLES/e.name;old_complete=read(original/'complete.json');refs={}
    assert read(original/'config.json')==cfg
    for name in ['source_train_examples.pt','source_val_examples.pt','source_train_gt.json','source_val_gt.json']:
        p=original/name;digest=sha(p);assert digest==old_complete['artifacts'][str(p)];refs[str(p)]=digest
    for role in ['source_train','source_val']:assert sha(e/(role+'_gt.json'))==refs[str(original/(role+'_gt.json'))]
    save(e/'source_examples_reference.json',dict(previous_complete=str(original/'complete.json'),previous_complete_sha256=sha(original/'complete.json'),artifacts=refs,read_only=True,only_boxes_truth_and_video_ids_reused=True))

    @torch.no_grad()
    def extract(f,m,boxes,levels):
        grid,delta=candidates(boxes)
        x,valid=context(f,m,grid.reshape(-1,2),levels[:,None].expand(-1,9).reshape(-1),model.fpn_strides)
        return grid,delta,x.reshape(len(boxes),9,-1),valid.reshape(len(boxes),9)

    def source_data(role):
        examples=torch.load(original/(role+'_examples.pt'),map_location='cpu');del examples['x']
        ds=base.dataset(cfg,e/(role+'_gt.json'),cfg['src_train_split'] if role=='source_train' else cfg['src_val_split'],False,False)
        by_video={}
        for index,vid in enumerate(examples['video_ids']):by_video.setdefault(vid,[]).append(index)
        out={k:[] for k in ['variants','delta','labels','quality','valid','video_ids']};selection={};visited=set()
        with torch.no_grad():
            for index in range(len(ds)):
                v=move_to_device([ds[index]],'cuda:0')[0];vid=v['metadata']['video_id']
                if vid not in by_video:continue
                visited.add(vid);indices=torch.tensor(by_video[vid]);indices=indices[torch.randperm(len(indices),generator=generator)[:32]];selection[vid]=indices.tolist()
                boxes=examples['boxes'][indices].cuda();truth=examples['gt'][indices].cuda()
                match,which=iou(truth,v['segments']).max(1)
                if not (match>.9999).all():raise ValueError('源样例GT与当前源标注坐标不匹配')
                labels=v['labels'][which];f,m=detector_features(model,v);levels=duration_levels(boxes,model.reg_range)
                grid,delta,x,valid=extract(f,m,boxes,levels)
                valid &= (grid[:,:,0]>=0)&(grid[:,:,1]<=v['feats'].shape[-1]);keep=valid[:,4]
                q=aligned_iou(grid.reshape(-1,2),truth[:,None].expand(-1,9,-1).reshape(-1,2)).reshape(-1,9)
                for name,value in [('variants',x),('delta',delta),('labels',labels),('quality',q),('valid',valid)]:out[name].append(value[keep].cpu())
                out['video_ids'].extend([vid]*int(keep.sum()));print('ENERGY_SOURCE_GROUPS',role,index,len(ds),int(keep.sum()),flush=True)
        assert visited==set(by_video)
        for name in ['variants','delta','labels','quality','valid']:out[name]=torch.cat(out[name])
        if not len(out['labels']):raise ValueError('没有有效联合源候选组')
        torch.save(out,e/(role+'_groups.pt'));save(e/(role+'_group_selection.json'),dict(indices=selection,groups=len(out['labels']),maximum_per_video=32,real_fpn_reextracted=True))
        return out
    train=source_data('source_train');val=source_data('source_val');assert set(train['video_ids']).isdisjoint(val['video_ids'])
    head=IntervalEnergy().cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001)
    counts={v:val['video_ids'].count(v) for v in set(val['video_ids'])};weights=torch.tensor([1/counts[v] for v in val['video_ids']],device='cuda:0')
    def batch(data,idx):
        x,d,labels,q,valid=[data[k][idx].cuda() for k in ['variants','delta','labels','quality','valid']]
        return x[:,4],x,d,labels,q,valid
    def validate():
        head.eval();losses=[];selected=[];before=[];oracle=[];moved=[];entropies=[]
        with torch.no_grad():
            for idx in torch.arange(len(val['labels'])).split(64):
                a,x,d,labels,q,valid=batch(val,idx);loss,energy=objective(head,a,x,d,labels,q,valid);choice=select_interval(energy,valid)
                losses.append(loss);selected.append(q.gather(1,choice[:,None]).squeeze(1));before.append(q[:,4]);oracle.append(q.masked_fill(~valid,-1).max(1).values);moved.append((choice!=4).float())
                p=(-energy).masked_fill(~valid,-1e9).softmax(1);entropies.append(-(p*p.clamp_min(1e-9).log()).sum(1))
        avg=lambda xs:float((torch.cat(xs)*weights).sum()/weights.sum())
        return dict(loss=avg(losses),mean_iou_before=avg(before),mean_iou_selected=avg(selected),mean_iou_grid_oracle=avg(oracle),moved_fraction=avg(moved),entropy=avg(entropies),source_validation_equal_video=True,not_independent_generalization_evidence=True)
    with torch.no_grad():
        a,x,d,labels,q,valid=batch(val,torch.arange(min(64,len(val['labels']))));assert select_interval(head(a,x,d,labels),valid).eq(4).all()
    report=validate();best=report['loss'];best_epoch=0;history=[dict(epoch=0,source_validation=best)]
    torch.save(dict(state_dict=head.state_dict(),epoch=0,specification=SPEC),e/'best_energy.pt');accepted=False
    for epoch in range(1,51):
        head.train()
        for idx in torch.randperm(len(train['labels']),generator=generator).split(64):
            a,x,d,labels,q,valid=batch(train,idx);before=base.tensor_digest(head.state_dict()) if not accepted else None
            loss,_=objective(head,a,x,d,labels,q,valid);loss=loss.mean();opt.zero_grad();loss.backward()
            grad=sum(float(p.grad.square().sum()) for p in head.parameters() if p.grad is not None)**.5
            if not torch.isfinite(loss) or not grad>0:raise ValueError('联合能量损失/梯度无效')
            opt.step()
            if not accepted:
                assert before!=base.tensor_digest(head.state_dict());save(e/'acceptance.json',dict(status='passed',source_loss=float(loss),gradient_norm=grad,branch_parameters_updated=True,initialization_exact_identity=True,joint_candidate_features_real=True,target_gt_used=False,guard=guard,guard_sha256=sha(guard)));accepted=True
        report=validate();history.append(dict(epoch=epoch,source_validation=report['loss']))
        if report['loss']<best:
            best=report['loss'];best_epoch=epoch;torch.save(dict(state_dict=head.state_dict(),epoch=epoch,specification=SPEC),e/'best_energy.pt')
        torch.save(dict(state_dict=head.state_dict(),optimizer=opt.state_dict(),history=history,generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0)),e/'latest_energy.pt')
        save(e/'energy_history.json',history);print('INTERVAL_ENERGY_EPOCH',epoch,report['loss'],report['mean_iou_selected'],report['mean_iou_before'],flush=True)
        if epoch-best_epoch>=15:break
    head.load_state_dict(torch.load(e/'best_energy.pt',map_location='cpu')['state_dict'],strict=True);head.eval();save(e/'source_mechanism.json',validate());del train,val

    @torch.no_grad()
    def forward_raw(v):
        f,m=detector_features(model,v);logits,_=model.cls_head(f,m);offsets,_=model.reg_head(f,m);points=model.point_generator(f);parts=[];levels=[]
        for level in range(len(f)):
            p=model.inference_single_video([points[level]],[m[level][0,0]],[logits[level][0].T],[offsets[level][0].T],[None],[None]);parts.append(p);levels.append(torch.full((len(p['scores']),),level,device='cuda:0',dtype=torch.long))
        raw={k:torch.cat([p[k] for p in parts]) for k in ['segments','scores','labels']};raw.update({k:v['metadata'][k] for k in ['video_id','fps','duration','feat_stride','feat_num_frames','use_temporal_grid_feats','video_length']})
        return f,m,raw,model.postprocessing([deepcopy(raw)])[0],torch.cat(levels)
    for domain in ['source','target']:
        ds=base.dataset(cfg,e/(domain+'_val_unlabeled.json'),cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'],False,True)
        columns={mode:{k:[] for k in ['video-id','t-start','t-end','label','score']} for mode in ['control','refined']};observations=[]
        with torch.no_grad():
            for index in range(len(ds)):
                v=move_to_device([ds[index]],'cuda:0')[0];assert v['segments'].numel()==0 and v['labels'].numel()==0
                f,m,raw,pred,levels=forward_raw(v)
                if index==0:
                    original=model(video_list_src=[v],video_list_tgt=[v],mode='test',domain=domain,curr_epoch=0);assert_same_prediction(pred,original[0][0])
                boxes=raw['segments'].clone();ids=raw['scores'].argsort(descending=True)[:200];moved=0
                for chosen in ids.split(32):
                    if not len(chosen):continue
                    grid,d,x,valid=extract(f,m,boxes[chosen],levels[chosen]);valid &= (grid[:,:,0]>=0)&(grid[:,:,1]<=v['feats'].shape[-1])
                    energy=head(x[:,4],x,d,raw['labels'][chosen]);pick=select_interval(energy,valid);boxes[chosen]=grid[torch.arange(len(chosen),device='cuda:0'),pick];moved+=int((pick!=4).sum())
                raw['segments']=boxes;new=model.postprocessing([raw])[0];observations.append(dict(video=v['metadata']['video_id'],raw_candidates=len(boxes),searched=len(ids),moved=moved))
                for mode,p in [('control',pred),('refined',new)]:
                    c=columns[mode];c['video-id'].extend([v['metadata']['video_id']]*len(p['scores']))
                    for k,z in [('t-start',p['segments'][:,0]),('t-end',p['segments'][:,1]),('label',p['labels']),('score',p['scores'])]:c[k].extend(z.cpu().tolist())
                print('ENERGY_INFERENCE',domain,index,len(ds),moved,flush=True)
        for mode,c in columns.items():np.savez(e/(domain+'_'+mode+'.npz'),**{k:np.asarray(v) for k,v in c.items()})
        save(e/(domain+'_search_observations.json'),observations)
    assert initial==base.tensor_digest(model.state_dict())
    save(e/'gpu_complete.json',dict(status='passed',frozen_detector_unchanged=True,best_epoch=best_epoch,stop_epoch=epoch,guard=guard,guard_sha256=sha(guard),code=code_identity(),target_gt_used=False))


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
    save(args.output/'plan.json',dict(schema=SCHEMA,code=identity,specification=SPEC,route=4,revision=0,
        maximum_revisions=2,gpus=[0,1,2,3],reference=str(REFERENCE),target_parameter_search=False))
    prepare(args.output,REFERENCE,INVENTORY,identity)
    for shift in SHIFTS:
        e=args.output/'formal'/shift;cfg=read(e/'config.json')
        db=read(cfg['baseline']['annotation_for_offline_evaluation'])['database']
        for role,splits in [('source_train',cfg['src_train_split']),('source_val',cfg['src_val_split'])]:
            wanted={s.lower() for s in splits};save(e/(role+'_gt.json'),dict(database={k:v for k,v in db.items() if v['subset'].lower() in wanted}),sort_keys=False)
    if args.previous:raise ValueError('路线四无部分恢复入口，禁止重送计划')
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
                    from tools.uncertainty_interval_energy import audit_entry
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
