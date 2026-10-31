"""路线三：四卡公共锁，两级源监督分布精修和完整离线评估。"""
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
from tools.uncertainty_distribution import SCHEMA,SPEC
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
    from tools.uncertainty_refiner import context,aligned_iou
    from tools.uncertainty_distribution import DistributionStage,objective,support_from_source,relative_target
    e=args.entry;cfg=read(e/'config.json');bound=read(e/'inputs.json')
    assert code_identity()==read(args.output/'plan.json')['code']
    reference=Path(bound['reference']);assert sha(reference/'best.pt')==bound['checkpoint_sha256']
    os.chdir(args.data_root);fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda()
    saved=torch.load(reference/'best.pt',map_location='cpu');model.load_state_dict(saved['ema_state_dict'],strict=True);del saved
    model.eval().requires_grad_(False);initial=base.tensor_digest(model.state_dict())
    generator=torch.Generator().manual_seed(123)
    original=SOURCE_EXAMPLES/e.name;completed=read(original/'complete.json');refs={}
    assert read(original/'config.json')==cfg
    for name in ['source_train_examples.pt','source_val_examples.pt','source_train_gt.json','source_val_gt.json']:
        p=original/name;assert sha(p)==completed['artifacts'][str(p)];refs[str(p)]=sha(p)
    for role in ['source_train','source_val']:assert sha(e/(role+'_gt.json'))==refs[str(original/(role+'_gt.json'))]
    save(e/'source_examples_reference.json',dict(previous_complete=str(original/'complete.json'),previous_complete_sha256=sha(original/'complete.json'),artifacts=refs,read_only=True))
    train=torch.load(original/'source_train_examples.pt',map_location='cpu');val=torch.load(original/'source_val_examples.pt',map_location='cpu')
    assert set(train['video_ids']).isdisjoint(val['video_ids'])
    radius=support_from_source(train['boxes'],train['gt'])
    save(e/'stage1_source_support.json',dict(radius=radius,derived_from='source_train_only',quantile=.95,
        coverage=float((relative_target(train['boxes'],train['gt']).abs()<=radius).all(1).float().mean())))
    heads=[];stage_reports=[]

    def train_stage(stage,training,validation,radius):
        head=DistributionStage(radius=radius).cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001)
        ids=validation['video_ids'];counts={v:ids.count(v) for v in set(ids)}
        weights=torch.tensor([1/counts[v] for v in ids],device='cuda:0')
        def validate():
            losses=[];before=[];after=[];variances=[];errors=[]
            head.eval()
            with torch.no_grad():
                for start in range(0,len(ids),256):
                    x,b,g=[validation[k][start:start+256].cuda() for k in ['x','boxes','gt']]
                    loss,p,var=objective(head,x,b,g);losses.append(loss);before.append(aligned_iou(b,g));after.append(aligned_iou(p,g))
                    variances.append(var);errors.append(relative_target(p,g).abs())
            avg=lambda xs:float((torch.cat(xs)*weights).sum()/weights.sum())
            return dict(loss=avg(losses),mean_iou_before=avg(before),mean_iou_after=avg(after),
                        variance=torch.cat(variances).cpu(),error=torch.cat(errors).cpu())
        with torch.no_grad():
            b=validation['boxes'][:256].cuda();p,_,_=head(validation['x'][:256].cuda(),b)
            assert torch.equal(p,b)
        report=validate();best=report['loss'];best_epoch=0;history=[dict(epoch=0,source_validation=best)]
        torch.save(dict(state_dict=head.state_dict(),epoch=0,specification=SPEC),e/('stage%d_best.pt'%stage))
        accepted=False
        for epoch in range(1,51):
            head.train()
            for indices in torch.randperm(len(training['boxes']),generator=generator).split(256):
                x,b,g=[training[k][indices].cuda() for k in ['x','boxes','gt']]
                before=base.tensor_digest(head.state_dict()) if not accepted else None
                loss,_,_=objective(head,x,b,g);loss=loss.mean();opt.zero_grad();loss.backward()
                grad=sum(float(p.grad.square().sum()) for p in head.parameters() if p.grad is not None)**.5
                if not torch.isfinite(loss) or not grad>0:raise ValueError('分布分支损失/梯度无效')
                opt.step()
                if not accepted:
                    assert before!=base.tensor_digest(head.state_dict())
                    save(e/('stage%d_acceptance.json'%stage),dict(status='passed',source_loss=float(loss),gradient_norm=grad,
                        branch_parameters_updated=True,initialization_exact_identity=True,genuine_updated_box_resampling=stage==2,
                        target_gt_used=False,guard=guard,guard_sha256=sha(guard)));accepted=True
            report=validate();history.append(dict(epoch=epoch,source_validation=report['loss']))
            if report['loss']<best:
                best=report['loss'];best_epoch=epoch
                torch.save(dict(state_dict=head.state_dict(),epoch=epoch,specification=SPEC),e/('stage%d_best.pt'%stage))
            torch.save(dict(state_dict=head.state_dict(),optimizer=opt.state_dict(),history=history,generator=generator.get_state(),
                            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0)),e/('stage%d_latest.pt'%stage))
            save(e/('stage%d_history.json'%stage),history);print('DISTRIBUTION_EPOCH',stage,epoch,report['loss'],flush=True)
            if epoch-best_epoch>=15:break
        head.load_state_dict(torch.load(e/('stage%d_best.pt'%stage),map_location='cpu')['state_dict'],strict=True);head.eval()
        report=validate();var=report.pop('variance');err=report.pop('error')
        correlation=float(np.corrcoef(var.flatten().numpy(),err.flatten().numpy())[0,1]) if float(var.std())>0 and float(err.std())>0 else None
        if correlation is not None and not np.isfinite(correlation):correlation=None
        report.update(stage=stage,best_epoch=best_epoch,stop_epoch=epoch,radius=radius,
                      source_validation_support_coverage=float((relative_target(validation['boxes'],validation['gt']).abs()<=radius).all(1).float().mean()),
                      variance_error_correlation=correlation,
                      equal_video_source_validation=True)
        return head,report

    head,report=train_stage(1,train,val,radius);heads.append(head);stage_reports.append(report)

    def resample(role,examples):
        ds=base.dataset(cfg,e/(role+'_gt.json'),cfg['src_train_split'] if role=='source_train' else cfg['src_val_split'],False,False)
        by_video={}
        for i,vid in enumerate(examples['video_ids']):by_video.setdefault(vid,[]).append(i)
        out={k:[] for k in ['x','boxes','gt','video_ids']};moved=0;changed=0;invalid=0;visited=set()
        with torch.no_grad():
            for index in range(len(ds)):
                v=move_to_device([ds[index]],'cuda:0')[0];vid=v['metadata']['video_id']
                if vid not in by_video:continue
                visited.add(vid);selected=torch.tensor(by_video[vid]);oldboxes=examples['boxes'][selected].cuda()
                x=examples['x'][selected];proposed=torch.cat([heads[0](z.cuda(),b)[0] for z,b in zip(x.split(256),oldboxes.split(256))])
                f,m=detector_features(model,v);levels=duration_levels(oldboxes,model.reg_range)
                fresh,valid=context(f,m,proposed,levels,model.fpn_strides)
                valid &= (proposed[:,0]>=0)&(proposed[:,1]<=v['feats'].shape[-1])
                moved+=int((proposed!=oldboxes).any(1).sum());changed+=int(((fresh-examples['x'][selected].cuda()).abs().max(1).values>1e-6).sum());invalid+=int((~valid).sum())
                out['x'].append(fresh[valid].cpu());out['boxes'].append(proposed[valid].cpu());out['gt'].append(examples['gt'][selected][valid.cpu()])
                out['video_ids'].extend([vid]*int(valid.sum()));print('RESAMPLE_STAGE2',role,index,len(ds),int(valid.sum()),flush=True)
        assert visited==set(by_video)
        for key in ['x','boxes','gt']:out[key]=torch.cat(out[key])
        if not len(out['boxes']):raise ValueError('第二级无有效源样本')
        torch.save(out,e/(role+'_stage2_examples.pt'))
        save(e/(role+'_resampling.json'),dict(updated_box_count=moved,changed_feature_count=changed,invalid_count=invalid,kept=len(out['boxes']),recomputed_from_real_fpn=True,same_original_level=True))
        return out
    train2=resample('source_train',train);val2=resample('source_val',val);del train,val
    radius2=support_from_source(train2['boxes'],train2['gt'],maximum=radius/2)
    save(e/'stage2_source_support.json',dict(radius=radius2,derived_from='stage1_best_source_train_outputs_only',quantile=.95,
        maximum=radius/2,coverage=float((relative_target(train2['boxes'],train2['gt']).abs()<=radius2).all(1).float().mean())))
    head,report=train_stage(2,train2,val2,radius2);heads.append(head);stage_reports.append(report);del train2,val2
    save(e/'source_mechanism.json',dict(stages=stage_reports,source_only=True,stage2_resampled=True,target_generalization_not_established=True))

    @torch.no_grad()
    def forward_raw(v):
        f,m=detector_features(model,v);logits,_=model.cls_head(f,m);offsets,_=model.reg_head(f,m);points=model.point_generator(f)
        parts=[];levels=[]
        for level in range(len(f)):
            p=model.inference_single_video([points[level]],[m[level][0,0]],[logits[level][0].T],[offsets[level][0].T],[None],[None])
            parts.append(p);levels.append(torch.full((len(p['scores']),),level,device='cuda:0',dtype=torch.long))
        raw={k:torch.cat([p[k] for p in parts]) for k in ['segments','scores','labels']}
        raw.update({k:v['metadata'][k] for k in ['video_id','fps','duration','feat_stride','feat_num_frames','use_temporal_grid_feats','video_length']})
        return f,m,raw,model.postprocessing([deepcopy(raw)])[0],torch.cat(levels)
    for domain in ['source','target']:
        ds=base.dataset(cfg,e/(domain+'_val_unlabeled.json'),cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'],False,True)
        columns={mode:{k:[] for k in ['video-id','t-start','t-end','label','score']} for mode in ['control','refined']}
        with torch.no_grad():
            for index in range(len(ds)):
                v=move_to_device([ds[index]],'cuda:0')[0];assert v['segments'].numel()==0 and v['labels'].numel()==0
                f,m,raw,pred,levels=forward_raw(v)
                if index==0:
                    original=model(video_list_src=[v],video_list_tgt=[v],mode='test',domain=domain,curr_epoch=0);assert_same_prediction(pred,original[0][0])
                boxes=raw['segments'].clone()
                for head in heads:
                    updated=boxes.clone()
                    for start in range(0,len(boxes),128):
                        b=boxes[start:start+128];x,valid=context(f,m,b,levels[start:start+128],model.fpn_strides)
                        p,_,_=head(x,b);updated[start:start+128]=torch.where(valid[:,None],p,b)
                    boxes=updated
                raw['segments']=boxes;new=model.postprocessing([raw])[0]
                for mode,p in [('control',pred),('refined',new)]:
                    c=columns[mode];c['video-id'].extend([v['metadata']['video_id']]*len(p['scores']))
                    for k,z in [('t-start',p['segments'][:,0]),('t-end',p['segments'][:,1]),('label',p['labels']),('score',p['scores'])]:c[k].extend(z.cpu().tolist())
                print('DISTRIBUTION_INFERENCE',domain,index,len(ds),flush=True)
        for mode,c in columns.items():np.savez(e/(domain+'_'+mode+'.npz'),**{k:np.asarray(v) for k,v in c.items()})
    assert initial==base.tensor_digest(model.state_dict())
    save(e/'gpu_complete.json',dict(status='passed',frozen_detector_unchanged=True,stages=stage_reports,
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
    save(args.output/'plan.json',dict(schema=SCHEMA,code=identity,specification=SPEC,route=3,revision=0,
        maximum_revisions=2,gpus=[0,1,2,3],reference=str(REFERENCE),target_parameter_search=False))
    prepare(args.output,REFERENCE,INVENTORY,identity)
    for shift in SHIFTS:
        e=args.output/'formal'/shift;cfg=read(e/'config.json')
        db=read(cfg['baseline']['annotation_for_offline_evaluation'])['database']
        for role,splits in [('source_train',cfg['src_train_split']),('source_val',cfg['src_val_split'])]:
            wanted={s.lower() for s in splits};save(e/(role+'_gt.json'),dict(database={k:v for k,v in db.items() if v['subset'].lower() in wanted}),sort_keys=False)
    if args.previous:raise ValueError('路线三无部分恢复入口，禁止重送计划')
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
                    from tools.uncertainty_distribution import audit_entry
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
