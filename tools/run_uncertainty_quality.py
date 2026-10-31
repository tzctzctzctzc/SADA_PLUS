"""路线五：四卡公共锁，源监督质量头和完整NMS前重评分评估。"""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.uncertainty_quality import SCHEMA,SPEC
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
    from tools.uncertainty_refiner import context
    from tools.uncertainty_quality import QualityHead,objective,rescore,evaluated_intervals,same_class_quality
    e=args.entry;cfg=read(e/'config.json');bound=read(e/'inputs.json')
    assert code_identity()==read(args.output/'plan.json')['code']
    ref=Path(bound['reference']);assert sha(ref/'best.pt')==bound['checkpoint_sha256']
    os.chdir(args.data_root);fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda()
    saved=torch.load(ref/'best.pt',map_location='cpu');model.load_state_dict(saved['ema_state_dict'],strict=True);del saved
    model.eval().requires_grad_(False);initial=base.tensor_digest(model.state_dict());generator=torch.Generator().manual_seed(123)
    @torch.no_grad()
    def forward_raw(v):
        f,m=detector_features(model,v);logits,_=model.cls_head(f,m);offsets,_=model.reg_head(f,m);points=model.point_generator(f);parts=[];levels=[]
        for level in range(len(f)):
            p=model.inference_single_video([points[level]],[m[level][0,0]],[logits[level][0].T],[offsets[level][0].T],[None],[None]);parts.append(p);levels.append(torch.full((len(p['scores']),),level,device='cuda:0',dtype=torch.long))
        raw={k:torch.cat([p[k] for p in parts]) for k in ['segments','scores','labels']};raw.update({k:v['metadata'][k] for k in ['video_id','fps','duration','feat_stride','feat_num_frames','use_temporal_grid_feats','video_length']})
        return f,m,raw,torch.cat(levels)
    def widths(boxes,levels):
        return (boxes[:,1]-boxes[:,0])/boxes.new_tensor(model.fpn_strides)[levels]
    def source_data(role):
        ds=base.dataset(cfg,e/(role+'_gt.json'),cfg['src_train_split'] if role=='source_train' else cfg['src_val_split'],False,False)
        keys=['x','labels','levels','scores','width','complete','quality'];out={k:[] for k in keys};out['video_ids']=[];inventory=[]
        with torch.no_grad():
            for index in range(len(ds)):
                v=move_to_device([ds[index]],'cuda:0')[0];f,m,raw,levels=forward_raw(v);chosen=[]
                for label in range(10):
                    ids=(raw['labels']==label).nonzero(as_tuple=True)[0];order=ids[raw['scores'][ids].argsort(descending=True)]
                    chosen.append(order[:20]);rest=order[20:]
                    if len(rest):chosen.append(rest[torch.randperm(len(rest),generator=generator)[:20].to(rest.device)])
                ids=torch.cat(chosen)
                if not len(ids):continue
                boxes=raw['segments'][ids];lev=levels[ids];x,complete=context(f,m,boxes,lev,model.fpn_strides)
                # 质量监督按实际评估的秒坐标和视频裁边匹配；其他类别重叠不能算正确。
                q=same_class_quality(evaluated_intervals(boxes,v['metadata']),raw['labels'][ids],
                                     evaluated_intervals(v['segments'],v['metadata']),v['labels'])
                values=dict(x=x,labels=raw['labels'][ids],levels=lev,scores=raw['scores'][ids],width=widths(boxes,lev),complete=complete,quality=q)
                for k in keys:out[k].append(values[k].cpu())
                out['video_ids'].extend([v['metadata']['video_id']]*len(ids));inventory.append(dict(video=v['metadata']['video_id'],raw=len(levels),selected=len(ids),indices=ids.cpu().tolist(),context_complete_fraction=float(complete.float().mean())))
                print('SOURCE_QUALITY_PROPOSALS',role,index,len(ds),len(ids),flush=True)
        for k in keys:out[k]=torch.cat(out[k])
        torch.save(out,e/(role+'_quality_examples.pt'));save(e/(role+'_proposal_selection.json'),inventory);return out
    train=source_data('source_train');val=source_data('source_val');assert set(train['video_ids']).isdisjoint(val['video_ids'])
    head=QualityHead().cuda();assert not {p.data_ptr() for p in head.parameters()} & {p.data_ptr() for p in model.parameters()}
    opt=torch.optim.AdamW(head.parameters(),lr=.001);counts={v:val['video_ids'].count(v) for v in set(val['video_ids'])};weights=torch.tensor([1/counts[v] for v in val['video_ids']],device='cuda:0')
    def batch(data,idx):return [data[k][idx].cuda() for k in ['x','labels','levels','scores','width','complete','quality']]
    def validate():
        head.eval();losses=[];briers=[];summaries=[];truth=[];confidence=[]
        with torch.no_grad():
            for ids in torch.arange(len(val['labels'])).split(256):
                args0=batch(val,ids);loss,logits,target=objective(head,*args0);p=logits.sigmoid()
                assert (p[:,1:]<=p[:,:-1]).all()
                losses.append(loss);briers.append((p-target).square().mean(1));summaries.append(p.mean(1).cpu());truth.append(args0[-1].cpu());confidence.append(args0[3].cpu())
        avg=lambda xs:float((torch.cat(xs)*weights).sum()/weights.sum())
        q=torch.cat(truth);summary=torch.cat(summaries);score=torch.cat(confidence);matched=q>=.1
        def corr(a,b):
            if len(a)<2 or float(a.std())==0 or float(b.std())==0:return None
            value=float(np.corrcoef(a.numpy(),b.numpy())[0,1]);return value if np.isfinite(value) else None
        return dict(loss=avg(losses),brier=avg(briers),quality_iou_correlation=corr(summary,q),
                    matched_quality_iou_correlation=corr(summary[matched],q[matched]),matched_original_score_iou_correlation=corr(score[matched],q[matched]),
                    source_validation_equal_video=True,matched_proposals=int(matched.sum()),samples=len(q),independent_target_calibration_established=False)
    best=float('inf');best_epoch=0;history=[];accepted=False
    for epoch in range(1,51):
        head.train()
        for ids in torch.randperm(len(train['labels']),generator=generator).split(256):
            args0=batch(train,ids);before=base.tensor_digest(head.state_dict()) if not accepted else None
            loss,logits,_=objective(head,*args0);loss=loss.mean();opt.zero_grad();loss.backward();grad=sum(float(p.grad.square().sum()) for p in head.parameters() if p.grad is not None)**.5
            if not torch.isfinite(loss) or not math.isfinite(grad):raise ValueError('源质量头损失/梯度非有限')
            opt.step()
            if not accepted and grad>0:
                assert before!=base.tensor_digest(head.state_dict());prob=logits.sigmoid();assert (prob[:,1:]<=prob[:,:-1]).all()
                save(e/'acceptance.json',dict(status='passed',source_loss=float(loss),gradient_norm=grad,branch_parameters_updated=True,
                    monotone_probabilities=True,target_gt_used=False,guard=guard,guard_sha256=sha(guard)));accepted=True
        if not accepted:raise ValueError('质量头无有效源监督更新')
        report=validate();history.append(dict(epoch=epoch,source_validation=report['loss'],source_brier=report['brier']))
        if report['loss']<best:
            best=report['loss'];best_epoch=epoch;torch.save(dict(state_dict=head.state_dict(),epoch=epoch,specification=SPEC),e/'best_quality.pt')
        torch.save(dict(state_dict=head.state_dict(),optimizer=opt.state_dict(),history=history,generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0)),e/'latest_quality.pt')
        save(e/'quality_history.json',history);print('LOCALIZATION_QUALITY_EPOCH',epoch,report['loss'],report['brier'],flush=True)
        if epoch-best_epoch>=15:break
    head.load_state_dict(torch.load(e/'best_quality.pt',map_location='cpu')['state_dict'],strict=True);head.eval();save(e/'source_mechanism.json',validate());del train,val
    for domain in ['source','target']:
        ds=base.dataset(cfg,e/(domain+'_val_unlabeled.json'),cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'],False,True)
        columns={mode:{k:[] for k in ['video-id','t-start','t-end','label','score']} for mode in ['control','refined']};observations=[]
        with torch.no_grad():
            for index in range(len(ds)):
                v=move_to_device([ds[index]],'cuda:0')[0];assert v['segments'].numel()==0 and v['labels'].numel()==0
                f,m,raw,levels=forward_raw(v);pred=model.postprocessing([deepcopy(raw)])[0]
                if index==0:
                    original=model(video_list_src=[v],video_list_tgt=[v],mode='test',domain=domain,curr_epoch=0);assert_same_prediction(pred,original[0][0])
                boxes=raw['segments'].clone();labels=raw['labels'].clone();scores=raw['scores'].clone();new_scores=scores.clone();complete_count=0
                for ids in torch.arange(len(scores),device='cuda:0').split(128):
                    if not len(ids):continue
                    x,complete=context(f,m,boxes[ids],levels[ids],model.fpn_strides);complete_count+=int(complete.sum())
                    logits=head(x,labels[ids],levels[ids],scores[ids],widths(boxes[ids],levels[ids]),complete);new_scores[ids]=rescore(scores[ids],logits.sigmoid())
                assert torch.isfinite(new_scores).all() and torch.equal(raw['segments'],boxes) and torch.equal(raw['labels'],labels)
                inversions=0;comparisons=0
                for label in range(10):
                    ids=(labels==label).nonzero(as_tuple=True)[0];ids=ids[::max(1,len(ids)//128)][:128]
                    a=scores[ids,None]-scores[None,ids];b=new_scores[ids,None]-new_scores[None,ids]
                    inversions+=int(((a*b)<0).sum());comparisons+=int((a!=0).sum())
                raw['scores']=new_scores;new=model.postprocessing([raw])[0]
                observations.append(dict(video=v['metadata']['video_id'],raw_candidates=len(scores),context_complete=complete_count,
                    within_class_rank_inversions=inversions,comparable_pairs=comparisons,raw_boundaries_unchanged=True,raw_labels_unchanged=True))
                for mode,p in [('control',pred),('refined',new)]:
                    c=columns[mode];c['video-id'].extend([v['metadata']['video_id']]*len(p['scores']))
                    for k,z in [('t-start',p['segments'][:,0]),('t-end',p['segments'][:,1]),('label',p['labels']),('score',p['scores'])]:c[k].extend(z.cpu().tolist())
                print('QUALITY_RESCORING_INFERENCE',domain,index,len(ds),inversions,flush=True)
        for mode,c in columns.items():np.savez(e/(domain+'_'+mode+'.npz'),**{k:np.asarray(v) for k,v in c.items()})
        save(e/(domain+'_quality_observations.json'),observations)
    assert initial==base.tensor_digest(model.state_dict())
    save(e/'gpu_complete.json',dict(status='passed',frozen_detector_unchanged=True,raw_boundaries_unchanged=True,raw_labels_unchanged=True,
        best_epoch=best_epoch,stop_epoch=epoch,guard=guard,guard_sha256=sha(guard),code=code_identity(),target_gt_used=False))


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
    save(args.output/'plan.json',dict(schema=SCHEMA,code=identity,specification=SPEC,route=5,revision=0,
        maximum_revisions=2,gpus=[0,1,2,3],reference=str(REFERENCE),target_parameter_search=False))
    prepare(args.output,REFERENCE,INVENTORY,identity)
    for shift in SHIFTS:
        e=args.output/'formal'/shift;cfg=read(e/'config.json')
        db=read(cfg['baseline']['annotation_for_offline_evaluation'])['database']
        for role,splits in [('source_train',cfg['src_train_split']),('source_val',cfg['src_val_split'])]:
            wanted={s.lower() for s in splits};save(e/(role+'_gt.json'),dict(database={k:v for k,v in db.items() if v['subset'].lower() in wanted}),sort_keys=False)
    if args.previous:raise ValueError('路线五无部分恢复入口，禁止重送计划')
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
                    from tools.uncertainty_quality import audit_entry
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
