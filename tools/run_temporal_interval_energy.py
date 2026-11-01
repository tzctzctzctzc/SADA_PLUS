"""四卡时序实例能量：完整源课程→真实联合训练→三路完整推理/CPU评估。

新增训练是从已有成熟Prototype出发的额外适配，绝不称cold等预算复现。
目标GT只在独立CPU evaluate阶段读取；所有GPU推理清单无annotations。
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.temporal_interval_energy import SPEC,SCHEMA
REFERENCE=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029')
SHIFTS=tuple('S%d'%i for i in range(1,7))


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
    from tools.temporal_interval_energy import TemporalIntervalEnergy,interval_iou,video_groups,score_intervals,energy_loss,search,install_joint
    e=args.entry;cfg=read(e/'config.json');bound=read(e/'inputs.json');identity=code_identity();assert identity==read(args.output/'plan.json')['code']
    os.chdir(args.data_root);fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    ref=Path(bound['reference']);assert sha(ref/'best.pt')==bound['checkpoint_sha256']
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda()
    saved=torch.load(ref/'best.pt',map_location='cpu');model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval().requires_grad_(False)
    initial=base.tensor_digest(model.state_dict());del saved
    generator=torch.Generator().manual_seed(123)

    def forward_raw(v,model):
        f,m=detector_features(model,v);logits,_=model.cls_head(f,m);offsets,_=model.reg_head(f,m);points=model.point_generator(f);parts=[];levels=[]
        for l in range(len(f)):
            p=model.inference_single_video([points[l]],[m[l][0,0]],[logits[l][0].T],[offsets[l][0].T],[None],[None]);parts.append(p);levels.append(torch.full((len(p['scores']),),l,device='cuda:0',dtype=torch.long))
        raw={k:torch.cat([p[k] for p in parts]) for k in ['segments','scores','labels']};raw.update({k:v['metadata'][k] for k in ['video_id','fps','duration','feat_stride','feat_num_frames','use_temporal_grid_feats','video_length']})
        return f,m,raw,torch.cat(levels)

    def cache(role,limit=None):
        path=e/(role+'_gt.json');splits=cfg['src_train_split'] if role=='source_train' else cfg['src_val_split'];ds=base.dataset(cfg,path,splits,False,False)
        output=e/'source_cache'/role;output.mkdir(parents=True);index=[];counts=dict(gt_instances=0,real_proposals=0,matched_real=0,unmatched_real=0,ambiguous_real=0)
        with torch.no_grad():
            for i in range(len(ds) if limit is None else min(limit,len(ds))):
                v=move_to_device([ds[i]],'cuda:0')[0];f,m,raw,lv=forward_raw(v,model);gt=v['segments'];classes=v['labels'];length=v['feats'].shape[-1]
                # 两种确定性anchor覆盖每个源GT，无按视频32组截断。
                anchors=[];truths=[];labels=[];instances=[];levels=[]
                for j,t in enumerate(gt):
                    d=(t[1]-t[0]).clamp_min(1)
                    for off in [t.new_tensor([-.2,0.]),t.new_tensor([0.,.2])]:
                        a=(t+d*off).clamp(0,length)
                        anchors.append(a);truths.append(t);labels.append(classes[j]);instances.append(j);levels.append(int(duration_levels(a[None],model.reg_range)[0]))
                counts['gt_instances']+=len(gt)
                for cls in raw['labels'].unique().tolist():
                    ids=(raw['labels']==cls).nonzero(as_tuple=True)[0];ids=ids[raw['scores'][ids].argsort(descending=True)[:20]]
                    for idx in ids:
                        a=raw['segments'][idx].clamp(0,length)
                        if a[1]-a[0]<1:continue
                        same=(classes==cls).nonzero(as_tuple=True)[0];instance=-1
                        if len(same):
                            overlaps=interval_iou(a[None],gt[same]);best,which=overlaps.max(0)
                            if best>=.1:
                                tied=int((overlaps>=best-.05).sum())>1
                                if not tied:instance=int(same[which])
                                else:counts['ambiguous_real']+=1
                        anchors.append(a);truths.append(gt[instance] if instance>=0 else a);labels.append(raw['labels'][idx]);instances.append(instance);levels.append(int(lv[idx]));counts['real_proposals']+=1;counts['matched_real' if instance>=0 else 'unmatched_real']+=1
                if not anchors:continue
                record=dict(features=[x.cpu() for x in f],masks=[x.cpu() for x in m],segments=gt.cpu(),gt_labels=classes.cpu(),anchors=torch.stack(anchors).cpu(),truths=torch.stack(truths).cpu(),classes=torch.stack(labels).cpu(),instances=torch.tensor(instances),levels=torch.tensor(levels),length=length,video_id=v['metadata']['video_id'])
                dest=output/(v['metadata']['video_id']+'.pt');torch.save(record,dest);index.append(dict(path=str(dest),sha256=sha(dest),video_id=record['video_id'],anchors=len(anchors),gt_instances=len(gt)))
                print('TEMPORAL_ENERGY_CACHE',role,i,len(ds),len(anchors),flush=True)
        assert index;save(e/(role+'_cache_index.json'),dict(videos=index,counts=counts,all_source_GT_covered=limit is None,diagnostic_limit=limit));return index

    train=cache('source_train',2 if args.stage=='probe' else None);val=cache('source_val',1 if args.stage=='probe' else None)
    assert set(x['video_id'] for x in train).isdisjoint(x['video_id'] for x in val)
    head=TemporalIntervalEnergy().cuda();opt=torch.optim.AdamW(head.parameters(),lr=SPEC['source_lr'])

    def parts(record,ids):
        f=[x.cuda() for x in record['features']];m=[x.cuda() for x in record['masks']]
        a,t,lv,cl,inst=[record[k][ids].cuda() for k in ['anchors','truths','levels','classes','instances']]
        gt=record['segments'].cuda();labels=record['gt_labels'].cuda()
        boxes,q,padding,kinds=video_groups(f,m,a,t,lv,cl,inst,gt,labels,model.fpn_strides,record['length'])
        score,accept,valid=score_intervals(head,f,m,a,boxes,lv,cl,model.fpn_strides)
        valid &= padding;loss=energy_loss(score,accept,q,valid,inst>=0)
        chosen=score.masked_fill(~valid,-torch.inf).argmax(1);selected=q.gather(1,chosen[:,None]).squeeze(1)
        return loss,selected,(inst>=0),q.masked_fill(~valid,-1).max(1).values,kinds

    def validate():
        head.eval();losses=[];ious=[];oracles=[]
        with torch.no_grad():
            for row in val:
                data=torch.load(row['path'],map_location='cpu');vl=[];vi=[];vo=[]
                for ids in torch.arange(len(data['anchors'])).split(SPEC['source_group_batch']):
                    loss,selected,matched,oracle,_=parts(data,ids);vl.extend(loss.tolist());vi.extend(selected[matched].tolist());vo.extend(oracle[matched].tolist())
                losses.append(float(np.mean(vl)))
                if vi:ious.append(float(np.mean(vi)));oracles.append(float(np.mean(vo)))
        return dict(loss=float(np.mean(losses)),selected_iou=float(np.mean(ious)),candidate_oracle_iou=float(np.mean(oracles)),population='source-val curriculum; includes GT proposals, not actual inference AP')

    report=validate();history=[dict(epoch=0,source_validation=report['loss'],**{k:v for k,v in report.items() if k!='loss'})];best=report['loss'];best_epoch=0
    def source_checkpoint(epoch):
        return dict(state_dict=head.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=deepcopy(history),specification=SPEC,generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),code=identity)
    torch.save(source_checkpoint(0),e/'best_energy.pt');accepted=False
    epochs=2 if args.stage=='probe' else SPEC['source_epochs']
    for epoch in range(1,epochs+1):
        head.train();running=[];kind_counts={}
        for i in torch.randperm(len(train),generator=generator).tolist():
            data=torch.load(train[i]['path'],map_location='cpu');order=torch.randperm(len(data['anchors']),generator=generator)
            if args.stage=='probe':order=order[:8]
            for ids in order.split(SPEC['source_group_batch']):
                before=base.tensor_digest(head.state_dict()) if not accepted else None
                loss,_,_,_,kinds=parts(data,ids);loss=loss.mean();opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.)
                assert torch.isfinite(loss) and torch.isfinite(norm);opt.step();running.append(float(loss))
                for k,count in zip(*kinds.unique(return_counts=True)):kind_counts[str(int(k))]=kind_counts.get(str(int(k)),0)+int(count)
                if not accepted:
                    assert norm>0 and before!=base.tensor_digest(head.state_dict());save(e/'acceptance.json',dict(status='passed',loss=float(loss),head_gradient_norm=float(norm),parameters_updated=True,temporal_token_count=40,candidate_types_seen=kind_counts,target_gt_used=False,guard=guard,guard_sha256=sha(guard)));accepted=True
        report=validate();history.append(dict(epoch=epoch,source_validation=report['loss'],train_loss=float(np.mean(running)),candidate_kind_counts=kind_counts,**{k:v for k,v in report.items() if k!='loss'}))
        if report['loss']<best:best=report['loss'];best_epoch=epoch;torch.save(source_checkpoint(epoch),e/'best_energy.pt')
        torch.save(source_checkpoint(epoch),e/'latest_energy.pt');save(e/'energy_history.json',history);print('TEMPORAL_ENERGY_EPOCH',epoch,json.dumps(history[-1]),flush=True)
        if args.stage!='probe' and epoch-best_epoch>=SPEC['patience']:break
    source_choice='latest_energy.pt' if args.stage=='probe' else 'best_energy.pt'
    head.load_state_dict(torch.load(e/source_choice,map_location='cpu')['state_dict'],strict=True);frozen_head=deepcopy(head.state_dict())
    save(e/'source_termination.json',dict(best_epoch=best_epoch,stop_epoch=epoch,history=history,diagnostic=args.stage=='probe'))
    assert base.tensor_digest(model.state_dict())==initial
    del model;torch.cuda.empty_cache()

    # 第二阶段是真实原检测器联合学习，完整source/target loader和原优化阶段。
    ctx=base.build(cfg);saved=torch.load(ref/'best.pt',map_location='cpu');ctx.model.load_state_dict(saved['ema_state_dict'],strict=True);ctx.model_ema.module.load_state_dict(saved['ema_state_dict'],strict=True)
    for network in [ctx.model,ctx.model_ema.module]:
        for name,value in saved.get('ema_controls',{}).items():
            if torch.is_tensor(value):getattr(network,name).copy_(value.cuda())
            elif isinstance(value,list):
                for current,item in zip(getattr(network,name),value):
                    if torch.is_tensor(current):current.copy_(item.to(current.device))
            else:setattr(network,name,deepcopy(value))
    del saved;install_joint(ctx,head,e/'joint_source_events.jsonl')
    joint=e/'joint';joint.mkdir();ctx.valid_metrics_src.reset()
    validation_loader=base.StepLoader(ctx.val_loader_src,1) if args.stage=='probe' else ctx.val_loader_src
    initial_loss=float(ctx._valid_epoch(0,validation_loader,domain='src'))
    best=initial_loss;joint_history=[[0,initial_loss]];joint_rows=[]
    torch.save(base.capture(ctx,cfg,0,joint_history),joint/'best.pt')
    joint_epochs=[1,5] if args.stage=='probe' else range(1,SPEC['joint_epochs']+1)
    for epoch in joint_epochs:
        base.epoch_setup(ctx,epoch);before=base.tensor_digest(ctx.model.backbone_source.state_dict())
        original_loaders=ctx.train_loader_src,ctx.train_loader_tgt
        # 原warmup首step的lr为0；至少两步才能验证优化器实际更新。
        if args.stage=='probe':ctx.train_loader_src,ctx.train_loader_tgt=[base.StepLoader(x,2) for x in original_loaders]
        ctx._train_epoch(epoch);ctx.train_loader_src,ctx.train_loader_tgt=original_loaders;metrics=ctx.train_metrics.result()
        assert before!=base.tensor_digest(ctx.model.backbone_source.state_dict())
        joint_rows.append(dict(epoch=epoch,source_cls=float(metrics['t_cls_loss']),source_reg=float(metrics['t_reg_loss']),energy_loss=float(metrics['t_temporal_energy_loss']),groups=float(metrics['t_temporal_energy_groups']),target_gt_cls_loss=0.,target_gt_reg_loss=0.,target_gt_downstream_loss=0.))
        if epoch==1:
            assert ctx.temporal_energy_state['gradient_checks'];save(e/'joint_acceptance.json',dict(status='passed',actual_source_fpn_gradient=ctx.temporal_energy_state['gradient_checks'],backbone_updated=True,energy_in_original_optimizers=True,target_gt_used=False,guard=guard,guard_sha256=sha(guard)))
        if epoch%5==0 or args.stage=='probe':
            ctx.valid_metrics_src.reset();vl=base.StepLoader(ctx.val_loader_src,1) if args.stage=='probe' else ctx.val_loader_src
            value=float(ctx._valid_epoch(epoch,vl,domain='src'));joint_history.append([epoch,value]);state=base.capture(ctx,cfg,epoch,joint_history);state['temporal_energy_state']=deepcopy(ctx.temporal_energy_state)
            if value<best:best=value;torch.save(state,joint/'best.pt')
            torch.save(state,joint/'latest.pt')
        save(joint/'metrics.json',joint_rows);save(joint/'source_history.json',joint_history);print('TEMPORAL_ENERGY_JOINT',epoch,joint_rows[-1],flush=True)
        if args.stage!='probe' and epoch%5==0:
            from tools.temporal_interval_energy import joint_decision
            stop=joint_decision(joint_history)
            if stop['reason']!='running':save(joint/'termination.json',stop);break
    if args.stage=='probe':
        import io
        state=base.capture(ctx,cfg,epoch,joint_history);buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0);expected=torch.load(buffer,map_location='cpu');base.restore(ctx,expected);actual=base.capture(ctx,cfg,epoch,joint_history);assert base.tensor_digest(actual)==base.tensor_digest(expected)
        # 真正边界梯度与实际多起点搜索，不将仅head更新当完整机制验收。
        sample=move_to_device([next(iter(ctx.val_loader_src))[0]],'cuda:0')[0];ctx.model.eval()
        f,m,raw,levels=forward_raw(sample,ctx.model);ids=raw['scores'].argsort(descending=True)[:2];head=ctx.model.temporal_energy.eval()
        boxes=raw['segments'][ids].detach();new,stats=search(head,f,m,boxes,levels[ids],raw['labels'][ids],ctx.model.fpn_strides,sample['feats'].shape[-1]);assert torch.isfinite(new).all()
        save(e/'probe_complete.json',dict(status='passed',code=identity,full_state_roundtrip=True,actual_joint_fpn_gradient=ctx.temporal_energy_state['gradient_checks'],source_backbone_updated=True,source_curriculum_all_GT=False,diagnostic_only=True,joint_probe_initial_head='latest diagnostic head, formal uses source-selected best',search=stats,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,target_gt_used=False,guard=guard,guard_sha256=sha(guard)));return
    # 三条完整推理路径：原参照 / 只训能量 / 联合训练，原类别分数和NMS均保留。
    best_joint=torch.load(joint/'best.pt',map_location='cpu');ctx.model.load_state_dict(best_joint['ema_state_dict'],strict=True);ctx.model.eval().requires_grad_(False);joint_model=ctx.model
    original=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda();original.load_state_dict(torch.load(ref/'best.pt',map_location='cpu')['ema_state_dict'],strict=True);original.eval().requires_grad_(False)
    frozen=TemporalIntervalEnergy().cuda();frozen.load_state_dict(frozen_head,strict=True);frozen.eval().requires_grad_(False)
    for domain in ['source','target']:
        ds=base.dataset(cfg,e/(domain+'_val_unlabeled.json'),cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'],False,True)
        columns={mode:{k:[] for k in ['video-id','t-start','t-end','label','score']} for mode in ['control','frozen','joint']};observations=[]
        for i in range(len(ds)):
            v=move_to_device([ds[i]],'cuda:0')[0];assert not v['segments'].numel() and not v['labels'].numel()
            for net,modes in [(original,['control','frozen']),(joint_model,['joint'])]:
                with torch.no_grad():f,m,raw,levels=forward_raw(v,net)
                for mode in modes:
                    candidate=deepcopy(raw)
                    if mode!='control':
                        energy=frozen if mode=='frozen' else joint_model.temporal_energy
                        for ids in torch.arange(len(raw['scores']),device='cuda:0').split(16):
                            b,stats=search(energy,f,m,raw['segments'][ids],levels[ids],raw['labels'][ids],net.fpn_strides,v['feats'].shape[-1]);candidate['segments'][ids]=b;observations.append(dict(video=v['metadata']['video_id'],mode=mode,**stats))
                    with torch.no_grad():pred=net.postprocessing([candidate])[0]
                    if mode=='control' and i==0:
                        expected=net(video_list_src=[v],video_list_tgt=[v],mode='test',domain=domain,curr_epoch=0)[0 if domain=='source' else 1][0];assert_same_prediction(pred,expected)
                    col=columns[mode];col['video-id'].extend([v['metadata']['video_id']]*len(pred['scores']))
                    for key,value in [('t-start',pred['segments'][:,0]),('t-end',pred['segments'][:,1]),('label',pred['labels']),('score',pred['scores'])]:col[key].extend(value.detach().cpu().tolist())
            print('TEMPORAL_ENERGY_INFERENCE',domain,i,len(ds),flush=True)
        for mode,col in columns.items():np.savez(e/(domain+'_'+mode+'.npz'),**{k:np.asarray(v) for k,v in col.items()})
        save(e/(domain+'_search_observations.json'),observations)
    save(e/'gpu_complete.json',dict(status='passed',code=identity,source_best=best_epoch,joint_best=best_joint['epoch'],target_gt_used=False,all_raw_candidates_searched=True,guard=guard,guard_sha256=sha(guard)))


def evaluate(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    from libs.evaluation import ANETdetection
    os.chdir(args.data_root);e=args.entry;cfg=read(e/'config.json');annotation=cfg['baseline']['annotation_for_offline_evaluation'];assert sha(annotation)==cfg['baseline']['annotation_sha256'];results={}
    for domain in ['source','target']:
        results[domain]={}
        for mode in ['control','frozen','joint']:
            p=e/(domain+'_'+mode+'.npz')
            with np.load(p,allow_pickle=False) as z:pred={k:z[k] for k in z.files}
            evaluator=ANETdetection(annotation,cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'],num_workers=2,action_subsampling='10_majority_classes');ap,mean,*_=evaluator.evaluate(pred,verbose=True)
            save(e/'evaluation'/(domain+'_'+mode+'.json'),dict(map=float(mean),map_by_tiou=ap.tolist(),prediction_sha256=sha(p)));results[domain][mode]=float(mean)
        reference=read(Path(read(e/'inputs.json')['reference'])/'evaluation'/(domain+'_val.json'))['map_0.1_to_0.5'];assert abs(results[domain]['control']-reference)<1e-7
    artifacts={str(p):sha(p) for p in e.rglob('*') if p.is_file() and p.suffix in ['.json','.pt','.npz'] and p.name!='complete.json'}
    save(e/'complete.json',dict(schema=SCHEMA,status='passed',shift=e.name,specification=SPEC,results=results,artifacts=artifacts,code=code_identity()))


def main(args):
    from tools.run_prototype_target_cache import prepare,INVENTORY
    from tools.run_sada_baseline import acquire
    identity=code_identity();args.output.mkdir(parents=True,exist_ok=False);save(args.output/'plan.json',dict(schema=SCHEMA,specification=SPEC,code=identity,method='full-source-curriculum-temporal-energy-with-joint-adaptation',extra_training_budget=True))
    prepare(args.output,REFERENCE,INVENTORY,identity)
    for s in SHIFTS:
        e=args.output/'formal'/s;cfg=read(e/'config.json');db=read(cfg['dataset']['json_file'])['database']
        for role,split in [('source_train',cfg['src_train_split']),('source_val',cfg['src_val_split'])]:save(e/(role+'_gt.json'),dict(database={k:v for k,v in db.items() if v['subset'] in split}),sort_keys=False)
    pending=list(SHIFTS);active={};evals=[];complete=[];probe_pending=list(SHIFTS);probe_reports={};accepted=False;minimum=12288
    def spawn(s,stage,card=None):
        e=args.output/('probes' if stage=='probe' else 'formal')/s
        if stage=='probe':
            e.mkdir(parents=True)
            for n in ['config.json','inputs.json','source_train_gt.json','source_val_gt.json']:save(e/n,read(args.output/'formal'/s/n))
        env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        if card:env.update(CUDA_VISIBLE_DEVICES=card['uuid'],SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
        with (e/(stage+'.log')).open('x') as log:p=subprocess.Popen([sys.executable,'-u','-B',__file__,'--output',str(args.output),'--entry',str(e),'--stage',stage],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return dict(p=p,stage=stage,shift=s,gpu=card['index'] if card else None)
    try:
        while pending or active or evals:
            for key,t in list(active.items()):
                rc=t['p'].poll()
                if rc is None:continue
                if t.get('ctx'):t['ctx'].__exit__(None,None,None)
                del active[key]
                if rc:raise RuntimeError('阶段失败 '+key+' exit='+str(rc))
                if t['stage']=='probe':
                    path=args.output/'probes'/t['shift']/'probe_complete.json';report=read(path);assert report['status']=='passed';minimum=max(minimum,int(report['peak_reserved_mib'])+4096);probe_reports[t['shift']]=dict(path=str(path),sha256=sha(path));accepted=len(probe_reports)==6
                    if accepted:save(args.output/'acceptance.json',dict(code=identity,reports=probe_reports,minimum_free_mib=minimum,specification=SPEC))
                elif t['stage']=='gpu':evals.append(t['shift'])
                else:
                    e=args.output/'formal'/t['shift'];from tools.temporal_interval_energy_audit import audit_entry
                    complete.append(audit_entry(e))
            for probe_shift in list(probe_pending):
                slot=acquire([0,1,2,3],args.data_root,args.output,minimum,identity['commit'])
                if slot:
                    ctx,card=slot;t=spawn(probe_shift,'probe',card);t['ctx']=ctx;active[probe_shift+'/probe']=t;probe_pending.remove(probe_shift)
                else:break
            if accepted:
                for s in list(pending):
                    slot=acquire([0,1,2,3],args.data_root,args.output,minimum,identity['commit'])
                    if not slot:break
                    ctx,card=slot;t=spawn(s,'gpu',card);t['ctx']=ctx;active[s+'/gpu']=t;pending.remove(s)
            while evals and sum(t['stage']=='evaluate' for t in active.values())<2:
                s=evals.pop(0);active[s+'/evaluate']=spawn(s,'evaluate')
            save(args.output/'state.json',dict(pid=os.getpid(),phase='formal' if accepted else 'real_gpu_acceptance',pending=pending,completed=complete,active={k:{f:v for f,v in t.items() if f not in ['p','ctx']}|{'pid':t['p'].pid} for k,t in active.items()}));time.sleep(3)
        assert len(complete)==6;save(args.output/'final_audit.json',dict(status='passed',count=6,results=complete));save(args.output/'state.json',dict(phase='complete',active={},completed=complete))
    except BaseException as err:save(args.output/'error.json',dict(error=str(err)));raise
    finally:
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        for t in active.values():
            try:t['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['p'].pid,signal.SIGKILL);t['p'].wait()
            if t.get('ctx'):t['ctx'].__exit__(None,None,None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--entry',type=Path);parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));parser.add_argument('--stage',choices=['probe','gpu','evaluate']);args=parser.parse_args();args.output=args.output.resolve()
    def stop(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.stage in ['probe','gpu']:gpu(args)
    elif args.stage=='evaluate':evaluate(args)
    else:main(args)
