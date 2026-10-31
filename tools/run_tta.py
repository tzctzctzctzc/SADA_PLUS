"""等待Prototype精确六方向终审，再按冻结顺序执行离线TTA；不修改训练工件。"""
import argparse
from copy import deepcopy
import hashlib
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
from tools.run_sada_baseline import acquire,prepare_file_limit
from tools.tta_core import SPEC,VARIANTS

SHIFTS=tuple('S%d'%i for i in range(1,7))
ACTIVE={}


def selection(candidates):
    if len(candidates)!=4 or any(len(x['scores'])!=6 for x in candidates):
        raise ValueError('源校准必须覆盖四种设置和全部六方向')
    expected={(lr,steps) for lr in SPEC['learning_rates'] for steps in SPEC['steps']}
    if ({(x['lr'],x['steps']) for x in candidates}!=expected or
            any(not 0<=v<=1 for x in candidates for v in x['scores'])):
        raise ValueError('源校准配置不唯一或mAP非法')
    # 先固定此规则，目标结果从不进入选择函数。
    return max(candidates,key=lambda x:(sum(x['scores'])/6,-x['steps'],-x['lr']))


def rows_to_columns(items):
    import numpy as np
    out={k:[] for k in ('video-id','t-start','t-end','label','score')}
    for item in items:
        count=len(item['scores']);out['video-id'].extend([item['video_id']]*count)
        out['t-start'].extend([x[0] for x in item['segments']]);out['t-end'].extend([x[1] for x in item['segments']])
        out['label'].extend(item['labels']);out['score'].extend(item['scores'])
    return {k:np.asarray(v) for k,v in out.items()}


def evaluate_predictions(items, annotation, splits):
    from libs.evaluation import ANETdetection
    evaluator=ANETdetection(str(annotation),splits,num_workers=2,action_subsampling='10_majority_classes')
    ap,score,*_=evaluator.evaluate(rows_to_columns(items),verbose=False)
    return dict(map=float(score),map_by_tiou=[float(x) for x in ap])


def load_model(cfg, checkpoint):
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    fix_random_seed(cfg['seed'],include_cuda=True);torch.cuda.manual_seed(cfg['seed'])
    saved=torch.load(checkpoint,map_location='cpu')
    if saved['config']!=cfg:raise ValueError('TTA必须使用checkpoint的原完整配置')
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').to('cuda:0')
    model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval()
    del saved
    return model


def source_bank(model,cfg,output):
    import torch
    from torch.nn import functional as F
    from tools.baseline_engine import dataset
    from libs.models.SADA.prototype_capacity import source_gt_proposals
    from tools.tta_core import dense,unlabeled
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],False)
    dimension=model.rq3_capacity.branch.classifier.in_features
    sums=torch.zeros(model.num_classes,dimension,device='cuda:0');counts=torch.zeros(model.num_classes,device='cuda:0')
    with torch.no_grad():
        for i in range(len(ds)):
            video=ds[i]
            proposals=source_gt_proposals([video],model.num_classes,len(model.fpn_strides),model.reg_range)
            if not len(proposals[0]['segments']):continue
            data=dense(model,unlabeled(video));proposal=proposals[0]
            for start in range(0,len(proposal['segments']),64):
                chunk={k:v[start:start+64].to('cuda:0') for k,v in proposal.items()}
                encoded=model.rq3_capacity.branch.encode(data['fpn'],[chunk],model.fpn_strides)['embedding']
                labels=chunk['targets'].argmax(-1);valid=chunk['targets'].sum(-1)>0
                sums.index_add_(0,labels[valid],encoded[valid]);counts.index_add_(0,labels[valid],torch.ones_like(labels[valid],dtype=torch.float))
            del data
    ready=counts>0
    if not ready[:10].all():raise ValueError('源训练参考缺少任务类别')
    bank=dict(centers=F.normalize(sums,dim=-1).cpu(),ready=ready.cpu(),counts=counts.cpu())
    torch.save(bank,output/'bank.pt')
    return dict(status='passed',source_train_videos=len(ds),classes=int(ready.sum()),
        source_manifest_sha256=sha(cfg['dataset']['json_file']),bank_sha256=sha(output/'bank.pt'))


def diagnostic(model,teacher,initial,cfg,job,bank):
    import torch
    from tools.baseline_engine import dataset,tensor_digest
    from tools.tta_core import dense,predict_full,unlabeled,adapt_video,adaptable_parameters,parameter_digest
    ds=dataset(cfg,job['metadata_manifest'],cfg['src_val_split'],False,True)
    expected=parameter_digest(adaptable_parameters(model));results=[];effective=False
    for i in range(min(4,len(ds))):
        video=unlabeled(ds[i])
        with torch.no_grad():
            official=model([video],[video],mode='test')[1][0]
            current=predict_full(model,video)
        for key in ('segments','scores','labels'):
            if not torch.equal(official[key],current[key]):
                raise ValueError('独立target路径与原正式推理不逐值相同: '+key)
        frozen_before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
        _,proof=adapt_video(model,teacher,initial,video,'prototype',1e-4,1,bank)
        effective |= proof['changed']
        if proof['initial_parameter_sha256']!=expected:raise ValueError('逐视频重置不精确')
        for n,p in model.named_parameters():
            if n in frozen_before and not torch.equal(p.detach().cpu(),frozen_before[n]):
                raise ValueError('冻结的头或辅助分支被更新')
        # 同一视频重置后，更新轨迹与输出都应一致，不依赖上一视频。
        first,one=adapt_video(model,teacher,initial,video,'prototype',1e-4,1,bank)
        second,two=adapt_video(model,teacher,initial,video,'prototype',1e-4,1,bank)
        if one!=two or any(not torch.equal(first[k],second[k]) for k in ('segments','scores','labels')):
            raise ValueError('逐视频重置后不能重现TTA结果')
        results.append(proof)
        if effective:break
    if not effective:raise ValueError('四个固定源验证视频中无真实TTA更新，需诊断，不放行正式评估')
    return dict(status='passed',exact_original_inference=True,episodic_reset=True,frozen_parameters=True,
        target_gt_seen=False,source_diagnostic=True,videos=results)


def worker(args):
    import torch
    from tools.baseline_gpu import first_cuda_guard
    from tools.baseline_engine import dataset
    from tools.tta_core import adaptable_parameters,adapt_video,unlabeled
    job=read(args.job);output=Path(job['output']);cfg=read(job['config'])
    if code_identity()!=job['code']:raise ValueError('执行代码与计划身份不符')
    if sha(job['config'])!=job['config_sha256']:raise ValueError('原训练配置改变')
    if job['specification']!=json.loads(json.dumps(SPEC)):raise ValueError('TTA冻结协议改变')
    if sha(job['checkpoint'])!=job['checkpoint_sha256']:raise ValueError('训练checkpoint发生变化')
    if sha(job['metadata_manifest'])!=job['metadata_sha256']:raise ValueError('无标签视频成员改变')
    if job['kind']=='target':
        frozen=Path(job['selection_file'])
        selected=read(frozen)
        if (sha(frozen)!=job['selection_sha256'] or selected['target_results_seen'] or
                job['lr']!=selected['chosen']['lr'] or job['steps']!=selected['chosen']['steps']):
            raise ValueError('必须先冻结纯源域选择，才能读取测试视频')
    # 父持物理锁，首次seed/model CUDA前再次核UUID/显存。
    guard=first_cuda_guard();os.chdir(args.data_root)
    model=load_model(cfg,job['checkpoint'])
    if job['kind']=='bank':
        result=source_bank(model,cfg,output)
    else:
        teacher=deepcopy(model).eval()
        for p in teacher.parameters():p.requires_grad_(False)
        named=adaptable_parameters(model);initial={n:p.detach().clone() for n,p in named.items()}
        bank=None
        if job.get('bank'):
            if sha(job['bank'])!=job['bank_sha256']:raise ValueError('源参考原型改变')
            bank={k:v.to('cuda:0') for k,v in torch.load(job['bank'],map_location='cpu').items()}
        if job['kind']=='diagnostic':
            result=diagnostic(model,teacher,initial,cfg,job,bank)
        else:
            splits=cfg['src_val_split'] if job['kind']=='calibration' else cfg['tgt_val_split']
            ds=dataset(cfg,job['metadata_manifest'],splits,False,True)
            videos=[];proofs=[];updated=0
            for i in range(len(ds)):
                video=unlabeled(ds[i])
                prediction,proof=adapt_video(model,teacher,initial,video,job['variant'],job['lr'],job['steps'],bank)
                item=dict(video_id=prediction['video_id'],segments=prediction['segments'].tolist(),
                    labels=prediction['labels'].tolist(),scores=prediction['scores'].tolist())
                videos.append(item);proofs.append(proof);updated+=int(proof['changed'])
                print('VIDEO',i+1,len(ds),item['video_id'],'updated',proof['changed'],flush=True)
            save(output/'predictions.json',videos);save(output/'adaptation_proofs.json',proofs)
            # 所有预测已写盘后，评估器才读取对应GT；目标GT不回传也不参与校准。
            result=evaluate_predictions(videos,job['evaluation_annotation'],splits)
            result.update(status='passed',videos=len(ds),updated_videos=updated,
                target_gt_seen_during_adaptation=False,prediction_sha256=sha(output/'predictions.json'),
                adaptation_proofs_sha256=sha(output/'adaptation_proofs.json'),peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    if 'map' in result and not 0<=result['map']<=1:raise ValueError('TTA mAP非有限或越界')
    save(output/'complete.json',dict(status='passed',job_sha256=sha(args.job),code=job['code'],
        guard=guard,guard_sha256=sha(guard),checkpoint_sha256=job['checkpoint_sha256'],result=result))


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    raise SystemExit(128+signum)


def audit_job(path):
    job=read(path);out=Path(job['output']);record=read(out/'complete.json')
    if record['status']!='passed' or record['job_sha256']!=sha(path) or record['code']!=job['code']:
        raise ValueError('TTA工件身份或SHA不符')
    if (sha(job['checkpoint'])!=record['checkpoint_sha256'] or sha(record['guard'])!=record['guard_sha256']
            or sha(job['config'])!=job['config_sha256']):
        raise ValueError('checkpoint或真实GPU凭证改变')
    guard=read(record['guard']);lease=read(guard['lease'])
    if sha(guard['lease'])!=guard['lease_sha256'] or lease['commit']!=job['code']['commit']:
        raise ValueError('GPU lease代码身份不符')
    if (guard['observed']['uuid']!=lease['observed']['uuid'] or
            guard['observed']['free_mib']<lease['minimum_free_mib']):
        raise ValueError('TTA首CUDA显存/UUID凭证不符')
    if job['kind']=='target' and sha(job['selection_file'])!=job['selection_sha256']:
        raise ValueError('正式TTA源选择改变')
    result=record['result']
    for artifact,field in [('predictions.json','prediction_sha256'),('adaptation_proofs.json','adaptation_proofs_sha256'),('bank.pt','bank_sha256')]:
        if field in result and sha(out/artifact)!=result[field]:raise ValueError('TTA结果SHA改变')
    if (out/'adaptation_proofs.json').exists():
        proofs=read(out/'adaptation_proofs.json')
        if any(x['target_gt_seen'] for x in proofs):raise ValueError('TTA使用目标GT')
        initials={x['initial_parameter_sha256'] for x in proofs}
        if len(initials)>1:raise ValueError('视频之间没有恢复同一个初始化')
    return dict(job=str(path),shift=job['shift'],variant=job.get('variant'),complete_sha256=sha(out/'complete.json'),**result)


def run_jobs(root,jobs,data,identity,phase):
    queue=list(jobs);completed=[]
    while queue or ACTIVE:
        for key,task in list(ACTIVE.items()):
            process=task['process']
            if process.poll() is None:continue
            task['context'].__exit__(None,None,None);del ACTIVE[key]
            if process.returncode:raise RuntimeError('TTA任务失败，保留工件: '+key)
            completed.append(audit_job(task['job']))
        for job_path in list(queue):
            job=read(job_path);out=Path(job['output'])
            if (out/'complete.json').exists():
                completed.append(audit_job(job_path));queue.remove(job_path);continue
            found=acquire([0,1],data,root,12288,identity['commit'])
            if not found:break
            context,lease=found
            env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
                TMPDIR=str(root/'tmp'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (out/'run.log').open('x') as stream:
                try:
                    process=subprocess.Popen([sys.executable,'-B',str(CODE/'tools/run_tta.py'),'--job',str(job_path),
                        '--data-root',str(data)],cwd=CODE,env=env,stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:context.__exit__(None,None,None);raise
            key=str(out.relative_to(root));ACTIVE[key]=dict(process=process,context=context,job=job_path,gpu=lease['index'])
            queue.remove(job_path)
        save(root/'state.json',dict(phase=phase,pid=os.getpid(),active={k:dict(pid=t['process'].pid,gpu=t['gpu'])
            for k,t in ACTIVE.items()},phase_completed=len(completed),phase_total=len(jobs)))
        if queue or ACTIVE:time.sleep(3)
    return completed


def write_job(root,relative,kind,shift,prototype,identity,metadata,annotation,**extra):
    entry=prototype/'formal'/shift/'prototype';out=root/relative;out.mkdir(parents=True,exist_ok=False)
    save(out/'job.json',dict(kind=kind,shift=shift,config=str(entry/'config.json'),config_sha256=sha(entry/'config.json'),checkpoint=str(entry/'best.pt'),
        checkpoint_sha256=sha(entry/'best.pt'),prototype_complete_sha256=sha(entry/'complete.json'),
        output=str(out),code=identity,specification=SPEC,metadata_manifest=str(metadata),metadata_sha256=sha(metadata),
        evaluation_annotation=str(annotation),**extra))
    return out/'job.json'


def main(args):
    root=args.output.resolve();prototype=args.prototype.resolve();data=args.data_root.resolve()
    identity=code_identity();root.mkdir(parents=True,exist_ok=False);(root/'tmp').mkdir()
    save(root/'runtime_resources.json',prepare_file_limit())
    save(root/'plan.json',dict(code=identity,specification=SPEC,prototype_root=str(prototype),
        target_evaluation_jobs=24,original_predictions='reuse-six-existing-complete-entries',
        calibration='four-settings-fixed-eight-source-validation-videos-per-shift-global-source-mean',
        target_gt_for_selection=False))
    try:
        while not (prototype/'final_audit.json').exists():
            old=read(prototype/'state.json')
            if old['phase'] in ('failed','stopped'):raise RuntimeError('Prototype未完成且已停止；TTA不接续')
            save(root/'state.json',dict(phase='waiting_prototype_final_audit',pid=os.getpid(),active={},prototype_complete=len(old['completed'])))
            time.sleep(30)
        final=read(prototype/'final_audit.json')
        if final['status']!='passed' or final['count']!=6 or {x['shift'] for x in final['results']}!=set(SHIFTS):
            raise ValueError('Prototype不是精确六方向终审')
        from tools.run_prototype_source_map import audit
        references={s:audit(prototype/'formal'/s/'prototype') for s in SHIFTS}
        save(root/'predecessor.json',dict(prototype_final_sha256=sha(prototype/'final_audit.json'),entries=references))
        manifests={}
        for s in SHIFTS:
            cfg=read(prototype/'formal'/s/'prototype/config.json')
            source=read(cfg['dataset']['json_file'])['database']
            source_ids=[v for v in source if source[v]['subset'].lower() in cfg['src_val_split']]
            source_ids=sorted(source_ids,key=lambda v:hashlib.sha256(('123:'+v).encode()).hexdigest())[:8]
            source_meta=root/'inputs'/s/'source_unlabeled.json';source_gt=root/'inputs'/s/'source_evaluation.json'
            target_meta=root/'inputs'/s/'target_unlabeled.json'
            save(source_gt,dict(database={v:source[v] for v in source_ids}))
            save(source_meta,dict(database={v:{k:x for k,x in source[v].items() if k!='annotations'} for v in source_ids}))
            raw=read(cfg['baseline']['annotation_for_offline_evaluation'])['database']
            save(target_meta,dict(database={v:{k:x for k,x in row.items() if k!='annotations'} for v,row in raw.items()
                if row['subset'].lower() in cfg['tgt_val_split']}))
            manifests[s]=dict(source_meta=source_meta,source_gt=source_gt,target_meta=target_meta,
                target_gt=cfg['baseline']['annotation_for_offline_evaluation'])
        banks=[write_job(root,'banks/'+s,'bank',s,prototype,identity,manifests[s]['source_meta'],manifests[s]['source_gt']) for s in SHIFTS]
        # 先用S1参考验收真实更新，避免未验收就铺开其余方向的GPU准备。
        run_jobs(root,banks[:1],data,identity,'source_reference_bank_S1')
        diag=write_job(root,'diagnostic/S1','diagnostic','S1',prototype,identity,manifests['S1']['source_meta'],manifests['S1']['source_gt'],
            bank=str(root/'banks/S1/bank.pt'),bank_sha256=sha(root/'banks/S1/bank.pt'))
        run_jobs(root,[diag],data,identity,'real_gpu_diagnostic')
        run_jobs(root,banks[1:],data,identity,'source_reference_banks_remaining')
        candidates=[]
        for lr in SPEC['learning_rates']:
            for steps in SPEC['steps']:
                trial='lr%s_steps%d'%(lr,steps)
                jobs=[write_job(root,'calibration/'+trial+'/'+s,'calibration',s,prototype,identity,
                    manifests[s]['source_meta'],manifests[s]['source_gt'],variant='self_train',lr=lr,steps=steps) for s in SHIFTS]
                results=run_jobs(root,jobs,data,identity,'source_calibration_'+trial)
                candidates.append(dict(lr=lr,steps=steps,scores=[r['map'] for r in results],audits=results))
        chosen=selection(candidates)
        frozen=root/'frozen_source_selection.json'
        save(frozen,dict(status='passed',selection='maximum-six-direction-source-validation-mean',
            chosen=chosen,candidates=candidates,target_results_seen=False,specification=SPEC,code=identity))
        all_results={}
        for variant in VARIANTS:
            jobs=[]
            for s in SHIFTS:
                options=dict(variant=variant,lr=chosen['lr'],steps=chosen['steps'],selection_file=str(frozen),selection_sha256=sha(frozen))
                if variant in ('prototype','uncertainty'):
                    options.update(bank=str(root/'banks'/s/'bank.pt'),bank_sha256=sha(root/'banks'/s/'bank.pt'))
                jobs.append(write_job(root,'formal/'+variant+'/'+s,'target',s,prototype,identity,
                    manifests[s]['target_meta'],manifests[s]['target_gt'],**options))
            results=run_jobs(root,jobs,data,identity,'target_'+variant)
            if len(results)!=6 or {r['shift'] for r in results}!=set(SHIFTS):
                raise ValueError('TTA方法未完成精确六方向')
            all_results[variant]=results
            save(root/'formal'/variant/'final_audit.json',dict(status='passed',count=6,results=results,
                mean_map=sum(x['map'] for x in results)/6,source_selection_sha256=sha(frozen)))
        save(root/'final_audit.json',dict(status='passed',count=24,results=all_results,original_prototype=references))
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),active={},count=24))
    except BaseException as error:
        save(root/'error.json',dict(type=type(error).__name__,message=str(error)))
        save(root/'state.json',dict(phase='stopped' if isinstance(error,SystemExit) else 'failed',pid=os.getpid(),active={}))
        raise
    finally:
        for task in ACTIVE.values():
            if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
        for task in ACTIVE.values():
            try:task['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(task['process'].pid,signal.SIGKILL);task['process'].wait()
            task['context'].__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prototype',type=Path);p.add_argument('--output',type=Path);p.add_argument('--job',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    args=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.job:worker(args)
    elif args.prototype and args.output:main(args)
    else:p.error('需要--prototype和--output，或--job')
