"""S1双教师源域留出验证；冻结特征局部探针，禁止当成正式UDA-TAL结果。"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import torch

from tools.rq4_artifacts import audit_source_chain, read, reference
from tools.rq4_protocol import build_cache, digest, file_sha, select_candidates, interval_iou
from tools.rq4_source_acceptance import save
from tools.rq4_temporal_probe_core import (
    ARMS, POLICY, phase_distribution, paired_distribution, real_indices, local_patches,
    check_training_payload, train_arms, score_predictions, score_routes,
    TemporalProbe, supervised_loss, source_bank, alignment_loss)

CODE=Path(__file__).resolve().parents[1]
VARIANT='rq4-source-heldout-temporal-prototype-probe-v1'


def code_identity():
    return {name:file_sha(CODE/'tools'/name) for name in
            ('rq4_temporal_probe_core.py','run_rq4_temporal_probe.py')}


def clean_head(expected):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
    if head!=expected or dirty:
        raise ValueError('audited clean worktree required')
    return head


def membership(cfg, database):
    result={key:sorted(vid for vid,row in database.items()
            if row['subset'].lower() in {s.lower() for s in cfg[split]})
            for key,split in (('train','src_train_split'),('heldout','src_val_split'),
                             ('target','tgt_train_split'))}
    if any(set(result[a])&set(result[b]) for a,b in
           (('train','heldout'),('train','target'),('heldout','target'))):
        raise ValueError('source/heldout/target membership overlap')
    return result


def teacher_pairs(request_path, folder, annotations, teacher):
    evidence=audit_source_chain(request_path,folder,annotations,teacher)
    manifest=read(request_path)
    replies=read(Path(folder)/'responses.json')
    identity=dict(proposal_checkpoint_sha256=manifest['checkpoint_sha256'],
        teacher_weights_sha256=digest(read(Path(folder)/'identity.json')['weights']),
        processor_sha256=read(Path(folder)/'identity.json')['processor_sha256'],
        input_manifest_sha256=file_sha(request_path),class_names_sha256=replies[0]['class_names_sha256'])
    cache=build_cache([r['candidate'] for r in manifest['requests']],replies,
        teacher=teacher,shift='S1',identity=identity,min_iou=0.,ambiguity_margin=0.)
    groups={}
    for row in cache['records']:
        c=row['candidate']; base,scale=c['candidate_id'].rsplit('-context',1)
        groups.setdefault((c['video_id'],base),{})[int(scale)]=row['target']
    if any(set(pair)!={2,3} for pair in groups.values()):
        raise ValueError('incomplete source contexts')
    return {key:[pair[2],pair[3]] for key,pair in groups.items()},evidence


def prepare(args):
    from libs.datasets import make_dataset
    clean_head(args.expected_head)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    cfg=read(args.baseline_config)
    if cfg['seed']!=6 or cfg['rq2_formal']['shift']!='S1':
        raise ValueError('fixed S1 seed6 required')
    dataset_cfg=deepcopy(cfg['dataset'])
    for name in ('json_file','feat_folder'):
        dataset_cfg[name]=str((args.data_root/dataset_cfg[name]).resolve())
    database=read(dataset_cfg['json_file'])['database']
    members=membership(cfg,database)
    manifest=read(args.requests)
    if manifest['shift']!='S1' or manifest['split']!='source_validation' or manifest['seed']!=6:
        raise ValueError('wrong diagnostic source requests')
    pool={}
    for request in manifest['requests']:
        c=deepcopy(request['candidate']);c['candidate_id']=c['candidate_id'].rsplit('-context',1)[0]
        if c['video_id'] not in members['heldout']:
            raise ValueError('teacher request is outside source heldout')
        if request['candidate']['candidate_id'].endswith('-context2'):
            pool[(c['video_id'],c['candidate_id'])]=c
    selected=select_candidates(pool.values(),POLICY['participation'])
    selected_ids={(r['video_id'],r['candidate_id']) for r in selected}
    pairs={};evidence={}
    for teacher in ARMS[2:]:
        pairs[teacher],evidence[teacher]=teacher_pairs(args.requests,args.source_root/teacher/'S1',
                                                     dataset_cfg['json_file'],teacher)
        if set(pairs[teacher])!=set(pool):
            raise ValueError('teacher pools differ')
    source=dict(x=[],roles=[],log_offsets=[],video_ids=[])
    unlabeled=dict(x=[],routes={name:[] for name in ARMS[1:]},video_ids=[],
                   candidate_ids=[],times=[],selected=[])
    truth=dict(candidates=[],roles=[])
    features=[];seen=set();skipped=[]
    for split in ('train','heldout'):
        key='src_train_split' if split=='train' else 'src_val_split'
        dataset=make_dataset(cfg['dataset_name'],False,cfg[key],**dataset_cfg)
        for video in dataset:
            meta=video['metadata'];vid=meta['video_id']
            if vid not in members[split]:
                raise ValueError('unexpected dataset membership')
            if split=='heldout' and not any(key[0]==vid for key in pool):
                continue
            path=Path(dataset_cfg['feat_folder'])/(str(dataset_cfg.get('file_prefix') or '')+
                                                  vid+dataset_cfg['file_ext'])
            if str(path) not in seen:
                features.append(reference(path));seen.add(str(path))
            fps=meta['fps'];stride=meta['feat_stride'];frames=meta['feat_num_frames']
            segments=(video['segments']*stride+.5*frames)/fps
            feats=video['feats']
            if split=='train':
                used=set()
                for gt in segments.tolist():
                    ids=real_indices(gt,fps,stride,frames,feats.shape[1],POLICY['source_points_per_instance'])
                    for idx in ids.tolist():
                        t=(idx*stride+.5*frames)/fps
                        # 重叠实例处不强制给同一点唯一阶段；仅源监督取点用GT。
                        if idx in used or int(((segments[:,0]<=t)&(segments[:,1]>=t)).sum())!=1:
                            continue
                        used.add(idx)
                        source['x'].append(local_patches(feats,torch.tensor([idx]))[0])
                        source['roles'].append(phase_distribution([t],gt)[0])
                        source['log_offsets'].append(torch.log1p(torch.tensor([t-gt[0],gt[1]-t]).clamp_min(0)))
                        source['video_ids'].append(vid)
            else:
                for identity,c in sorted(pool.items()):
                    if identity[0]!=vid:continue
                    ids=real_indices(c['proposal'],fps,stride,frames,feats.shape[1],POLICY['candidate_points'])
                    if not len(ids):
                        skipped.append(dict(video_id=vid,candidate_id=c['candidate_id'],reason='no_real_grid_point'))
                        continue
                    times=(ids.float()*stride+.5*frames)/fps
                    first=len(unlabeled['x']);count=len(ids)
                    unlabeled['x'].extend(local_patches(feats,ids).unbind(0))
                    unlabeled['times'].extend(times.tolist())
                    unlabeled['video_ids'].extend([vid]*count)
                    unlabeled['candidate_ids'].extend([c['candidate_id']]*count)
                    unlabeled['selected'].extend([identity in selected_ids]*count)
                    unlabeled['routes']['proposal'].extend(phase_distribution(times,c['proposal']).unbind(0))
                    for teacher in ARMS[2:]:
                        unlabeled['routes'][teacher].extend(paired_distribution(times,pairs[teacher][identity]).unbind(0))
                    # 以下只写入独立评分工件；不写入unlabeled/训练参数。
                    ranked=sorted([(interval_iou(c['proposal'],s),i) for i,s in enumerate(segments.tolist())],reverse=True)
                    gt=segments[ranked[0][1]].tolist() if ranked and ranked[0][0]>0 else None
                    if len(ranked)>1 and abs(ranked[0][0]-ranked[1][0])<1e-8:
                        gt=None
                    truth['roles'].extend(phase_distribution(times,gt).tolist())
                    truth['candidates'].append(dict(candidate_id=c['candidate_id'],video_id=vid,
                        row_indices=list(range(first,first+count)),gt=gt,proposal=c['proposal'],
                        selected=identity in selected_ids,duration=meta['duration']))
            print('PROBE_PREPARE',split,vid,len(source['x']),len(unlabeled['x']),flush=True)
    for key in ('x','roles','log_offsets'):source[key]=torch.stack(source[key])
    unlabeled['x']=torch.stack(unlabeled['x'])
    unlabeled['times']=torch.tensor(unlabeled['times'])
    unlabeled['selected']=torch.tensor(unlabeled['selected'],dtype=torch.bool)
    unlabeled['routes']={k:torch.stack(v) for k,v in unlabeled['routes'].items()}
    check_training_payload(source,unlabeled)
    torch.save(source,root/'source_train.pt');torch.save(unlabeled,root/'unlabeled.pt')
    save(root/'scoring_only.json',truth)
    # 静态诊断先保存，但训练入口不加载此结果，不扫描参数。
    save(root/'routing_diagnostic.json',score_routes(unlabeled,truth))
    plan=dict(schema=VARIANT,diagnostic_only=True,formal_rq4=False,shift='S1',policy=POLICY,
        code_commit=args.expected_head,code_files=code_identity(),source_evidence=evidence,
        baseline_config=reference(args.baseline_config),annotations=reference(dataset_cfg['json_file']),
        features=features,membership=members,target_videos_loaded=[],target_gt_used=False,
        inputs={name:reference(root/name) for name in ('source_train.pt','unlabeled.pt','scoring_only.json')},
        selected_ids=sorted(selected_ids),pool_count=len(pool),selected_count=len(selected_ids),
        source_points=len(source['x']),heldout_points=len(unlabeled['x']),skipped=skipped,
        architecture='frozen SlowFast local 9-grid feature probe; not full SADA/FPN training',
        fit='source_train only; source_validation features used transductively without labels',
        scoring='fixed original-proposal matched instance; class-agnostic conditional localization, not TAL mAP',
        selection='fixed last step; no validation-based checkpoint or hyperparameter selection',
        limitations=['single shift/seed, finite source proposal pool; not evidence of cross-domain TAL gain',
            'three role labels are continuous interpolation, not hard equal-third action segmentation'])
    save(root/'plan.json',plan)
    print('PROBE_CPU_PREPARE_PASS',plan['source_points'],plan['heldout_points'],flush=True)


def audit_inputs(root,expected_head):
    clean_head(expected_head);plan=read(root/'plan.json')
    if plan['schema']!=VARIANT or plan['policy']!=POLICY or plan['code_files']!=code_identity():
        raise ValueError('probe protocol/code differs from preparation')
    for ref in list(plan['inputs'].values())+plan['features']+[plan['annotations'],plan['baseline_config']]:
        if file_sha(ref['path'])!=ref['sha256']:
            raise ValueError('probe input SHA changed: '+ref['path'])
    for source in plan['source_evidence'].values():
        for ref in source.values():
            if file_sha(ref['path'])!=ref['sha256']:raise ValueError('source evidence SHA changed')
    source=torch.load(plan['inputs']['source_train.pt']['path'],map_location='cpu',weights_only=True)
    unlabeled=torch.load(plan['inputs']['unlabeled.pt']['path'],map_location='cpu',weights_only=True)
    check_training_payload(source,unlabeled)
    return plan,source,unlabeled


def real_cpu_audit(args):
    root=args.output.resolve()
    plan,source,unlabeled=audit_inputs(root,args.expected_head)
    torch.manual_seed(POLICY['seed'])
    model=TemporalProbe(source['x'].shape[1])
    main=supervised_loss(model,source['x'][:64],source['roles'][:64],source['log_offsets'][:64])
    main.backward()
    if not torch.isfinite(main) or model.offset.weight.grad.norm()<=0:
        raise ValueError('real source localization branch has no finite gradient')
    bank=source_bank(model,source['x'],source['roles'],'cpu')
    gradients={}
    for arm in ARMS[1:]:
        model.zero_grad(set_to_none=True)
        idx=torch.where(unlabeled['selected'] & (unlabeled['routes'][arm].sum(-1)>0))[0][:64]
        if not len(idx):raise ValueError('no actual routed points: '+arm)
        aux=alignment_loss(model(unlabeled['x'][idx])[0],bank,unlabeled['routes'][arm][idx])
        aux.backward()
        gradient=float(model.project.weight.grad.norm())
        if not math.isfinite(gradient) or gradient<=0:raise ValueError('real shared encoder has no gradient')
        gradients[arm]=gradient
    proof=dict(status='passed',round=3,scope='real source feature CPU forward/backward and SHA audit',
        code_commit=args.expected_head,code_files=code_identity(),plan_sha256=file_sha(root/'plan.json'),
        source_points=plan['source_points'],heldout_points=plan['heldout_points'],
        gradients=gradients,scoring_gt_loaded=False,gpu_started=False)
    save(root/'real_cpu_audit.json',proof)
    print('PROBE_REAL_INPUT_AUDIT_PASS',json.dumps(proof),flush=True)


def run(args):
    import fcntl
    from tools.rq3_gpu import gpu_slot,guard_before_cuda
    root=args.output.resolve()
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'training_started.json').exists():
            raise ValueError('existing run; preserve outputs rather than restart')
        plan,source,unlabeled=audit_inputs(root,args.expected_head)
        proof=read(root/'real_cpu_audit.json')
        if proof['status']!='passed' or proof['code_files']!=code_identity() or \
                proof['plan_sha256']!=file_sha(root/'plan.json'):
            raise ValueError('real CPU audit required before single-GPU experiment')
        save(root/'training_started.json',dict(pid=os.getpid(),head=args.expected_head,time=time.time(),
            expected_gpu=0,plan_sha256=file_sha(root/'plan.json')))
        def stop(*_):raise RuntimeError('explicit stop; preserve probe outputs')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        started=time.monotonic()
        artifacts={}
        def emit(*values):print('TEMPORAL_PROBE',*values,flush=True)
        def save_arm(arm,model,bank,history,predictions):
            out=root/arm;out.mkdir(exist_ok=False)
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=state,source_bank=bank.cpu(),policy=POLICY),out/'probe.pt')
            torch.save(predictions,out/'predictions.pt');save(out/'history.json',history)
            artifacts[arm]={name:reference(out/name) for name in ('probe.pt','predictions.pt','history.json')}
        try:
            with gpu_slot('0',args.data_root,wait=True) as card:
                os.environ.update(CUDA_VISIBLE_DEVICES=card['uuid'],RQ3_GPU_INDEX='0',RQ3_GPU_UUID=card['uuid'])
                os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                guard_before_cuda()
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=False
                torch.backends.cudnn.benchmark=False
                predictions=train_arms(source,unlabeled,'cuda:0',emit,save_arm)
                save(root/'all_arms_frozen.json',dict(artifacts=artifacts,
                    evaluation_gt_loaded=False,card=card,code_commit=args.expected_head))
                # 先冻结所有结果，最后一次性读GT评分。任何组不能据此重训。
                truth=read(root/'scoring_only.json')
                scores={arm:score_predictions(result,unlabeled,truth) for arm,result in predictions.items()}
                for arm,score in scores.items():save(root/arm/'metrics.json',score)
                report=dict(schema=VARIANT,status='complete',formal_rq4=False,arms=scores,
                    routing_diagnostic=score_routes(unlabeled,truth),policy=POLICY,
                    artifacts=artifacts,code_commit=args.expected_head,plan_sha256=file_sha(root/'plan.json'),
                    elapsed_seconds=time.monotonic()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                    target_gt_used=False,heldout_gt_used_for_optimization=False,
                    interpretation='mechanism-only; no TAL mAP, no cross-domain gain claim')
                save(root/'result.json',report)
                print('TEMPORAL_PROBE_COMPLETE',json.dumps({a:{k:scores[a][k] for k in
                    ('mean_iou','normalized_endpoint_error','role_accuracy')} for a in ARMS}),flush=True)
        except BaseException as exc:
            save(root/'error.json',dict(error=repr(exc),preserve_outputs=True))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','audit','run'))
    for name in ('output','data-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-head',required=True)
    for name in ('baseline-config','requests','source-root'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.mode=='prepare':prepare(args)
    elif args.mode=='audit':real_cpu_audit(args)
    else:run(args)


if __name__=='__main__':main()
