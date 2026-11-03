"""固定头的源fit/val只读诊断；无优化、阈值选择、目标数据或新增AP评估。"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha, code_identity

ORIGIN = Path('/vepfs/vlms/background_ap_cost_source6_bd16507_1103')
RATES = (.005, .01, .02, .05)
SPEC = dict(schema='background-fit-validation-gap-readonly-v1', source_only=True, target_read=False,
    fixed_heads='bd16507 fixed20 control and ap_cost', rates=RATES, no_training=True,
    AUC='video macro; exclude videos without both labels and report denominator',
    cost='source fit marginal loss-weighted mean BG score; not AP or additive AP effect',
    selection='all four fixed per-class budgets; no winner selection')


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool); scores = np.asarray(scores, dtype=float)
    npos = int(labels.sum()); nneg = len(labels)-npos
    if not npos or not nneg:
        return None
    order = np.argsort(scores, kind='stable'); sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0)+1]
    ends = np.r_[starts[1:], len(order)]
    ranks = np.empty(len(order), dtype=float)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start+1+end)/2
    return float((ranks[labels].sum()-npos*(npos+1)/2)/(npos*nneg))


def summarize(rows, mode):
    videos=[]; all_scores=[]; all_bg=[]; all_labels=[]; all_delta=[]
    for row in rows:
        bg=np.asarray(row['bg'],dtype=bool); score=np.asarray(row['pbg'][mode],dtype=float) if mode!='confidence' else 1-np.asarray(row['scores'],dtype=float)
        delta=np.asarray(row['delta'],dtype=float) if 'delta' in row else None
        item=dict(video=row['video'],n=len(bg),BG=int(bg.sum()),AUC=auc(bg,score))
        if delta is not None:
            harm=np.maximum(-delta,0)*(~bg); benefit=np.maximum(delta,0)*bg
            item['harm_weighted_BG_score']=float(np.dot(harm,score)/harm.sum()) if harm.sum()>0 else None
            item['benefit_weighted_BG_score']=float(np.dot(benefit,score)/benefit.sum()) if benefit.sum()>0 else None
            all_delta.extend(delta.tolist())
        videos.append(item);all_scores.extend(score.tolist());all_bg.extend(bg.tolist());all_labels.extend(row['labels'])
    values=[x['AUC'] for x in videos if x['AUC'] is not None]
    score=np.asarray(all_scores);bg=np.asarray(all_bg);labels=np.asarray(all_labels);budgets=[]
    for rate in RATES:
        chosen=[]
        for c in range(10):
            ids=np.flatnonzero(labels==c).tolist();ids.sort(key=lambda i:(-score[i],i));chosen.extend(ids[:int(rate*len(ids))])
        chosen=np.asarray(sorted(chosen),dtype=int)
        item=dict(rate=rate,deleted=len(chosen),BG_removed=int(bg[chosen].sum()),nonBG_removed=int((~bg[chosen]).sum()),selected_indices=chosen.tolist())
        if all_delta:
            harm=np.maximum(-np.asarray(all_delta),0)*(~bg)
            item['deleted_harm_cost_share_proxy_not_AP']=float(harm[chosen].sum()/harm.sum()) if harm.sum()>0 else None
        budgets.append(item)
    result=dict(video_macro_AUC=float(np.mean(values)) if values else None,AUC_valid_videos=len(values),total_videos=len(rows),candidates=len(bg),BG=int(bg.sum()),videos=videos,all_fixed_budgets=budgets)
    if all_delta:
        for key in ('harm_weighted_BG_score','benefit_weighted_BG_score'):
            v=[x[key] for x in videos if x[key] is not None];result[key+'_macro']=float(np.mean(v)) if v else None
    return result


def worker(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.hard_background_modules import HardBackgroundHead
    from tools.audit_background_ap_cost import make_frame, background
    from tools.run_foreground_calibration_oracle import load_truth
    parent=ORIGIN/entry.name;start=time.time()
    old_audit=read(ORIGIN/'independent_final_audit.json');assert old_audit['passed']
    assert sha(ORIGIN/'independent_results'/(entry.name+'.json'))==old_audit['per_shift_audit_sha256'][entry.name]
    checkpoint=parent/'fixed20_heads.pt';gpu=read(parent/'gpu_complete.json')
    assert sha(checkpoint)==gpu['artifacts'][str(checkpoint)]
    cost=read(parent/'costs.json');state=torch.load(checkpoint,map_location='cpu')
    guard=first_cuda_guard(entry)
    heads={m:HardBackgroundHead().cuda().eval() for m in ('control','ap_cost')}
    for m,h in heads.items():h.load_state_dict(state['heads'][m])
    before={m:base.tensor_digest(h.state_dict()) for m,h in heads.items()}
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(0)
    old_predictions=torch.load(parent/'source_val_predictions.pt',map_location='cpu')
    validation={m:{x['video']:x for x in rows} for m,rows in old_predictions.items()}
    with torch.no_grad():
        # 全验证集逐Tensor复现冻结头的原读出，然后才做fit诊断。
        for record in cost['validation']:
            data=torch.load(record['path'],map_location='cpu');ids=data['scores']>=.3
            for m,h in heads.items():
                p=torch.zeros(len(ids))
                if ids.any():p[ids]=h(data['feature'].cuda(),data['valid'].cuda(),data['boxes'][ids].cuda(),data['labels'][ids].cuda(),data['scores'][ids].cuda())[:,0].sigmoid().cpu()
                assert torch.equal(p,validation[m][record['video']]['pbg'])
        fit=[]
        for row in cost['rows']:
            data=torch.load(row['record']['path'],map_location='cpu');ids=torch.tensor(row['hard_indices'],dtype=torch.long)
            probabilities={}
            for m,h in heads.items():
                probabilities[m]=h(data['feature'].cuda(),data['valid'].cuda(),data['boxes'][ids].cuda(),data['labels'][ids].cuda(),data['scores'][ids].cuda())[:,0].sigmoid().cpu().tolist() if len(ids) else []
            fit.append(dict(video=row['record']['video'],scores=data['scores'][ids].tolist(),labels=data['labels'][ids].tolist(),bg=row['bg'],delta=row['delta'],pbg=probabilities))
    assert before=={m:base.tensor_digest(h.state_dict()) for m,h in heads.items()}
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state(0))
    assert sha(checkpoint)==gpu['artifacts'][str(checkpoint)]
    cfg=read('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103/'+entry.name+'/config.json')
    gt=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);val=[]
    for row in old_predictions['control']:
        frame=make_frame([row]);mask=row['scores'].numpy()>=.3;bg=background(gt,frame)[mask]
        val.append(dict(video=row['video'],scores=row['scores'][mask].tolist(),labels=row['labels'][mask].tolist(),bg=bg.tolist(),pbg={m:validation[m][row['video']]['pbg'][mask].tolist() for m in heads}))
    assert not {x['video'] for x in fit}&{x['video'] for x in val}
    rows_path=entry/'readout.json';save(rows_path,dict(fit=fit,validation=val))
    result={role:{m:summarize(rows,m) for m in ('control','ap_cost','confidence')} for role,rows in [('fit',fit),('validation',val)]}
    save(entry/'complete.json',dict(passed=True,spec=SPEC,results=result,rows_sha256=sha(rows_path),origin_checkpoint_sha256=sha(checkpoint),origin_costs_sha256=sha(parent/'costs.json'),origin_final_audit_sha256=sha(ORIGIN/'independent_final_audit.json'),guard=guard,guard_sha256=sha(guard),head_digest=before,parameters_unchanged=True,RNG_unchanged=True,all_validation_predictions_exact=True,seconds=time.time()-start,peak_mib=torch.cuda.max_memory_reserved()/2**20))
    print('COMPLETE',entry.name,time.time()-start,flush=True)


def run(root):
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity(),origin=str(ORIGIN)))
    pending=[f'S{i}' for i in range(1,7)];active={}
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for task in active.values():
            if task['p'].poll() is None:os.killpg(task['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        while pending or active:
            for s,t in list(active.items()):
                if t['p'].poll() is None:continue
                t['log'].close();del active[s]
                if t['p'].returncode:raise RuntimeError(s+' readout failed')
            for g in (2,0,3,1):
                if not pending:break
                if g in [t['gpu'] for t in active.values()]:continue
                obs=snapshot(g)
                if obs['free_mib']<12288:continue
                s=pending.pop(0);e=root/s;e.mkdir();lease=e/'readout_lease.json'
                save(lease,dict(owner_pid=os.getpid(),index=g,uuid=obs['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=obs['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                log=(e/'readout.log').open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(e)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[s]=dict(p=p,gpu=g,log=log)
            save(root/'state.json',dict(owner_pid=os.getpid(),pending=pending,active={s:dict(pid=t['p'].pid,gpu=t['gpu']) for s,t in active.items()}))
            time.sleep(3)
        results={f'S{i}':read(root/f'S{i}'/'complete.json') for i in range(1,7)}
        save(root/'final.json',dict(spec=SPEC,results=results,complete_sha256={s:sha(root/s/'complete.json') for s in results}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for task in active.values():
            if task['p'].poll() is None:os.killpg(task['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);args=p.parse_args()
    worker(args.entry) if args.entry else run(args.root)
