"""同一固定阈值网格的无召回约束AP容量参照；只读主队列，不改训练/预测。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_foreground_calibration_oracle import SHIFTS,REFERENCE,OLD_PILOT


def select_unconstrained(cal):
    """仅已有target_train网格；移除99.5%TP约束，其余tie规则完全一致。"""
    grid=cal['grid'];details=cal['per_class'];classes=cal['classes'];rules={}
    for c in classes:
        rows=details[str(c)]['grid']
        i=max(range(len(grid)),key=lambda i:(rows[i]['map'],rows[i]['kept'],grid[i][0],-grid[i][1]))
        if not any(details[str(c)]['baseline']['tp']):
            i=next(i for i,r in enumerate(grid) if r==[1.01,.1] or r==(1.01,.1))
        rules[str(c)]=dict(index=i,rule=grid[i])
    i=max(range(len(grid)),key=lambda i:(sum(details[str(c)]['grid'][i]['map'] for c in classes),
        sum(details[str(c)]['grid'][i]['kept'] for c in classes),grid[i][0],-grid[i][1]))
    return dict(global_index=i,global_rule=grid[i],per_class=rules,classes=classes,
        objective='target_train_AP_max_without_TP_retention_constraint',target_val_for_selection=False)


def freeze(source,out):
    path=source/'calibration_frozen.json';cal=read(path);result=select_unconstrained(cal)
    result.update(calibration_sha256=sha(path),target_train_predictions_sha256=cal['predictions_sha256'],code=code_identity())
    save(out/'unconstrained_frozen.json',result)


def evaluate(source,out):
    import numpy as np
    import torch
    from tools.foreground_oracle_ap import prediction_frame,ClassAP,keep_rule,background_mask
    from tools.run_foreground_calibration_oracle import load_truth
    from libs.evaluation import ANETdetection
    frozen=read(out/'unconstrained_frozen.json');assert sha(source/'calibration_frozen.json')==frozen['calibration_sha256']
    path=source/'target_val_predictions.pt';receipt=read(source/'target_val_gpu_complete.json')
    assert sha(path)==receipt['predictions_sha256'];cfg=read(source/'config.json')
    frame=prediction_frame(torch.load(path,map_location='cpu'))
    truth=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'])
    bg=background_mask(truth,frame);global_keep=keep_rule(frame,frozen['global_rule']);class_keep=np.ones(len(frame),dtype=bool)
    for c in frozen['classes']:
        positions=np.flatnonzero(frame.label.to_numpy()==c)
        class_keep[positions]=keep_rule(frame.iloc[positions],frozen['per_class'][str(c)]['rule'])
    masks=dict(global_unconstrained=global_keep,class_unconstrained=class_keep)
    results={};baseline=read(REFERENCE/source.name/'evaluation/target_val.json')['map_0.1_to_0.5']
    for name,mask in masks.items():
        evaluator=ANETdetection(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],
                                num_workers=1,action_subsampling='10_majority_classes')
        values=evaluator.evaluate(frame.loc[mask,['video-id','t-start','t-end','label','score']].copy(),verbose=False)
        details={};total_AP=[]
        for c in sorted(truth.label.unique()):
            positions=np.flatnonzero(frame.label.to_numpy()==c);f=frame.iloc[positions].reset_index(drop=True)
            engine=ClassAP(truth[truth.label==c],f)
            before=engine.evaluate(np.ones(len(f),dtype=bool));after=engine.evaluate(mask[positions])
            details[str(c)]=dict(before=before,after=after);total_AP.extend(after['ap'])
        assert abs(np.mean(total_AP)-float(values[1]))<1e-7
        results[name]=dict(map=float(values[1]),delta_pp=100*(float(values[1])-baseline),
            kept=int(mask.sum()),background_removed=int((bg&~mask).sum()),nonbackground_removed=int((~bg&~mask).sum()),
            per_class=details)
    save(out/'complete.json',dict(shift=source.name,role='target_train_calibration_Oracle_capacity_not_UDA',
        results=results,target_val_used_for_selection=False,
        artifacts={str(p):sha(p) for p in [out/'unconstrained_frozen.json',source/'target_val_gpu_complete.json',path,
            source/'calibration_frozen.json',source/'target_train_predictions.pt']}))


def watch(source,root):
    import fcntl
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active=None;completed=[]
    def stop(sig,frame):
        if active and active['process'].poll() is None:os.killpg(active['process'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        while len(completed)<6:
            for s in SHIFTS:
                e=root/s
                if (source/s/'calibration_frozen.json').exists() and not (e/'unconstrained_frozen.json').exists():freeze(source/s,e)
            if active and active['process'].poll() is not None:
                active['log'].close()
                if active['process'].returncode:raise RuntimeError(active['shift']+' capacity CPU failed')
                r=read(root/active['shift']/'complete.json')
                for path,digest in r['artifacts'].items():assert sha(path)==digest
                completed.append(active['shift']);active=None
            # 等旧两CPU队列全收尾，避免与主队列合计超过2路完整AP。
            old_done=all((p/'final.json').exists() for p in [OLD_PILOT,Path('/vepfs/vlms/sada_negative_curriculum6_f20e133_1103')])
            if not active and old_done:
                for s in SHIFTS:
                    if s in completed or not (root/s/'unconstrained_frozen.json').exists() or not (source/s/'target_val_gpu_complete.json').exists():continue
                    log=(root/s/'evaluate.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
                    proc=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--evaluate','--source',str(source/s),'--root',str(root/s)],
                        env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active=dict(process=proc,shift=s,log=log);break
            save(root/'state.json',dict(pid=os.getpid(),completed=completed,active=None if not active else dict(shift=active['shift'],pid=active['process'].pid),old_CPU_queues_finished=old_done))
            if len(completed)<6:time.sleep(10)
        rows=[read(root/s/'complete.json') for s in SHIFTS]
        save(root/'final.json',dict(count=6,results=rows,unconstrained_target_train_calibration_Oracle=True,
            means={name:sum(r['results'][name]['delta_pp'] for r in rows)/6 for name in ['global_unconstrained','class_unconstrained']}))
    except Exception as e:
        save(root/'error.json',dict(error=repr(e)));raise
    finally:
        if active and active['process'].poll() is None:os.killpg(active['process'].pid,signal.SIGTERM)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    os.chdir('/vepfs/vlms/SADA_PLUS')
    if a.evaluate:evaluate(a.source,a.root)
    else:
        a.root.mkdir(exist_ok=False,parents=True)
        save(a.root/'plan.json',dict(code=code_identity(),source=str(a.source),
            rule='maximize target_train grid AP with no .995 TP constraint; same tie/fallback rules',
            source_training_or_predictions_changed=False,target_val_used_for_selection=False))
        watch(a.source,a.root)
