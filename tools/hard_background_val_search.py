"""用户授权的target-val-tuned参数容量搜索；只读现有四头输出，不回流训练。"""
import argparse,os,sys,time,subprocess,signal
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.hard_background_modules import MODES
from tools.run_hard_background_pilots import SOURCE,SHIFTS
PILOT=Path('/vepfs/vlms/hard_background_pilots24_8e80d74_1103')


def grid():
    import numpy as np
    thresholds=np.unique(np.r_[np.arange(0,1.001,.05),.9,.925,.95,.975,.99,.995,.999,.9999,1.01])
    rows=[dict(kind='delete',cutoff=c,beta=b,threshold=float(t))
          for c in [.3,.35,.4,.5,.7] for b in [0.,.25,.5,1.] for t in thresholds]
    rows += [dict(kind='soft',cutoff=c,alpha=a,beta=b)
             for c in [.3,.4,.5] for a in [0.,.25,.5,1.,2.] for b in [0.,.25,.5,1.,2.]]
    return rows


def transform(scores,pbg,quality,rule):
    import numpy as np
    scores=np.asarray(scores,dtype=float);p=np.clip(pbg,1e-8,1-1e-8);q=np.clip(quality,1e-8,1-1e-8)
    high=scores>=rule['cutoff'];keep=np.ones(len(scores),dtype=bool);new=scores.copy()
    if rule['kind']=='delete':
        z=np.log(p)-np.log1p(-p)-rule['beta']*(np.log(q)-np.log1p(-q))
        risk=1/(1+np.exp(-np.clip(z,-80,80)))
        keep=~(high&(risk>rule['threshold']))
    else:
        new[high]=scores[high]*(1-p[high])**rule['alpha']*q[high]**rule['beta']
    return keep,new


def evaluate(entry):
    import numpy as np
    import torch
    from tools.foreground_oracle_ap import ClassAP,prediction_frame
    from tools.background_payoff_diagnosis import summarize_ap
    from tools.run_foreground_calibration_oracle import load_truth
    s=entry.name;parent=PILOT/s;cfg=read(SOURCE/s/'config.json');receipt=read(parent/'gpu_complete.json')
    path=parent/'target_val_predictions.pt';assert sha(path)==receipt['artifacts'][str(path)]
    predictions=torch.load(path,map_location='cpu');truth=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'])
    choices=grid();classes=sorted(int(c) for c in truth.label.unique());maps=np.zeros((len(MODES),len(choices),len(classes)))
    counts=np.zeros_like(maps,dtype=np.int64);baselines=[];best_rows={}
    for mi,mode in enumerate(MODES):
        rows=predictions[mode];frame=prediction_frame(rows)
        frame['quality']=np.concatenate([r['quality'].numpy() for r in rows])
        baseline=[]
        for ci,c in enumerate(classes):
            f=frame[frame.label==c].reset_index(drop=True);engine=ClassAP(truth[truth.label==c],f)
            baseline.append(engine.evaluate(np.ones(len(f),dtype=bool))['map'])
            for i,rule in enumerate(choices):
                keep,score=transform(f.score.to_numpy(),f.pbg.to_numpy(),f.quality.to_numpy(),rule)
                result=summarize_ap(engine,keep,score)
                maps[mi,i,ci]=result['map'];counts[mi,i,ci]=int(keep.sum())
            print('VAL_GRID_CLASS',s,mode,c,flush=True)
        baselines.append(float(np.mean(baseline)))
        idx=max(range(len(choices)),key=lambda i:(float(maps[mi,i].mean()),int(counts[mi,i].sum()),-i))
        per=[max(range(len(choices)),key=lambda i:(maps[mi,i,c],counts[mi,i,c],-i)) for c in range(len(classes))]
        best_rows[mode]=dict(baseline=baselines[-1],global_map=float(maps[mi,idx].mean()),
            global_delta_pp=100*(float(maps[mi,idx].mean())-baselines[-1]),global_index=idx,global_rule=choices[idx],
            per_class_map=float(np.mean([maps[mi,i,c] for c,i in enumerate(per)])),
            per_class_delta_pp=100*(float(np.mean([maps[mi,i,c] for c,i in enumerate(per)]))-baselines[-1]),
            per_class_rules={str(c):dict(index=i,rule=choices[i]) for c,i in zip(classes,per)})
    assert max(baselines)-min(baselines)<1e-12
    path_out=entry/'grid.npz';np.savez_compressed(path_out,maps=maps,kept=counts,baseline=np.array(baselines),classes=np.array(classes))
    save(entry/'complete.json',dict(role='target_val_supervised_parameter_search_development_not_UDA',code=code_identity(),
        choices=len(choices),modes=list(MODES),results=best_rows,classes=classes,source_training_changed=False,
        target_validation_labels_used_for_selection=True,
        artifacts={str(p):sha(p) for p in [path_out,path,parent/'gpu_complete.json',entry.parent/'plan.json']}))


def audit(entry):
    import numpy as np
    d=read(entry/'complete.json')
    for path,digest in d['artifacts'].items():assert sha(path)==digest
    a=np.load(entry/'grid.npz');choices=grid()
    for mi,mode in enumerate(MODES):
        idx=max(range(len(choices)),key=lambda i:(float(a['maps'][mi,i].mean()),int(a['kept'][mi,i].sum()),-i))
        assert d['results'][mode]['global_index']==idx
        assert abs(d['results'][mode]['global_delta_pp']-100*(a['maps'][mi,idx].mean()-a['baseline'][mi]))<1e-12
    return dict(passed=True,complete_sha256=sha(entry/'complete.json'),results=d['results'])


def watch(root):
    import fcntl
    import numpy as np
    root.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(code=code_identity(),choices=grid(),modes=list(MODES),input=str(PILOT),
        user_authorized_target_val_parameter_search=True,target_val_tuned_development=True,
        inference_information_scope='only_original_score>=.3_has_learned_risk; no_lower_pool_claim',
        primary_report='one_variant_and_one_parameter_set_shared_across_all6_directions',
        additional_reports='per_direction_global_and_per_class_parameter_capacity',original_train_only_calibration_preserved=True))
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active=None;done=[]
    def stop(sig,frame):
        if active and active['p'].poll() is None:os.killpg(active['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        while len(done)<6:
            if active and active['p'].poll() is not None:
                active['log'].close()
                if active['p'].returncode:raise RuntimeError(active['shift']+' val parameter search failed')
                save(root/'audits'/(active['shift']+'.json'),audit(root/active['shift']));done.append(active['shift']);active=None
            if active is None:
                for s in SHIFTS:
                    if s in done or not (PILOT/s/'gpu_complete.json').exists():continue
                    e=root/s;e.mkdir(exist_ok=True);log=(e/'evaluate.log').open('a')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
                    p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--evaluate','--entry',str(e)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active=dict(p=p,log=log,shift=s);break
            save(root/'state.json',dict(pid=os.getpid(),completed=done,active=None if not active else dict(pid=active['p'].pid,shift=active['shift'])))
            if len(done)<6:time.sleep(5)
        arrays=[np.load(root/s/'grid.npz') for s in SHIFTS]
        values=np.mean(np.stack([a['maps'].mean(2) for a in arrays]),axis=0);baseline=float(np.mean([a['baseline'][0] for a in arrays]))
        mi,ri=max(((m,i) for m in range(len(MODES)) for i in range(len(grid()))),key=lambda x:(values[x],-x[0],-x[1]))
        save(root/'final.json',dict(count=6,role='target_val_tuned_development_not_formal_UDA',
            shared_mode=MODES[mi],shared_rule=grid()[ri],shared_rule_index=ri,mean_map=float(values[mi,ri]),baseline_mean=baseline,
            mean_delta_pp=100*(values[mi,ri]-baseline),
            per_shift_delta_pp={s:100*(arrays[j]['maps'][mi,ri].mean()-arrays[j]['baseline'][mi]) for j,s in enumerate(SHIFTS)},
            results=[read(root/s/'complete.json') for s in SHIFTS]))
    except Exception as e:save(root/'error.json',dict(error=repr(e)));raise
    finally:
        if active and active['p'].poll() is None:os.killpg(active['p'].pid,signal.SIGTERM)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--evaluate',action='store_true');p.add_argument('--entry',type=Path);a=p.parse_args()
    if a.run:watch(a.run)
    else:
        os.chdir('/vepfs/vlms/SADA_PLUS');evaluate(a.entry)
