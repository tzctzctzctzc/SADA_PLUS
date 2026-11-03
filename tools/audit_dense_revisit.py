"""独立CPU检查新监督范围、源归一化、全部预算/删框反事实和保存状态。"""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_dense_revisit import SOURCE,UDA,DIAG,LAYER,MODES,TASKS


def union_targets(length,boxes,labels):
    out=np.zeros((length,97),np.float64);left=np.arange(length)-.5;right=left+1
    for c in set(labels):
        intervals=sorted((float(a),float(b)) for (a,b),k in zip(boxes,labels) if k==c and b>a);merged=[]
        for a,b in intervals:
            if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
            else:merged.append([a,b])
        for a,b in merged:out[:,c]+=np.maximum(0,np.minimum(right,b)-np.maximum(left,a))
    return out.clip(0,1)


def input_audit(root):
    import torch
    prep=read(root/'prepare_complete.json');results={}
    for s in ('S1','S2'):
        path=root/'inputs'/s/'data.pt';assert sha(path)==prep['results'][s]['data_sha256'];data=torch.load(path,map_location='cpu');roles=data['roles'];ids={role:[r['video'] for r in rows] for role,rows in roles.items()}
        ranked=sorted(ids['fit']+ids['heldout'],key=lambda v:(hashlib.sha256(('head-holdout-v1|'+v).encode()).hexdigest(),v));held=set(ranked[:int(np.ceil(.2*len(ranked)))]);assert set(ids['heldout'])==held
        for i,a in enumerate(ids):
            for b in list(ids)[i+1:]:assert not set(ids[a])&set(ids[b])
        cfg=read(SOURCE/s/'config.json');database=read(cfg['dataset']['json_file'])['database'];count=0;annot=0;densemax=0.;total=0;sums=np.zeros(2304,np.float64);sq=sums.copy();raws={}
        for role,rows in roles.items():
            if role.startswith('target'):
                manifest=read(UDA/'inputs'/s/(role+'_unlabeled.json'));assert set(ids[role])==set(manifest['database']);assert all('annotations' not in x for x in manifest['database'].values())
            for r in rows:
                p=Path(r['raw_path']);assert sha(p)==r['raw_sha256'];count+=1;raws[r['video']]=r['raw_sha256']
                assert r['hard'].dtype==torch.bool and torch.equal(r['hard'],r['scores']>=.3)
                if role!='fit':assert not {'dense','annotations','bg','delta'}&set(r);continue
                raw=np.load(p)['feats'].astype(np.float32)[::r['downsample']];total+=len(raw);sums+=raw.astype(np.float64).sum(0);sq+=np.square(raw.astype(np.float64)).sum(0)
                meta=r['video_metadata'];ann=database[r['video']]['annotations'];boxes=(np.array([a['segment'] for a in ann])*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride'];labels=[int(a['label_id']) for a in ann]
                assert np.allclose(boxes,r['annotations']['segments'],atol=1e-3,rtol=1e-6);assert labels==r['annotations']['labels'];annot+=len(labels)
                # 以正式保存的float32 grid为独立NumPy标签参考，另核秒到grid变换。
                expected=union_targets(len(raw),r['annotations']['segments'],labels);err=np.abs(expected-r['dense'].numpy()).max();densemax=max(densemax,float(err));assert err<1e-5
                assert len(r['delta'])==int(r['hard'].sum())==len(r['bg'])
        mean=sums/total;std=np.sqrt(np.maximum(sq/total-mean**2,0)).clip(1e-4)
        assert np.allclose(mean,data['mean'].numpy(),atol=1e-6,rtol=1e-6) and np.allclose(std,data['std'].numpy(),atol=1e-6,rtol=1e-6)
        original=torch.load(DIAG/s/'inputs.pt',map_location='cpu');cost=read(DIAG/s/'fit_costs_and_pairs.json');cursor=0
        for role in original:
            for a,b in zip(roles[role],original[role]):
                assert a['video']==b['video']
                for k in ('segments','scores','labels','hard'):assert torch.equal(a[k],b[k])
                if role=='fit':
                    n=int(a['hard'].sum());assert np.array_equal(a['delta'].numpy(),np.asarray(cost['delta'][cursor:cursor+n],np.float32));assert np.array_equal(a['bg'].numpy().astype(bool),np.asarray(cost['BG'][cursor:cursor+n]));cursor+=n
        results[s]=dict(passed=True,data_sha256=sha(path),members=ids,raw_files_SHA_checked=count,raw_sha256=raws,full_source_annotations_checked=annot,all_97_dense_target_cells_numpy_checked=True,max_dense_error=densemax,normalization_sourceFIT_only=True)
        save(root/'independent_input_audit.json',dict(passed=len(results)==2,results=results,audit_code=code_identity()));print('DENSE_INPUT_AUDITED',s,count,annot,flush=True)


def select(frame,eligible,rate,confidence):
    result=[];cls=frame.label.to_numpy();score=frame.score.to_numpy();value=score if confidence else -frame.pbg.to_numpy()
    for c in range(10):
        ids=np.flatnonzero((cls==c)&(score>=.3));n=int(rate*len(ids));allowed=ids[eligible[ids]]
        ranked=sorted(allowed.tolist(),key=lambda i:(float(value[i]),i));result.extend(ranked[:n])
    return np.array(sorted(result),int)


def independent_bg(frame,truth):
    bg=np.ones(len(frame),bool)
    for vid,ids in frame.groupby('video-id',sort=False).indices.items():
        ids=np.asarray(ids);a=frame.iloc[ids][['t-start','t-end']].to_numpy(float);b=truth[truth['video-id']==vid][['t-start','t-end']].to_numpy(float)
        if not len(b):continue
        inter=np.maximum(0,np.minimum(a[:,None,1],b[None,:,1])-np.maximum(a[:,None,0],b[None,:,0]));union=(a[:,1]-a[:,0])[:,None]+(b[:,1]-b[:,0])[None]-inter
        bg[ids]=(inter/np.maximum(union,1e-12)).max(1)<.01
    return bg


def audit_results(root,s):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import prediction_frame
    from tools.audit_background_ap_cost import IndependentAP
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    folder=root/'independent_results';folder.mkdir(exist_ok=True);report=read(root/(s+'_evaluation.json'));roles=torch.load(root/'inputs'/s/'data.pt',map_location='cpu')['roles'];cfg=read(SOURCE/s/'config.json');gt=read(SOURCE/s/'source_truth.json')
    truth={role:pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c)) for r in roles[role] for b,c in zip(gt[r['video']]['segments'],gt[r['video']]['labels'])],columns=['video-id','t-start','t-end','label']) for role in ('fit','heldout')}
    truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);truth['target_val']=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'])
    outputs={};gpu_report={}
    for mode in MODES:
        entry=root/(s+'_'+mode);gpu=read(entry/'gpu_complete.json');assert sha(entry/'predictions.pt')==gpu['predictions_sha256'];assert sha(entry/'epoch020.pt')==gpu['checkpoint_sha256'];assert sha(entry/'kernel_complete.json')==gpu['kernel_sha256']
        kernel=read(entry/'kernel_complete.json');assert kernel['passed'] and kernel['complete_model_optimizer_RNG_next_step_exact'];assert sha(entry/'kernel.pt')==kernel['kernel_sha256']
        guard=read(gpu['guard']);assert sha(gpu['guard'])==gpu['guard_sha256'];lease=read(entry/'lease_train.json');assert lease['owner_pid']==read(root/'state.json')['owner_pid']
        assert sha(entry/'lease_train.json')==guard['lease_sha256'];assert lease['uuid']==guard['observed']['uuid'];assert guard['observed']['free_mib']>=12288
        assert guard['owner_pid']==lease['owner_pid'] and guard['observed']['index']==lease['index']
        history=read(entry/'history.json');assert [r['epoch'] for r in history]==list(range(1,21))
        for epoch in (1,5,10,20):
            state=torch.load(entry/f'epoch{epoch:03d}.pt',map_location='cpu');assert state['history']==history[:epoch] and state['step']==epoch*len(roles['fit'])
            assert all(k in state for k in ('model','optimizer','order_rng','target_order','augmentation_rng','python_rng','numpy_rng','cpu_rng','cuda_rng','inputs_sha256'))
            assert state['inputs_sha256']==sha(root/'inputs'/s/'data.pt')
            from tools.dense_revisit_math import DenseEvidence
            names=[n for n,_ in DenseEvidence().named_parameters()]
            for i,v in state['optimizer']['state'].items():
                expected=epoch*sum(bool(r['hard'].any()) for r in roles['fit']) if names[i].startswith('utility.') else state['step']
                assert int(v['step'])==expected
        assert os.stat(entry/'latest.pt').st_ino==os.stat(entry/'epoch020.pt').st_ino
        outputs[mode]=torch.load(entry/'predictions.pt',map_location='cpu');gpu_report[mode]=dict(gpu_sha256=sha(entry/'gpu_complete.json'),kernel_sha256=gpu['kernel_sha256'],checkpoint_sha256=gpu['checkpoint_sha256'],guard_sha256=gpu['guard_sha256'])
    result={}
    for role in ('fit','heldout','validation','target_val'):
        frames={};eligible=None
        for m in MODES:
            rows=outputs[m][role];assert len(rows)==len(roles[role])
            for a,b in zip(rows,roles[role]):
                assert a['video']==b['video']
                for k in ('segments','scores','labels'):assert torch.equal(a[k],b[k])
                assert not a['eligible'][b['scores']<.3].any() and (a['pbg'][b['scores']<.3]==0).all()
            frames[m]=prediction_frame(rows);v=torch.cat([r['eligible'] for r in rows]).numpy()
            if eligible is None:eligible=v
            else:assert np.array_equal(eligible,v)
        frame=frames['dense10'];bg=independent_bg(frame,truth[role]);engines={};pos={};baseline=[]
        for c in range(10):
            ids=np.flatnonzero(frame.label.to_numpy()==c);pos[c]=ids;engines[c]=IndependentAP(truth[role][truth[role].label==c],frame.iloc[ids].reset_index(drop=True));baseline.append(engines[c].ap(np.ones(len(ids),bool)))
        def effect(indices):
            keep=np.ones(len(frame),bool);keep[indices]=False;a=[engines[c].ap(keep[pos[c]]) for c in range(10)]
            return np.asarray(a),float(100*(np.mean(a)-np.mean(baseline)))
        result[role]={}
        for mode in (*MODES,'confidence'):
            f=frame if mode=='confidence' else frames[mode];old=report['results'][role][mode];assert np.allclose(baseline,old['original_per_class_AP'],atol=1e-7,rtol=0);auc=[];scores=1-f.score.to_numpy() if mode=='confidence' else f.pbg.to_numpy()
            for _,ids in f.groupby('video-id',sort=False).indices.items():
                ids=np.asarray(ids);ids=ids[f.iloc[ids].score.to_numpy()>=.3];a=independent_auc(bg[ids],scores[ids])
                if a is not None:auc.append(a)
            assert len(auc)==old['AUC_valid_videos'] and abs(np.mean(auc)-old['video_macro_AUC'])<1e-10
            budgets=[]
            for op in old['budgets']:
                selected=select(f,eligible,op['rate'],mode=='confidence');assert selected.tolist()==op['selected'];a,actual=effect(selected);_,benefit=effect(selected[bg[selected]]);_,harm=effect(selected[~bg[selected]])
                assert np.allclose(a,op['per_class_AP'],atol=1e-7,rtol=0)
                for x,y in ((actual,op['delta_pp']),(benefit,op['BG_only_delta_pp']),(harm,op['nonBG_only_delta_pp'])):assert abs(x-y)<1e-7
                assert int(bg[selected].sum())==op['BG_removed'] and int((~bg[selected]).sum())==op['nonBG_removed']
                budgets.append(dict(rate=op['rate'],delta_pp=actual,BG_only_delta_pp=benefit,nonBG_only_delta_pp=harm,BG_removed=int(bg[selected].sum()),nonBG_removed=int((~bg[selected]).sum())))
            result[role][mode]=dict(video_macro_AUC=float(np.mean(auc)),valid_videos=len(auc),budgets=budgets)
        if role=='target_val':
            baseline_old=read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/s/'evaluation'/'target_val.json')['map_0.1_to_0.5'];assert abs(np.mean(baseline)-baseline_old)<1e-7
        print('DENSE_RESULTS_AUDITED',s,role,flush=True)
    save(folder/(s+'.json'),dict(passed=True,report_sha256=sha(root/(s+'_evaluation.json')),GPU=gpu_report,results=result,all4budgets_and_counterfactuals_numpy_equal=True,all_original_arrays_exact=True,scope='S1/S2 historicdevelopment; not independent finalbenchmark'))


def run(root):
    while not (root/'prepare_complete.json').exists():
        if (root/'error.json').exists():raise RuntimeError('producer failed during prepare')
        time.sleep(3)
    input_audit(root)
    while not (root/'final.json').exists():
        if (root/'error.json').exists():raise RuntimeError('producer failed; preserve all artifacts, do not auto-restart')
        time.sleep(3)
    final=read(root/'final.json');assert set(final['all8GPU_sha256'])==set(TASKS)
    for task,digest in final['all8GPU_sha256'].items():assert sha(root/task/'gpu_complete.json')==digest
    for s in ('S1','S2'):audit_results(root,s)
    save(root/'independent_final_audit.json',dict(passed=True,audit_code=code_identity(),production_final_sha256=sha(root/'final.json'),input_audit_sha256=sha(root/'independent_input_audit.json'),result_sha256={s:sha(root/'independent_results'/(s+'.json')) for s in ('S1','S2')},exact_GPU_tasks=8,exact_shifts=2,exact_modes=4,exact_rates=4,target_train_GT_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
