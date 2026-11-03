"""独立重算全部操作点，核风格随机流、目标无标签使用与组对手完整状态。"""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_style_robust import DENSE,MODES,TASKS,SOURCE


def check_states(root,s,roles):
    import torch
    from tools import baseline_engine as base
    from tools.run_uda_background import TargetStream
    results={};fit=roles['fit'];groups=sorted({r['video'].split('_')[0] for r in fit});gi=np.array([groups.index(r['video'].split('_')[0]) for r in fit]);counts=np.bincount(gi);old=read(DENSE/(s+'_dense10')/'gpu_complete.json')
    assert sha(DENSE/'independent_final_audit.json')=='3cf9fcc7eb3f06e2bc557af745816847bbde67f2725ab5e2820a86a0664f0141'
    for m in MODES:
        entry=root/(s+'_'+m);gpu=read(entry/'gpu_complete.json');kernel=read(entry/'kernel_complete.json');guard=read(gpu['guard']);lease=read(entry/'lease.json')
        assert sha(entry/'gpu_complete.json')==read(root/'final.json')['allGPU_sha256'][s+'_'+m]
        for filename,key in [('kernel_complete.json','kernel_sha256'),('epoch020.pt','checkpoint_sha256'),('predictions.pt','predictions_sha256')]:assert sha(entry/filename)==gpu[key]
        assert sha(entry/'kernel.pt')==kernel['kernel_sha256'];assert sha(gpu['guard'])==gpu['guard_sha256']==kernel['guard_sha256'];assert sha(entry/'lease.json')==guard['lease_sha256']
        assert guard['owner_pid']==lease['owner_pid']==read(root/'state.json')['owner_pid'];assert guard['observed']['uuid']==lease['uuid'] and guard['observed']['free_mib']>=12288
        assert all(kernel[k] for k in ('passed','disabled_old_loss_allgrad_AdamW_exact','real_style_or_DRO_update','complete_model_optimizer_style_RNG_adversary_nextstep_exact','target_cursor_draws_exact','CPUtransfer_vs_resident_update_exact'))
        assert gpu['initial_digest']==old['initial_digest'];history=read(entry/'history.json');assert [v['epoch'] for v in history]==list(range(1,21))
        order=torch.Generator().manual_seed(123);rng=np.random.RandomState(311110);stream=TargetStream(len(roles['target_train']));effective=torch.zeros(len(roles['target_train']),dtype=torch.long);q=np.ones(len(groups),np.float32)/len(groups)
        for epoch,row in enumerate(history,1):
            mixed=0;source_other_participant=0
            for i in torch.randperm(len(fit),generator=order).tolist():
                apply=bool(rng.rand()<.5);lam=float(rng.beta(.1,.1));u=float(rng.rand());ti=int(stream.draw(1)[0]);choices=np.flatnonzero(gi!=gi[i]);si=int(choices[min(len(choices)-1,int(u*len(choices)))]);assert gi[si]!=gi[i]
                if m!='groupdro' and apply:
                    mixed+=1
                    if m in ('targetmix','targetdro'):effective[ti]+=1
                    else:source_other_participant+=1
                assert 0<=lam<=1
            assert mixed==row['mixed_videos'] and row['steps']==len(fit);assert np.allclose(q,row['q_before'],atol=5e-7,rtol=0)
            if m in ('groupdro','targetdro'):
                q=q*np.exp(np.float32(.01)*np.asarray(row['group_losses'],np.float32));q=q/q.sum()
            assert np.allclose(q,row['q_after'],atol=5e-7,rtol=0)
            if epoch in (1,5,10,20):
                state=torch.load(entry/f'epoch{epoch:03d}.pt',map_location='cpu');assert state['history']==history[:epoch] and state['step']==epoch*len(fit)
                assert state['groups']==groups and state['group_video_counts']==counts.tolist() and state['group_ids']==gi.tolist();assert state['old_dense_GPU_sha256']==sha(DENSE/(s+'_dense10')/'gpu_complete.json')
                assert state['inputs_sha256']==sha(DENSE/'inputs'/s/'data.pt');assert np.array_equal(state['adv_probs'].numpy(),np.asarray(row['q_after'],np.float32));assert torch.equal(effective,state['effective_target_seen'])
                expected_rng=rng.get_state();actual_rng=state['style_rng'];assert expected_rng[0]==actual_rng[0] and np.array_equal(expected_rng[1],actual_rng[1]) and expected_rng[2:]==actual_rng[2:]
                assert torch.equal(order.get_state(),state['order_rng']);expected_stream=stream.state();saved=state['target_order'];assert expected_stream['cursor']==saved['cursor']
                for key in ('generator','order','seen'):assert torch.equal(expected_stream[key],saved[key])
                assert all(key in state for key in ('python_rng','numpy_rng','cpu_rng','cuda_rng'))
                assert state['cpu_rng'].dtype==torch.uint8 and state['cuda_rng'].dtype==torch.uint8
                for v in state['optimizer']['state'].values():assert int(v['step'])==state['step']
                if epoch==20:assert base.tensor_digest(state['model'])==gpu['final_digest']
        assert int((effective>0).sum())==gpu['effective_target_videos'];assert os.stat(entry/'epoch020.pt').st_ino==os.stat(entry/'latest.pt').st_ino
        results[m]=dict(complete_sha256=sha(entry/'gpu_complete.json'),kernel_sha256=gpu['kernel_sha256'],checkpoint_sha256=gpu['checkpoint_sha256'],guard_sha256=gpu['guard_sha256'],participants=groups,effective_target_videos=int((effective>0).sum()),style_draws_cursor_and_group_adversary_independent_equal=True)
        print('STYLE_STATES_AUDITED',s,m,flush=True)
    return results


def audit(root,s):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import prediction_frame
    from tools.audit_background_ap_cost import IndependentAP
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    data=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu');roles=data['roles'];states=check_states(root,s,roles);outputs={m:torch.load(root/(s+'_'+m)/'predictions.pt',map_location='cpu') for m in MODES};outputs['erm']=torch.load(DENSE/(s+'_dense10')/'predictions.pt',map_location='cpu')
    cfg=read(SOURCE/s/'config.json');gt=read(SOURCE/s/'source_truth.json');truth={role:pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c)) for r in roles[role] for b,c in zip(gt[r['video']]['segments'],gt[r['video']]['labels'])],columns=['video-id','t-start','t-end','label']) for role in ('fit','heldout')}
    truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);truth['target_val']=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split']);report=read(root/(s+'_evaluation.json'));result={}
    for role in ('fit','heldout','validation','target_val'):
        frames={};eligible=None
        for m,out in outputs.items():
            assert len(out[role])==len(roles[role])
            for a,b in zip(out[role],roles[role]):
                assert a['video']==b['video']
                for k in ('segments','scores','labels'):assert torch.equal(a[k],b[k])
                assert (a['pbg'][b['scores']<.3]==0).all() and not a['eligible'][b['scores']<.3].any()
            frames[m]=prediction_frame(out[role]);e=torch.cat([r['eligible'] for r in out[role]]).numpy()
            if eligible is None:eligible=e
            else:assert np.array_equal(e,eligible)
        f=frames['erm'];bg=np.ones(len(f),bool)
        for vid,ids in f.groupby('video-id',sort=False).indices.items():
            ids=np.asarray(ids);a=f.iloc[ids][['t-start','t-end']].to_numpy(float);b=truth[role][truth[role]['video-id']==vid][['t-start','t-end']].to_numpy(float)
            if len(b):
                inter=np.maximum(0,np.minimum(a[:,None,1],b[None,:,1])-np.maximum(a[:,None,0],b[None,:,0]));union=(a[:,1]-a[:,0])[:,None]+(b[:,1]-b[:,0])[None]-inter;bg[ids]=(inter/np.maximum(union,1e-12)).max(1)<.01
        pos={};engines={};baseline=[]
        for c in range(10):
            ids=np.flatnonzero(f.label.to_numpy()==c);pos[c]=ids;engines[c]=IndependentAP(truth[role][truth[role].label==c],f.iloc[ids].reset_index(drop=True));baseline.append(engines[c].ap(np.ones(len(ids),bool)))
        def effect(chosen):
            keep=np.ones(len(f),bool);keep[chosen]=False;a=[engines[c].ap(keep[pos[c]]) for c in range(10)];return a,float(100*(np.mean(a)-np.mean(baseline)))
        result[role]={}
        for m in (*outputs,'confidence'):
            frame=f if m=='confidence' else frames[m];old=report['results'][role][m];assert np.allclose(baseline,old['original_per_class_AP'],atol=1e-7,rtol=0);value=1-frame.score.to_numpy() if m=='confidence' else frame.pbg.to_numpy();aucs=[];budgets=[]
            for _,ids in frame.groupby('video-id',sort=False).indices.items():
                ids=np.asarray(ids);ids=ids[frame.iloc[ids].score.to_numpy()>=.3];a=independent_auc(bg[ids],value[ids])
                if a is not None:aucs.append(a)
            assert len(aucs)==old['AUC_valid_videos'] and abs(np.mean(aucs)-old['video_macro_AUC'])<1e-10
            for op in old['budgets']:
                chosen=[]
                for c in range(10):
                    pool=np.flatnonzero((frame.label.to_numpy()==c)&(frame.score.to_numpy()>=.3));n=int(len(pool)*op['rate']);allowed=pool[eligible[pool]];key=frame.score.to_numpy() if m=='confidence' else -frame.pbg.to_numpy();chosen+=sorted(allowed.tolist(),key=lambda i:(float(key[i]),i))[:n]
                chosen=np.asarray(sorted(chosen),int);assert chosen.tolist()==op['selected'];ap,delta=effect(chosen);_,benefit=effect(chosen[bg[chosen]]);_,harm=effect(chosen[~bg[chosen]])
                assert np.allclose(ap,op['per_class_AP'],atol=1e-7,rtol=0)
                for a,b in ((delta,op['delta_pp']),(benefit,op['BG_only_delta_pp']),(harm,op['nonBG_only_delta_pp'])):assert abs(a-b)<1e-7
                assert int(bg[chosen].sum())==op['BG_removed'] and int((~bg[chosen]).sum())==op['nonBG_removed'];budgets.append(op)
            result[role][m]=dict(baseline=float(np.mean(baseline)),video_macro_AUC=float(np.mean(aucs)),valid_videos=len(aucs),budgets=budgets)
        if role=='target_val':assert abs(np.mean(baseline)-read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/s/'evaluation'/'target_val.json')['map_0.1_to_0.5'])<1e-7
        print('STYLE_AP_AUDITED',s,role,flush=True)
    folder=root/'independent_results';folder.mkdir(exist_ok=True);save(folder/(s+'.json'),dict(passed=True,states=states,results=result,report_sha256=sha(root/(s+'_evaluation.json')),all_fixed_AP_and_counterfactuals_independent=True,original_arrays_exact=True,target_train_GT_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    while not (root/'final.json').exists():
        if (root/'error.json').exists():raise RuntimeError('producer failed; no automatic restart')
        time.sleep(3)
    final=read(root/'final.json');assert set(final['allGPU_sha256'])==set(TASKS)
    for task,digest in final['allGPU_sha256'].items():assert sha(root/task/'gpu_complete.json')==digest
    for s in ('S1','S2'):audit(root,s)
    save(root/'independent_final_audit.json',dict(passed=True,audit_code=code_identity(),production_final_sha256=sha(root/'final.json'),style_diagnosis_sha256=sha(root/'style_shift_diagnosis.json'),prior_complete_input_audit_sha256=sha(DENSE/'independent_input_audit.json'),result_sha256={s:sha(root/'independent_results'/(s+'.json')) for s in ('S1','S2')},exact_shifts=2,exact_new_methods=4,exact_GPU_jobs=8,all_fixed_rates=4,target_train_GT_read=False))
