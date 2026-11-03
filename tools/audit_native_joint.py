"""原生SGP联合背景学习：全teacher多尺度与源标签、warmup/优化器/完整AP独审。"""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_native_joint import DENSE,DIAG,SOURCE,MODES,TASKS
LAYER=Path('/vepfs/vlms/highscore_layer_signals_bdcc520_1103')


def teacher_audit(root,s):
    import torch
    report=read(root/s/'complete.json');assert sha(root/s/'teacher.pt')==report['teacher_sha256'];guard=read(report['guard']);lease=read(root/s/'lease_extract.json')
    assert sha(report['guard'])==report['guard_sha256'] and sha(root/s/'lease_extract.json')==guard['lease_sha256'];assert guard['owner_pid']==lease['owner_pid']==read(root/'state.json')['owner_pid'];assert guard['observed']['uuid']==lease['uuid'] and guard['observed']['free_mib']>=12288
    teacher=torch.load(root/s/'teacher.pt',map_location='cpu');data=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu');old=torch.load(DIAG/s/'inputs.pt',map_location='cpu');assert teacher['input_sha256']==sha(DENSE/'inputs'/s/'data.pt');assert len(teacher['rows'])==len(old['fit'])
    bounds={r['video']:r for r in read(LAYER/(s+'_source')/'complete.json')['records']};truth=read(SOURCE/s/'source_truth.json');maxerr=0.;anchors=0
    for a,b,r in zip(teacher['rows'],old['fit'],data['roles']['fit']):
        assert a['video']==b['video']==r['video'];assert torch.equal(a['seq'][0],b['seq']) and torch.equal(a['valid'][0],b['valid'])
        for k in ('metadata','bg','quality'):assert torch.equal(a[k],b[k])
        assert torch.equal(a['delta'],r['delta']);anchors+=int(((a['bg']==0)&(a['delta']<0)).sum())
        rec=bounds[a['video']];assert sha(rec['path'])==rec['sha256'];bound=torch.load(rec['path'],map_location='cpu');v=a['valid'][:,:,4:20];x=a['seq'][:,:,4:20];mean=((x*v[...,None]).sum(2)/v.sum(2,keepdim=True).clamp_min(1)).permute(1,0,2)
        reference=bound['all_FPN_inner'];err=float((mean-reference).abs().max());maxerr=max(maxerr,err);assert torch.allclose(mean,reference,atol=2e-5,rtol=1e-6)
        gt=truth[a['video']];boxes=np.array(gt['segments'],float).reshape(-1,2);labels=np.array(gt['labels']);query=r['grid_boxes'][r['hard']].numpy().astype(float)
        inter=np.maximum(0,np.minimum(query[:,None,1],boxes[None,:,1])-np.maximum(query[:,None,0],boxes[None,:,0]));union=(query[:,1]-query[:,0])[:,None]+(boxes[:,1]-boxes[:,0])[None]-inter;iou=inter/np.maximum(union,1e-12)
        expected=iou.max(1)<.01;assert np.array_equal(expected,a['bg'].numpy().astype(bool));q=np.array([iou[i,labels==int(c)].max() if (labels==int(c)).any() else 0 for i,c in enumerate(r['labels'][r['hard']])]);assert np.allclose(q,a['quality'].numpy(),atol=3e-6,rtol=0)
    return dict(teacher_sha256=report['teacher_sha256'],source_videos=len(teacher['rows']),critical_nonBG_anchors=anchors,FPN0_all_samples_bitwise_equal=True,all_pyramid_inner_means_compared_to_prior_audited_native=True,max_pyramid_mean_error=maxerr,all_BG_quality_independent=True)


def states_audit(root,s,rows):
    import torch
    from tools import baseline_engine as base
    from tools.run_foreground_calibration_oracle import REFERENCE
    teacher=read(root/s/'complete.json');original=torch.load(REFERENCE/s/'best.pt',map_location='cpu')['ema_state_dict'];result={}
    for mode in MODES:
        e=root/(s+'_'+mode);gpu=read(e/'gpu_complete.json');assert sha(e/'gpu_complete.json')==read(root/'final.json')['allGPU_sha256'][s+'_'+mode]
        for file,key in [('kernel_complete.json','kernel_sha256'),('epoch005.pt','checkpoint_sha256'),('predictions.pt','predictions_sha256')]:assert sha(e/file)==gpu[key]
        kernel=read(e/'kernel_complete.json');assert kernel['passed'] and kernel['cached_vs_native_loss_allgrad_update_exact'] and kernel['complete_model_optimizer_RNG_nextstep_exact'];assert sha(e/'kernel.pt')==kernel['kernel_sha256'];assert kernel['real_joint_backbone_update']==(mode=='joint')
        guard=read(gpu['guard']);lease=read(e/'lease_train.json');assert sha(gpu['guard'])==gpu['guard_sha256']==kernel['guard_sha256'];assert sha(e/'lease_train.json')==guard['lease_sha256'];assert guard['owner_pid']==lease['owner_pid']==read(root/'state.json')['owner_pid'];assert guard['observed']['uuid']==lease['uuid'] and guard['observed']['free_mib']>=12288
        history=read(e/'history.json');assert [r['epoch'] for r in history]==list(range(1,6));order=torch.Generator().manual_seed(123);changed={}
        for epoch,row in enumerate(history,1):
            torch.randperm(len(rows),generator=order);assert row['joint_backbone']==(mode=='joint' and epoch>1)
            if mode=='frozen' or epoch==1:assert row['anchor']==0
            if epoch not in (1,3,5):continue
            state=torch.load(e/f'epoch{epoch:03d}.pt',map_location='cpu');assert state['history']==history[:epoch] and state['epoch']==epoch and state['step']==epoch*len(rows);assert torch.equal(order.get_state(),state['order_rng'])
            assert state['teacher_sha256']==teacher['teacher_sha256'];assert state['native_checkpoint_sha256']==teacher['native_checkpoint_sha256'];assert state['input_sha256']==sha(DENSE/'inputs'/s/'data.pt')
            assert all(k in state for k in ('python_rng','numpy_rng','cpu_rng','cuda_rng'));assert state['cpu_rng'].dtype==state['cuda_rng'].dtype==torch.uint8
            body=set(state['optimizer']['param_groups'][0]['params']);head=set(state['optimizer']['param_groups'][1]['params']);assert not body&head
            for i,v in state['optimizer']['state'].items():
                if i in body:assert mode=='joint' and int(v['step'])==(epoch-1)*len(rows)
                else:assert i in head and int(v['step'])==epoch*len(rows)
            count=0
            for name,value in state['model'].items():
                if name.startswith('backbone.'):ref=original['backbone_source.'+name[len('backbone.'):]]
                elif name.startswith('neck.'):ref=original['neck_source.'+name[len('neck.'):]]
                else:continue
                same=torch.equal(value,ref);count+=not same
                if mode=='frozen' or epoch==1:assert same
            if mode=='joint' and epoch>1:assert count>0
            changed[str(epoch)]=count
            if epoch==5:assert base.tensor_digest(state['model'])==gpu['final_digest']
        assert os.stat(e/'epoch005.pt').st_ino==os.stat(e/'latest.pt').st_ino;assert gpu['backbone_neck_changed']==(mode=='joint') and gpu['saved_model_GPU_predictions_bitwise_equal']
        result[mode]=dict(GPU_sha256=sha(e/'gpu_complete.json'),checkpoint_sha256=gpu['checkpoint_sha256'],kernel_sha256=gpu['kernel_sha256'],changed_body_tensors_by_epoch=changed,all_step_counts_and_warmup_exact=True,original_teacher_weights_preserved_when_frozen=True)
        print('NATIVE_STATES_AUDITED',s,mode,flush=True)
    return result


def result_audit(root,s):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import prediction_frame
    from tools.audit_background_ap_cost import IndependentAP
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    roles=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu')['roles'];teachers=teacher_audit(root,s);states=states_audit(root,s,roles['fit']);outputs={m:torch.load(root/(s+'_'+m)/'predictions.pt',map_location='cpu') for m in MODES};cfg=read(SOURCE/s/'config.json');gt=read(SOURCE/s/'source_truth.json')
    truth={role:pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c)) for r in roles[role] for b,c in zip(gt[r['video']]['segments'],gt[r['video']]['labels'])],columns=['video-id','t-start','t-end','label']) for role in ('fit','heldout')};truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);truth['target_val']=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split']);report=read(root/(s+'_evaluation.json'));results={}
    for role in ('fit','heldout','validation','target_val'):
        frames={};eligible=None
        for m,out in outputs.items():
            assert len(out[role])==len(roles[role])
            for a,b in zip(out[role],roles[role]):
                assert a['video']==b['video']
                for k in ('segments','scores','labels'):assert torch.equal(a[k],b[k])
                assert (a['pbg'][b['scores']<.3]==0).all() and not a['eligible'][b['scores']<.3].any()
            frames[m]=prediction_frame(out[role]);v=torch.cat([r['eligible'] for r in out[role]]).numpy()
            if eligible is None:eligible=v
            else:assert np.array_equal(v,eligible)
        frame=frames['frozen'];bg=np.ones(len(frame),bool)
        for vid,ids in frame.groupby('video-id',sort=False).indices.items():
            ids=np.asarray(ids);a=frame.iloc[ids][['t-start','t-end']].to_numpy(float);b=truth[role][truth[role]['video-id']==vid][['t-start','t-end']].to_numpy(float)
            if len(b):
                inter=np.maximum(0,np.minimum(a[:,None,1],b[None,:,1])-np.maximum(a[:,None,0],b[None,:,0]));union=(a[:,1]-a[:,0])[:,None]+(b[:,1]-b[:,0])[None]-inter;bg[ids]=(inter/np.maximum(union,1e-12)).max(1)<.01
        pos={};engines={};baseline=[]
        for c in range(10):
            ids=np.flatnonzero(frame.label.to_numpy()==c);pos[c]=ids;engines[c]=IndependentAP(truth[role][truth[role].label==c],frame.iloc[ids].reset_index(drop=True));baseline.append(engines[c].ap(np.ones(len(ids),bool)))
        def effect(chosen):
            keep=np.ones(len(frame),bool);keep[chosen]=False;a=[engines[c].ap(keep[pos[c]]) for c in range(10)];return a,float(100*(np.mean(a)-np.mean(baseline)))
        results[role]={}
        for m in (*MODES,'confidence'):
            f=frame if m=='confidence' else frames[m];old=report['results'][role][m];assert np.allclose(baseline,old['original_per_class_AP'],atol=1e-7,rtol=0);score=1-f.score.to_numpy() if m=='confidence' else f.pbg.to_numpy();auc=[];budgets=[]
            for _,ids in f.groupby('video-id',sort=False).indices.items():
                ids=np.asarray(ids);ids=ids[f.iloc[ids].score.to_numpy()>=.3];a=independent_auc(bg[ids],score[ids])
                if a is not None:auc.append(a)
            assert len(auc)==old['AUC_valid_videos'] and abs(np.mean(auc)-old['video_macro_AUC'])<1e-10
            for op in old['budgets']:
                chosen=[]
                for c in range(10):
                    pool=np.flatnonzero((f.label.to_numpy()==c)&(f.score.to_numpy()>=.3));n=int(len(pool)*op['rate']);allowed=pool[eligible[pool]];key=f.score.to_numpy() if m=='confidence' else -f.pbg.to_numpy();chosen+=sorted(allowed.tolist(),key=lambda i:(float(key[i]),i))[:n]
                chosen=np.asarray(sorted(chosen),int);assert chosen.tolist()==op['selected'];ap,delta=effect(chosen);_,gain=effect(chosen[bg[chosen]]);_,harm=effect(chosen[~bg[chosen]])
                assert np.allclose(ap,op['per_class_AP'],atol=1e-7,rtol=0)
                for a,b in ((delta,op['delta_pp']),(gain,op['BG_only_delta_pp']),(harm,op['nonBG_only_delta_pp'])):assert abs(a-b)<1e-7
                assert int(bg[chosen].sum())==op['BG_removed'] and int((~bg[chosen]).sum())==op['nonBG_removed'];budgets.append(op)
            results[role][m]=dict(baseline=float(np.mean(baseline)),video_macro_AUC=float(np.mean(auc)),valid_videos=len(auc),budgets=budgets)
        if role=='target_val':assert abs(np.mean(baseline)-read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/s/'evaluation'/'target_val.json')['map_0.1_to_0.5'])<1e-7
        print('NATIVE_AP_AUDITED',s,role,flush=True)
    folder=root/'independent_results';folder.mkdir(exist_ok=True);save(folder/(s+'.json'),dict(passed=True,teacher=teachers,states=states,results=results,report_sha256=sha(root/(s+'_evaluation.json')),all_original_arrays_and_fixed_AP_independent=True,target_train_GT_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    while not (root/'final.json').exists():
        if (root/'error.json').exists():raise RuntimeError('producer failed; no automatic restart')
        time.sleep(3)
    final=read(root/'final.json');assert set(final['allGPU_sha256'])==set(TASKS)
    for task,digest in final['allGPU_sha256'].items():assert sha(root/task/'gpu_complete.json')==digest
    for s in ('S1','S2'):result_audit(root,s)
    save(root/'independent_final_audit.json',dict(passed=True,audit_code=code_identity(),production_final_sha256=sha(root/'final.json'),results_sha256={s:sha(root/'independent_results'/(s+'.json')) for s in ('S1','S2')},exact_shifts=2,exact_methods=2,exact_GPU_training_jobs=4,exact_rates=4,all_teacher_pyramid_means_compared_to_prior_audited_native=True,all_checkpoint_warmup_step_counts_and_original_arrays_checked=True,target_train_GT_read=False))
