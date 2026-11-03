"""完整GT bank身份、全raw bank描述符、每视频固定3查询及完整AP独立审计。"""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_semantic_memory import DENSE,TASKS


def descriptor_numpy(values,boxes):
    t=(np.arange(16)+.5)/16;g=boxes[:,0,None]+(boxes[:,1]-boxes[:,0])[:,None]*t
    lo=np.floor(g).astype(int);hi=lo+1;valid=(lo>=0)&(hi<len(values));a=g-lo
    x=(values[lo.clip(0,len(values)-1)]*(1-a[...,None])+values[hi.clip(0,len(values)-1)]*a[...,None])*valid[...,None];bins=[]
    for k in (0,4,8,12):
        z=x[:,k:k+4].sum(1)/np.maximum(1,valid[:,k:k+4].sum(1,keepdims=True));z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12);bins.append(z)
    z=np.concatenate(bins,1);return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12),valid.sum(1)>=4


def bank_boxes(r):
    b=np.array(r['annotations']['segments'],np.float32).reshape(-1,2);c=np.array(r['annotations']['labels'],int);tasks=b[c<10];keep=c<10
    if len(tasks):
        inter=np.maximum(0,np.minimum(b[:,None,1],tasks[None,:,1])-np.maximum(b[:,None,0],tasks[None,:,0]));union=(b[:,1]-b[:,0])[:,None]+(tasks[:,1]-tasks[:,0])[None]-inter;keep|=(c>=10)&((inter/np.maximum(union,1e-12)).max(1)<.01)
    else:keep|=c>=10
    pure=r['grid_boxes'][r['hard']][r['bg'].bool()].numpy();return np.concatenate((b[keep],pure)),np.concatenate((c[keep],np.full(len(pure),97,int)))


def distances_numpy(query,bank,labels,video,qvideo):
    d=np.maximum(0,(query*query).sum(1)[:,None]+(bank*bank).sum(1)[None]-2*query@bank.T);columns=[]
    for c in range(98):
        ids=np.flatnonzero((labels==c)&(video!=qvideo))
        columns.append(np.sort(d[:,ids],axis=1)[:,:min(5,len(ids))].mean(1) if len(ids) else np.full(len(query),np.inf))
    z=np.stack(columns,1);fg=z[:,:10].min(1);return np.stack((fg,fg-z[:,10:].min(1)),1)


def audit(root,s):
    import torch,pandas as pd
    from tools.run_dense_revisit import SOURCE
    from tools.foreground_oracle_ap import prediction_frame
    from tools.audit_background_ap_cost import IndependentAP
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    data=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu');roles=data['roles'];model_outputs={};records={};raw_max=0.;replay_max=0.;query_count=0
    for kind in ('raw','learned'):
        entry=root/(s+'_'+kind);gpu=read(entry/'gpu_complete.json');assert gpu['readonly_parameters_RNG_unchanged'] and gpu['zero_optimizer_steps']
        for path,key in [('bank.pt','bank_sha256'),('predictions.pt','predictions_sha256'),('kernel_complete.json','kernel_sha256')]:assert sha(entry/path)==gpu[key]
        kernel=read(entry/'kernel_complete.json');assert kernel['passed'] and kernel['self_video_excluded'] and kernel['exact_float64_GPU_and_independent_NumPy_fullbank3queries'];assert kernel['bank_sha256']==gpu['bank_sha256']
        guard=read(gpu['guard']);lease=read(entry/'lease.json');assert sha(gpu['guard'])==gpu['guard_sha256'] and sha(entry/'lease.json')==guard['lease_sha256'];assert guard['owner_pid']==lease['owner_pid']==read(root/'state.json')['owner_pid'];assert guard['observed']['uuid']==lease['uuid'] and guard['observed']['free_mib']>=12288
        bank=torch.load(entry/'bank.pt',map_location='cpu');f=bank['features'].double().numpy();cl=bank['labels'].numpy();vid=bank['video'].numpy();cursor=0;out=torch.load(entry/'predictions.pt',map_location='cpu')
        for r in roles['fit']:
            b,c=bank_boxes(r)
            # 有效采样数只取决于原始时间长度；无重新采样或GT选查询。
            raw=np.load(r['raw_path'])['feats'].astype(np.float32)[::r['downsample']];values=(raw-data['mean'].numpy())/data['std'].numpy();z,valid=descriptor_numpy(values,b);n=int(valid.sum());expected=c[valid]
            assert np.array_equal(cl[cursor:cursor+n],expected) and (vid[cursor:cursor+n]==bank['video_ids'][r['video']]).all()
            if kind=='raw':
                err=float(np.max(np.abs(z[valid]-f[cursor:cursor+n]))) if n else 0.;raw_max=max(raw_max,err);assert err<3e-6
            cursor+=n
        assert cursor==len(f)
        if kind=='raw':
            for role in ('fit','heldout','validation','target_val'):
                for i,r in enumerate(roles[role]):
                    hard=np.flatnonzero(r['hard'].numpy())
                    if not len(hard):continue
                    ids=np.unique([0,len(hard)//2,len(hard)-1]);chosen=hard[ids];raw=np.load(r['raw_path'])['feats'].astype(np.float32)[::r['downsample']];values=(raw-data['mean'].numpy())/data['std'].numpy()
                    q,valid=descriptor_numpy(values,r['grid_boxes'][chosen].numpy());value=distances_numpy(q,f,cl,vid,bank['video_ids'][r['video']]);query_count+=len(chosen)
                    for col,m in enumerate(('foreground','dual')):
                        expected=out[m][role][i]['pbg'][chosen].numpy();err=np.abs(value[valid,col]-expected[valid]).max() if valid.any() else 0.;replay_max=max(replay_max,float(err));assert err<3e-6
        for m,r in out.items():model_outputs[kind+'_'+m]=r
        records[kind]=dict(GPU_sha256=sha(entry/'gpu_complete.json'),bank_sha256=sha(entry/'bank.pt'),all_bank_labels_video_members_independent_equal=True,full_raw_descriptors_checked=kind=='raw',learned_encoder_native_replay=kind=='learned' and gpu['all_learned_original_dense_predictions_bitwise_replayed'])
    cfg=read(SOURCE/s/'config.json');gt=read(SOURCE/s/'source_truth.json');truth={role:pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c)) for r in roles[role] for b,c in zip(gt[r['video']]['segments'],gt[r['video']]['labels'])],columns=['video-id','t-start','t-end','label']) for role in ('fit','heldout')}
    truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);truth['target_val']=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split']);report=read(root/(s+'_evaluation.json'));result={}
    for role in ('fit','heldout','validation','target_val'):
        frames={};eligible=None
        for m,outputs in model_outputs.items():
            rows=outputs[role];assert len(rows)==len(roles[role])
            for a,b in zip(rows,roles[role]):
                assert a['video']==b['video']
                for k in ('segments','scores','labels'):assert torch.equal(a[k],b[k])
                assert (a['pbg'][b['scores']<.3]==0).all()
            frames[m]=prediction_frame(rows);v=torch.cat([r['eligible'] for r in rows]).numpy()
            if eligible is None:eligible=v
            else:assert np.array_equal(eligible,v)
        frame=next(iter(frames.values()));bg=np.ones(len(frame),bool)
        for v,ids in frame.groupby('video-id',sort=False).indices.items():
            ids=np.asarray(ids);a=frame.iloc[ids][['t-start','t-end']].to_numpy(float);b=truth[role][truth[role]['video-id']==v][['t-start','t-end']].to_numpy(float)
            if len(b):
                inter=np.maximum(0,np.minimum(a[:,None,1],b[None,:,1])-np.maximum(a[:,None,0],b[None,:,0]));union=(a[:,1]-a[:,0])[:,None]+(b[:,1]-b[:,0])[None]-inter;bg[ids]=(inter/np.maximum(union,1e-12)).max(1)<.01
        engines={};pos={};baseline=[]
        for c in range(10):
            ids=np.flatnonzero(frame.label.to_numpy()==c);pos[c]=ids;engines[c]=IndependentAP(truth[role][truth[role].label==c],frame.iloc[ids].reset_index(drop=True));baseline.append(engines[c].ap(np.ones(len(ids),bool)))
        def effect(selected):
            keep=np.ones(len(frame),bool);keep[selected]=False;return float(100*(np.mean([engines[c].ap(keep[pos[c]]) for c in range(10)])-np.mean(baseline)))
        result[role]={}
        for m in (*frames,'confidence'):
            f=frame if m=='confidence' else frames[m];old=report['results'][role][m];assert abs(np.mean(baseline)-old['baseline'])<1e-7;auc=[];value=1-f.score.to_numpy() if m=='confidence' else f.pbg.to_numpy();budgets=[]
            for _,ids in f.groupby('video-id',sort=False).indices.items():
                ids=np.asarray(ids);ids=ids[f.iloc[ids].score.to_numpy()>=.3];a=independent_auc(bg[ids],value[ids])
                if a is not None:auc.append(a)
            assert len(auc)==old['AUC_valid_videos'] and abs(np.mean(auc)-old['video_macro_AUC'])<1e-10
            for op in old['budgets']:
                chosen=[]
                for c in range(10):
                    pool=np.flatnonzero((f.label.to_numpy()==c)&(f.score.to_numpy()>=.3));n=int(len(pool)*op['rate']);available=pool[eligible[pool]];key=f.score.to_numpy() if m=='confidence' else -f.pbg.to_numpy();chosen+=sorted(available.tolist(),key=lambda i:(float(key[i]),i))[:n]
                chosen=np.array(sorted(chosen),int);assert chosen.tolist()==op['selected'];a=effect(chosen);b=effect(chosen[bg[chosen]]);c=effect(chosen[~bg[chosen]])
                for x,y in ((a,op['delta_pp']),(b,op['BG_only_delta_pp']),(c,op['nonBG_only_delta_pp'])):assert abs(x-y)<1e-7
                assert int(bg[chosen].sum())==op['BG_removed'];assert len(chosen)-int(bg[chosen].sum())==op['nonBG_removed'];budgets.append(op)
            result[role][m]=dict(video_macro_AUC=float(np.mean(auc)),valid_videos=len(auc),budgets=budgets)
        if role=='target_val':assert abs(np.mean(baseline)-read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/s/'evaluation'/'target_val.json')['map_0.1_to_0.5'])<1e-7
        print('MEMORY_AUDITED',s,role,flush=True)
    folder=root/'independent_results';folder.mkdir(exist_ok=True);save(folder/(s+'.json'),dict(passed=True,results=result,banks=records,raw_descriptor_full_bank_max_error=raw_max,raw_fixed3queries_pervideo_checked=query_count,raw_fixed_query_score_max_error=replay_max,report_sha256=sha(root/(s+'_evaluation.json')),subset_effects_not_additive=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    while not (root/'final.json').exists():
        if (root/'error.json').exists():raise RuntimeError('producer failed; preserve artifacts, no autorestart')
        time.sleep(3)
    final=read(root/'final.json')
    for t,digest in final['allGPU_sha256'].items():assert sha(root/t/'gpu_complete.json')==digest
    for s in ('S1','S2'):audit(root,s)
    save(root/'independent_final_audit.json',dict(passed=True,audit_code=code_identity(),producer_final_sha256=sha(root/'final.json'),result_sha256={s:sha(root/'independent_results'/(s+'.json')) for s in ('S1','S2')},raw_full_bank_descriptor_checked=True,raw_query_scores_scope='fixed first/middle/last3 per physical video, not allqueries',learned_bank_scope='full labels/video identity, producer native encoder replay and complete GPU/NumPy3-query kernel, not independent alllearned descriptors',all_AP_budgets_counterfactuals_independent=True,target_train_GT_read=False))
