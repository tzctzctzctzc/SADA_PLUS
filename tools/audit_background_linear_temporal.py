"""独立NumPy描述符、保存头重放、全源AP审计；不读目标、不选预算。"""
import sys
import time
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.audit_background_ap_cost import IndependentAP,make_frame,background,selected_indices
from tools.audit_background_fit_gap import independent_auc
from libs.evaluation.evaluator_anet import load_gt_seg_from_json

ORIGIN=Path('/vepfs/vlms/background_ap_cost_source6_bd16507_1103')
SOURCE=Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
CURVE=Path('/vepfs/vlms/background_holdout_curve6_837d6fb_1103')
MODES=('max_linear','temporal_abs','temporal_signed')
RATES=(.005,.01,.02,.05)


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,sort_keys=True));t.replace(p)


def numpy_descriptors(feature,valid,boxes,labels,scores):
    """独立重建原坐标/有效插值/统计量；没有导入生产描述符函数。"""
    f=np.asarray(feature);b=np.asarray(boxes);length=b[:,1]-b[:,0];samples=[];masks=[];maxima=[];means=[];coverage=[]
    for left,right,n in ((b[:,0]-.25*length,b[:,0],4),(b[:,0],b[:,1],16),(b[:,1],b[:,1]+.25*length,4)):
        fraction=(np.arange(n,dtype=np.float32)+np.float32(.5))/np.float32(n)
        position=left[:,None]+(right-left)[:,None]*fraction
        floor=np.floor(position).astype(np.int64);ceil=np.ceil(position).astype(np.int64)
        weight=position-floor.astype(np.float32);lo=np.clip(floor,0,f.shape[1]-1);hi=np.clip(ceil,0,f.shape[1]-1)
        mask=(position>=0)&(position<=f.shape[1]-1)&valid[lo]&valid[hi]
        values=(f[:,lo]*(1-weight)[None]+f[:,hi]*weight[None]).transpose(1,2,0)
        maximum=np.max(np.where(mask[:,:,None],values,-np.inf),axis=1)
        maximum=np.where(mask.any(1)[:,None],maximum,0)
        mean=(values*mask[:,:,None]).sum(1)/np.maximum(mask.sum(1,keepdims=True),1)
        samples.append(values);masks.append(mask);maxima.append(maximum);means.append(mean);coverage.append(mask.mean(1,keepdims=True))
    early_mask,late_mask=masks[1][:,:8],masks[1][:,8:]
    early=(samples[1][:,:8]*early_mask[:,:,None]).sum(1)/np.maximum(early_mask.sum(1,keepdims=True),1)
    late=(samples[1][:,8:]*late_mask[:,:,None]).sum(1)/np.maximum(late_mask.sum(1,keepdims=True),1)
    pair=early_mask.any(1)&late_mask.any(1);difference=(late-early)*pair[:,None]
    p=np.clip(scores,1e-5,1-1e-5)
    meta=np.concatenate((np.eye(10,dtype=np.float32)[labels],(np.log(p/(1-p))/5)[:,None],np.log1p(length/max(1,f.shape[1]))[:,None]),axis=1)
    l,i,r=maxima;max_rep=np.concatenate((i-l,i,i-r,*coverage,meta),axis=1)
    l,i,r=means
    def temporal(d):return np.concatenate((i-l,i,i-r,d,*coverage,pair[:,None],meta),axis=1)
    return dict(max_linear=max_rep,temporal_abs=temporal(np.abs(difference)),temporal_signed=temporal(difference))


def audit(root,s):
    entry=root/s;done=read(entry/'complete.json');gpu=read(entry/'gpu_complete.json');split=read(entry/'split.json');launch=read(root/'_launch.json')
    assert done['passed'] and gpu['passed'] and sha(entry/'gpu_complete.json')==done['gpu_complete_sha256']
    assert sha(entry/'split.json')==done['split_sha256']==gpu['split_sha256']
    assert gpu['spec']['schema']=='background-linear-temporal-source-v1' and not gpu['spec']['target_read']
    assert sha(CURVE/'independent_final_audit.json')==gpu['curve_audit_sha256'] and read(CURVE/'independent_final_audit.json')['passed']
    assert sha(CURVE/s/'split.json')==split['old_split_sha256']
    cost=read(ORIGIN/s/'costs.json');assert sha(ORIGIN/s/'costs.json')==gpu['origin_costs_sha256']==split['origin_costs_sha256']
    ids=sorted(r['record']['video'] for r in cost['rows']);ranked=sorted(ids,key=lambda v:(hashlib.sha256(('head-holdout-v1|'+v).encode()).hexdigest(),v))
    assert split['heldout']==sorted(ranked[:(len(ids)+4)//5]) and split['fit']==sorted(set(ids)-set(split['heldout']))
    assert not set(ids)&{r['video'] for r in cost['validation']}
    assert gpu['probe_video']==next(r['record']['video'] for r in cost['rows'] if r['record']['video'] in split['fit'] and r['hard_indices'])
    assert sha(entry/'kernel_complete.json')==gpu['kernel_sha256'];kernel=read(entry/'kernel_complete.json');assert kernel['passed'] and kernel['timed_updates_exact']
    benchmark=read(entry/'aggregate_benchmark.json');assert benchmark['numerically_exact'] and benchmark['cache_faster'] and benchmark['repeats']==30
    assert kernel['cached_three_head_step_seconds']==benchmark['cached_seconds']<benchmark['shared_direct_seconds']==kernel['shared_direct_three_head_step_seconds']
    assert sha(entry/'kernel_descriptors.pt')==kernel['descriptor_sha256']
    for m in MODES:assert sha(entry/(m+'_kernel.pt'))==kernel['reports'][m]['checkpoint_sha256']
    for report in (kernel,gpu):
        assert sha(report['guard'])==report['guard_sha256'];g=read(report['guard']);lease=read(g['lease'])
        assert sha(g['lease'])==g['lease_sha256'] and g['owner_pid']==lease['owner_pid']==launch['pid']
        assert lease['commit']==launch['head'] and g['observed']['uuid']==lease['uuid'] and g['observed']['index']==lease['index'] and g['observed']['free_mib']>=12288
    assert gpu['readout_parameter_and_RNG_unchanged'] and gpu['parameters']==dict(max_linear=3104,temporal_abs=4130,temporal_signed=4130)
    history=read(entry/'history.json');assert sha(entry/'history.json')==gpu['history_sha256'] and [h['epoch'] for h in history]==list(range(1,21))
    checkpoints={}
    for epoch in (1,5,10,20):
        path=entry/f'epoch{epoch:03d}.pt';state=torch.load(path,map_location='cpu')
        assert {'heads','optimizers','history','order_rng','python_rng','numpy_rng','cpu_rng','cuda_rng','split_sha256','spec'}<=set(state)
        assert state['history']==history[:epoch] and state['split_sha256']==sha(entry/'split.json') and json.loads(json.dumps(state['spec']))==gpu['spec']
        assert set(state['heads'])==set(MODES)==set(state['optimizers'])
        for m,opt in state['optimizers'].items():
            assert sum(t.numel() for t in state['heads'][m].values())==gpu['parameters'][m]
            assert all(float(v['step'])==epoch*gpu['fit_steps_per_epoch'] for v in opt['state'].values()) and opt['state']
        checkpoints[str(epoch)]=sha(path)
    assert checkpoints['20']==gpu['saved']['20']['checkpoint_sha256']
    assert (entry/'latest.pt').stat().st_ino==(entry/'epoch020.pt').stat().st_ino
    weights=state['heads'];prediction=entry/'predictions020.pt';assert sha(prediction)==gpu['saved']['20']['predictions_sha256']
    predictions=torch.load(prediction,map_location='cpu');assert set(predictions)=={'fit','heldout','validation'}
    assert sha(entry/'descriptors.pt')==gpu['descriptors_sha256'];prepared=torch.load(entry/'descriptors.pt',map_location='cpu')
    assert set(prepared)==set(predictions)
    originals={r:{} for r in prepared};descriptors_checked=0;descriptor_error=0.;readout_error=0.
    source_rows={r['record']['video']:r for r in cost['rows']}
    for role,records in [('source',[r['record'] for r in cost['rows']]),('validation',cost['validation'])]:
        for rec in records:
            v=rec['video'];dest=role if role=='validation' else ('fit' if v in split['fit'] else 'heldout')
            raw=torch.load(rec['path'],map_location='cpu');hard=raw['scores']>=.3
            indexed={r['video']:r for r in prepared[dest]};assert len(indexed)==len(prepared[dest]);p=indexed[v]
            for k in ('scores','labels'):assert torch.equal(raw[k],p[k])
            assert torch.equal(raw['segments' if role=='validation' else 'boxes'],p['segments']) and torch.equal(hard,p['hard'])
            if role=='source':
                row=source_rows[v];assert torch.where(hard)[0].tolist()==row['hard_indices'] and torch.equal(p['bg'],torch.tensor(row['bg'])) and torch.equal(p['quality'],torch.tensor(row['quality']))
            reps=numpy_descriptors(raw['feature'].numpy(),raw['valid'].numpy(),raw['boxes'][hard].numpy(),raw['labels'][hard].numpy(),raw['scores'][hard].numpy())
            for m in MODES:
                saved=p['descriptors'][m].numpy();assert saved.shape==reps[m].shape
                if saved.size:descriptor_error=max(descriptor_error,float(np.abs(saved-reps[m]).max()))
                assert np.allclose(saved,reps[m],atol=2e-5,rtol=1e-6)
            originals[dest][v]=p;descriptors_checked+=1
    assert descriptors_checked==len(cost['rows'])+len(cost['validation'])
    assert set(originals['fit'])==set(split['fit']) and set(originals['heldout'])==set(split['heldout'])
    gtrows=read(SOURCE/s/'source_truth.json');assert sha(SOURCE/s/'source_truth.json')==cost['source_truth_sha256']
    cfg=read(SOURCE/s/'config.json');valgt=load_gt_seg_from_json(cfg['dataset']['json_file'],cfg['src_val_split'],action_subsampling='10_majority_classes');valgt=valgt[valgt['t-start']!=valgt['t-end']].reset_index(drop=True)
    truth={role:pd.DataFrame([(v,float(b[0]),float(b[1]),int(c)) for v in split[role] for b,c in zip(gtrows[v]['segments'],gtrows[v]['labels'])],columns=['video-id','t-start','t-end','label']) for role in ('fit','heldout')};truth['validation']=valgt
    reports={}
    for role,modes in predictions.items():
        assert set(modes)==set(MODES)
        for mode,rows in modes.items():
            assert {r['video'] for r in rows}==set(originals[role]) and len(rows)==len(originals[role])
            w=weights[mode]['classifier.weight'].numpy();bias=weights[mode]['classifier.bias'].numpy()
            for row in rows:
                p=originals[role][row['video']]
                for k in ('scores','labels','segments'):assert torch.equal(row[k],p[k])
                assert (row['pbg'][~p['hard']]==0).all() and torch.isfinite(row['pbg']).all()
                logits=p['descriptors'][mode].numpy()@w.T+bias
                replay=1/(1+np.exp(-logits[:,0].astype(np.float64)))
                actual=row['pbg'][p['hard']].numpy()
                if len(actual):readout_error=max(readout_error,float(np.abs(actual-replay).max()))
                assert np.allclose(actual,replay,atol=3e-6,rtol=1e-6)
        frames={m:make_frame(rows) for m,rows in modes.items()};f=frames[MODES[0]]
        assert all(f.drop(columns='pbg').equals(frame.drop(columns='pbg')) for frame in frames.values())
        bg=background(truth[role],f);engines={};positions={};baseline=[]
        for c in range(10):
            idx=np.flatnonzero(f.label.to_numpy()==c);positions[c]=idx;engines[c]=IndependentAP(truth[role][truth[role].label==c],f.iloc[idx].reset_index(drop=True));baseline.append(engines[c].ap(np.ones(len(idx),dtype=bool)).tolist())
        if role=='validation':assert np.allclose(baseline,read(ORIGIN/'independent_results'/f'{s}.json')['original_per_class_AP'],atol=1e-7,rtol=0)
        reports[role]={}
        for mode in (*MODES,'confidence'):
            frame=frames[MODES[0] if mode=='confidence' else mode];reported=done['results']['20'][role][mode];aucs=[]
            for v,idx in frame.groupby('video-id',sort=False).indices.items():
                idx=np.asarray(idx);idx=idx[frame.iloc[idx].score.to_numpy()>=.3]
                a=independent_auc(bg[idx],1-frame.iloc[idx].score.to_numpy() if mode=='confidence' else frame.iloc[idx].pbg.to_numpy())
                if a is not None:aucs.append(a)
            assert len(aucs)==reported['AUC_valid_videos'] and abs(np.mean(aucs)-reported['video_macro_AUC'])<1e-12
            assert abs(np.mean(baseline)-reported['baseline'])<1e-7
            budgets=[]
            for rate,rep in zip(RATES,reported['all_fixed_budgets']):
                assert rate==rep['rate'];selected=selected_indices(frame,rate,mode=='confidence');keep=np.ones(len(frame),dtype=bool);keep[selected]=False
                ap=[engines[c].ap(keep[positions[c]]).tolist() for c in range(10)]
                assert abs(np.mean(ap)-rep['map'])<1e-7 and abs(100*(np.mean(ap)-np.mean(baseline))-rep['delta_pp'])<1e-7
                assert int(bg[selected].sum())==rep['BG_removed'] and int((~bg[selected]).sum())==rep['nonBG_removed']
                budgets.append(dict(rate=rate,delta_pp=rep['delta_pp'],per_class_AP=ap,BG_removed=rep['BG_removed'],nonBG_removed=rep['nonBG_removed']))
            reports[role][mode]=dict(video_macro_AUC=reported['video_macro_AUC'],AUC_valid_videos=len(aucs),budgets=budgets)
    out=dict(passed=True,complete_sha256=sha(entry/'complete.json'),gpu_complete_sha256=sha(entry/'gpu_complete.json'),split_sha256=sha(entry/'split.json'),checkpoint_sha256=checkpoints,descriptor_sha256=sha(entry/'descriptors.pt'),all_video_descriptors_independently_rebuilt=descriptors_checked,max_descriptor_abs_error=descriptor_error,all_saved_predictions_numpy_linear_replay=True,max_readout_abs_error=readout_error,all_original_arrays_exact=True,all_checkpoint_state_and_history=True,results=reports)
    save(root/'independent_results'/f'{s}.json',out);print('AUDITED',s,flush=True);return out


def main(root):
    results={}
    for i in range(1,7):
        s=f'S{i}'
        while not (root/s/'complete.json').exists():
            assert not (root/'error.json').exists();time.sleep(5)
        # 合计最多两路完整AP；只有生产评估少于两路且确实存活才占一席。
        while True:
            state=read(root/'state.json');active=state.get('active',{})
            live=sum(Path('/proc',str(t['pid'])).exists() for t in active.values()) if state.get('stage')=='evaluate' else 0
            if state.get('stage')=='complete' or (state.get('stage')=='evaluate' and not state.get('pending') and live<2):break
            time.sleep(5)
        results[s]=audit(root,s)
    while not (root/'final.json').exists():
        assert not (root/'error.json').exists();time.sleep(3)
    final=read(root/'final.json');assert set(final['results'])==set(results)
    for s in results:assert final['complete_sha256'][s]==results[s]['complete_sha256'] and final['results'][s]==read(root/s/'complete.json')
    means={role:{m:dict(AUC=float(np.mean([v['results'][role][m]['video_macro_AUC'] for v in results.values()])),delta_pp=[float(np.mean([v['results'][role][m]['budgets'][i]['delta_pp'] for v in results.values()])) for i in range(4)]) for m in (*MODES,'confidence')} for role in ('fit','heldout','validation')}
    save(root/'independent_final_audit.json',dict(passed=True,source_only=True,final_sha256=sha(root/'final.json'),audits={s:sha(root/'independent_results'/f'{s}.json') for s in results},means=means))
    print(json.dumps(means),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
