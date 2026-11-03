"""独立重算全部源留出曲线，复用已测试的NumPy匹配，不选择时点。"""
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
from tools.audit_background_ap_cost import IndependentAP, make_frame, background, selected_indices
from tools.audit_background_fit_gap import independent_auc
from libs.evaluation.evaluator_anet import load_gt_seg_from_json

ORIGIN=Path('/vepfs/vlms/background_ap_cost_source6_bd16507_1103')
SOURCE=Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
EPOCHS=(1,5,10,20)
RATES=(.005,.01,.02,.05)


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,sort_keys=True));tmp.replace(p)


def audit(root,s):
    entry=root/s;done=read(entry/'complete.json');gpu=read(entry/'gpu_complete.json');split=read(entry/'split.json')
    assert sha(entry/'gpu_complete.json')==done['gpu_complete_sha256'] and sha(entry/'split.json')==gpu['split_sha256']==done['split_sha256']
    cost=read(ORIGIN/s/'costs.json');assert sha(ORIGIN/s/'costs.json')==gpu['origin_costs_sha256']==split['origin_costs_sha256']
    ids=sorted(r['record']['video'] for r in cost['rows']);ranked=sorted(ids,key=lambda v:(hashlib.sha256(('head-holdout-v1|'+v).encode()).hexdigest(),v))
    assert split['heldout']==sorted(ranked[:(len(ids)+4)//5]) and split['fit']==sorted(set(ids)-set(split['heldout']))
    assert not set(ids)&{r['video'] for r in cost['validation']}
    assert sha(entry/'probe/kernel_complete.json')==gpu['probe_sha256'];probe=read(entry/'probe/kernel_complete.json');assert probe['passed'] and probe['control_old_loss_and_grad_exact']
    assert {r['record']['video'] for r in read(entry/'probe/costs.json')['rows']}==set(split['fit'])
    for report in (probe,gpu):
        assert sha(report['guard'])==report['guard_sha256'];g=read(report['guard']);lease=read(g['lease'])
        assert sha(g['lease'])==g['lease_sha256'] and g['owner_pid']==lease['owner_pid']==2730024
        assert g['observed']['uuid']==lease['uuid'] and g['observed']['index']==lease['index'] and g['observed']['free_mib']>=12288
    assert gpu['initial_digest']==read(ORIGIN/s/'gpu_complete.json')['initial_digest'] and gpu['readout_parameter_and_RNG_unchanged']
    history=read(entry/'history.json');assert sha(entry/'history.json')==gpu['history_sha256'] and [h['epoch'] for h in history]==list(range(1,21))
    gtrows=read(SOURCE/s/'source_truth.json');assert sha(SOURCE/s/'source_truth.json')==cost['source_truth_sha256']
    cfg=read(SOURCE/s/'config.json');valgt=load_gt_seg_from_json(cfg['dataset']['json_file'],cfg['src_val_split'],action_subsampling='10_majority_classes');valgt=valgt[valgt['t-start']!=valgt['t-end']].reset_index(drop=True)
    truth={r:pd.DataFrame([(v,float(b[0]),float(b[1]),int(c)) for v in split[r] for b,c in zip(gtrows[v]['segments'],gtrows[v]['labels'])],columns=['video-id','t-start','t-end','label']) for r in ('fit','heldout')};truth['validation']=valgt
    originals={r:{} for r in ('fit','heldout','validation')}
    for row in cost['rows']:
        video=row['record']['video'];x=torch.load(row['record']['path'],map_location='cpu');role='fit' if video in split['fit'] else 'heldout'
        originals[role][video]={k:x[k] for k in ('boxes','scores','labels')}
    for row in cost['validation']:
        x=torch.load(row['path'],map_location='cpu');originals['validation'][row['video']]={k:x[k] for k in ('segments','scores','labels')}
    engines={};positions={};baseline={};backgrounds={};reference_frames={};reports={}
    for epoch in EPOCHS:
        checkpoint=entry/f'epoch{epoch:03d}.pt';prediction=entry/f'predictions{epoch:03d}.pt'
        assert sha(checkpoint)==gpu['saved'][str(epoch)]['checkpoint_sha256'] and sha(prediction)==gpu['saved'][str(epoch)]['predictions_sha256']
        state=torch.load(checkpoint,map_location='cpu')
        assert {'heads','optimizers','history','order_rng','python_rng','numpy_rng','cpu_rng','cuda_rng','split_sha256','spec'}<=set(state)
        assert state['history']==history[:epoch] and state['split_sha256']==sha(entry/'split.json')
        for opt in state['optimizers'].values():
            assert opt['state'] and all(float(v['step'])==epoch*gpu['fit_steps_per_epoch'] for v in opt['state'].values())
        predictions=torch.load(prediction,map_location='cpu');assert set(predictions)==({'fit','heldout','validation'} if epoch==20 else {'fit','heldout'})
        reports[str(epoch)]={}
        for role,modes in predictions.items():
            for mode,rows in modes.items():
                assert {r['video'] for r in rows}==set(originals[role]) and len(rows)==len(originals[role])
                for row in rows:
                    x=originals[role][row['video']]
                    for k in ('scores','labels'):assert torch.equal(row[k],x[k])
                    assert torch.equal(row['segments'],x['segments' if role=='validation' else 'boxes'])
                    assert torch.isfinite(row['pbg']).all() and (row['pbg'][row['scores']<.3]==0).all()
            frames={m:make_frame(rows) for m,rows in modes.items()};f=frames['control'];raw=f.drop(columns='pbg')
            assert raw.equals(frames['ap_cost'].drop(columns='pbg'))
            if role not in engines:
                reference_frames[role]=raw;engines[role]={};positions[role]={};baseline[role]=[];backgrounds[role]=background(truth[role],f)
                for c in range(10):
                    ids=np.flatnonzero(f.label.to_numpy()==c);positions[role][c]=ids;engines[role][c]=IndependentAP(truth[role][truth[role].label==c],f.iloc[ids].reset_index(drop=True))
                    baseline[role].append(engines[role][c].ap(np.ones(len(ids),dtype=bool)).tolist())
                if role=='validation':
                    old=read(ORIGIN/'independent_results'/f'{s}.json')
                    assert np.allclose(baseline[role],old['original_per_class_AP'],atol=1e-7,rtol=0)
            else:assert raw.equals(reference_frames[role])
            bg=backgrounds[role];permode={}
            for mode in ('control','ap_cost','confidence'):
                frame=frames['control' if mode=='confidence' else mode];reported=done['results'][str(epoch)][role][mode];auc_values=[]
                for video,ids in frame.groupby('video-id',sort=False).indices.items():
                    ids=np.asarray(ids);ids=ids[frame.iloc[ids].score.to_numpy()>=.3]
                    a=independent_auc(bg[ids],1-frame.iloc[ids].score.to_numpy() if mode=='confidence' else frame.iloc[ids].pbg.to_numpy())
                    if a is not None:auc_values.append(a)
                assert len(auc_values)==reported['AUC_valid_videos'] and abs(np.mean(auc_values)-reported['video_macro_AUC'])<1e-12
                assert abs(np.mean(baseline[role])-reported['baseline'])<1e-7
                budgets=[]
                for rate,rep in zip(RATES,reported['all_fixed_budgets']):
                    assert rate==rep['rate'];selected=selected_indices(frame,rate,mode=='confidence');keep=np.ones(len(frame),dtype=bool);keep[selected]=False
                    ap=[engines[role][c].ap(keep[positions[role][c]]).tolist() for c in range(10)]
                    assert abs(np.mean(ap)-rep['map'])<1e-7 and abs(100*(np.mean(ap)-np.mean(baseline[role]))-rep['delta_pp'])<1e-7
                    assert int(bg[selected].sum())==rep['BG_removed'] and int((~bg[selected]).sum())==rep['nonBG_removed']
                    budgets.append(dict(rate=rate,delta_pp=rep['delta_pp'],per_class_AP=ap,BG_removed=rep['BG_removed'],nonBG_removed=rep['nonBG_removed']))
                permode[mode]=dict(video_macro_AUC=reported['video_macro_AUC'],AUC_valid_videos=len(auc_values),budgets=budgets)
            reports[str(epoch)][role]=permode
    assert (entry/'latest.pt').stat().st_ino==(entry/'epoch020.pt').stat().st_ino
    out=dict(passed=True,complete_sha256=sha(entry/'complete.json'),gpu_complete_sha256=sha(entry/'gpu_complete.json'),split_sha256=sha(entry/'split.json'),all_original_arrays_exact=True,all_checkpoint_state_and_history=True,results=reports)
    save(root/'independent_results'/f'{s}.json',out);print('AUDITED',s,flush=True);return out


def main(root):
    results={}
    for i in range(1,7):
        s=f'S{i}'
        while not (root/s/'complete.json').exists():
            assert not (root/'error.json').exists(), 'production failure; preserve completed stages'
            time.sleep(5)
        results[s]=audit(root,s)
    while not (root/'final.json').exists():
        assert not (root/'error.json').exists()
        time.sleep(3)
    final=read(root/'final.json');assert set(final['results'])==set(results)
    for s in results:assert final['complete_sha256'][s]==results[s]['complete_sha256'] and final['results'][s]==read(root/s/'complete.json')
    means={}
    for epoch in EPOCHS:
        means[str(epoch)]={}
        for role in ('fit','heldout','validation') if epoch==20 else ('fit','heldout'):
            means[str(epoch)][role]={m:dict(AUC=float(np.mean([v['results'][str(epoch)][role][m]['video_macro_AUC'] for v in results.values()])),delta_pp=[float(np.mean([v['results'][str(epoch)][role][m]['budgets'][i]['delta_pp'] for v in results.values()])) for i in range(4)]) for m in ('control','ap_cost','confidence')}
    save(root/'independent_final_audit.json',dict(passed=True,source_only=True,final_sha256=sha(root/'final.json'),audits={s:sha(root/'independent_results'/f'{s}.json') for s in results},means=means))
    print(json.dumps(means),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
