"""汇总预先固定层对照；原混淆对只解释机制，不作独立收益估计或选择样本。"""
import argparse
from pathlib import Path
import sys
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha
from tools.run_layer_signals import STAGES,MODES,PROBES,EXTRACT,DIAG


def run(root):
    import torch
    audit=read(root/'independent_final_audit.json');assert audit['passed'] and sha(root/'final.json')==audit['final_sha256']
    results={s:{} for s in ('S1','S2')};contrasts={};pairs={};geometry={};times={}
    for task in PROBES:
        assert sha(root/'independent_results'/(task+'.json'))==audit['audits'][task]
        j=read(root/'independent_results'/(task+'.json'));assert j['passed'] and sha(root/task/'complete.json')==j['complete_sha256']
        s=task.split('_')[0]
        for role,modes in j['results'].items():
            dest=results[s].setdefault(role,{})
            for m,a in modes.items():
                if m in dest:assert dest[m]==a
                dest[m]=a
        times[task]=read(root/task/'gpu_complete.json')['seconds']
    for task in EXTRACT:
        assert sha(root/task/'independent_native.json')==audit['extraction_native_sha256'][task]
        times[task]=read(root/task/'complete.json')['seconds']
    for s in ('S1','S2'):
        for role,modes in results[s].items():assert set(modes)==set(MODES)|{'confidence'}
        contrasts[s]={}
        for role,modes in results[s].items():
            out={}
            for a,b in [('raw_RP512','FPN0'),('embedding','FPN0'),('embedding','SGP'),('SGP','FPN0'),('duration_scale','FPN0')]+[(x+'_centered',x) for x in STAGES]:
                aa=modes[a];bb=modes[b];ca=aa['conditional'];cb=bb['conditional']
                assert (ca['valid_videos'],ca['BG_anchors'],ca['pairs'])==(cb['valid_videos'],cb['BG_anchors'],cb['pairs'])
                out[a+' minus '+b]=dict(video_macro_AUC_difference=aa['video_macro_AUC']-bb['video_macro_AUC'],conditional_AUC_difference=ca['video_macro_conditional_AUC']-cb['video_macro_conditional_AUC'],
                    AP_delta_pp_difference=[u['delta_pp']-v['delta_pp'] for u,v in zip(aa['budgets'],bb['budgets'])],conditional_support=ca)
            contrasts[s][role]=out
        original=torch.load(DIAG/s/'inputs.pt',map_location='cpu');cost=read(DIAG/s/'fit_costs_and_pairs.json');table=np.array([[p['BG'],p['foreground']] for p in cost['pairing']['pairs']],dtype=int)
        lookup={}
        for domain in ('source','validation'):
            for rec in read(root/(s+'_'+domain)/'complete.json')['records']:lookup[rec['video']]=rec
        pairs[s]={}
        for stage in STAGES:
            rows=[torch.load(lookup[r['video']]['path'],map_location='cpu')['representations'][stage].numpy() for r in original['fit']]
            x=np.concatenate(rows).astype(np.float64);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);cos=(x[table[:,0]]*x[table[:,1]]).sum(1)
            pairs[s][stage]=dict(pairs=len(table),cosine_mean=float(cos.mean()),cosine_q25_q50_q75=np.quantile(cos,[.25,.5,.75]).tolist())
        geometry[s]={}
        for role in ('fit','heldout','validation'):
            geometry[s][role]={}
            for family in ('early','temporal'):
                geometry[s][role].update(read(root/(s+'_'+family)/(role+'_geometry.json'))['geometry'])
    summary=dict(schema='highscore-layer-mechanism-result-v1',source_only=True,target_read=False,independent_final_audit_sha256=sha(root/'independent_final_audit.json'),
        spec=read(root/'plan.json')['spec'],results=results,prespecified_contrasts=contrasts,frozen_fit_AP_critical_pair_cosines=pairs,
        descriptive_geometry=geometry,elapsed_seconds_by_worker=times,
        limitations=['frozenFPN0 mean pool and linear probes differ from old nonlinear multi-region head; this diagnoses available readout signal rather than a single-variable replacement of every old head',
            'fixed random raw projection may discard information; no negative probe can prove absent information',
            'original fit pairs were selected for nearest FPN0 foreground, so their cross-layer cosines are descriptive and selection-biased',
            'video-centering removes action-distribution as well as scene information, and cosine clustering is not a causal scene-shortcut test',
            'conditional pairs share anchors/videos and denominators differ across roles; no significance or target generalization assertion',
            'all fixed budgets and all probes retained; sourceval observation is not a license to select best probe/threshold'])
    save(root/'mechanism_summary.json',summary)
    print('SUMMARY',sha(root/'mechanism_summary.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
