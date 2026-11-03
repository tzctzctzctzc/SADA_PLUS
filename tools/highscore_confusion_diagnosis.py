"""只读源域高分混淆诊断：完整AP反事实、全fit单候选边际、难背景/关键前景配对。"""
import argparse
from pathlib import Path
import sys
import time
import numpy as np

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_background_four_routes import OLD,ORIGIN,SOURCE

PRIOR=Path('/vepfs/vlms/background_four_routes_S1_c99105c_1103')
RATES=(.005,.01,.02,.05)
SPEC=dict(schema='highscore-background-AP-confusion-source-v1',source_only=True,target_read=False,
    pool='original postNMS score>=.3',BG='any-task maxIoU<.01',
    AP_cost='full source FIT set, class-wise delete-one then re-sort/re-match at.1:.1:.5, fixed10-class mean; all low-score replacements retained',
    pair_anchor='pure BG with positive full-fit AP deletion benefit',
    pair_foreground='nonBG with negative full-fit AP deletion delta, same predicted class, different physical video',
    matching='fit-only per-predicted-class original-score deciles and grid-duration quartiles; identical two bins required; no fallback',
    hard_mining='within matched cells choose up to4 foreground candidates with largest raw-inner-feature cosine; deterministic original-index ties',
    selection='sourcefit labels only for supervision; no val/heldout labels used to form pairs, no parameter/threshold sweep',
    caution='single-deletion marginal is not additive; pairing weights are training proxies, full AP remains the acceptance metric')


def matched_pairs(labels,scores,lengths,videos,bg,delta,features):
    """固定分层后用原视觉特征找混淆对；和旧top32原分数配对不同。"""
    normalized=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    scorebin=np.zeros(len(scores),int);lengthbin=np.zeros(len(scores),int);edges={}
    for c in np.unique(labels):
        ids=np.flatnonzero(labels==c)
        se=np.quantile(scores[ids],np.arange(1,10)/10)
        le=np.quantile(lengths[ids],(.25,.5,.75))
        scorebin[ids]=np.searchsorted(se,scores[ids],side='right')
        lengthbin[ids]=np.searchsorted(le,lengths[ids],side='right')
        edges[str(c)]=dict(score=se.tolist(),length=le.tolist())
    result=[];unpaired=[]
    for b in np.flatnonzero(bg&(delta>0)):
        possible=np.flatnonzero((~bg)&(delta<0)&(labels==labels[b])&(videos!=videos[b])&
            (scorebin==scorebin[b])&(lengthbin==lengthbin[b]))
        if not len(possible):unpaired.append(int(b));continue
        similarities=normalized[possible]@normalized[b]
        order=np.lexsort((possible,-similarities))[:4]
        for j in order:
            f=int(possible[j]);result.append(dict(BG=int(b),foreground=f,cosine=float(similarities[j]),
                score_gap=float(abs(scores[b]-scores[f])),log_duration_gap=float(abs(np.log(lengths[b]/lengths[f]))),
                BG_benefit=float(delta[b]),FG_harm=float(-delta[f]),score_bin=int(scorebin[b]),length_bin=int(lengthbin[b]),label=int(labels[b])))
    return dict(pairs=result,unpaired_BG=unpaired,fit_only_edges=edges)


def prepare_inputs(root,shift):
    import torch
    from tools import baseline_engine as base
    from tools.background_linear_temporal import region_samples,descriptors
    if shift=='S1':
        assert sha(PRIOR/'independent_final_audit.json')=='0e7477e9a65cd7e010faf0c0209b0d26e82545624b191dd2c8538756e6a17ea6'
        original=read(PRIOR/'prepare_complete.json');assert sha(PRIOR/'prepared.pt')==original['prepared_sha256']
        roles=torch.load(PRIOR/'prepared.pt',map_location='cpu')['roles']
    else:
        cost=read(ORIGIN/shift/'costs.json');split=read(OLD/shift/'split.json');rows={r:[] for r in ('fit','heldout','validation')}
        lookup={r['record']['video']:r for r in cost['rows']}
        for role,records in [('source',[r['record'] for r in cost['rows']]),('validation',cost['validation'])]:
            for rec in records:
                raw=torch.load(rec['path'],map_location='cpu');v=rec['video'];hard=raw['scores']>=.3
                dest='validation' if role=='validation' else ('fit' if v in split['fit'] else 'heldout')
                vals,masks=region_samples(raw['feature'],raw['valid'],raw['boxes'][hard])
                desc=descriptors(raw['feature'],raw['valid'],raw['boxes'][hard],raw['labels'][hard],raw['scores'][hard])['max_linear']
                x=dict(video=v,segments=raw['segments' if role=='validation' else 'boxes'],scores=raw['scores'],labels=raw['labels'],
                    hard=hard,seq=torch.cat(vals,1),valid=torch.cat(masks,1),metadata=desc[:,-12:],
                    inner=(vals[1]*masks[1][...,None]).sum(1)/masks[1].sum(1,keepdim=True).clamp_min(1))
                if role=='source':x.update(bg=torch.tensor(lookup[v]['bg']),quality=torch.tensor(lookup[v]['quality']))
                rows[dest].append(x)
        roles=rows
    base.checkpoint(root/shift/'inputs.pt',roles)
    return roles


def run(root,shifts):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import ClassAP,prediction_frame,background_mask
    from tools.audit_background_ap_cost import selected_indices,IndependentAP
    from tools.run_foreground_calibration_oracle import load_truth
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,shifts=shifts,code=code_identity()))
    for shift in shifts:
        entry=root/shift;entry.mkdir();began=time.monotonic();roles=prepare_inputs(root,shift)
        truth=read(SOURCE/shift/'source_truth.json');split=read(OLD/shift/'split.json');cfg=read(SOURCE/shift/'config.json')
        gt={r:pd.DataFrame([(v,float(b[0]),float(b[1]),int(c)) for v in split[r] for b,c in zip(truth[v]['segments'],truth[v]['labels'])],columns=['video-id','t-start','t-end','label']) for r in ('fit','heldout')}
        gt['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split'])
        frames={}
        for role,items in roles.items():frames[role]=prediction_frame([{k:r[k] for k in ('video','segments','scores','labels')}|dict(pbg=torch.zeros(len(r['scores']))) for r in items])
        fit=frames['fit'];bg=background_mask(gt['fit'],fit);hard=fit.score.to_numpy()>=.3;hardids=np.flatnonzero(hard);deltas=np.zeros(len(fit));baseline=[];checks=[]
        for c in range(10):
            ids=np.flatnonzero(fit.label.to_numpy()==c);frame=fit.iloc[ids];truth_c=gt['fit'][gt['fit'].label==c]
            engine=ClassAP(truth_c,frame);allkeep=np.ones(len(ids),bool);base=engine.evaluate(allkeep)['map'];baseline.append(base)
            ref=IndependentAP(truth_c,frame.reset_index(drop=True));assert abs(ref.ap(allkeep).mean()-base)<1e-7
            for i in np.flatnonzero(hard[ids]):
                keep=allkeep.copy();keep[i]=False
                deltas[ids[i]]=(engine.evaluate(keep)['map']-base)/10
            # 每类各一个真实有益BG/有害FG交叉用独立NumPy重算，完整cost保留。
            for mask in (bg[ids]&(deltas[ids]>0),~bg[ids]&(deltas[ids]<0)):
                options=np.flatnonzero(mask)
                if len(options):
                    i=int(options[0]);keep=allkeep.copy();keep[i]=False
                    value=(ref.ap(keep).mean()-ref.ap(allkeep).mean())/10
                    assert abs(value-deltas[ids[i]])<1e-10;checks.append(dict(index=int(ids[i]),delta=float(value)))
        feature=torch.cat([r['inner'] for r in roles['fit']]).numpy();lengths=(fit['t-end']-fit['t-start']).to_numpy()[hardids]
        pairing=matched_pairs(fit.label.to_numpy()[hardids],fit.score.to_numpy()[hardids],lengths,fit['video-id'].to_numpy()[hardids],bg[hardids],deltas[hardids],feature)
        save(entry/'fit_costs_and_pairs.json',dict(spec=SPEC,inputs_sha256=sha(entry/'inputs.pt'),source_truth_sha256=sha(SOURCE/shift/'source_truth.json'),old_split_sha256=sha(OLD/shift/'split.json'),
            baseline=float(np.mean(baseline)),hard_original_indices=hardids.tolist(),BG=bg[hardids].tolist(),delta=deltas[hardids].tolist(),pairing=pairing,independent_cost_spots=checks,
            summary=dict(high_candidates=int(hard.sum()),BG=int(bg[hard].sum()),beneficial_BG=int((bg&hard&(deltas>0)).sum()),harmful_foreground=int((~bg&hard&(deltas<0)).sum()),pairs=len(pairing['pairs']),paired_BG=len(set(p['BG'] for p in pairing['pairs'])))))
        # S1已存所有路线全预算：拆开删BG/删nonBG后的完整AP，效应不可相加。
        effects={}
        if shift=='S1':
            outputs={route:torch.load(PRIOR/route/'predictions.pt',map_location='cpu') for route in ('pauc','asl','knn','causal')}
            for role,frame in frames.items():
                b=background_mask(gt[role],frame);engines={};positions={};base=[]
                for c in range(10):
                    idx=np.flatnonzero(frame.label.to_numpy()==c);positions[c]=idx;engines[c]=ClassAP(gt[role][gt[role].label==c],frame.iloc[idx]);base.append(engines[c].evaluate(np.ones(len(idx),bool))['map'])
                def effect(remove):
                    keep=np.ones(len(frame),bool);keep[remove]=False
                    return float(100*(np.mean([engines[c].evaluate(keep[positions[c]])['map'] for c in range(10)])-np.mean(base)))
                effects[role]={}
                for route,modes in outputs.items():
                    for mode,rows in modes[role].items():
                        f=prediction_frame(rows);key=route+'/'+mode;effects[role][key]=[]
                        for rate in RATES:
                            idx=selected_indices(f,rate)
                            effects[role][key].append(dict(rate=rate,actual_delta_pp=effect(idx),BG_only_delta_pp=effect(idx[b[idx]]),nonBG_only_delta_pp=effect(idx[~b[idx]]),BG_removed=int(b[idx].sum()),nonBG_removed=int((~b[idx]).sum())))
                    if 'confidence' not in effects[role]:
                        effects[role]['confidence']=[]
                        for rate in RATES:
                            idx=selected_indices(frame,rate,True)
                            effects[role]['confidence'].append(dict(rate=rate,actual_delta_pp=effect(idx),BG_only_delta_pp=effect(idx[b[idx]]),nonBG_only_delta_pp=effect(idx[~b[idx]]),BG_removed=int(b[idx].sum()),nonBG_removed=int((~b[idx]).sum())))
        save(entry/'diagnosis.json',dict(passed=True,source_only=True,fit_costs_sha256=sha(entry/'fit_costs_and_pairs.json'),effects=effects,seconds=time.monotonic()-began,AP_counterfactual_effects_not_additive=True))
        print('DIAGNOSED',shift,read(entry/'fit_costs_and_pairs.json')['summary'],flush=True)
    save(root/'complete.json',dict(passed=True,spec=SPEC,results={s:sha(root/s/'diagnosis.json') for s in shifts}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shifts',nargs='+',default=['S1','S2']);a=p.parse_args();run(a.root,a.shifts)
