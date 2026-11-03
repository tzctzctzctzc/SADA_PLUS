"""冻结四方案的UDA比较：目标收益、同数量confidence、源副作用全部保留。"""
import argparse
from pathlib import Path
import sys
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha
from tools.run_uda_background import SPEC,TASKS


def run(root):
    audit=read(root/'independent_final_audit.json');assert audit['passed'] and sha(root/'final.json')==audit['final_sha256'];results={};comparison={};means={};timings={};signals={}
    for task in TASKS:
        path=root/'independent_results'/(task+'.json');assert sha(path)==audit['audits'][task];j=read(path);assert sha(root/task/'complete.json')==j['complete_sha256'];results[task]=j['results'];gpu=read(root/task/'gpu_complete.json')
        timings[task]=dict(end_to_end_training_worker_seconds=gpu['seconds'],training_loop_seconds=sum(h['seconds'] for h in read(root/task/'history.json')),parameters=gpu['parameters'],target_candidates=gpu['target_train_candidates'],target_candidates_seen=gpu['target_train_candidates_seen'])
    for shift in SPEC['shifts']:
        for mode in SPEC['modes']:
            task=shift+'_'+mode;comparison[task]={}
            for role in ('fit','heldout','validation','target_val'):
                actual=results[task][role][mode];reference=results[shift+'_ontology'][role]['ontology'];confidence=results[task][role]['confidence'];comparison[task][role]=[]
                for a,b,c in zip(actual['budgets'],reference['budgets'],confidence['budgets']):
                    assert a['rate']==b['rate']==c['rate'];comparison[task][role].append(dict(rate=a['rate'],actual_delta_pp=a['delta_pp'],ontology_delta_pp=b['delta_pp'],confidence_delta_pp=c['delta_pp'],
                        actual_minus_ontology_pp=a['delta_pp']-b['delta_pp'],actual_minus_confidence_pp=a['delta_pp']-c['delta_pp'],BG_removed=a['BG_removed'],nonBG_removed=a['nonBG_removed']))
    for mode in SPEC['modes']:
        means[mode]={role:[float(np.mean([results[s+'_'+mode][role][mode]['budgets'][i]['delta_pp'] for s in SPEC['shifts']])) for i in range(4)] for role in ('fit','heldout','validation','target_val')}
        signals[mode]=[]
        for i,rate in enumerate(SPEC['rates']):
            points=[comparison[s+'_'+mode]['target_val'][i] for s in SPEC['shifts']]
            signals[mode].append(dict(rate=rate,both_shifts_positive=all(p['actual_delta_pp']>0 for p in points),both_shifts_above_confidence=all(p['actual_minus_confidence_pp']>0 for p in points),
                both_shifts_above_ontology=all(p['actual_minus_ontology_pp']>0 for p in points),not_significance_or_independent_generalization=True))
    mechanism={}
    for task in TASKS:
        history=read(root/task/'history.json');mechanism[task]={k:dict(epoch1=history[0][k],epoch5=history[4][k],epoch20=history[-1][k]) for k in ('ordinary','pair','semantic','coral','consistency')}
    summary=dict(schema='highscore-source-ontology-UDA-result-v1',UDA_training=True,target_train_GT_read=False,independent_final_audit_sha256=sha(root/'independent_final_audit.json'),spec=SPEC,
        all_results=results,paired_comparisons=comparison,equal_shift_mean_delta_pp=means,all_prespecified_target_signal_checks=signals,mechanism_diagnostics=mechanism,timings=timings,
        limitations=['target validation has historical development exposure; all8models and all4rates frozen before this round CPU GT evaluation, no result-driven tuning',
            'S1/S2 exploratory results do not equal the six-direction final mean; no significance assertion from two shifts',
            'domain ratio assumes covariate shift and may misweight conditional/label shifts; conditionalCORAL may still mix action/background distributions',
            'EMA consistency uses soft model outputs, not verified targetBG labels; feature noise is an augmentation hypothesis',
            'lower adaptation loss/mean entropy is not proof of AP benefit; all positive and negative points retained'],
        scope=dict(source_heads='same new97verb auxiliary and APcritical source supervision',target_train='106+221 unlabeled videos, all highscore candidates visited',target_val='30+50 full videos, CPU evaluation only after8heads frozen',
                   efficiency='frozen source/target24-point descriptors cached once; each training worker copies fit/target tensors toGPU once; measured wall times are not whole-campaign speedups'))
    save(root/'scientific_decision.json',summary);print('RESULT',sha(root/'scientific_decision.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
