"""四个预先固定idea和原FPN配对头逐点比较，保留全部正负操作点。"""
import argparse
from pathlib import Path
import sys
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha
from tools.run_signal_ideas import SPEC,TASKS,MODES,BASE


def run(root):
    audit=read(root/'independent_final_audit.json');assert audit['passed'] and sha(root/'final.json')==audit['final_sha256']
    results={};comparisons={};times={};means={}
    for task in TASKS:
        p=root/'independent_results'/(task+'.json');assert sha(p)==audit['audits'][task];j=read(p);assert j['passed'] and sha(root/task/'complete.json')==j['complete_sha256']
        shift,mode=task.split('_');reference=read(BASE/(shift+'_pairs')/'complete.json');results[task]=j['results'];comparisons[task]={}
        for role in ('fit','heldout','validation'):
            actual=j['results'][role][mode];old=reference['results'][role]['pairs'];comparisons[task][role]=[]
            for x,y in zip(actual['budgets'],old['budgets']):
                assert x['rate']==y['rate'];comparisons[task][role].append(dict(rate=x['rate'],actual_delta_pp=x['delta_pp'],original_FPN_delta_pp=y['delta_pp'],difference_pp=x['delta_pp']-y['delta_pp'],
                    BG_removed=x['BG_removed'],nonBG_removed=x['nonBG_removed'],original_FPN_BG_removed=y['BG_removed'],original_FPN_nonBG_removed=y['nonBG_removed']))
        gpu=read(root/task/'gpu_complete.json');times[task]=dict(end_to_end_seconds=gpu['seconds'],training_loop_seconds=sum(h['seconds'] for h in read(root/task/'history.json')),parameters=gpu['parameters'])
    for mode in MODES:
        means[mode]={role:[float(np.mean([results[s+'_'+mode][role][mode]['budgets'][i]['delta_pp'] for s in SPEC['shifts']])) for i in range(4)] for role in ('fit','heldout','validation')}
    source_positive={m:all(v>0 for v in means[m]['validation']) for m in MODES}
    summary=dict(schema='highscore-four-signal-idea-results-v1',source_only=True,target_read=False,spec=SPEC,independent_final_audit_sha256=sha(root/'independent_final_audit.json'),
        all_results=results,paired_reference_comparisons=comparisons,equal_shift_mean_delta_pp=means,timings=times,
        decision=dict(stable_positive_all_prespecified_sourceval_points=source_positive,
            no_automatic_target_or_full_training=True,
            rule='No picking one shift/budget as winner; lower loss than the reference is not an AP gain over the original detector. All four ideas are reported.',
            interpretation='Fit AP gains show this fixed objective can learn these input representations; negative heldout/sourceval AP does not establish that information is absent or that all background methods fail.',
            next_constraint='Do not launch another blind frozen-head coefficient/threshold sweep; any new round must address transferable feature learning or critical-foreground rejection risk with a fixed short budget.'))
    save(root/'scientific_decision.json',summary);print('RESULT',sha(root/'scientific_decision.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
