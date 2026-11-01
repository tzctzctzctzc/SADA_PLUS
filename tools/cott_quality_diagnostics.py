"""只读离线诊断：不同模型置信度不可直接等同，固定网格/相同数量比较。"""
import argparse
import os
import sys
import time
import subprocess
from datetime import datetime,timezone
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.cott_protocol import *
from tools.evaluate_cott_quality import audit_video

DIAGNOSTICS=dict(score_grid=[0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.],
    per_video_budgets=[5,10,20,40],tiou=[.1,.2,.3,.4,.5],
    equal_count='per video keep teacher/direct top N, N=all CoTT target-class predictions; never invent padding predictions',
    stagewise='all primitive target-class intervals (unscored) vs all refined before NMS vs final',
    no_best_grid_selection=True,target_used=False,source_scope_only=True,
    inference_changed=False,note='descriptive diagnostics, cannot choose thresholds or a student checkpoint from these results')

def counts(pred,gt,t):
    rows,used=match_predictions(pred,gt,t);n=len(rows);g=len(gt);tp=len(used)
    return dict(tp=tp,fp=n-tp,fn=g-tp,count=n,precision=tp/n if n else 0.,
        recall=tp/g if g else 0.,f1=2*tp/(n+g) if n+g else 0.)

def top_per_video(pred,budgets):
    result=[]
    for v,n in sorted(budgets.items()):
        result += sorted([p for p in pred if p['video_id']==v],key=lambda p:(-p['confidence'],p['start'],p['end'],p['label']))[:n]
    return result

def assess(methods,gt,ids):
    curves={m:{str(c):{str(t):counts([p for p in pp if p['confidence']>=c],gt,t)
        for t in DIAGNOSTICS['tiou']} for c in DIAGNOSTICS['score_grid']} for m,pp in methods.items()}
    fixed_budget={str(n):{m:{str(t):counts(top_per_video(pp,{v:n for v in ids}),gt,t)
        for t in DIAGNOSTICS['tiou']} for m,pp in methods.items()} for n in DIAGNOSTICS['per_video_budgets']}
    numbers={v:sum(p['video_id']==v for p in methods['cott']) for v in ids}
    equal_count={m:{str(t):counts(top_per_video(pp,numbers),gt,t) for t in DIAGNOSTICS['tiou']}
        for m,pp in methods.items()}
    return dict(score_grid=curves,fixed_per_video_budget=fixed_budget,
        cott_generated_count_per_video=numbers,equal_count_to_cott=equal_count)

def analyze(root,done):
    root=Path(root);manifest=read(root/'manifest.json');plan=read(root/'plan.json')
    assert sha(root/'manifest.json')==plan['manifest_sha256']
    assert sha(root/'evaluation_inputs/source_only.json')==plan['evaluation_inputs_sha256']
    inputs=read(root/'evaluation_inputs/source_only.json');pred={};receipt={}
    for v in sorted(done):
        pred[v],_=audit_video(root,v);receipt[v]=sha(root/'videos'/v/'complete.json')
    out={}
    for s,selected in manifest['selection'].items():
        ids=sorted(set(selected)&done)
        if not ids:continue
        gt=[g for g in inputs[s]['ground_truth'] if g['video_id'] in ids]
        methods={m:[p for v in ids for p in pred[v][m]] for m in ['cott','direct']}
        methods['teacher']=[p for p in inputs[s]['teacher_predictions'] if p['video_id'] in ids]
        primitive=[dict(p,video_id=v,confidence=1.) for v in ids
            for p in read(root/'videos'/v/'primitive_intervals.json')['intervals'] if p['label']!='OTHER']
        refined=[p for v in ids for p in pred[v]['raw_refined'] if p['label']!='OTHER']
        out[s]=dict(videos=ids,complete_shift=len(ids)==len(selected),**assess(methods,gt,ids),
            stages={m:{str(t):counts(pp,gt,t) for t in DIAGNOSTICS['tiou']}
                for m,pp in [('primitive',primitive),('raw_refined',refined),('final_cott',methods['cott'])]})
    return dict(schema='cott-fixed-quality-diagnostics-v1',spec=DIAGNOSTICS,
        code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        manifest_sha256=sha(root/'manifest.json'),video_complete_sha256=receipt,by_shift=out,
        complete_videos=len(done),planned_unique_videos=len(manifest['videos']))

def main(root,watch=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    root=Path(root);code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
    specpath=root/'monitor/quality_diagnostics_plan.json'
    if specpath.exists():assert read(specpath)['spec']==DIAGNOSTICS
    else:save(specpath,dict(spec=DIAGNOSTICS,code=code,
        declared_at_utc=datetime.now(timezone.utc).isoformat(),
        generated_results_present_at_declaration=len(list((root/'videos').glob('*/complete.json')))))
    previous={}
    while True:
        complete={p.parent.name:sha(p) for p in (root/'videos').glob('*/complete.json')}
        if complete and complete!=previous:
            report=analyze(root,set(complete));stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            path=root/'monitor/quality_diagnostics'/f'{stamp}.json';save(path,report)
            save(root/'monitor/quality_diagnostics_latest.json',dict(path=str(path),sha256=sha(path),complete_videos=len(complete)))
            previous=complete;print('DIAGNOSTICS',len(complete),str(path),flush=True)
        if (root/'complete.json').exists() or not watch:return
        if (root/'error.json').exists() or root.with_name(root.name+'_supervisor_error.json').exists():
            raise RuntimeError('inference pipeline reported an error; diagnostics did not alter or restart it')
        time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--watch',action='store_true')
    a=p.parse_args();main(a.root,a.watch)
