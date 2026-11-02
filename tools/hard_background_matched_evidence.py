"""固定四个删除预算，观察模块排序本身；不选阈值/不取验证集最大值。"""
import argparse,os,sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.hard_background_modules import MODES
from tools.run_hard_background_pilots import SOURCE,SHIFTS
PILOT=Path('/vepfs/vlms/hard_background_pilots24_8e80d74_1103')
BUDGETS=(.005,.01,.02,.05)


def mask_at_budget(labels,scores,risk,fraction):
    import numpy as np
    mask=np.ones(len(scores),dtype=bool)
    for c in np.unique(labels):
        idx=np.flatnonzero((labels==c)&(scores>=.3));n=int(np.floor(len(idx)*fraction))
        order=np.lexsort((idx,-np.asarray(risk)[idx]));mask[idx[order[:n]]]=False
    return mask


def analyze(s,root):
    import numpy as np
    import torch
    from tools.foreground_oracle_ap import ClassAP,prediction_frame,background_mask
    from tools.run_foreground_calibration_oracle import load_truth
    e=PILOT/s;cfg=read(SOURCE/s/'config.json');done=read(e/'gpu_complete.json');results={}
    for domain,split in [('source_val',cfg['src_val_split']),('target_val',cfg['tgt_val_split'])]:
        path=e/(domain+'_predictions.pt');assert sha(path)==done['artifacts'][str(path)]
        rows=torch.load(path,map_location='cpu');truth=load_truth(cfg['dataset']['json_file'] if domain=='source_val' else cfg['baseline']['annotation_for_offline_evaluation'],split)
        reports={}
        for mode in MODES:
            f=prediction_frame(rows[mode]);bg=background_mask(truth,f);curves=[]
            engines={int(c):ClassAP(truth[truth.label==c],f[f.label==c].reset_index(drop=True)) for c in sorted(truth.label.unique())}
            base={c:engine.evaluate(np.ones(len(engine.scores),dtype=bool)) for c,engine in engines.items()}
            baseline=float(np.mean([v['map'] for v in base.values()]))
            for b in BUDGETS:
                keep=mask_at_budget(f.label.to_numpy(),f.score.to_numpy(),f.pbg.to_numpy(),b)
                values=[];before=[];after=[]
                for c,engine in engines.items():
                    positions=np.flatnonzero(f.label.to_numpy()==c);r=engine.evaluate(keep[positions]);values.append(r['map']);before.append(base[c]['tp']);after.append(r['tp'])
                curves.append(dict(fraction=b,delta_pp=100*(float(np.mean(values))-baseline),
                    removed=int((~keep).sum()),BG_removed=int((~keep&bg).sum()),nonBG_removed=int((~keep&~bg).sum()),
                    tp_change_by_tiou=(np.sum(after,axis=0)-np.sum(before,axis=0)).tolist()))
            reports[mode]=dict(baseline=baseline,all_fixed_budgets=curves)
        # 各版本预算按同一原候选集确定，必须实际删相同数量，不能靠全忽略获胜。
        for i,b in enumerate(BUDGETS):assert len({reports[m]['all_fixed_budgets'][i]['removed'] for m in MODES})==1
        results[domain]=reports
    save(root/(s+'.json'),dict(shift=s,results=results,parameter_search=False,no_best_budget_selection=True,
        target_val_labels_only_for_observation=True,artifacts={str(e/(d+'_predictions.pt')):sha(e/(d+'_predictions.pt')) for d in ['source_val','target_val']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();os.chdir('/vepfs/vlms/SADA_PLUS')
    a.root.mkdir(exist_ok=False,parents=True)
    save(a.root/'plan.json',dict(code=code_identity(),budgets=BUDGETS,scope='score>=.3; equal_counts_per_predicted_class',
        parameter_search=False,no_best_budget_selection=True,not_formal_UDA=True))
    completed=[]
    for s in SHIFTS:
        if (PILOT/s/'gpu_complete.json').exists():analyze(s,a.root);completed.append(s)
    save(a.root/'complete.json',dict(completed=completed,all_budget_points_reported=True,no_argmax_or_parameter_update=True))
