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


def removed_by_video(frame,keep,bg):
    import numpy as np
    result=[]
    for video,positions in frame.groupby('video-id').indices.items():
        removed=positions[~keep[positions]]
        result.append(dict(video=video,removed=len(removed),BG_removed=int(bg[removed].sum()),
                           nonBG_removed=int((~bg[removed]).sum())))
    observed=[r['BG_removed']/r['removed'] for r in result if r['removed']]
    return dict(videos=result,affected_videos=len(observed),
                video_macro_BG_precision=float(np.mean(observed)) if observed else None)


def analyze(s,root):
    import numpy as np
    import torch
    from tools.foreground_oracle_ap import ClassAP,prediction_frame,background_mask
    from tools.run_foreground_calibration_oracle import load_truth
    e=PILOT/s;cfg=read(SOURCE/s/'config.json');done=read(e/'gpu_complete.json');results={}
    for domain,split in [('source_val',cfg['src_val_split']),('target_val',cfg['tgt_val_split'])]:
        path=e/(domain+'_predictions.pt');assert sha(path)==done['artifacts'][str(path)]
        rows=torch.load(path,map_location='cpu');truth=load_truth(cfg['dataset']['json_file'] if domain=='source_val' else cfg['baseline']['annotation_for_offline_evaluation'],split)
        reports={};reference=prediction_frame(rows[MODES[0]])
        bg=background_mask(truth,reference)
        engines={int(c):ClassAP(truth[truth.label==c],reference[reference.label==c].reset_index(drop=True)) for c in sorted(truth.label.unique())}
        base={c:engine.evaluate(np.ones(len(engine.scores),dtype=bool)) for c,engine in engines.items()}
        baseline=float(np.mean([v['map'] for v in base.values()]))
        for mode in (*MODES,'confidence'):
            f=reference.copy() if mode=='confidence' else prediction_frame(rows[mode])
            assert f.iloc[:,:5].equals(reference.iloc[:,:5]),'原候选必须完全相同'
            curves=[]
            for b in BUDGETS:
                # confidence对照删最低原分数；预算、候选范围与小头完全相同。
                risk=-f.score.to_numpy() if mode=='confidence' else f.pbg.to_numpy()
                keep=mask_at_budget(f.label.to_numpy(),f.score.to_numpy(),risk,b)
                values=[];before=[];after=[]
                for c,engine in engines.items():
                    positions=np.flatnonzero(f.label.to_numpy()==c);r=engine.evaluate(keep[positions]);values.append(r['map']);before.append(base[c]['tp']);after.append(r['tp'])
                curves.append(dict(fraction=b,delta_pp=100*(float(np.mean(values))-baseline),
                    removed=int((~keep).sum()),BG_removed=int((~keep&bg).sum()),nonBG_removed=int((~keep&~bg).sum()),
                    tp_change_by_tiou=(np.sum(after,axis=0)-np.sum(before,axis=0)).tolist(),
                    per_video=removed_by_video(f,keep,bg)))
            reports[mode]=dict(baseline=baseline,all_fixed_budgets=curves)
        # 各版本预算按同一原候选集确定，必须实际删相同数量，不能靠全忽略获胜。
        for i,b in enumerate(BUDGETS):
            assert len({r['all_fixed_budgets'][i]['removed'] for r in reports.values()})==1
            for mode in MODES:
                r=reports[mode]['all_fixed_budgets'][i]
                r['delta_vs_confidence_pp']=r['delta_pp']-reports['confidence']['all_fixed_budgets'][i]['delta_pp']
                r['delta_vs_conditional_pp']=r['delta_pp']-reports['conditional']['all_fixed_budgets'][i]['delta_pp']
        results[domain]=reports
    save(root/(s+'.json'),dict(shift=s,results=results,parameter_search=False,no_best_budget_selection=True,
        target_val_labels_only_for_observation=True,artifacts={str(e/(d+'_predictions.pt')):sha(e/(d+'_predictions.pt')) for d in ['source_val','target_val']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();os.chdir('/vepfs/vlms/SADA_PLUS')
    a.root.mkdir(exist_ok=False,parents=True)
    save(a.root/'plan.json',dict(code=code_identity(),budgets=BUDGETS,scope='score>=.3; equal_counts_per_predicted_class',
        parameter_search=False,no_best_budget_selection=True,not_formal_UDA=True,
        simple_control='remove_lowest_original_score_same_class_counts',
        interpretation='all_four_budgets_are_observations; no_budget_or_variant_argmax'))
    completed=[]
    for s in SHIFTS:
        if (PILOT/s/'gpu_complete.json').exists():analyze(s,a.root);completed.append(s)
    save(a.root/'complete.json',dict(completed=completed,all_budget_points_reported=True,no_argmax_or_parameter_update=True))
