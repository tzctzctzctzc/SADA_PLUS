"""训练完成后独立CPU审计最终推理预测；不导入或修改运行中训练模块。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tools.baseline_protocol import read,save,sha
from tools.audit_head_replacements import metrics
from libs.evaluation.evaluator_anet import load_gt_seg_from_json


def audit(root,shift):
    entry=root/shift;complete=read(entry/'complete.json');job=read(root/'job_audits'/(shift+'.json'))
    assert complete['passed'] and job['passed'] and job['complete_sha256']==sha(entry/'complete.json')
    for file,digest in complete['artifacts'].items():assert sha(entry/file)==digest
    cfg=read(entry/'config.json');output=root/'fixed_readout'/shift;result=read(output/'result.json')
    assert result['full_dataset'] and result['optimizer_steps']==0
    assert result['checkpoint_sha256']==sha(entry/'best.pt') and result['predictions_sha256']==sha(output/'predictions.pt')
    pred=pd.DataFrame(torch.load(output/'predictions.pt',map_location='cpu'))
    gt=load_gt_seg_from_json(cfg['baseline']['annotation_for_offline_evaluation'],split=cfg['tgt_val_split'],action_subsampling='10_majority_classes')
    gt=gt[gt['t-start']!=gt['t-end']].reset_index(drop=True);independent=metrics(gt,pred)
    assert np.max(np.abs(np.asarray(independent['map_by_tiou'])-np.asarray(result['by_iou'])))<1e-7
    assert abs(independent['map']*100-result['map_percent'])<1e-7
    save(root/'final_readout_audits'/(shift+'.json'),dict(passed=True,shift=shift,independent=independent,
        prediction_sha256=sha(output/'predictions.pt'),checkpoint_sha256=sha(entry/'best.pt'),training_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--shift',choices=['all']+['S'+str(i) for i in range(1,7)],default='all');args=p.parse_args()
    shifts=['S'+str(i) for i in range(1,7)] if args.shift=='all' else [args.shift]
    for shift in shifts:audit(args.root,shift)
    if args.shift=='all':
        reports={s:read(args.root/'final_readout_audits'/(s+'.json')) for s in shifts};assert all(v['passed'] for v in reports.values())
        mean=sum(v['independent']['map'] for v in reports.values())/6*100
        assert abs(mean-read(args.root/'summary.json')['fixed_readout_mean_percent'])<1e-7
        save(args.root/'final_readout_audit.json',dict(passed=True,count=6,mean_percent=mean,delta_vs_strongest_pp=mean-56.22733524999254,single_seed=True,target_validation_selected=True))
