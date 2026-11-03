"""固定既有预测，全10类比较原/快速AP、TP/FP/FN和随机状态；不跑新预测。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha
from tools.baseline_engine import tensor_digest
from tools.native_ap_fast import compute_average_precision_detection as fast
from libs.evaluation.evaluator_anet import compute_average_precision_detection as native,load_gt_seg_from_json


def benchmark(entry,output):
    cfg=read(entry/'config.json');results={}
    for domain,splits in [('source',cfg['src_val_split']),('target',cfg['tgt_val_split'])]:
        source=entry/'evaluation'/(domain+'_predictions.pt')
        columns=torch.load(source,map_location='cpu');pred=pd.DataFrame(columns)
        gt=load_gt_seg_from_json(cfg['baseline']['annotation_for_offline_evaluation'],split=splits,
            action_subsampling='10_majority_classes')
        gt=gt[gt['t-start']!=gt['t-end']].reset_index(drop=True)
        timings={'native':0.,'fast':0.};classes={}
        before=tensor_digest((random.getstate(),np.random.get_state(),torch.get_rng_state()))
        for label in sorted(gt.label.unique()):
            truth=gt[gt.label==label].reset_index(drop=True);prediction=pred[pred.label==label].reset_index(drop=True)
            started=time.monotonic();expected=native(truth,prediction);timings['native']+=time.monotonic()-started
            started=time.monotonic();actual=fast(truth,prediction);timings['fast']+=time.monotonic()-started
            assert type(expected)==type(actual)
            if isinstance(expected,tuple):
                assert all(np.array_equal(a,b,equal_nan=True) for a,b in zip(expected,actual)),(domain,int(label))
            else:assert np.array_equal(expected,actual,equal_nan=True)
            classes[str(int(label))]=dict(predictions=len(prediction),GT=len(truth),all_AP_TP_FP_FN_exact=True)
        after=tensor_digest((random.getstate(),np.random.get_state(),torch.get_rng_state()));assert before==after
        results[domain]=dict(classes=classes,timings_seconds=timings,speedup=timings['native']/timings['fast'],
            prediction_sha256=sha(source),source_entry=str(entry),CPU_Python_NumPy_rng_unchanged=True)
        print('AP_BENCHMARK',domain,json.dumps(results[domain]),flush=True)
    assert all(v['speedup']>1.05 for v in results.values())
    save(output,dict(passed=True,code_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        scope='full selected S1 source and target saved prediction pools, every supported GT class; not a GPU training speedup',
        results=results,fast_code_sha256=sha(CODE/'tools/native_ap_fast.py')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('entry',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--slot-root',type=Path)
    a=p.parse_args();lock=None
    if a.slot_root:
        import fcntl
        while lock is None:
            for i in range(2):
                candidate=(a.slot_root/f'private_CPU_evaluation_{i}.lock').open('a')
                try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:candidate.close()
                else:lock=candidate;break
            if lock is None:time.sleep(5)
    try:benchmark(a.entry,a.output)
    finally:
        if lock:lock.close()
