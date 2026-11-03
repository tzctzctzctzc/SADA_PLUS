"""只读结果审计：完整GT口径、原源选模、完整状态及配对数据流。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.release_data_protocol import SPEC,SCHEMA,assert_pair,release_decision,stream_pair
from tools.audit_background_ap_cost import IndependentAP,overlaps
from libs.evaluation.evaluator_anet import load_gt_seg_from_json


def independent_metrics(gt,pred):
    classes=sorted(gt.label.unique());assert classes
    values=[]
    for c in classes:
        truth=gt[gt.label==c].reset_index(drop=True);prediction=pred[pred.label==c].reset_index(drop=True)
        values.append(IndependentAP(truth,prediction).ap(np.ones(len(prediction),dtype=bool)))
    ap=np.asarray(values).mean(0)
    return dict(map_by_tiou=ap.tolist(),map=float(ap.mean()),GT_classes=[int(c) for c in classes])


def audit(root,task):
    entry=root/task;cfg=read(entry/'config.json');binding=read(entry/'origin_binding.json')
    complete=read(entry/'complete.json');assert cfg['release_data_protocol']==SPEC
    for path,digest in complete['artifacts'].items():assert sha(entry/path)==digest,path
    for path,digest in binding['inputs'].items():assert sha(path)==digest,path
    shift,variant=task.split('_');original=read(Path(binding['origin'])/'config.json')
    assert all(cfg[k]==original[k] for k in ('model','opt','loader','train_cfg','seed','init_rand_seed'))
    if variant=='prototype':assert cfg['prototype_capacity']==original['prototype_capacity']
    else:assert 'prototype_capacity' not in cfg
    annotation=cfg['baseline']['annotation_for_offline_evaluation'];db=read(annotation)['database']
    target=read(cfg['baseline']['target_manifest'])['database']
    expected={k:v for k,v in db.items() if v['subset'].lower() in cfg['tgt_train_split']}
    assert list(target.items())==list(expected.items()) and any(v.get('annotations') for v in target.values())
    source=read(cfg['dataset']['json_file'])['database']
    assert list(source.items())==list(read(original['dataset']['json_file'])['database'].items())
    assert not set(source)&set(target)
    assert not set(target)&{k for k,v in db.items() if v['subset'].lower() in cfg['tgt_val_split']}
    cpu=root/'inputs'/shift/'CPU_release_reference.json';assert read(cpu)['passed']
    native=CODE/'libs/trainer/base_trainer_da.py'
    author=subprocess.check_output(['git','show','a96a15b0d0c905a553c7a2b14f607c9485ebe758:libs/trainer/base_trainer_da.py'],cwd=CODE)
    assert native.read_bytes().replace(b'\r\n',b'\n')==author.replace(b'\r\n',b'\n')
    runtime=read(entry/'release_runtime.json');assert runtime['raw_train_file_sha256']==sha(native)
    history=read(entry/'termination.json')['history'];decision=release_decision(history)
    assert decision['reason']!='running' and decision==read(entry/'termination.json')['decision']
    probe=read(entry/'probe.json');assert probe['passed'] and probe['fixed_batch_full_next_update_exact']
    assert probe['target_labels_removed_same_objective_and_all_gradients']
    initial=read(entry/'formal_initialization.json')
    assert not initial['loaded_trained_weights'] and initial['initial_state_sha256']==probe['initial_state_sha256']
    assert initial['detector_initial_sha256']==binding['historical_detector_initial_sha256']
    owner=read(root/'state.json')['owner_pid']
    for stage in ('train','evaluate'):
        guard=read(read(entry/(stage+'_complete.json'))['guard']);lease=read(guard['lease'])
        assert guard['owner_pid']==lease['owner_pid']==owner and lease['commit']==cfg['baseline']['code']['commit']
        assert guard['lease_sha256']==sha(guard['lease']) and guard['observed']['uuid']==lease['observed']['uuid']
        assert guard['observed']['free_mib']>=lease['minimum_free_mib']
    states={}
    for name in ('best','latest'):
        state=torch.load(entry/(name+'.pt'),map_location='cpu')
        assert state['schema']==SCHEMA and state['config']==cfg
        assert len(state['optimizers'])==len(state['schedulers'])==3
        assert all(k in state['rng'] for k in ('python','numpy','cpu','cuda','loaders'))
        assert len(state['rng']['cuda'])==1 and state['loader_resume']=='unsupported_persistent_worker_RNG_not_captured'
        assert all(torch.equal(state['rng']['cpu'],s) for s in state['rng']['loaders'].values())
        expected_epoch=decision['best_epoch'] if name=='best' else decision['epoch']
        assert state['epoch']==expected_epoch and state['history']==[x for x in history if x[0]<=expected_epoch]
        states[name]=dict(epoch=state['epoch'],state_dict_keys=len(state['state_dict']))
        del state
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,decision['epoch']+1))
    for r in rows:
        stream=entry/'streams'/f"epoch{r['epoch']:03d}.json"
        assert sha(stream)==r['data_stream_sha256']
        assert len(read(stream))==cfg['baseline']['epoch_budget']['iterations_per_epoch']
    results={}
    for role,splits in [('source',cfg['src_val_split']),('target',cfg['tgt_val_split'])]:
        report=read(entry/'evaluation'/(role+'_val.json'))
        assert report['split']==splits and report['prediction_splits']==splits
        assert report['checkpoint_sha256']==sha(entry/'best.pt') and report['selected_epoch']==decision['best_epoch']
        columns=torch.load(entry/'evaluation'/(role+'_predictions.pt'),map_location='cpu');pred=pd.DataFrame(columns)
        members={k for k,v in db.items() if v['subset'].lower() in splits}
        assert set(pred['video-id'])<=members and len(pred)==report['prediction_count'] and report['videos']==len(members)
        gt=load_gt_seg_from_json(annotation,split=splits,action_subsampling='10_majority_classes')
        gt=gt[gt['t-start']!=gt['t-end']].reset_index(drop=True)
        result=independent_metrics(gt,pred)
        assert np.max(np.abs(np.asarray(result['map_by_tiou'])-np.asarray(report['map_by_tiou'])))<1e-7
        assert abs(result['map']-report['map_0.1_to_0.5'])<1e-7
        result.update(GT_videos=int(gt['video-id'].nunique()),prediction_video_scope=len(members),
            prediction_videos_without_task_GT=int((~pred['video-id'].isin(set(gt['video-id']))).sum()))
        results[role]=result
    save(root/'independent_results'/(task+'.json'),dict(passed=True,complete_sha256=sha(entry/'complete.json'),
        results=results,decision=decision,checkpoint_states=states,
        evaluation='full_prediction_scope_and_full_corresponding_GT',independent_NumPy_matcher=True,
        descriptor_scope='four actual video samples per shift checked vs official Dataset/truncate; not all crops',
        debug_AP_scope='debug prediction artifacts bound; only final selected-checkpoint AP independently recomputed'))
    print('RELEASE_AUDITED',task,results['target']['map'],flush=True)


def run(root):
    while not (root/'prepare_complete.json').exists():
        if (root/'error.json').exists() or (root/'stop.json').exists():return
        time.sleep(10)
    tasks=read(root/'plan.json')['tasks'];done=set()
    while not (root/'stop.json').exists():
        for task in tasks:
            if task not in done and (root/task/'complete.json').exists():audit(root,task);done.add(task)
        save(root/'independent_audit_progress.json',dict(owner_pid=os.getpid(),audited=sorted(done)))
        if (root/'final.json').exists():
            final=read(root/'final.json');assert done==set(final['completed'])
            paired={};summaries={}
            for s in range(1,7):
                a,b=root/f'S{s}_sada',root/f'S{s}_prototype'
                if (a/'complete.json').exists() and (b/'complete.json').exists():
                    assert_pair(read(a/'config.json'),read(b/'config.json'))
                    streams=stream_pair(*[[json.loads(x) for x in (e/'metrics.jsonl').read_text().splitlines()] for e in (a,b)])
                    left,right=[read(root/'independent_results'/(e.name+'.json')) for e in (a,b)]
                    paired[f'S{s}']=dict(delta_prototype_minus_sada_pp=100*(right['results']['target']['map']-left['results']['target']['map']),
                        data_streams=streams)
            for mode in ('sada','prototype'):
                reports=[read(root/'independent_results'/(f'S{s}_{mode}.json')) for s in range(1,7) if f'S{s}_{mode}' in done]
                summaries[mode]=dict(count=len(reports),mean_percent=float(np.mean([r['results']['target']['map'] for r in reports])*100) if reports else None)
            save(root/'independent_final_audit.json',dict(passed=len(done)==12,code=code_identity(),
                producer_final_sha256=sha(root/'final.json'),failed=final['failed'],
                reports={t:sha(root/'independent_results'/(t+'.json')) for t in sorted(done)},
                pairs=paired,summaries=summaries,scope='native training; full matching GT evaluation, same metric scope as corrected baseline'))
            return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
