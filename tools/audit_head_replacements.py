"""只读单CPU审计十二项；源选模/完整工件及独立NumPy AP，不改训练。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,termination,code_identity
from tools.run_head_replacements import TASKS,ORIGIN
from tools.audit_background_ap_cost import IndependentAP,overlaps
from libs.evaluation.evaluator_anet import load_gt_seg_from_json


def metrics(gt,pred):
    classes=sorted(gt.label.unique())
    assert classes==list(range(10))
    ap=np.asarray([IndependentAP(gt[gt.label==c].reset_index(drop=True),pred[pred.label==c].reset_index(drop=True)).ap(
        np.ones(int((pred.label==c).sum()),dtype=bool)) for c in classes]).mean(0)
    high=pred[pred.score>=.3].reset_index(drop=True);bg=np.ones(len(high),dtype=bool)
    for video,ids in high.groupby('video-id',sort=False).indices.items():
        truth=gt[gt['video-id']==video]
        if len(truth):
            matrix=overlaps(high.iloc[ids][['t-start','t-end']].to_numpy(float),truth[['t-start','t-end']].to_numpy(float))
            bg[ids]=matrix.max(1)<.01
    return dict(map_by_tiou=ap.tolist(),map=float(ap.mean()),highscore_count=len(high),
                highscore_pure_task_background=int(bg.sum()),highscore_nonBG=int((~bg).sum()),
                pool_caveat='detector outputs differ between methods; counts are not matched-pool recall')


def audit(root,task):
    e=root/task;cfg=read(e/'config.json');binding=read(e/'origin_binding.json');complete=read(e/'complete.json')
    for path,digest in complete['artifacts'].items():assert sha(e/path)==digest,path
    for path,digest in binding['inputs'].items():assert sha(path)==digest,path
    original=read(Path(binding['origin'])/'config.json')
    assert cfg['model']==original['model'] and cfg['opt']==original['opt'] and cfg['loader']==original['loader']
    assert cfg['prototype_capacity']==original['prototype_capacity'] and cfg['seed']==original['seed']==123
    src=read(cfg['dataset']['json_file'])['database'];tgt=read(cfg['baseline']['target_manifest'])['database']
    assert not set(src)&set(tgt) and all('annotations' not in row for row in tgt.values())
    owner=read(root/'_launch.json')['parent_pid_snapshot']
    for stage in ('train','evaluate'):
        p=read(e/(stage+'_complete.json'));assert sha(p['guard'])==p['guard_sha256']
        g=read(p['guard']);assert sha(g['lease'])==g['lease_sha256'];lease=read(g['lease'])
        assert g['owner_pid']==lease['owner_pid']==owner and lease['commit']==cfg['baseline']['code']['commit']
        assert g['observed']['uuid']==lease['observed']['uuid'] and g['observed']['free_mib']>=16384
    probe=read(e/'probe.json');initial=read(e/'formal_initialization.json')
    assert probe['passed'] and probe['full_next_update_replayed'] and initial['passed'] and not initial['loaded_trained_checkpoint']
    assert probe['initial_state_sha']==initial['initial_state_sha']
    assert probe['historical_base_initial_sha']==initial['base_initial_sha']==binding['historical_initial_state_sha']
    assert sha(e/'probe_state.pt')==probe['probe_state_sha256']
    term=read(e/'termination.json');decision=termination(term['history'])
    rows=[json.loads(x) for x in (e/'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,decision['epoch']+1))
    assert [[r['epoch'],r['source_validation_loss']] for r in rows if 'source_validation_loss' in r]==term['history']
    assert all(r['target_gt_cls_loss']==r['target_gt_reg_loss']==r['target_gt_downstream_loss']==0 for r in rows)
    for name,epoch in (('best.pt',decision['best_epoch']),('latest.pt',decision['epoch'])):
        saved=torch.load(e/name,map_location='cpu');assert saved['config']==cfg and saved['epoch']==epoch
        assert len(saved['optimizers'])==len(saved['schedulers'])==3 and len(saved['rng']['cuda'])==1
        assert set(saved['rng'])=={'python','numpy','cpu','cuda','loaders'}
        weights=saved['state_dict']
        if cfg['head_replacement']['variant']=='trident':
            assert weights['reg_head.offset_head.conv.weight'].shape[0]==34
            assert 'start_head.cls_head.conv.weight' in weights and 'end_head.cls_head.conv.weight' in weights
        else:assert weights['task_interaction.cls_output.weight'].abs().sum()>0
        del saved
    results={}
    for domain,splits in (('source',cfg['src_val_split']),('target',cfg['tgt_val_split'])):
        report=read(e/'evaluation'/(domain+'_val.json'))
        assert report['checkpoint_sha256']==sha(e/'best.pt') and report['selected_epoch']==decision['best_epoch']
        columns=torch.load(e/'evaluation'/(domain+'_predictions.pt'),map_location='cpu');pred=pd.DataFrame(columns)
        annotation=cfg['baseline']['annotation_for_offline_evaluation']
        database=read(annotation)['database'];members={k for k,v in database.items() if v['subset'].lower() in splits}
        assert set(pred['video-id'])<=members and len(pred)==report['prediction_count']
        gt=load_gt_seg_from_json(annotation,split=splits,action_subsampling='10_majority_classes')
        gt=gt[gt['t-start']!=gt['t-end']].reset_index(drop=True)
        result=metrics(gt,pred)
        assert np.max(np.abs(np.asarray(result['map_by_tiou'])-np.asarray(report['map_by_tiou'])))<1e-7
        assert abs(result['map']-report['map_0.1_to_0.5'])<1e-7
        results[domain]=result
    save(root/'independent_results'/(task+'.json'),dict(passed=True,results=results,decision=decision,
         complete_sha256=sha(e/'complete.json'),delta_vs_baseline_pp=100*(results['target']['map']-binding['historical_target_map']),
         independent_AP_matching=True,GT_loader_reused=True,base_initial_hash_is_producer_GPU_assertion=True))
    print('HEAD_AUDITED',task,results['target']['map'],flush=True)


def run(root):
    done=set()
    while not (root/'stop.json').exists():
        for task in TASKS:
            if task not in done and (root/task/'complete.json').exists():audit(root,task);done.add(task)
        save(root/'independent_audit_progress.json',dict(audited=sorted(done),owner_pid=os.getpid()))
        if (root/'final.json').exists():
            final=read(root/'final.json');assert done==set(final['completed'])
            summaries={}
            for variant in ('trident','interaction'):
                reports=[read(root/'independent_results'/(f'S{s}_{variant}.json')) for s in range(1,7)
                         if f'S{s}_{variant}' in done]
                summaries[variant]=dict(count=len(reports),all_six_complete=len(reports)==6,
                    target_mean_percent=float(np.mean([r['results']['target']['map'] for r in reports])*100) if reports else None,
                    mean_delta_pp=float(np.mean([r['delta_vs_baseline_pp'] for r in reports])) if reports else None)
            save(root/'independent_final_audit.json',dict(passed=len(done)==12,code=code_identity(),
                producer_final_sha256=sha(root/'final.json'),reports={t:sha(root/'independent_results'/(t+'.json')) for t in sorted(done)},
                summaries=summaries,failed=final['failed'],selection='source validation only, all results retained'))
            return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
