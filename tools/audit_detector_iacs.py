"""源监督/完整训练状态/无标签游标与完整原生检测AP的独立审计。"""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_detector_iacs import TASKS,MODES,READOUTS,SPEC,SOURCE,DENSE


def quality_numpy(classes,offsets,predicted):
    inter=np.minimum(offsets[:,0],predicted[:,0])+np.minimum(offsets[:,1],predicted[:,1])
    iou=np.clip(inter/np.maximum(offsets.sum(-1)+predicted.sum(-1)-inter,1e-8),0,1)
    return classes*(iou[:,None]>=np.array([.1,.2,.3,.4,.5],np.float32)).mean(-1)[:,None]


def states(root,s):
    import os,torch
    from tools import baseline_engine as base
    from tools.run_uda_background import TargetStream
    from tools.run_foreground_calibration_oracle import REFERENCE
    prep=read(root/'prepare.json')['shifts'][s];folder=root/'inputs';sourcefile=folder/(s+'_source.pt');unfile=folder/(s+'_unlabeled.pt')
    assert sha(sourcefile)==prep['source_sha256'] and sha(unfile)==prep['unlabeled_sha256'];rows=torch.load(sourcefile,map_location='cpu');data=torch.load(unfile,map_location='cpu');truth=read(SOURCE/s/'source_truth.json');prior=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu')
    assert sha(DENSE/'inputs'/s/'data.pt')==prep['prior_audited_input_sha256']==read(DENSE/'prepare_complete.json')['results'][s]['data_sha256'];assert sha(SOURCE/s/'source_truth.json')==prep['source_truth_sha256'];assert sha(SOURCE/s/'config.json')==prep['source_config_sha256']
    assert [r['video'] for r in rows]==prep['source_videos']==sorted(truth);assert set(truth)=={r['video'] for role in ('fit','heldout') for r in prior['roles'][role]}
    from tools.run_dense_revisit import raw_load
    raw_index={r['video']:r for role in ('fit','heldout') for r in prior['roles'][role]}
    for r in rows:
        gt=truth[r['video']];assert np.array_equal(r['gt_segments'].numpy(),np.asarray(gt['segments'],np.float32).reshape(-1,2));assert np.array_equal(r['gt_labels'].numpy(),np.asarray(gt['labels'],np.int64));assert r['raw'].shape[1]==2304 and not {'target_gt','bg','delta'}&set(r);assert torch.equal(r['raw'],raw_load(raw_index[r['video']]))
    for role,rs in [('target_train',data['target_train'])]+list(data['evaluation'].items()):
        assert not set(truth)&{r['video'] for r in rs};assert [r['video'] for r in rs]==[r['video'] for r in prior['roles'][role]]
        for r,old in zip(rs,prior['roles'][role]):
            assert not {'annotations','bg','quality','dense','delta'}&set(r)
            for key in ('segments','scores','labels'):assert torch.equal(r[key],old[key])
            for key in ('raw_sha256','video_metadata','raw_path'):assert r[key]==old[key]
    owner=read(root/'state.json')['owner_pid'];final=read(root/'final.json');results={}
    original_sha=sha(REFERENCE/s/'best.pt')
    for mode in MODES:
        e=root/(s+'_'+mode);g=read(e/'gpu_complete.json');assert sha(e/'gpu_complete.json')==final['allGPU_sha256'][e.name];assert g['passed'] and g['source_videos']==len(rows) and g['all_predictions_from_saved_EMA20'] and g['readout_parameters_RNG_unchanged'] and not g['target_train_GT_read']
        for f,k in [('kernel_complete.json','kernel_sha256'),('epoch020.pt','checkpoint_sha256'),('predictions.pt','predictions_sha256'),('semantic_teacher_logits.pt','semantic_logits_sha256')]:assert sha(e/f)==g[k]
        k=read(e/'kernel_complete.json');assert all(k[x] for x in ('passed','original_pretrained_decode_arrays_bitwise_equal','original_focal_loss_exact_when_disabled','authors_VFL_used_directly','real_backbone_classifier_regressor_updates','full_model_EMA_dualopts_controls_RNG_targetcache_nextstep_exact'));assert sha(e/'kernel.pt')==k['kernel_sha256']
        guard=read(g['guard']);lease=read(e/'lease_train.json');assert sha(g['guard'])==g['guard_sha256']==k['guard_sha256'] and sha(e/'lease_train.json')==guard['lease_sha256'];assert guard['owner_pid']==lease['owner_pid']==owner and guard['observed']['uuid']==lease['uuid'] and guard['observed']['free_mib']>=12288
        h=read(e/'history.json');assert [a['epoch'] for a in h]==list(range(1,21));order=torch.Generator().manual_seed(123);stream=TargetStream(len(data['target_train']));checkpoints={};previous={}
        for epoch in range(1,21):
            torch.randperm(len(rows),generator=order)
            if epoch>5:stream.draw(len(rows))
            assert h[epoch-1]['steps']==len(rows) and h[epoch-1]['alignment_active']==(epoch>5)
            if epoch<=5:assert h[epoch-1]['adv']==0
            if epoch not in (1,5,10,20):continue
            p=e/f'epoch{epoch:03d}.pt';st=torch.load(p,map_location='cpu');assert st['epoch']==epoch and st['step']==epoch*len(rows) and st['history']==h[:epoch];assert st['source_input_sha256']==prep['source_sha256'] and st['unlabeled_input_sha256']==prep['unlabeled_sha256'] and st['original_checkpoint_sha256']==original_sha
            assert torch.equal(order.get_state(),st['order_rng']);expected=stream.state();assert expected['cursor']==st['target_order']['cursor']
            for key in ('generator','order','seen'):assert torch.equal(expected[key],st['target_order'][key])
            for key in ('python_rng','numpy_rng','cpu_rng','cuda_rng','controls'):assert key in st
            assert st['cpu_rng'].dtype==st['cuda_rng'].dtype==torch.uint8
            assert not any(n.startswith('head.') for n in st['model']);assert all(any(n.startswith(v+'.') for n in st['model']) for v in ('backbone','neck','cls_head','reg_head'))
            for v in st['optimizer']['state'].values():assert int(v['step'])==epoch*len(rows)
            if epoch<=5:assert not st['disc_optimizer']['state'] and not st['semantic_cache']
            for v in st['disc_optimizer']['state'].values():assert 0<int(v['step'])<=max(0,epoch-5)*len(rows)
            if epoch>5:assert set(st['semantic_cache'])=={r['video'] for r,seen in zip(data['target_train'],stream.seen) if seen>0}
            assert set(st['model'])==set(st['ema']);checkpoints[str(epoch)]=sha(p)
            if epoch==20:
                assert base.tensor_digest(st['model'])==g['final_model_digest'] and base.tensor_digest(st['ema'])==g['final_EMA_digest'];assert base.tensor_digest(st['semantic_cache'])==base.tensor_digest(torch.load(e/'semantic_teacher_logits.pt',map_location='cpu'))
            previous=st
        assert os.stat(e/'latest.pt').st_ino==os.stat(e/'epoch020.pt').st_ino
        assert g['target_train_draws']==15*len(rows) and g['target_train_seen']==len(data['target_train'])
        results[mode]=dict(all_checkpoint_SHA256=checkpoints,GPU_sha256=sha(e/'gpu_complete.json'),source_videos=len(rows),target_unlabeled_videos=len(data['target_train']),full20_dualopt_EMA_RNG_history_and_order_checked=True)
        print('IACS_STATE_AUDITED',s,mode,flush=True)
    return results


def run(root):
    import torch,pandas as pd
    from tools.audit_background_ap_cost import IndependentAP,official
    from tools.run_foreground_calibration_oracle import load_truth,REFERENCE
    while not (root/'final.json').exists():
        if (root/'error.json').exists():raise RuntimeError('producer failed; no restart')
        time.sleep(3)
    link=read(root/'independent_replay_link.json');replayroot=Path(link['root'])
    while not (replayroot/'final.json').exists():
        if (replayroot/'error.json').exists():raise RuntimeError('independent savedEMA replay failed')
        time.sleep(3)
    rf=read(replayroot/'final.json');assert rf['passed'] and set(rf['results_sha256'])==set(TASKS);replay_owner=read(replayroot/'state.json')['owner_pid']
    for task in TASKS:
        p=replayroot/task/'complete.json';r=read(p);assert sha(p)==rf['results_sha256'][task] and r['passed'] and r['optimizer_steps']==0 and r['CPUCUDA_RNG_unchanged'];g=read(r['guard']);lease=read(replayroot/task/'lease_train.json');assert sha(r['guard'])==r['guard_sha256'] and sha(replayroot/task/'lease_train.json')==g['lease_sha256'];assert g['owner_pid']==lease['owner_pid']==replay_owner and g['observed']['uuid']==lease['uuid'] and g['observed']['free_mib']>=12288
        assert r['origin_GPU_sha256']==sha(root/task/'gpu_complete.json') and r['checkpoint_sha256']==sha(root/task/'epoch020.pt') and r['predictions_sha256']==sha(root/task/'predictions.pt');assert all(c['all_three_methods_and_arrays_bitwise_equal'] for c in r['cases'])
    final=read(root/'final.json');assert set(final['allGPU_sha256'])==set(TASKS);reports={}
    for s in ('S1','S2'):
        checked=states(root,s);prod=read(root/(s+'_evaluation.json'));assert sha(root/(s+'_evaluation.json'))==final['evaluation_sha256'][s];cfg=read(SOURCE/s/'config.json');data=torch.load(root/'inputs'/(s+'_unlabeled.pt'),map_location='cpu');outputs={m:torch.load(root/(s+'_'+m)/'predictions.pt',map_location='cpu') for m in MODES};results={}
        for role,rows in data['evaluation'].items():
            truth=load_truth(cfg['dataset']['json_file'] if role=='validation' else cfg['baseline']['annotation_for_offline_evaluation'],cfg['src_val_split'] if role=='validation' else cfg['tgt_val_split']);methods={'original':rows};methods.update({m+'/'+v:outputs[m][role][v] for m in MODES for v in READOUTS});results[role]={}
            for name,rs in methods.items():
                assert [r['video'] for r in rs]==[r['video'] for r in rows];assert all(torch.isfinite(r['segments']).all() and torch.isfinite(r['scores']).all() and len(r['segments'])==len(r['scores'])==len(r['labels']) for r in rs)
                frame=pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c),float(v)) for r in rs for b,c,v in zip(r['segments'],r['labels'],r['scores'])],columns=['video-id','t-start','t-end','label','score']);curves=[]
                for c in range(10):
                    a=truth[truth.label==c].reset_index(drop=True);b=frame[frame.label==c].reset_index(drop=True);value=IndependentAP(a,b).ap(np.ones(len(b),bool));ref=official(a,b) if len(a) and len(b) else np.zeros(5);assert np.allclose(value,ref,atol=1e-7,rtol=0);curves.append(value)
                old=prod['results'][role][name];assert np.allclose(curves,old['per_class_tiou_AP'],atol=1e-7,rtol=0) and abs(np.mean(curves)-old['map'])<1e-7;assert abs(old['delta_pp']-100*(old['map']-prod['results'][role]['original']['map']))<1e-7;results[role][name]=old
            if role=='target_val':assert abs(results[role]['original']['map']-read(REFERENCE/s/'evaluation/target_val.json')['map_0.1_to_0.5'])<1e-7
        reports[s]=dict(states=checked,results=results);save(root/'independent_results'/(s+'.json'),reports[s]);print('IACS_AP_AUDITED',s,flush=True)
    means={role:{name:float(np.mean([reports[s]['results'][role][name]['delta_pp'] for s in ('S1','S2')])) for name in reports['S1']['results'][role]} for role in ('validation','target_val')}
    save(root/'independent_final_audit.json',dict(passed=True,code=code_identity(),production_final_sha256=sha(root/'final.json'),results_sha256={s:sha(root/'independent_results'/(s+'.json')) for s in reports},mean_delta_pp=means,exact_new_trainings=4,exact_shifts=2,readouts=list(READOUTS),target_train_GT_read=False,source='allsourceTRAIN; no head-onlyheldout evaluation remains',replay_final_sha256=sha(replayroot/'final.json'),full_saved_EMA_all_videos_three_methods_GPU_arrays_bitwise_reproduced=True,scope='S1/S2 historically exposed development; not fullsix final; originalNMS reused, no independentNMS rewrite'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
