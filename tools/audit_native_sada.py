"""独立审计全新SADA训练状态、目标无标签游标、原点分配及完整配对AP。"""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools import audit_native_joint as audit
from tools.run_native_sada import ORIGIN,MODES,TASKS


def independent_labels(points,segments,labels,center_sample,radius,reg_distance):
    points=np.asarray(points,np.float32);segments=np.asarray(segments,np.float32).reshape(-1,2);labels=np.asarray(labels,int);n=len(points)
    if not len(segments):return np.zeros((n,10),np.float32)
    lengths=np.broadcast_to((segments[:,1]-segments[:,0])[None],(n,len(segments))).copy();left=points[:,0,None]-segments[None,:,0];right=segments[None,:,1]-points[:,0,None]
    if center_sample=='radius':
        middle=np.float32(.5)*(segments[:,0]+segments[:,1]);low=np.maximum(middle[None]-points[:,3,None]*np.float32(radius),segments[None,:,0]);high=np.minimum(middle[None]+points[:,3,None]*np.float32(radius),segments[None,:,1]);inside=np.minimum(points[:,0,None]-low,high-points[:,0,None])>0
    else:inside=np.minimum(left,right)>0
    lengths[~inside]=np.inf
    if reg_distance:
        distance=np.maximum(left,right);lengths[~((distance>=points[:,1,None])&(distance<=points[:,2,None]))]=np.inf
    shortest=lengths.min(1);matches=((lengths<=shortest[:,None]+np.float32(.001))&np.isfinite(lengths)).astype(np.float32)
    return np.minimum(matches@np.eye(10,dtype=np.float32)[labels],1)


def states(root,s,rows):
    import torch
    from tools import baseline_engine as base
    from tools.run_native_joint import SOURCE,DENSE
    from tools.run_uda_background import TargetStream
    from tools.run_foreground_calibration_oracle import REFERENCE
    cfg=read(SOURCE/s/'config.json');truth=read(SOURCE/s/'source_truth.json');original=torch.load(REFERENCE/s/'best.pt',map_location='cpu')['ema_state_dict'];data=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu');target=data['roles']['target_train'];results={};owner=read(root/'state.json')['owner_pid'];prefix=[]
    for mode in MODES:
        e=root/(s+'_'+mode);gpu=read(e/'gpu_complete.json');active=mode!='control'
        assert sha(e/'gpu_complete.json')==read(root/'final.json')['allGPU_sha256'][e.name]
        for file,key in [('kernel_complete.json','kernel_sha256'),('epoch020.pt','checkpoint_sha256'),('predictions.pt','predictions_sha256'),('source_point_labels.pt','source_point_labels_sha256'),('fresh_epoch5_equivalence.json','fresh_epoch5_equivalence_sha256')]:assert sha(e/file)==gpu[key]
        k=read(e/'kernel_complete.json');assert k['passed'] and k['disabled_DA_native_loss_allgrad_AdamW_exact'] and k['complete_model_twoopts_order_target_PythonNumPyCPUCUDA_RNG_reconstruction_nextstep_exact'];assert k['original_SADA_conditional_loss_allgrad_dualAdamW_exact']==active and sha(e/'kernel.pt')==k['kernel_sha256']
        assert k['legacy_sync_loss_allgrad_dualAdamW_and_RNG_exact']==active
        if active:
            assert sha(e/'efficiency_benchmark.json')==k['efficiency_benchmark_sha256'];benchmark=read(e/'efficiency_benchmark.json');assert benchmark['passed_numeric'] and benchmark['full_model_dualopt_grad_and_RNG_exact']
            assert len(benchmark['legacy_seconds'])==len(benchmark['optimized_seconds'])==10 and sum(benchmark['legacy_seconds'])>sum(benchmark['optimized_seconds']) and benchmark['speedup']>1
        assert gpu['no_background_checkpoint_or_history_loaded_for_training'] and gpu['discriminator_or_embeddings_changed']==active and gpu['saved_model_GPU_predictions_bitwise_equal'] and gpu['readout_parameters_RNG_unchanged']
        g=read(gpu['guard']);lease=read(e/'lease_train.json');assert sha(gpu['guard'])==gpu['guard_sha256']==k['guard_sha256'] and sha(e/'lease_train.json')==g['lease_sha256'];assert g['owner_pid']==lease['owner_pid']==owner and g['observed']['uuid']==lease['uuid'] and g['observed']['free_mib']>=12288
        h=read(e/'history.json');assert [a['epoch'] for a in h]==list(range(1,21));assert all(a['alignment_active']==(active and a['epoch']>5) and a['joint_backbone']==(a['epoch']>1) and a['steps']==len(rows) for a in h)
        assert all(a['adv']==0 for a in h[:5]);order=torch.Generator().manual_seed(123);stream=TargetStream(len(target))
        for epoch in range(1,21):
            torch.randperm(len(rows),generator=order)
            if active and epoch>5:stream.draw(len(rows))
            if epoch not in (1,5,10,20):continue
            state=torch.load(e/f'epoch{epoch:03d}.pt',map_location='cpu');assert state['epoch']==epoch and state['step']==epoch*len(rows) and state['history']==h[:epoch] and torch.equal(order.get_state(),state['order_rng'])
            actual=state['target_order'];expected=stream.state();assert actual['cursor']==expected['cursor']
            for field in ('generator','order','seen'):assert torch.equal(actual[field],expected[field])
            assert all(name in state for name in ('model','optimizer','disc_optimizer','python_rng','numpy_rng','cpu_rng','cuda_rng'));assert state['cpu_rng'].dtype==state['cuda_rng'].dtype==torch.uint8
            assert state['input_sha256']==sha(DENSE/'inputs'/s/'data.pt') and state['source_teacher_sha256']==sha(ORIGIN/s/'teacher.pt')
            o=state['optimizer'];body=set(o['param_groups'][0]['params']);head=set(o['param_groups'][1]['params'])
            for index,v in o['state'].items():assert int(v['step'])==((epoch-1)*len(rows) if index in body else epoch*len(rows)) and index in body|head
            if not active or epoch<=5:assert not state['disc_optimizer']['state']
            for value in state['disc_optimizer']['state'].values():assert 0<int(value['step'])<=max(0,epoch-5)*len(rows)
            core={n:v for n,v in state['model'].items() if n.startswith(('backbone.','neck.','head.'))}
            for name,value in state['model'].items():
                if name.startswith('cls_head.'):assert torch.equal(value,original[name])
                if (not active or epoch<=5) and (name.startswith('disc_dom_conditional_adv.') or name=='learnable_embeddings'):assert torch.equal(value,original[name])
                if epoch==1 and name.startswith('backbone.'):assert torch.equal(value,original['backbone_source.'+name[len('backbone.'):]])
                if epoch==1 and name.startswith('neck.'):assert torch.equal(value,original['neck_source.'+name[len('neck.'):]])
            if epoch==5 or not active and epoch==20:
                prior=ORIGIN/(s+'_jointcost')/f'epoch{epoch:03d}.pt';old=torch.load(prior,map_location='cpu');assert base.tensor_digest(core)==base.tensor_digest(old['model']) and base.tensor_digest(o)==base.tensor_digest(old['optimizer'])
                assert os.stat(e/f'epoch{epoch:03d}.pt').st_ino!=os.stat(prior).st_ino
                if epoch==5:prefix.append(base.tensor_digest(core))
            if epoch==20:assert base.tensor_digest(state['model'])==gpu['final_digest'] and base.tensor_digest(core)==gpu['final_core_digest']
        assert os.stat(e/'epoch020.pt').st_ino==os.stat(e/'latest.pt').st_ino
        assert gpu['target_draws']==(15*len(rows) if active else 0) and gpu['target_seen']==(len(target) if active else 0)
        assignments=torch.load(e/'source_point_labels.pt',map_location='cpu');assert set(assignments)==({r['video'] for r in rows} if active else set())
        for vid,record in assignments.items():
            points=torch.cat(record['points']).numpy();expected=independent_labels(points,truth[vid]['segments'],truth[vid]['labels'],record['center_sample'],record['radius'],record['reg_distance']);actual=torch.cat(record['labels']).numpy();assert np.array_equal(expected,actual)
            for level,(point,mask,label) in enumerate(zip(record['points'],record['masks'],record['labels'])):
                p=point.numpy();assert np.array_equal(p[:,0],np.arange(len(p))*2**level) and np.all(p[:,3]==2**level) and len(mask)==len(p)==len(label)
                assert np.array_equal(p[:,1:3],np.broadcast_to(np.array(cfg['model']['regression_range'][level],np.float32),(len(p),2)))
        results[mode]=dict(all_original_SADA_point_labels_independent_NumPy_exact=True,source_point_videos=len(assignments),all_fresh_epoch5_weights_and_optimizer_exact=True,no_checkpoint_prefix_imported=True,complete_model_dualopt_and_all_RNG_checked=True,target_cyclic_all_seen=active,GPU_sha256=sha(e/'gpu_complete.json'),checkpoint_sha256=sha(e/'epoch020.pt'))
        print('NATIVE_SADA_STATE_AUDITED',s,mode,flush=True)
    assert len(set(prefix))==1
    return results


def run(root):
    while not (root/'final.json').exists():
        if (root/'error.json').exists():raise RuntimeError('producer failed; no restart')
        time.sleep(3)
    old=read(ORIGIN/'independent_final_audit.json');assert old['passed'];assert sha(ORIGIN/'final.json')==old['production_final_sha256'];assert set(read(root/'final.json')['allGPU_sha256'])==set(TASKS)
    audit.MODES=MODES;audit.states_audit=states;audit.teacher_audit=lambda root,s:dict(reused_unchanged_original_input_and_teacher_audit_sha256=sha(ORIGIN/'independent_final_audit.json'),original_shift_audit_sha256=old['results_sha256'][s])
    for s in ('S1','S2'):
        assert sha(ORIGIN/'independent_results'/(s+'.json'))==old['results_sha256'][s];audit.result_audit(root,s)
        result=read(root/'independent_results'/(s+'.json'));prior=read(ORIGIN/'independent_results'/(s+'.json'))
        for role in result['results']:
            for a,b in zip(result['results'][role]['control']['budgets'],prior['results'][role]['jointcost']['budgets']):assert a==b
    save(root/'independent_final_audit.json',dict(passed=True,code=code_identity(),production_final_sha256=sha(root/'final.json'),results_sha256={s:sha(root/'independent_results'/(s+'.json')) for s in ('S1','S2')},exact_shifts=2,exact_methods=3,exact_fresh_trainings=6,exact_rates=4,all_source_point_assignment_independent=True,all_fresh_prefixes_and_target_streams_exact=True,all_original_AP_controls_reproduced=True,target_train_GT_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
