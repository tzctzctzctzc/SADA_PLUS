"""完整产品读出/原生重放证据、不可变源状态和全部操作点AP独审。"""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools import audit_native_joint as audit
from tools.run_quality_guard import ORIGIN,BASEMODES,MODES,JOBS


def independent_risk(p,q):
    return p*(np.float32(1)-q)


def readonly_audit(root,s,rows):
    import torch
    owner=read(root/'state.json')['owner_pid'];plan=read(root/'plan.json');results={}
    for m in BASEMODES:
        entry=root/(s+'_'+m);paired=root/(s+'_'+m+'guard');old=ORIGIN/entry.name;r=read(entry/'gpu_complete.json');g=read(r['guard']);lease=read(entry/'lease_replay.json');og=read(old/'gpu_complete.json')
        assert r['passed'] and r['optimizer_steps']==0 and r['parameters_CPUCUDA_RNG_unchanged'] and r['all_original_BG_and_eligibility_bitwise_replayed'] and r['all_BG_quality_readouts_disk_reconstruction_bitwise_exact']
        assert r['all_sampled_head_logits_disk_replay_bitwise_equal'] and r['first3_duplicate_native_logits_bitwise_equal']
        assert len(r['direct_duplicate_seconds'])==len(r['cached_duplicate_seconds'])==3 and sum(r['direct_duplicate_seconds'])>sum(r['cached_duplicate_seconds'])
        reuse=plan['reused_complete'].get(entry.name)
        expected_owner=reuse['source_owner_pid'] if reuse else owner
        expected_commit=reuse['source_commit'] if reuse else plan['code']['commit']
        budget=read(entry/'throughput.json');assert budget['limit']==(300 if reuse else 1200) and budget['one_pass_and_diskhead_replay_estimate_seconds']<budget['limit']
        if reuse:
            source=Path(reuse['source_root'])/entry.name
            assert sha(entry/'gpu_complete.json')==reuse['GPU_sha256'] and sha(entry/'readout.pt')==reuse['readout_sha256']
            for f in ('predictions.pt','readout.pt','gpu_complete.json','throughput.json'):assert os.stat(entry/f).st_ino==os.stat(source/f).st_ino
        assert sha(old/'epoch020.pt')==r['source_checkpoint_sha256']==og['checkpoint_sha256'] and r['final_digest']==og['final_digest']
        assert sha(old/'gpu_complete.json')==r['origin_GPU_sha256'] and sha(ORIGIN/'independent_final_audit.json')==r['origin_final_audit_sha256']
        assert sha(r['guard'])==r['guard_sha256'] and sha(entry/'lease_replay.json')==g['lease_sha256'];assert g['owner_pid']==lease['owner_pid']==expected_owner
        assert g['observed']['uuid']==lease['uuid'] and g['observed']['free_mib']>=12288 and lease['commit']==expected_commit
        assert sha(entry/'readout.pt')==r['readout_sha256'];assert sha(entry/'predictions.pt')==r['original_predictions_sha256']==og['predictions_sha256'];assert sha(paired/'predictions.pt')==r['guarded_predictions_sha256']
        assert os.stat(entry/'predictions.pt').st_ino==os.stat(old/'predictions.pt').st_ino
        raw=torch.load(entry/'readout.pt',map_location='cpu');original=torch.load(entry/'predictions.pt',map_location='cpu');protected=torch.load(paired/'predictions.pt',map_location='cpu');count=0
        for role in ('fit','heldout','validation','target_val'):
            assert len(raw[role])==len(original[role])==len(protected[role])
            for a,b,c in zip(raw[role],original[role],protected[role]):
                assert a['video']==b['video']==c['video']
                assert torch.equal(a['pbg'],b['pbg']) and torch.equal(a['eligible'],b['eligible']) and torch.equal(a['eligible'],c['eligible'])
                for k in ('segments','scores','labels'):assert torch.equal(b[k],c[k])
                q=a['quality'].numpy();p=a['pbg'].numpy();assert ((q>=0)&(q<=1)).all() and np.isfinite(q).all()
                expected=independent_risk(p,q);assert np.array_equal(expected,a['risk'].numpy()) and np.array_equal(expected,c['pbg'].numpy());assert (q[b['scores'].numpy()<.3]==0).all();count+=1
        results[m]=dict(readout_sha256=r['readout_sha256'],GPU_sha256=sha(entry/'gpu_complete.json'),source_checkpoint_sha256=r['source_checkpoint_sha256'],full_videos_checked=count,all_float32_products_exact=True,all_original_arrays_and_pBG_exact=True,optimizer_steps=0)
    return results


def run(root):
    while not (root/'final.json').exists():
        if (root/'error.json').exists():raise RuntimeError('producer failed; no automatic restart')
        time.sleep(3)
    orig=read(ORIGIN/'independent_final_audit.json');assert orig['passed'];assert sha(ORIGIN/'final.json')==orig['production_final_sha256']
    for s,digest in orig['results_sha256'].items():assert sha(ORIGIN/'independent_results'/(s+'.json'))==digest
    final=read(root/'final.json');assert set(final['allGPU_sha256'])==set(JOBS) and final['optimizer_steps']==0 and final['GPU_replays']==8
    plan=read(root/'plan.json');assert len(plan['reused_complete'])==6 and len(plan['remaining'])==2 and plan['prior_own_group_exited']
    for task,digest in final['allGPU_sha256'].items():assert sha(root/task/'gpu_complete.json')==digest
    audit.MODES=MODES;audit.states_audit=readonly_audit
    audit.teacher_audit=lambda root,s:dict(prior_full_teacher_state_AP_audit_sha256=sha(ORIGIN/'independent_final_audit.json'),prior_shift_audit_sha256=orig['results_sha256'][s],reused_unchanged_source_audit=True)
    for s in ('S1','S2'):
        audit.result_audit(root,s)
        r=read(root/'independent_results'/(s+'.json'));old=read(ORIGIN/'independent_results'/(s+'.json'))
        for role in r['results']:
            for m in (*BASEMODES,'confidence'):
                for a,b in zip(r['results'][role][m]['budgets'],old['results'][role][m]['budgets']):assert a==b
    save(root/'independent_final_audit.json',dict(passed=True,code=code_identity(),origin_audit_sha256=sha(ORIGIN/'independent_final_audit.json'),production_final_sha256=sha(root/'final.json'),results_sha256={s:sha(root/'independent_results'/(s+'.json')) for s in ('S1','S2')},exact_GPU_readonly_replays=8,exact_existing_models=4,exact_readouts=8,exact_shifts=2,exact_rates=4,all_original_baseline_decisions_and_AP_reproduced=True,all_quality_product_values_NumPy_exact=True,optimizer_steps=0,target_train_GT_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
