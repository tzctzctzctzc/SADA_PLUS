from pathlib import Path
import json
import pytest
from tools.temporal_energy_resume import prepare_resume
from tools.baseline_protocol import save,read,sha


def fixture(old,spec):
    snapshot=old/'monitor/snapshot';snapshot.mkdir(parents=True)
    receipt=dict(all_owned_processes_exited=True,parent={'pid':99999991},
        workers={},snapshot_root=str(snapshot),saved={})
    save(old/'plan.json',{'specification':spec});save(old/'inputs.json',{})
    save(old/'acceptance.json',{'minimum_free_mib':15254})
    for s in ['S1','S2','S3','S4','S5','S6']:
        e=old/'formal'/s;e.mkdir(parents=True)
        for name in ['config.json','inputs.json','source_train_gt.json','source_val_gt.json']:
            save(e/name,{})
        for name in ['source_val_unlabeled.json','target_val_unlabeled.json','target_train_unlabeled.json']:
            save(e/name,{'database':{'v':{'duration':1}}})
    e=snapshot/'S1';e.mkdir()
    for name in ['best_energy.pt','latest_energy.pt','energy_history.json','acceptance.json']:
        (e/name).write_bytes(name.encode())
    for role in ['source_train','source_val']:save(old/'formal/S1'/(role+'_cache_index.json'),{})
    receipt['saved']['S1']={'epoch':7,'artifacts':{p.name:sha(p) for p in e.iterdir()}}
    save(snapshot/'receipt.json',receipt)
    save(old/'monitor/authorized_pause_latest.json',{'receipt':str(snapshot/'receipt.json')})
    return receipt


def test_resume_copies_exact_prefix_and_leaves_pending_shifts_fresh(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';new.mkdir();spec={'schema':'test'};fixture(old,spec)
    assert prepare_resume(new,old,{'commit':'new'},spec)==15254
    assert read(new/'formal/S1/resume.json')['prefix_epoch']==7
    assert (new/'formal/S1/latest_energy.pt').read_bytes()==b'latest_energy.pt'
    assert not (new/'formal/S2/resume.json').exists()
    assert read(new/'acceptance.json')['new_worker_gpu_equivalence_required']


def test_resume_rejects_modified_checkpoint(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';new.mkdir();spec={'schema':'test'};fixture(old,spec)
    (old/'monitor/snapshot/S1/latest_energy.pt').write_bytes(b'changed')
    with pytest.raises(AssertionError):prepare_resume(new,old,{'commit':'new'},spec)
