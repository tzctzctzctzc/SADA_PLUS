import json
from pathlib import Path

import pytest
from tools import run_rq3_enhancement_dual as dual
from tools import run_rq3_enhancement as single
from tools.rq2_dense_protocol import sha256_file
from tools.rq3_enhancement import ORDER, specification


def test_only_one_own_worker_per_gpu_and_next_shift_fills_finished_card():
    assert dual.dispatch_pending(['S1','S2','S3'],{},['2','1']) == [('2','S1'),('1','S2')]
    assert dual.dispatch_pending(['S3','S4'],{'a':{'gpu':'2'}},['2','1']) == [('1','S3')]
    assert dual.dispatch_pending(['S3','S4'],{'a':{'gpu':'1'}},['2','1']) == [('2','S3')]
    assert dual.dispatch_pending(['S3'],{'a':{'gpu':'1'},'b':{'gpu':'2'}},['2','1']) == []


@pytest.mark.parametrize('variant',ORDER)
@pytest.mark.parametrize('shift',['S1','S4'])
def test_parallel_uses_same_sequential_command_and_stopping_budget(tmp_path,variant,shift):
    ref=dict(rq2_formal={'target_manifest':{'path':'target.json'}},
             rq3_auxiliary=dict(cache_path='cache.json',manifest_path='manifest.json',
                                calibration_path='calibration.json'))
    if shift=='S4':
        ref['rq3_auxiliary']['early_stopping']={'patience_epochs':15}
    name,run_root,common,command=single.entry_command(tmp_path,tmp_path,variant,shift,ref)
    assert command[command.index('--rq3-early-stop-patience')+1] == ('0' if shift=='S1' else '15')
    assert command[command.index('--rq3-batch-size')+1]=='2'
    assert command[command.index('-s')+1]=='6'
    assert command[command.index('--rq3-enhancement')+1]==variant
    assert str(run_root)==str(tmp_path/'model_results'/name)
    assert dual.single.entry_command is single.entry_command


def test_partial_or_changed_acceptance_cannot_release_second_student(tmp_path):
    report=dict(status='passed',specification=specification('control'),gpu_admission={})
    single.save(tmp_path/'acceptance.json',report)
    proof=dict(code_commit='head',variant='control',inputs_sha256='inputs',
               acceptance_sha256=sha256_file(tmp_path/'acceptance.json'))
    single.save(tmp_path/'acceptance_ready.json',proof)
    dual.ready_acceptance(tmp_path,'control','head','inputs')
    for head,inputs in [('wrong','inputs'),('head','changed')]:
        with pytest.raises(ValueError):
            dual.ready_acceptance(tmp_path,'control',head,inputs)
    report['status']='failed'
    single.save(tmp_path/'acceptance.json',report)
    with pytest.raises(ValueError):
        dual.ready_acceptance(tmp_path,'control','head','inputs')
