import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tools.resume_prototype_weights_four import EXPECTED,remaining_jobs,process_status,has_finished,stop_frozen_parent
from tools.resume_prototype_weights_four import resource_policy,find_entries,dependency_results


def test_exact_remainder_never_repeats_completed_or_inflight():
    completed=[(v,f'S{i}') for v in ['weight_020','weight_040'] for i in [1,2]]
    legacy=[('weight_020','S3')]
    result=remaining_jobs(completed,legacy,legacy)
    assert len(result)==7 and not set(result)&(set(completed)|set(legacy))
    assert set(result)|set(completed)|set(legacy)==EXPECTED


def test_unknown_partial_requires_inspection_not_fresh_training():
    with pytest.raises(ValueError):remaining_jobs([],[],[('weight_020','S3')])
    with pytest.raises(ValueError):remaining_jobs([('weight_005','S1')],[],[])
    with pytest.raises(ValueError):remaining_jobs([('weight_020','S1')],[('weight_020','S1')],[])


def test_reused_pid_is_not_signalled():
    current=process_status(os.getpid())
    with pytest.raises(ValueError):has_finished(os.getpid(),'wrong-start-time')
    assert not has_finished(os.getpid(),current['start_ticks'])


def test_no_inflight_is_supported():
    assert has_finished(None,None)
    assert len(remaining_jobs([],[],[]))==12


def test_measured_training_budget_allows_10GiB_but_evaluation_stays_12(tmp_path):
    from tools.baseline_protocol import save,sha
    p=tmp_path/'weight_020/result.json';save(p,dict(phases={'1':dict(peak_reserved_mib=4128.),'5':dict(peak_reserved_mib=4472.)}))
    acceptance=dict(report_sha256={str(p):sha(p)})
    cfg=dict(loader=dict(batch_size=2),dataset=dict(input_dim=2304,max_seq_len=2304))
    policy=resource_policy(acceptance,[cfg]*6)
    assert policy['training_minimum_free_mib']==9216 and policy['evaluation_minimum_free_mib']==12288
    assert 10052>=policy['training_minimum_free_mib'] and 10052<policy['evaluation_minimum_free_mib']
    cfg['loader']['batch_size']=4
    with pytest.raises(ValueError):resource_policy(acceptance,[cfg])


def test_lineage_includes_old_results_and_rejects_duplicate_directories(tmp_path):
    from tools.baseline_protocol import save
    old=tmp_path/'old';new=tmp_path/'new';entry=old/'formal/weight_020/S1';entry.mkdir(parents=True)
    save(new/'predecessor.json',dict(previous_root=str(old)))
    assert find_entries(new)['weight_020','S1']==entry
    (new/'formal/weight_020/S1').mkdir(parents=True)
    with pytest.raises(ValueError):find_entries(new)


def test_dependent_keeper_must_finish_before_its_parent_can_drain(tmp_path):
    from tools.baseline_protocol import save
    p=tmp_path/'status.json';save(p,dict(phase='waiting_original_stage'))
    record=dict(dependent_keepers=[dict(pid=None,start_ticks=None,status_path=str(p))])
    assert dependency_results(record) is None
    save(p,dict(phase='complete',results=[dict(variant='weight_020',shift='S3')]))
    assert len(dependency_results(record))==1


def test_fixed_training_padding_matches_memory_evidence():
    import ast
    from tools.baseline_protocol import CODE
    tree=ast.parse((CODE/'libs/models/SADA/meta_archs.py').read_text())
    preprocess=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='preprocessing')
    branch=next(n for n in preprocess.body if isinstance(n,ast.If))
    assert ast.unparse(branch.test)=="mode == 'train'"
    assert any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='max_len' for t in n.targets)
               and isinstance(n.value,ast.Attribute) and n.value.attr=='max_seq_len' for n in branch.body)


@pytest.mark.skipif(os.name=='nt',reason='真实Linux进程组/冻结语义')
def test_freezing_supervisor_does_not_interrupt_child_and_drain_is_safe(tmp_path):
    program='''import subprocess,sys,signal,time,os
from pathlib import Path
folder=Path(sys.argv[1])
child=subprocess.Popen([sys.executable,'-c',"import time;from pathlib import Path;time.sleep(1.);Path("+repr(str(folder/'child_complete'))+").write_text('ok')"])
(folder/'child_pid').write_text(str(child.pid))
def stop(s,f):raise SystemExit(143)
signal.signal(signal.SIGTERM,stop)
try:
 while True:time.sleep(.1)
finally:
 if child.poll() is None:child.terminate()
 child.wait()
'''
    parent=subprocess.Popen([sys.executable,'-B','-c',program,str(tmp_path)],start_new_session=True)
    record=None
    try:
        until=time.monotonic()+10
        while not (tmp_path/'child_pid').exists():
            assert time.monotonic()<until;time.sleep(.02)
        child=int((tmp_path/'child_pid').read_text());cstatus=process_status(child)
        os.kill(parent.pid,signal.SIGSTOP)
        while process_status(parent.pid)['state']!='T':time.sleep(.01)
        record=dict(parent=parent.pid,parent_start_ticks=process_status(parent.pid)['start_ticks'],
            children=[dict(pid=child,start_ticks=cstatus['start_ticks'])])
        with pytest.raises(ValueError):stop_frozen_parent(record)
        while not has_finished(child,cstatus['start_ticks']):
            assert time.monotonic()<until;time.sleep(.02)
        assert (tmp_path/'child_complete').read_text()=='ok'
        stop_frozen_parent(record);parent.wait(timeout=5)
        assert parent.returncode==143
    finally:
        if parent.poll() is None:
            parent.terminate();os.kill(parent.pid,signal.SIGCONT);parent.wait(timeout=5)
