from copy import deepcopy
import hashlib
import io
import json

import pytest

from tools.cold_oracle_resume import paired_config,validate_prefix


def fixture():
    digest='0'*64;lines=[]
    for step in range(1,7):
        event=dict(step=step,epoch=(step-1)//2+1,observations=[])
        digest=hashlib.sha256((digest+json.dumps(event,sort_keys=True)).encode()).hexdigest()
        lines.append((json.dumps(dict(event,digest=digest))+'\n').encode())
    saved=dict(epoch=2,history=[],model_controls=dict(cold_oracle_state=dict(step=4,digest=json.loads(lines[3])['digest'])))
    return saved,[dict(epoch=1),dict(epoch=2)],lines


def test_resume_keeps_exact_committed_ledger_prefix_and_discards_partial_epoch():
    saved,rows,lines=fixture()
    assert validate_prefix(saved,rows,2,iter(lines))==b''.join(lines[:4])


def test_resume_rejects_history_missing_epoch_and_damaged_chain():
    saved,rows,lines=fixture()
    with pytest.raises(AssertionError):validate_prefix(saved,rows[1:],2,iter(lines))
    with pytest.raises(AssertionError):validate_prefix(saved,rows,2,iter(lines[:3]))
    lines[1]=lines[1].replace(b'"step": 2',b'"step": 3')
    with pytest.raises(AssertionError):validate_prefix(saved,rows,2,iter(lines))


def test_resume_only_allows_execution_code_metadata_change():
    old=dict(baseline=dict(code='a',iterations=7),seed=123)
    new=deepcopy(old);new['baseline']['code']='b';paired_config(old,new)
    new['seed']=124
    with pytest.raises(AssertionError):paired_config(old,new)


def test_matrix_only_resumes_missing_job_and_preserves_original_files(tmp_path):
    import os
    import torch
    from tools.baseline_protocol import save,sha
    from tools.cold_oracle_resume import prepare_matrix
    old=tmp_path/'old';new=tmp_path/'new';results=[]
    saved,rows,lines=fixture()
    for mode in ('joint','classification','localization'):
        for shift in ('S1','S2','S3','S4','S5','S6'):
            entry=old/'formal'/mode/shift;entry.mkdir(parents=True)
            save(entry/'config.json',dict(baseline=dict(code='original',epoch_budget=dict(iterations_per_epoch=2))))
            if (mode,shift)==('joint','S6'):
                torch.save(saved,entry/'latest.pt');save(entry/'metrics.json',rows)
                (entry/'oracle_events.jsonl').write_bytes(b''.join(lines))
            else:
                save(entry/'complete.json',dict(status='complete'))
                results.append(dict(mode=mode,shift=shift,complete_sha256=sha(entry/'complete.json')))
    save(old/'progress.json',dict(results=results));save(old/'acceptance.json',dict(status='passed'))
    pause=tmp_path/'pause.json';save(pause,dict(status='stopped',root=str(old),processes=[]))
    configs,done,pending,signals=prepare_matrix(old,new,'new',pause)
    assert len(done)==17 and list(configs)==[('joint','S6')]
    assert pending==[('joint','S6','train',None)] and not signals
    entry=new/'formal/joint/S6'
    assert (entry/'oracle_events.jsonl').read_bytes()==b''.join(lines[:4])
    assert os.path.samefile(entry/'latest.pt',old/'formal/joint/S6/latest.pt')
    before=sha(old/'formal/joint/S6/latest.pt')
    # 原子替换新保存点不能污染旧工件。
    temp=entry/'latest.pt.tmp';torch.save(dict(epoch=3),temp);temp.replace(entry/'latest.pt')
    assert sha(old/'formal/joint/S6/latest.pt')==before
