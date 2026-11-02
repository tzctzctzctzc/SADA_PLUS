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
