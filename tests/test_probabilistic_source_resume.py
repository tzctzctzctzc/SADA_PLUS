from copy import deepcopy
import json
import pytest
import torch
from tools import resume_probabilistic_source_six as recovery


def sample():
    saved=dict(epoch=10,history=[[5,.8],[10,.7]],rng=dict(cuda=[torch.ones(8,dtype=torch.uint8)],
        loaders={k:None for k in ('source','target','validation','perturbation')}))
    rows=[dict(epoch=i,**({'source_validation_loss':.8 if i==5 else .7} if i%5==0 else {})) for i in range(1,15)]
    return saved,rows


def test_prefix_retains_full_patience_history_and_drops_unsaved_epochs():
    saved,rows=sample();retained=recovery.prefix(saved,rows)
    assert [r['epoch'] for r in retained]==list(range(1,11))
    saved['history']=[[5,.8],[10,.9]]
    with pytest.raises(ValueError):recovery.prefix(saved,rows)
    saved,rows=sample();saved['rng']['cuda']=[]
    with pytest.raises(ValueError):recovery.prefix(saved,rows)


def test_compatible_only_allows_code_and_temporary_path_changes():
    a=dict(baseline=dict(code={'commit':'old'},tmpdir='/old',shift='S4'),seed=123,lr=.001)
    b=deepcopy(a);b['baseline'].update(code={'commit':'new'},tmpdir='/new');recovery.compatible(a,b)
    b['lr']=.002
    with pytest.raises(ValueError):recovery.compatible(a,b)


def test_clone_keeps_original_files_weights_and_history(tmp_path):
    old=tmp_path/'old';entry=tmp_path/'new';old.mkdir();entry.mkdir();saved,rows=sample()
    cfg=dict(baseline=dict(code={'commit':'old'},tmpdir='/old'),seed=123)
    saved.update(config=cfg,state_dict={'weight':torch.tensor([1.,2.])})
    for n in ('best','latest'):torch.save(saved,old/(n+'.pt'))
    (old/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (old/'train.log').write_text('original log');(old/'initialization.json').write_text('{}')
    before={p.name:p.read_bytes() for p in old.iterdir()}
    new=deepcopy(cfg);new['baseline'].update(code={'commit':'new'},tmpdir='/new')
    recovery.clone_partial(old,entry,new)
    actual=torch.load(entry/'latest.pt');assert actual['config']==new
    assert torch.equal(actual['state_dict']['weight'],saved['state_dict']['weight'])
    assert actual['history']==saved['history']
    assert all(p.read_bytes()==before[p.name] for p in old.iterdir())
    assert len((entry/'metrics.jsonl').read_text().splitlines())==10
