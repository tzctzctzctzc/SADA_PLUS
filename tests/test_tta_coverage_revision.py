"""候选覆盖修订仍要求跨视图类别和边界共识，不添加未选背景监督。"""
from copy import deepcopy
from pathlib import Path
from runpy import run_path
import torch
from tools import tta_core as core

helpers=run_path(str(Path(__file__).with_name('test_tta.py')))


def test_sub_half_confidence_candidates_can_enter_but_need_two_views(monkeypatch):
    model=type('Model',(),dict(max_div_factor=8))()
    v=core.unlabeled(helpers['video'](128))
    monkeypatch.setattr(core,'dense',lambda m,v:v)
    calls=[]
    def candidates(m,output,begin,end,total):
        calls.append(begin)
        return [dict(segment=torch.tensor([30.,50.]),label=0,level=0,score=torch.tensor(.2))]
    monkeypatch.setattr(core,'candidates',candidates)
    result=core.consensus(model,v)
    assert len(result)==1 and result[0]['support']==3 and abs(result[0]['probability']-.2)<1e-6
    calls.clear()
    def only_one(m,output,begin,end,total):
        calls.append(begin)
        return candidates(m,output,begin,end,total) if len(calls)==1 else []
    monkeypatch.setattr(core,'candidates',only_one)
    assert not core.consensus(model,v)


def test_revision_really_admits_point_two_confidence(tmp_path):
    model=helpers['small_model'](tmp_path);v=core.unlabeled(helpers['video'](64))
    output=core.dense(model,v)
    for logit,offset in zip(output['logits'],output['offsets']):
        with torch.no_grad():logit.fill_(-20);offset.fill_(1.)
    with torch.no_grad():output['logits'][0][0,32,0]=torch.logit(torch.tensor(.2))
    result=core.candidates(model,output,0,64,64)
    assert result and result[0]['label']==0 and .19<float(result[0]['score'])<.21
