from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from test_dyfadet_text_integration import textconfig
from test_dyfadet_uda import fixed_pseudo
from tools import dyfadet_text_staged_engine as e
from tools.dyfadet_text_staged_protocol import decision,stage_config
from tools.evaluate_text_source_selection import evaluate

def test_source_random_epoch0_cannot_win_and_ap_is_maximized():
    h=[[0,.99],[5,.5],[10,.6],[15,.55],[20,.59],[25,.58]]
    assert decision(h[:1],'source_pretrain')['best_epoch'] is None
    d=decision(h,'source_pretrain');assert d['best_epoch']==10 and d['reason']=='source_AP_patience'
    assert all(decision(h[:i],'source_pretrain')['reason']=='running' for i in range(1,len(h)))
    assert decision([[0,.6],[5,.5],[10,.55],[15,.59]],'uda')['best_epoch']==0

def test_stages_have_explicit_different_budgets_without_mutating_config():
    c={'opt':{'learning_rate':.1},'benchmark':{}}
    old=deepcopy(c);a=stage_config(c,'source_pretrain');b=stage_config(c,'uda')
    assert c==old and a['opt']['epochs']==50 and b['opt']['epochs']==30
    with pytest.raises(AssertionError):decision([[0,.5],[7,.6]],'uda')

@pytest.mark.parametrize('variant',['tifad','pda'])
def test_trained_student_handoff_discards_random_ema_and_has_exact_restore(variant,tmp_path,monkeypatch):
    cfg=textconfig(tmp_path,variant);ctx=e.build(cfg,'source_pretrain',device='cpu',workers=0)
    src=next(iter(ctx['train']));before=e.base.digest(ctx['model'].semantic.state_dict())
    for _ in range(4):e.eng.update(ctx,src)
    assert before!=e.base.digest(ctx['model'].semantic.state_dict())
    s=deepcopy(e.capture(ctx,1,[],[]))
    fresh=e.eng.build(stage_config(cfg,'uda'),'uda','cpu',0)
    proof=e.bootstrap(fresh,s)
    assert proof['teacher_equals_trained_student'] and not fresh['opt'].state
    assert e.base.digest(fresh['ema'].module.state_dict())!=e.base.digest(s['ema'])
    with pytest.raises(AssertionError):e.bootstrap(fresh,dict(s,epoch=0))
    monkeypatch.setattr(e.eng,'pseudo',fixed_pseudo)
    target=next(iter(fresh['target']))
    for _ in range(4):e.eng.update(fresh,src,target)
    checkpoint=deepcopy(e.capture(fresh,1,[],[]));e.restore(fresh,deepcopy(checkpoint))
    assert e.base.digest(e.capture(fresh,1,[],[]))==e.base.digest(checkpoint)
    a,_=e.eng.update(fresh,src,target);da=e.base.digest(e.capture(fresh,1,[],[]))
    e.restore(fresh,deepcopy(checkpoint));b,_=e.eng.update(fresh,src,target)
    assert a==b and da==e.base.digest(e.capture(fresh,1,[],[]))

def test_full_source_ap_uses_final_scores_and_only_source_manifest(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    gt=tmp_path/'source.json';manifest=tmp_path/'unlabeled.json'
    gt.write_text(json.dumps({'database':{'v':{'subset':'validation','annotations':[{'segment':[1.,2.],'label_id':0}]}}}))
    manifest.write_text(json.dumps({'database':{'v':{'subset':'validation'}}}))
    values=[]
    for i,score in enumerate([[.9,.8],[.8,.9]]):
        p=tmp_path/f'{i}.npz';out=tmp_path/f'{i}.json'
        np.savez_compressed(p,**{'video-id':np.array(['v','v']),'t-start':np.array([1.,3.]),'t-end':np.array([2.,4.]),'label':np.array([0,0]),'score':np.array(score)})
        evaluate(str(gt),str(p),str(manifest),str(out));r=json.loads(out.read_text());values.append(r['map_0.1_to_0.5'])
        assert not r['target_GT_used'] and r['annotation']==str(gt)
    assert values==[1.,.5]
