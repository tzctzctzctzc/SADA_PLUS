import sys
from pathlib import Path
from copy import deepcopy
import io
import json
import random
import numpy as np
import pytest
import torch

from tools.dyfadet_protocol import CODE,VENDOR,SPEC,decision,terminal,read,sha
sys.path.insert(0,str(VENDOR))
from libs.core import load_config
from tools.dyfadet_engine import BenchmarkDataset,build,step,capture,restore,digest,dense,validation

def config(tmp_path):
    cfg=load_config(str(VENDOR/'configs/epic_slowfast_verb.yaml'))
    cfg['dataset'].update(num_classes=10,input_dim=16,max_seq_len=64,feat_folder=str(tmp_path))
    cfg['model'].update(num_classes=10,input_dim=16,max_seq_len=64,embd_dim=32,fpn_dim=32,head_dim=32,mlp_dim=64,max_buffer_len_factor=4.)
    cfg['model']['train_cfg']=cfg['train_cfg'];cfg['model']['test_cfg']=cfg['test_cfg']
    cfg['loader']['num_workers']=0
    data={'database':{}}
    for i in range(4):
        key='v'+str(i);np.savez(tmp_path/(key+'.npz'),feats=np.arange(96*16,dtype=np.float32).reshape(96,16)/1000)
        data['database'][key]=dict(subset='source',duration=52.,annotations=[dict(label_id=2,label='x',segment=[1.,12.])])
    p=tmp_path/'source.json';p.write_text(json.dumps(data))
    cfg['benchmark']=dict(steps_per_epoch=4,source_train=str(p),source_val=str(p));return cfg

def test_upstream_unmodified():
    for p,h in read(VENDOR/'upstream.json')['files'].items():assert sha(VENDOR/p)==h

def test_first_legal_stop():
    history=[[5,3.],[10,2.],[15,2.1],[20,2.2],[25,2.3]]
    assert terminal(history)['best_epoch']==10
    with pytest.raises(AssertionError):terminal(history+[[30,1.]])
    with pytest.raises(AssertionError):decision([[5,1],[15,.9]])

def test_data_coordinates_and_unlabeled(tmp_path):
    cfg=config(tmp_path);ds=BenchmarkDataset(cfg,cfg['benchmark']['source_train'])
    v=ds[0];assert v['feats'].shape==(16,96)
    torch.testing.assert_close(v['segments'],torch.tensor([[(30.-16)/16,(360.-16)/16]]))
    p=tmp_path/'unlabeled.json';db=read(cfg['benchmark']['source_train'])
    for x in db['database'].values():x.pop('annotations')
    p.write_text(json.dumps(db));u=BenchmarkDataset(cfg,p,unlabeled=True)[0]
    assert 'segments' not in u and 'labels' not in u
    with pytest.raises(AssertionError):BenchmarkDataset(cfg,cfg['benchmark']['source_train'],unlabeled=True)

def test_real_model_resume_and_gt_independence(tmp_path):
    torch.set_num_threads(2);cfg=config(tmp_path);ctx=build(cfg,device='cpu',workers=0)
    batch=next(iter(ctx['train']));before=digest(ctx['model'].state_dict())
    for _ in range(3):loss,norms=step(ctx,batch,True)
    assert before!=digest(ctx['model'].state_dict()) and all(v>0 for v in norms.values())
    b=io.BytesIO();torch.save(capture(ctx,cfg,1,[],[]),b);b.seek(0);saved=torch.load(b)
    restore(ctx,deepcopy(saved));assert digest(capture(ctx,cfg,1,[],[]))==digest(saved)
    l1,_=step(ctx,batch);d1=digest(capture(ctx,cfg,1,[],[]))
    restore(ctx,deepcopy(saved));l2,_=step(ctx,batch);d2=digest(capture(ctx,cfg,1,[],[]))
    assert l1==l2 and d1==d2
    m=ctx['model'];m.eval();v=next(iter(ctx['val']))
    with torch.no_grad():
        a=m(v);b=m.inference(v,*dense(m,v))
        u=[{k:x for k,x in item.items() if k not in ('segments','labels')} for item in v]
        c=m.inference(u,*dense(m,u))
    assert digest(a)==digest(b)==digest(c)
    old=m.loss_normalizer;rng=digest((torch.get_rng_state(),random.getstate(),np.random.get_state()))
    value=validation(m,ctx['val']);assert np.isfinite(value) and m.loss_normalizer==old
    assert rng==digest((torch.get_rng_state(),random.getstate(),np.random.get_state()))

def test_evaluator_matches_known_perfect_ap(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('eval_test',CODE/'libs/evaluation/evaluator_anet.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p=tmp_path/'gt.json';p.write_text(json.dumps({'database':{'a':dict(subset='source',annotations=[dict(label_id=2,segment=[1,2])]),'b':dict(subset='target',annotations=[dict(label_id=2,segment=[3,4])])}}))
    ev=m.ANETdetection(str(p),['target'],num_workers=1,action_subsampling='10_majority_classes')
    pred={'video-id':np.array(['b']),'t-start':np.array([3.]),'t-end':np.array([4.]),'label':np.array([2]),'score':np.array([.9])}
    assert ev.evaluate(pred,verbose=False)[1]==1.

def test_empty_source_video_remains_negative(tmp_path):
    torch.set_num_threads(2);cfg=config(tmp_path);p=Path(cfg['benchmark']['source_train']);db=read(p)
    for v in db['database'].values():v['annotations']=[]
    p.write_text(json.dumps(db));ctx=build(cfg,device='cpu',workers=0);batch=next(iter(ctx['train']))
    assert len(ctx['train'].dataset)==4 and all(v['segments'].shape==(0,2) for v in batch)
    losses,_=step(ctx,batch);assert np.isfinite(losses['final_loss']) and losses['reg_loss']==0.
