from copy import deepcopy
from pathlib import Path
import importlib.util,random,json
import numpy as np
import pytest
import torch
from tools import dyfadet_uda_engine as e
from tools.dyfadet_uda_protocol import decision,terminal,SPEC

spec=importlib.util.spec_from_file_location('so_test_helper',Path(__file__).with_name('test_dyfadet_so.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def config(tmp_path):
    torch.set_num_threads(2);cfg=helper.config(tmp_path)
    cfg['opt'].update(learning_rate=1e-5,eta_min=1e-6,warmup=False,warmup_epochs=0,epochs=30)
    ref=tmp_path/'reference';ref.mkdir();ctx=e.base.build(cfg,device='cpu',workers=0)
    e.base.checkpoint(ref/'best.pt',e.base.capture(ctx,cfg,0,[[0,1.]],[]))
    db=e.read(cfg['benchmark']['source_train'])
    for v in db['database'].values():v.pop('annotations')
    path=tmp_path/'target.json';e.save(path,db)
    cfg['benchmark'].update(target_train=str(path),reference=str(ref));return cfg

def fixed_pseudo(teacher,videos):
    return [dict(segments=torch.tensor([[4.,12.]]),labels=torch.tensor([2]),scores=torch.tensor([.9])) for _ in videos]

def test_target_ignore_and_no_normalizer_update(tmp_path):
    cfg=config(tmp_path);ctx=e.build(cfg,'uda','cpu',0);m=ctx['model'];m.eval()
    videos=next(iter(ctx['target']))[:1];out=e.base.dense(m,videos);points,masks,cls,reg=out
    cls=[x.detach().requires_grad_() for x in cls];reg=[x.detach().requires_grad_() for x in reg]
    normal=m.loss_normalizer;pred=fixed_pseudo(m,videos);c,r,n=e.positive_loss(m,(points,masks,cls,reg),pred)
    assert n>0; (c+r).backward();assert m.loss_normalizer==normal
    gt,_=m.label_points(points,[p['segments'] for p in pred],[p['labels'] for p in pred]);pos=torch.stack(gt).sum(-1)>0
    grad=torch.cat([x.grad for x in cls],1);assert torch.count_nonzero(grad[~pos])==0 and torch.count_nonzero(grad[pos])>0
    altered=[];start=0
    for x in cls:
        y=x.detach().clone();mask=pos[:,start:start+x.shape[1]];y[~mask]+=100;altered.append(y);start+=x.shape[1]
    c2,r2,n2=e.positive_loss(m,(points,masks,altered,reg),pred)
    assert c.item()==c2.item() and r.item()==r2.item() and n==n2

def test_empty_pseudo_has_zero_target_gradient(tmp_path):
    cfg=config(tmp_path);ctx=e.build(cfg,'uda','cpu',0);m=ctx['model'];m.train();v=next(iter(ctx['target']))
    p=[dict(segments=torch.zeros(0,2),labels=torch.zeros(0,dtype=torch.long)) for _ in v]
    c,r,n=e.positive_loss(m,e.base.dense(m,v),p);(c+r).backward()
    assert c.item()==r.item()==n==0
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in m.parameters())

def test_real_teacher_batch_uses_unlabeled_local_grid(tmp_path):
    cfg=config(tmp_path);ctx=e.build(cfg,'uda','cpu',0);videos=next(iter(ctx['target']))
    predictions=e.pseudo(ctx['ema'].module,videos);assert len(predictions)==2
    for p,v in zip(predictions,videos):
        assert not p['segments'].requires_grad and bool((p['scores']>=.5).all())
        assert bool((p['segments']>=0).all()) and bool((p['segments']<=v['crop_length']).all())

def test_uniform_crop_and_channel_mask_preserve_coordinates(tmp_path):
    cfg=config(tmp_path);ds=e.TargetDataset(cfg,cfg['benchmark']['target_train']);random.seed(3);v=ds[0]
    assert 57<=v['feats'].shape[1]<=64 and 'segments' not in v
    full=e.base.BenchmarkDataset(cfg,cfg['benchmark']['target_train'],unlabeled=True)[0]['feats']
    s=v['crop_start_grid'];assert torch.equal(v['feats'],full[:,s:s+v['crop_length']])
    out=e.strong([v],torch.Generator().manual_seed(1))[0]
    assert out['crop_start_grid']==s and out['feats'].shape==v['feats'].shape
    assert all(torch.equal(a,b/(1-SPEC['channel_dropout'])) or torch.count_nonzero(a)==0 for a,b in zip(out['feats'],v['feats']))

def test_target_rng_does_not_consume_source_stream():
    torch.manual_seed(17);random.seed(18);np.random.seed(19)
    ctx=dict(device='cpu',target_rng=e.rng_get('cpu'));before=e.base.digest(e.rng_get('cpu'))
    with e.target_stream(ctx):torch.rand(10);random.random();np.random.rand()
    assert e.base.digest(e.rng_get('cpu'))==before and e.base.digest(ctx['target_rng'])!=before

def test_disabled_path_exact_and_all_modules_target_gradient(tmp_path,monkeypatch):
    cfg=config(tmp_path);ctx=e.build(cfg,'uda','cpu',0);src=next(iter(ctx['train']));tgt=next(iter(ctx['target']))
    state=deepcopy(e.capture(ctx,0,[],[]));state['probe']=True
    loss,_=e.base.step(ctx,src);d=e.base.digest(e.base.capture(ctx,cfg,0,[],[]))
    e.restore(ctx,deepcopy(state));other,_=e.update(ctx,src)
    assert all(loss[k]==other[k] for k in loss) and d==e.base.digest(e.base.capture(ctx,cfg,0,[],[]))
    e.restore(ctx,deepcopy(state));monkeypatch.setattr(e,'pseudo',fixed_pseudo)
    _,norms=e.update(ctx,src,tgt,True)
    assert set(norms)==set(e.MODULES) and all(v>0 and np.isfinite(v) for v in norms.values())
    assert all(p.grad is None for p in ctx['ema'].module.parameters())

def test_full_rng_optimizer_restore_next_step(tmp_path,monkeypatch):
    cfg=config(tmp_path);ctx=e.build(cfg,'uda','cpu',0);monkeypatch.setattr(e,'pseudo',fixed_pseudo)
    src=next(iter(ctx['train']));tgt=next(iter(ctx['target']))
    for _ in range(4):e.update(ctx,src,tgt)
    state=deepcopy(e.capture(ctx,1,[],[]));e.restore(ctx,deepcopy(state))
    assert e.base.digest(e.capture(ctx,1,[],[]))==e.base.digest(state)
    l1,_=e.update(ctx,src,tgt);d1=e.base.digest(e.capture(ctx,1,[],[]))
    e.restore(ctx,deepcopy(state));l2,_=e.update(ctx,src,tgt);d2=e.base.digest(e.capture(ctx,1,[],[]))
    assert l1==l2 and d1==d2
    bad=deepcopy(state);bad['step']=3
    with pytest.raises(AssertionError):e.restore(ctx,bad)

def test_legal_epoch0_and_matched_budget():
    h=[[0,1.],[5,1.1],[10,1.2],[15,1.3]]
    assert terminal(h)['best_epoch']==0 and terminal(h)['epoch']==15
    with pytest.raises(AssertionError):terminal(h+[[20,.9]])
    with pytest.raises(AssertionError):decision([[0,1.],[10,.9]])

def test_ledger_chains_exact_members(tmp_path):
    ctx=dict(step=1,ledger_digest='');v=[dict(video_id='src')];t=[dict(video_id='tgt',crop_start_grid=3,crop_length=8)]
    p=tmp_path/'events.jsonl';e.record(ctx,p,v,t,dict(target_positive_points=3,target_proposals=2,target_classes=[0]*10))
    first=json.loads(p.read_text());assert first['source_videos']==['src'] and first['target_videos']==['tgt']
    ctx['step']=2;e.record(ctx,p,v,None,dict(target_positive_points=0,target_proposals=0,target_classes=[0]*10))
    second=json.loads(p.read_text().splitlines()[1]);assert second['previous']==first['sha256'] and second['target_videos']==[]

def test_numerical_check_rejects_wrong_gradient_optimizer_or_rng():
    a=dict(model={'x':torch.tensor([1.])},ema={'x':torch.tensor([1.])},optimizer=dict(param_groups=[],state={0:dict(step=torch.tensor(1.),exp_avg=torch.tensor([.1]),exp_avg_sq=torch.tensor([.01]))}),rng=torch.tensor([1]))
    g={'x':torch.tensor([.2])};assert e.numerical_step_check(a,deepcopy(a),g,g,1e-5)['exact']
    bad=deepcopy(a);bad['rng'][0]=2
    with pytest.raises(AssertionError):e.numerical_step_check(a,bad,g,g,1e-5)
    bad=deepcopy(a);bad['optimizer']['state'][0]['exp_avg']+=.01
    with pytest.raises(AssertionError):e.numerical_step_check(a,bad,g,g,1e-5)
    with pytest.raises(AssertionError):e.numerical_step_check(a,deepcopy(a),g,{'x':g['x']+.01},1e-5)
