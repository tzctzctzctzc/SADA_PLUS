from copy import deepcopy
import json
import random
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.context_mix import (VARIANTS, FeatureBank, background_gaps, draw_pair, install,
                               mixed_loss, patch_record, prepare_table, specification)
from tools.baseline_protocol import CODE, read, save, sha
from tools.run_prototype_context_mix import assert_pair, changed_config, audit_ledger


def table_fixture(tmp_path):
    for name in ('s','background','validation'):
        np.savez(tmp_path/(name+'.npz'), feats=np.random.default_rng(7).normal(size=(96,8)).astype('float32'))
    src = tmp_path/'source.json'
    save(src, dict(database=dict(s=dict(subset='train',fps=1,duration=96,annotations=[
        dict(label='take',label_id=0,segment=[13.25,21.75]),
        dict(label='put',label_id=1,segment=[15.5,17.25])]),
        background=dict(subset='train',fps=1,duration=96,annotations=[]),
        validation=dict(subset='val',fps=1,duration=96,annotations=[
            dict(label='take',label_id=0,segment=[1.,3.])]))))
    target = tmp_path/'target.json';save(target,dict(database=dict(t=dict(subset='target',fps=1,duration=96))))
    inventory = tmp_path/'inventory.json'
    save(inventory,[dict(path=str(tmp_path/(name+'.npz')),sha256=sha(tmp_path/(name+'.npz')))
                    for name in ('s','background','validation')])
    cfg = dict(seed=123,src_train_split=['train'],src_val_split=['val'],
               baseline=dict(shift='S1',source_manifest_sha256=sha(src),target_manifest=str(target),
                             target_manifest_sha256=sha(target)),
               dataset=dict(json_file=str(src),feat_folder=str(tmp_path),file_ext='.npz',file_prefix=None,
                   action_subsampling='10_majority_classes',feat_stride=1,downsample_rate=1,
                   default_fps=1,num_frames=2,max_seq_len=64))
    path = tmp_path/'table.json'
    return cfg, path, prepare_table(cfg, inventory, path)


def target_fixture():
    return dict(feats=torch.full((8,64),-5.),segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long),
                metadata=dict(video_id='t',crop_start=11,crop_feat_len=64,fps=1,feat_stride=1,feat_num_frames=2))


def test_background_gaps_handle_nested_and_video_edge_gt():
    assert background_gaps([[-2.,3.2],[1.,2.],[5.,8.],[7.,10.2],[15.,25.]],20)==[[4,5],[11,15]]


def test_gt_float_endpoints_complete_overlaps_and_native_boundaries():
    rec = patch_record([[12.25,20.75],[14.5,16.25]],[0,1],0,64,3)
    assert rec['start']==9 and rec['end']==24
    assert rec['segments']==[[3.25,11.75],[5.5,7.25]] and rec['labels']==[0,1]
    assert 0<rec['main'][0]<rec['main'][1]<rec['end']-rec['start']
    assert patch_record([[12.25,20.75],[8.,13.]],[0,1],0,64,3) is None
    assert patch_record([[.25,5.75]],[0],0,64,3) is None


def test_tables_source_train_only_and_target_annotation_rejected(tmp_path):
    cfg,path,table = table_fixture(tmp_path)
    assert set(table['source_train'])=={'s','background'}
    assert table['source_validation_excluded']==['validation']
    assert {r['video_id'] for r in table['videos']}=={'s','background'}
    assert table['summary']['unique_eligible_actions']>0
    raw=read(cfg['baseline']['target_manifest']);raw['database']['t']['annotations']=[]
    save(cfg['baseline']['target_manifest'],raw)
    cfg['baseline']['target_manifest_sha256']=sha(cfg['baseline']['target_manifest'])
    with pytest.raises(ValueError,match='annotations'):prepare_table(cfg,tmp_path/'inventory.json',tmp_path/'evil.json')


def test_paired_views_preserve_all_gt_and_do_not_mutate_context(tmp_path):
    _,_,table = table_fixture(tmp_path);bank=FeatureBank(table)
    target=target_fixture();before=target['feats'].clone();rng_before=torch.get_rng_state().clone()
    views,e=draw_pair(bank,target,random.Random(42),'target')
    lo,hi=e['inserted']
    assert torch.equal(target['feats'],before) and torch.equal(rng_before,torch.get_rng_state())
    assert torch.equal(views[0]['feats'][:,:lo],before[:,:lo])
    assert torch.equal(views[1]['feats'][:,hi:],before[:,hi:])
    assert views[1]['segments'].shape==(0,2)
    assert lo<e['main'][0]<e['main'][1]<hi
    assert len(e['segments'])==2 and e['labels']==[0,1]
    assert all(lo<=a<b<=hi for a,b in e['segments'])


def test_all_variants_use_identical_draw_plan_with_private_rng(tmp_path):
    _,_,table=table_fixture(tmp_path);target=target_fixture();bank=FeatureBank(table)
    py=random.getstate();np_state=deepcopy(np.random.get_state());ts=torch.get_rng_state().clone()
    draws=[draw_pair(bank,target,random.Random(19),specification(v)['context']) for v in VARIANTS]
    assert all(e==draws[0][1] for _,e in draws)
    a,b=draws[0][0],draws[1][0];lo,hi=draws[0][1]['inserted']
    assert torch.equal(a[0]['feats'][:,lo:hi],b[0]['feats'][:,lo:hi])
    assert torch.equal(a[1]['feats'][:,lo:hi],b[1]['feats'][:,lo:hi])
    assert random.getstate()==py and torch.equal(ts,torch.get_rng_state())
    assert all(np.array_equal(a,b) for a,b in zip(np_state,np.random.get_state()))


def test_target_gt_rejected_and_source_feature_sha_checked(tmp_path):
    _,_,table=table_fixture(tmp_path);bank=FeatureBank(table);target=target_fixture()
    target['segments']=torch.tensor([[1.,2.]])
    with pytest.raises(ValueError):draw_pair(bank,target,random.Random(2),'target')
    path=table['videos'][0]['path'];np.savez(path,feats=np.zeros((96,8),dtype='float32'))
    with pytest.raises(ValueError,match='SHA'):bank.load(0)


@pytest.mark.parametrize('change',['lr','seed','batch','policy','cache','spec'])
def test_only_declared_changes_allowed(tmp_path,change):
    _,path,_=table_fixture(tmp_path)
    old=dict(seed=123,loader=dict(batch_size=2),model=dict(adv_method=['None']),opt=dict(learning_rate=.0001),
        baseline=dict(code={},tmpdir='/old',policy=__import__('tools.baseline_protocol',fromlist=['POLICY']).POLICY),
        prototype_capacity=dict(prepared_cache='fixed'))
    cfg=changed_config(old,'target_context',path,{},'/new');assert_pair(cfg,old)
    if change=='lr':cfg['opt']['learning_rate']=.001
    elif change=='seed':cfg['seed']=6
    elif change=='batch':cfg['loader']['batch_size']=4
    elif change=='policy':cfg['baseline']['policy']=dict(selection='target_map')
    elif change=='cache':cfg['prototype_capacity']['prepared_cache']='changed'
    else:cfg['context_mix']['specification']['mix_weight']=.5
    with pytest.raises(ValueError):assert_pair(cfg,old)


def test_real_sada_mixed_loss_uses_original_heads_and_restores_main_normalizer(tmp_path):
    from libs.models.SADA.meta_archs import SADA
    _,_,table=table_fixture(tmp_path)
    cfg=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']()
    cfg['model']['adv_method']=['None'];cfg['model']['droppath']=0.
    model=SADA(cfg,'cpu')
    views,event=draw_pair(FeatureBank(table),target_fixture(),random.Random(42),'target')
    before=model.loss_normalizer,model.loss_normalizer_pseudo_labels
    loss,norm,metrics=mixed_loss(model,views,event,250.,5,.1,True)
    assert (model.loss_normalizer,model.loss_normalizer_pseudo_labels)==before
    assert 0<norm<250 and metrics['reg_raw']>0 and metrics['pair_points']>0
    assert metrics['fpn_gradient_norm']>0 and torch.isfinite(loss)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    initial=deepcopy(model.reg_head.state_dict());loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.neck_source.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.cls_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.reg_head.parameters())
    assert all(p.grad is None for p in model.disc_dom_conditional_adv.parameters())
    opt.step();assert any(not torch.equal(v,model.reg_head.state_dict()[k]) for k,v in initial.items())


def test_outside_patch_logits_are_ignored_in_loss_and_normalization(tmp_path,monkeypatch):
    from libs.models.SADA.meta_archs import SADA
    import tools.context_mix as module
    _,_,table=table_fixture(tmp_path)
    cfg=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']()
    cfg['model']['adv_method']=['None'];model=SADA(cfg,'cpu')
    views,event=draw_pair(FeatureBank(table),target_fixture(),random.Random(42),'target')
    features,masks,points,logits,offsets,act=module.supervised_predictions(model,views)
    lo,hi=event['inserted']
    def call(shift):
        new=[(x+((p[:,0]<lo)|(p[:,0]>=hi))[None,:,None]*shift).detach().requires_grad_()
             for x,p in zip(logits,points)]
        monkeypatch.setattr(module,'supervised_predictions',lambda *a:(features,masks,points,new,offsets,act))
        loss,norm,_=mixed_loss(model,views,event,250.,5,0.)
        grads=torch.autograd.grad(loss,new,retain_graph=True)
        assert all(g[:,(p[:,0]<lo)|(p[:,0]>=hi)].abs().sum()==0 for g,p in zip(grads,points))
        return loss.detach(),norm
    a,an=call(0.);b,bn=call(100.)
    torch.testing.assert_close(a,b,rtol=0,atol=0);assert an==bn


def test_install_keeps_parameters_base_losses_and_rng_unchanged(tmp_path):
    from libs.models.SADA.meta_archs import SADA
    cfg0,path,table=table_fixture(tmp_path)
    cfg=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']()
    cfg.update(seed=123,baseline=cfg0['baseline'])
    cfg['model']['adv_method']=['None']
    cfg['context_mix']=dict(variant='target_context_pair',specification=specification('target_context_pair'),
                            table=str(path),table_sha256=sha(path))
    torch.manual_seed(123);plain=SADA(cfg,'cpu')
    torch.manual_seed(123);model=SADA(cfg,'cpu');ctx=SimpleNamespace(model=model)
    install(ctx,cfg);model.context_mix_state['step']=3
    source=target_fixture();source=deepcopy(source)
    source['metadata']['video_id']='s';source['segments']=torch.tensor([[15.,35.]])
    source['labels']=torch.tensor([0])
    torch.manual_seed(73)
    a=plain(video_list_src=[source],video_list_tgt=[target_fixture()],mode='train',curr_epoch=5)[0]
    rng=torch.get_rng_state().clone()
    torch.manual_seed(73)
    b=model(video_list_src=[source],video_list_tgt=[target_fixture()],mode='train',curr_epoch=5)[0]
    assert torch.equal(rng,torch.get_rng_state())
    assert plain.loss_normalizer==model.loss_normalizer
    for key in ('cls_loss','reg_loss'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert b['downstream_loss']>a['downstream_loss'] and b['mix_fpn_gradient_norm']>0
    assert list(model.state_dict())==list(plain.state_dict())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in plain.state_dict().items())


def test_completion_audit_reconstructs_continuous_gt_and_event_hash_chain(tmp_path):
    import hashlib
    _,_,table=table_fixture(tmp_path);bank=FeatureBank(table);rng=random.Random(42)
    entry=tmp_path/'entry';entry.mkdir()
    save(entry/'config.json',dict(baseline=dict(epoch_budget=dict(iterations_per_epoch=4)),
                                 context_mix=dict(specification=specification('target_context'))))
    save(entry/'termination.json',dict(decision=dict(epoch=3)))
    events=[];digest='0'*64
    for epoch in range(1,4):
        _,event=draw_pair(bank,target_fixture(),rng,'target')
        event.update(step=epoch*4,epoch=epoch)
        digest=hashlib.sha256((digest+json.dumps(event,sort_keys=True,separators=(',',':'))).encode()).hexdigest()
        events.append(dict(event,context='target',digest=digest))
    ledger=entry/'mix_events.jsonl'
    ledger.write_text(''.join(json.dumps(e)+'\n' for e in events))
    saved=dict(step=12,events=3,event_digest=digest)
    audit_ledger(entry,table,saved)
    events[-1]['segments'][0][0]+=.125
    ledger.write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(ValueError,match='GT'):
        audit_ledger(entry,table,saved)


def test_private_state_survives_checkpoint_roundtrip_without_new_model_weights(tmp_path):
    import io
    from tools.baseline_engine import controls,tensor_digest
    from libs.models.SADA.meta_archs import SADA
    base,path,_=table_fixture(tmp_path)
    cfg=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']()
    cfg.update(seed=123,baseline=base['baseline'])
    cfg['model']['adv_method']=['None']
    cfg['context_mix']=dict(variant='source_context',specification=specification('source_context'),
                            table=str(path),table_sha256=sha(path))
    model=SADA(cfg,'cpu');install(SimpleNamespace(model=model),cfg)
    saved=controls(model);data=io.BytesIO();torch.save(saved,data);data.seek(0)
    restored=torch.load(data,map_location='cpu')
    assert tensor_digest(saved)==tensor_digest(restored)
    model.context_mix_state=restored['context_mix_state']
    a=random.Random();a.setstate(saved['context_mix_state']['rng'])
    b=random.Random();b.setstate(model.context_mix_state['rng'])
    assert [a.random() for _ in range(8)]==[b.random() for _ in range(8)]
