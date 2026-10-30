from copy import deepcopy
import random
from runpy import run_path
import torch
import pytest
from tools.baseline_protocol import CODE,save,sha
from libs.models.SADA.background_oracle import crop_annotations,SCHEMA,POLICY


def test_oracle_crop_is_fixed_and_does_not_consume_rng():
    record=dict(annotations=[dict(segment=[5.,15.],label_id=0),dict(segment=[0.,2.],label_id=1),dict(segment=[5.,15.],label_id=20)])
    meta=dict(fps=1.,feat_stride=1.,feat_num_frames=2,crop_start=6,crop_feat_len=8)
    before=random.getstate();rng=torch.get_rng_state().clone();segments,labels=crop_annotations(record,meta,set(range(10)),.5)
    assert segments.tolist()==[[0.,8.]] and labels.tolist()==[0]
    assert random.getstate()==before and torch.equal(rng,torch.get_rng_state())


def model_config(tmp_path,oracle=True,empty=False):
    cfg=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']()
    cfg['prototype_capacity']=run_path(str(CODE/'tests/test_prototype_capacity.py'))['configuration'](tmp_path,'control')
    gt=tmp_path/'oracle.json';manifest=tmp_path/'unlabeled.json'
    save(manifest,{'database':{'t':{'subset':'target'}}})
    save(gt,dict(schema=SCHEMA,shift='S1',database={'t':dict(subset='target',fps=1.,annotations=[] if empty else [dict(segment=[16.,36.],label_id=0)])}))
    cfg['baseline']=dict(shift='S1',target_manifest=str(manifest),target_gt_used=oracle)
    cfg['tgt_train_split']=['target']
    if oracle:cfg['background_oracle']=dict(schema=SCHEMA,policy=deepcopy(POLICY),target_train_gt=str(gt),target_train_gt_sha256=sha(gt))
    return cfg


@pytest.mark.parametrize('empty',[False,True])
def test_real_sada_oracle_keeps_initialization_source_losses_and_input(tmp_path,monkeypatch,empty):
    from libs.models.SADA.meta_archs import SADA
    native=torch.tensor
    monkeypatch.setattr(torch,'tensor',lambda *a,**kw:(lambda x:x.clone() if x.requires_grad and x.device.type=='cpu' else x)(native(*a,**kw)))
    cfg=model_config(tmp_path,True,empty);plain=deepcopy(cfg);plain.pop('background_oracle');plain['baseline']['target_gt_used']=False
    torch.manual_seed(123);old=SADA(plain,'cpu');torch.manual_seed(123);new=SADA(cfg,'cpu')
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    source,target=run_path(str(CODE/'tests/test_prototype_capacity.py'))['videos']();before=deepcopy(target)
    torch.manual_seed(777);a=old(video_list_src=[source],video_list_tgt=[target],mode='train',curr_epoch=5)[0]
    torch.manual_seed(777);b=new(video_list_src=[source],video_list_tgt=[target],mode='train',curr_epoch=5)[0]
    assert torch.equal(a['cls_loss'],b['cls_loss']) and torch.equal(a['reg_loss'],b['reg_loss'])
    assert torch.equal(a['rq3_loss'],b['rq3_loss']) and target['segments'].numel()==target['labels'].numel()==0
    assert torch.equal(target['feats'],before['feats']) and target['metadata']==before['metadata']
    assert all(mask.ndim==2 for mask in new.oracle_target_positive)
    removed=sum(float(v) for k,v in b.items() if k.startswith('oracle_bg_removed_'))
    if empty:
        assert removed==0
        for key in a:assert torch.equal(a[key],b[key]),key
    else:assert removed>0
    for key,value in b.items():
        if key.startswith('oracle_bg_before_'):
            level=key[len('oracle_bg_before_'):]
            assert value==b['oracle_bg_removed_'+level]+b['oracle_bg_after_'+level]
    b['downstream_loss'].backward()
    assert new.reg_head.offset_head.conv.weight.grad is not None


def test_oracle_ordinary_target_gt_still_rejected(tmp_path):
    from libs.models.SADA.background_oracle import BackgroundOracle
    cfg=model_config(tmp_path);obj=BackgroundOracle(cfg['background_oracle'],cfg)
    video=dict(segments=torch.ones(1,2),labels=torch.zeros(1,dtype=torch.long))
    with pytest.raises(ValueError):obj.positive_masks(None,[video],[],[])


def test_batch2_masks_do_not_broadcast_between_crops_and_do_not_use_class_identity(tmp_path):
    from libs.models.SADA.meta_archs import SADA
    from tools.baseline_protocol import read
    cfg=model_config(tmp_path);model=SADA(cfg,'cpu');obj=model.background_oracle
    _,target=run_path(str(CODE/'tests/test_prototype_capacity.py'))['videos']()
    first=deepcopy(target);second=deepcopy(target);first['metadata']['video_length']=128
    second['metadata'].update(video_length=128,crop_start=64)
    x=torch.arange(64.).reshape(-1,1);points=[torch.cat((x,x*0,x*0+10000,x*0+1),1)]
    masks=[torch.ones(2,1,64,dtype=torch.bool)]
    actual=obj.positive_masks(model,[first,second],points,masks)
    assert actual[0].shape==(2,64) and actual[0][0].any() and not actual[0][1].any()
    obj.database['t']['annotations'][0]['label_id']=1
    changed=obj.positive_masks(model,[first,second],points,masks)
    assert torch.equal(actual[0],changed[0])


def test_training_entry_compiles():
    path=CODE/'tools/run_background_oracle.py';compile(path.read_text(encoding='utf8'),str(path),'exec')


def test_oracle_configuration_cannot_change_training_budget(tmp_path):
    from tools.background_oracle_protocol import config_for,assert_pair
    base=dict(seed=123,loader=dict(batch_size=2),opt=dict(learning_rate=.0001),baseline=dict(code={},tmpdir='old',target_gt_used=False),prototype_capacity=dict(variant='control'))
    gt=tmp_path/'gt.json';save(gt,{})
    cfg=config_for(base,gt,{'commit':'new'},tmp_path);assert cfg['baseline']['target_gt_used'] is True
    for field,value in [('seed',6),('loader',dict(batch_size=4)),('opt',dict(learning_rate=.001))]:
        bad=deepcopy(cfg);bad[field]=value
        with pytest.raises(ValueError):assert_pair(bad,base)
