from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.release_data_protocol import SPEC,manifests,configuration,assert_pair,release_decision,stream_pair


def test_manifest_keeps_target_train_GT_and_source_order():
    cfg=dict(src_train_split=['s'],src_val_split=['sv'],tgt_train_split=['t'],tgt_val_split=['tv'])
    db={k:dict(subset=v,annotations=[dict(label_id=0,segment=[1,2])])
        for k,v in [('z','s'),('a','t'),('c','sv'),('d','tv')]}
    source,target=manifests(db,cfg)
    assert list(source['database'])==['z','c']
    assert list(target['database'])==['a'] and target['database']['a']['annotations']


def test_pair_differs_only_by_original_prototype():
    cfg=dict(seed=123,init_rand_seed=123,loader=dict(batch_size=2),dataset={},
        baseline={},model=dict(adv_method=['conditional_rev']),
        prototype_capacity=dict(variant='control'))
    kwargs=dict(identity={'commit':'test'},source='source',target='target',tmpdir='tmp',
                hashes=dict(source='sha1',target='sha2'))
    plain=configuration(cfg,'sada',**kwargs);prototype=configuration(cfg,'prototype',**kwargs)
    assert_pair(plain,prototype)
    assert plain['baseline']['policy']['valid_every']==2
    assert plain['release_data_protocol']==SPEC and 'prototype_capacity' in cfg


def test_release_stop_first_bad_validation():
    assert release_decision([[2,1.],[4,.9]])['reason']=='running'
    assert release_decision([[2,1.],[4,1.]])['best_epoch']==2
    assert release_decision([[2,1.],[4,1.]])['reason']=='source_first_non_improvement'
    with pytest.raises(AssertionError):release_decision([[2,1.],[4,1.],[6,.9]])
    assert release_decision([[2,'inf']])['best_epoch'] is None


def test_column_sort_has_author_nested_interval_pairing():
    segments=torch.tensor([[0.,10.],[2.,3.],[1.,8.]])
    labels=torch.tensor([1,2,3]);ordered,indices=torch.sort(segments,dim=0)
    assert torch.equal(ordered,torch.tensor([[0.,3.],[1.,8.],[2.,10.]]))
    assert torch.equal(labels[indices[:,0]],torch.tensor([1,3,2]))


class RandomDataset(torch.utils.data.Dataset):
    def __len__(self):return 7
    def __getitem__(self,index):return dict(index=index,draw=(random.random(),float(np.random.rand()),float(torch.rand(()))))


def test_shared_default_generator_persistent_loader_matches_author():
    from libs.datasets.datasets import make_data_loader
    from tools.baseline_engine import loader
    def stream(factory):
        torch.manual_seed(123)
        # 包括模型构造消耗CPU默认随机流，而不是另建一个seed123的generator。
        torch.rand(11)
        source=factory(RandomDataset(),True,torch.default_generator,2,1)
        target=factory(RandomDataset(),True,torch.default_generator,2,1)
        result=[];pids=None
        try:
            for epoch in range(2):
                result.append((list(source),list(target)))
                now=tuple(w.pid for obj in (source,target) for w in obj._iterator._workers)
                if pids is not None:assert now==pids
                pids=now
        finally:
            for obj in (source,target):obj._iterator._shutdown_workers()
        return result
    assert stream(make_data_loader)==stream(lambda *a:loader(*a,released=True))


def test_native_train_keeps_validation_debug_order_and_stop(monkeypatch):
    import logging
    import wandb
    from libs.trainer.base_trainer_da import BaseTrainerDA
    events=[]
    class EmptyMetrics:
        def reset(self):pass
        def result(self):return {}
    class Stub(BaseTrainerDA):
        def __init__(self):
            self.start_epoch=1;self.pretraining=False;self.epochs=50;self.valid_freq=self.test_freq=2
            self.cfg=dict(train_cfg=dict(warmup_adv=5),model=dict(weight_sharing=True))
            self.save_period=1;self.logger=logging.getLogger('test');self.train_metrics=EmptyMetrics()
            self.valid_metrics_src=EmptyMetrics();self.valid_metrics_tgt=EmptyMetrics()
            self.train_loader_tgt2='target_train';self.val_loader_src='source_val'
            self.train_loader_src2='source_train';self.val_loader_tgt='target_val'
            self.optimizer_gen_source=self.optimizer_gen_source_alignment=self.optimizer_disc=type('O',(),{'param_groups':[dict(lr=1.)]})()
        def _train_epoch(self,epoch,**kwargs):events.append(('train',epoch))
        def _valid_epoch(self,epoch,loader,domain='src'):
            events.append(('validate',epoch,loader));return 1.
        def _test_epoch(self,epoch,loader,mode,domain):events.append(('test',epoch,loader));return {}
        def save_ckpt(self,mode,epoch):events.append(('save',epoch,mode))
    monkeypatch.setattr(wandb,'log',lambda *a,**kw:None)
    trainer=Stub();trainer.train()
    assert [e[1] for e in events if e[0]=='train']==[1,2,3,4]
    assert [e[2] for e in events if e[0]=='validate']==['target_train','source_val']*2
    assert [e[2] for e in events if e[0]=='test']==['source_train','source_val','target_train','target_val']
    assert events[-1]==('save',4,'latest')


def test_stream_compare_only_common_completed_epochs():
    assert stream_pair([{'data_stream_sha256':'a'}],[{'data_stream_sha256':'a'},{'data_stream_sha256':'b'}])==dict(common_epochs=1,passed=True)


def test_pair_allows_only_equivalent_core_execution_versions():
    plain={'model':{'loss':1},'baseline':{'code':{'commit':'old','core_files':{'model':'same'}}}}
    proto=deepcopy(plain);proto['prototype_capacity']={'variant':'control'}
    proto['baseline']['code']['commit']='fast'
    assert_pair(plain,proto)
    proto['model']['loss']=2
    with pytest.raises(AssertionError):assert_pair(plain,proto)
