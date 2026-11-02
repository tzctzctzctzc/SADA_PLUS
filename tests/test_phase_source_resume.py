from copy import deepcopy
import random
import numpy as np
import pytest
import torch
from tools.phase_source_resume import restore_source_head


def test_source_adam_dropout_random_stream_and_best_resume_exactly(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    torch.manual_seed(13);random.seed(13);np.random.seed(13)
    a=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1));oa=torch.optim.AdamW(a.parameters(),lr=.001)
    def step(model,opt):
        opt.zero_grad();x=torch.randn(5,3)+random.random()+np.random.rand();loss=model(x).square().mean();loss.backward();opt.step();return loss.detach()
    for _ in range(3):step(a,oa)
    history=[dict(epoch=i,source_candidate_score=3-i,quality=dict(value=i)) for i in range(3)]
    saved=deepcopy(dict(spec={'test':True},epoch=2,history=history,state_dict=a.state_dict(),optimizer=oa.state_dict(),
        rng_cpu=torch.get_rng_state(),rng_cuda=None,rng_python=random.getstate(),rng_numpy=np.random.get_state(),cache_index_sha256='index'))
    expected=[step(a,oa) for _ in range(2)]
    b=deepcopy(a);ob=torch.optim.AdamW(b.parameters(),lr=.001)
    h,score,epoch,best,quality,start=restore_source_head(b,ob,saved,{'test':True},'index')
    actual=[step(b,ob) for _ in range(2)]
    assert all(torch.equal(x,y) for x,y in zip(expected,actual)) and start==3 and epoch==2 and h==history
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    assert all(torch.equal(best[k],saved['state_dict'][k]) for k in best)
    saved['history'][0]['source_candidate_score']=0
    with pytest.raises(AssertionError):restore_source_head(b,ob,saved,{'test':True},'index')
