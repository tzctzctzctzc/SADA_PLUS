import torch
import pytest
from libs.utils.inference_scope import reporting_forward


class BothStreams(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.arange(8,dtype=torch.float32))
        self.fail=fail
        self.backbone_calls=0
        self.postprocessed=[]

    def forward(self,video_list_src,video_list_tgt,mode):
        self.backbone_calls+=1
        # 两路前向始终执行，模拟真实SADA固定batch/算子顺序。
        source=video_list_src[0]*self.weight
        target=video_list_tgt[0]*self.weight+1
        if self.fail:
            raise RuntimeError('evaluation failed')
        domain=getattr(self,'_rq3_reporting_domain',None)
        values=[]
        for name,value in [('src',source),('tgt',target)]:
            if domain is None or domain==name:
                self.postprocessed.append(name)
                values.append(value.sort().values)
            else:
                values.append(None)
        return tuple(values)


@pytest.mark.parametrize('domain,index',[('src',0),('tgt',1)])
def test_selected_output_rng_and_parameters_identical(domain,index):
    model=BothStreams().eval()
    videos=[torch.arange(8,dtype=torch.float32)]
    before=torch.random.get_rng_state().clone()
    parameters={k:v.clone() for k,v in model.state_dict().items()}
    reference=model(video_list_src=videos,video_list_tgt=videos,mode='test')
    model.postprocessed=[]
    result=reporting_forward(model,videos,domain)
    assert torch.equal(result[index],reference[index])
    assert result[1-index] is None
    assert model.postprocessed==[domain] and model.backbone_calls==2
    assert torch.equal(before,torch.random.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in parameters.items())
    assert not hasattr(model,'_rq3_reporting_domain')


def test_nested_state_and_exception_are_restored():
    model=BothStreams(fail=True)
    model._rq3_reporting_domain='tgt'
    with pytest.raises(RuntimeError):
        reporting_forward(model,[torch.ones(8)],'src')
    assert model._rq3_reporting_domain=='tgt'
    with pytest.raises(ValueError):
        reporting_forward(model,[],'both')
