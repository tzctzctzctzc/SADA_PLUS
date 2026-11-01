import unittest
from copy import deepcopy
import torch
from torch import nn
from tools.charades_speed import fast_ema,batched_metrics,epoch_function,enabled
from libs.utils import ModelEma2
from libs.evaluation.metrics import MetricTracker

class SpeedTests(unittest.TestCase):
    def test_only_measured_faster_path_enabled(self):
        cfg=dict(benchmark='charades-ego-third-to-first',prototype_capacity={'variant':'control'},model={'adv_method':['None']})
        self.assertTrue(enabled(cfg));cfg['model']['adv_method']=['reverse'];self.assertFalse(enabled(cfg))
    def test_ema_preserves_repeated_alias_updates(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.a=nn.Linear(4,3);self.b=self.a;self.c=self.a
                self.register_buffer('ready',torch.tensor([False,True]))
        a=Model();b=Model();x=ModelEma2(deepcopy(a),.999);y=ModelEma2(deepcopy(a),.999)
        for _ in range(20):
            with torch.no_grad():
                b.a.weight.add_(.01);b.ready.logical_not_()
            x.update(b);fast_ema(y,b)
            for k,v in x.module.state_dict().items():self.assertTrue(torch.equal(v,y.module.state_dict()[k]),k)

    def test_batched_metrics_exact_original_values(self):
        a=MetricTracker('t_loss');b=MetricTracker('t_loss')
        for i in range(20):
            losses=dict(a=torch.tensor(i/17,dtype=torch.float32),b=torch.tensor(i,dtype=torch.int64),c=torch.tensor(i/13,dtype=torch.float64))
            for name,value in losses.items():
                if not a.contains('t_'+name):a.add_new('t_'+name)
                a.update('t_'+name,value.item())
            batched_metrics(b,losses)
        self.assertEqual(a.result(),b.result())

    def test_original_epoch_single_logging_replacement(self):
        self.assertTrue(callable(epoch_function()))

if __name__=='__main__':unittest.main()
