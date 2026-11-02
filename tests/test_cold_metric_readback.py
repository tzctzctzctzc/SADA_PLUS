import ast
from pathlib import Path
import subprocess

import torch

from libs.evaluation.metrics import MetricTracker
from tools.cold_metric_readback import ScalarMetricTracker, batch_scalars


def test_metric_accumulation_matches_original_after_each_update():
    native=MetricTracker('a','b');fast=ScalarMetricTracker('a','b')
    for cycle in range(2):
        for i in range(137):
            key=('a','b','new')[i%3]
            for tracker in (native,fast):
                if not tracker.contains(key):tracker.add_new(key)
                tracker.update(key,float(torch.tensor((i-50)/71)),n=i%4+1)
            assert native.result()==fast.result()
            assert all(native.avg(k)==fast.avg(k) for k in native.result())
        native.reset();fast.reset()
        assert native.result()==fast.result()


def test_batched_scalars_preserve_each_scalar_value():
    values={'a':torch.tensor(.1),'b':torch.tensor(.3,dtype=torch.float64),'c':torch.tensor(1.)}
    assert batch_scalars(values)=={k:v.item() for k,v in values.items()}


def test_native_training_control_flow_unchanged():
    root=Path(__file__).resolve().parents[1]
    original=subprocess.check_output(['git','show','87c165a:libs/trainer/da_trainer.py'],cwd=root).decode('utf8')
    current=(root/'libs/trainer/da_trainer.py').read_text(encoding='utf8')
    def training(tree):
        return next(n for n in ast.walk(ast.parse(tree)) if isinstance(n,ast.FunctionDef) and n.name=='_train_epoch')
    class Native(ast.NodeTransformer):
        def visit_If(self,node):
            if 'equivalent_metrics_enabled' in ast.unparse(node.test):return node.orelse
            return self.generic_visit(node)
    assert ast.dump(training(original),include_attributes=False)==ast.dump(Native().visit(training(current)),include_attributes=False)


def test_nan_replacement_keeps_finite_loss_and_gradient():
    x=torch.randn(17,requires_grad=True);loss=x.square().mean()
    expected=torch.autograd.grad(loss,x,retain_graph=True)[0]
    replaced=torch.where(torch.isnan(loss),torch.zeros_like(loss),loss)
    actual=torch.autograd.grad(replaced,x)[0]
    assert torch.equal(loss,replaced) and torch.equal(expected,actual)
