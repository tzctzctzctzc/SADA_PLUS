"""等价指标累计及批量标量读回；不参与模型或优化器计算。"""
import torch


class ScalarMetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer=writer
        self.values={key:[0,0,0] for key in keys}
    def reset(self):
        for key in self.values:self.values[key]=[0,0,0]
    def update(self,key,value,n=1):
        if self.writer is not None:self.writer.add_scalar(key,value)
        item=self.values[key];item[0]+=value*n;item[1]+=n;item[2]=item[0]/item[1]
    def avg(self,key):return self.values[key][2]
    def result(self):return {key:value[2] for key,value in self.values.items()}
    def contains(self,key):return key in self.values
    def add_new(self,key):self.values[key]=[0,0,0]


def batch_scalars(values):
    names=list(values)
    packed=torch.stack([values[name].detach().reshape(()) for name in names]).cpu().tolist()
    return dict(zip(names,packed))


def configure(ctx, enabled=True):
    if not hasattr(ctx,'native_metric_trackers'):
        ctx.native_metric_trackers={key:getattr(ctx,key) for key in ('train_metrics','valid_metrics_src','valid_metrics_tgt')}
        ctx.fast_metric_trackers={key:ScalarMetricTracker(*value.result(),writer=value.writer) for key,value in ctx.native_metric_trackers.items()}
    ctx.equivalent_metrics_enabled=enabled
    for key,value in (ctx.fast_metric_trackers if enabled else ctx.native_metric_trackers).items():setattr(ctx,key,value)
