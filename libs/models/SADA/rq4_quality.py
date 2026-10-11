"""独立源域定位质量模型；不向学生、EMA或冻结视频特征反传。"""
import torch
from torch import nn
from torch.nn import functional as F

GROUPS = 32
DIM = GROUPS * 4 + 6


def descriptor(features, boxes, centers, strides):
    """crop-local坐标。只池化真实SlowFast特征，不产生虚拟关系位置。"""
    features = features.detach().float()
    if features.ndim != 2 or features.shape[0] != 2304:
        raise ValueError('quality model requires frozen 2304-D SlowFast features')
    reduced = features.reshape(GROUPS, -1, features.shape[-1]).mean(1)
    grid = torch.arange(features.shape[-1], device=features.device)
    rows = []
    for box, center, stride in zip(boxes.detach(), centers.detach(), strides.detach()):
        start, end = box.unbind(); duration = (end-start).clamp_min(1e-6)
        middle = (start+end)/2
        ranges = ((start,end),(start,middle),(middle,end),(start-duration/2,end+duration/2))
        pooled = []
        for left,right in ranges:
            keep = (grid >= left) & (grid <= right)
            pooled.append(reduced[:,keep].mean(1) if keep.any() else reduced.new_zeros(GROUPS))
        visible = (torch.minimum(end,end.new_tensor(features.shape[-1]-1))-
                   torch.maximum(start,start.new_zeros(()))).clamp_min(0)/duration
        geometry = torch.stack((torch.log1p(duration), torch.log1p(stride),
            (center-start)/duration, (end-center)/duration, visible.clamp_max(1),
            torch.log1p(duration/stride.clamp_min(1e-6))))
        rows.append(torch.cat((*pooled,geometry)))
    return torch.stack(rows) if rows else features.new_empty((0,DIM))


class LocalizationQuality(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.register_buffer('mean',torch.zeros(DIM))
        self.register_buffer('scale',torch.ones(DIM))
        self.net = nn.Sequential(nn.Linear(DIM,hidden),nn.GELU(),nn.Linear(hidden,3))

    def forward(self, x):
        raw = self.net((x.detach()-self.mean)/self.scale)
        return torch.cat((raw[:,:1].sigmoid(),F.softplus(raw[:,1:])),dim=-1)


def source_targets(boxes, gt):
    """定位IoU及按预测时长归一化的起/终点绝对误差；不学习动作类别。"""
    if not len(gt):
        return torch.cat((boxes.new_zeros((len(boxes),1)),boxes.new_ones((len(boxes),2))),1)
    overlap=(torch.minimum(boxes[:,None,1],gt[None,:,1])-
             torch.maximum(boxes[:,None,0],gt[None,:,0])).clamp_min(0)
    duration=(boxes[:,1]-boxes[:,0]).clamp_min(1e-6)
    union=duration[:,None]+(gt[:,1]-gt[:,0])[None]-overlap
    quality,index=(overlap/union.clamp_min(1e-6)).max(1)
    errors=(boxes-gt[index]).abs()/duration[:,None]
    errors=torch.where((quality>0)[:,None],errors,torch.ones_like(errors))
    return torch.cat((quality[:,None],errors.clamp_max(2)),1)


def fit_quality(x, y, *, epochs, learning_rate, seed=6):
    # 独立局部RNG；创建/拟合质量模块不得改变学生初始化或DataLoader随机流。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=LocalizationQuality()
        model.mean.copy_(x.mean(0)); model.scale.copy_(x.std(0,unbiased=False).clamp_min(1e-4))
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
        for _ in range(epochs):
            prediction=model(x)
            loss=F.smooth_l1_loss(prediction,y)
            if not torch.isfinite(loss): raise ValueError('nonfinite source quality fit')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval().requires_grad_(False)
    return model


def calibrate(prediction, truth, video_ids, *, quantile):
    """逐视频最坏残差的源验证分位数；不是目标域coverage保证。"""
    if not 0 < quantile < 1: raise ValueError('calibration quantile must be explicit in (0,1)')
    residual=torch.cat(((prediction[:,:1]-truth[:,:1]).clamp_min(0),
                         (truth[:,1:]-prediction[:,1:]).clamp_min(0)),1)
    grouped=[]
    for video in sorted(set(video_ids)):
        indices=torch.tensor([i for i,v in enumerate(video_ids) if v==video])
        grouped.append(residual[indices].amax(0))
    if len(grouped)<2: raise ValueError('source calibration needs at least two independent videos')
    margin=torch.quantile(torch.stack(grouped),quantile,dim=0,interpolation='higher')
    return dict(quantile=quantile,quality_margin=float(margin[0]),
        error_margin=margin[1:].tolist(),calibration_videos=len(grouped),
        target_coverage_guaranteed=False)
