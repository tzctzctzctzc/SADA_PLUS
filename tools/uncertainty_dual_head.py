"""路线二首版：独立密集端点专家B，源GT训练、逐端点不确定性融合。"""
import torch
from torch import nn
from torch.nn import functional as F
SCHEMA='prototype-independent-endpoint-head-v1'
SPEC=dict(route=2,revision=0,seed=123,frozen_detector=True,source_only=True,
          input='three-real-FPN-neighbours-plus-level-onehot',hidden=256,
          max_points_per_video=2048,batch=512,lr=.001,max_epochs=50,patience=15,
          source_validation='equal-video-Laplace-NLL-plus-IoU',
          calibration='equal-video-source-validation-endpoint-MAE-and-scale-ratio',
          fusion='per-endpoint-precision-times-agreement-times-foreground',target_gt=False,
          target_training=False,original_prototype_unchanged=True)

@torch.no_grad()
def point_context(features,masks):
    xs=[];valid=[];levels=[]
    for level,(f,m) in enumerate(zip(features,masks)):
        seq=f[0].T;mask=m[0].reshape(-1).bool();seq=seq*mask[:,None];pad=F.pad(seq.T,(1,1)).T
        z=torch.cat([pad[:-2],seq,pad[2:]],1)
        one=F.one_hot(torch.full((len(seq),),level,device=f.device,dtype=torch.long),len(features)).to(f.dtype)
        xs.append(torch.cat([z,one],1));valid.append(mask);levels.append(torch.full((len(seq),),level,device=f.device,dtype=torch.long))
    return torch.cat(xs),torch.cat(valid),torch.cat(levels)

class EndpointHead(nn.Module):
    def __init__(self,channels=512,levels=6):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(channels*3+levels),nn.Linear(channels*3+levels,256),
            nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,4))
    def forward(self,x):
        out=self.net(x)
        return F.softplus(out[:,:2])+.001,F.softplus(out[:,2:])+.01

def offset_iou(a,b):
    return torch.minimum(a,b).sum(1)/torch.maximum(a,b).sum(1).clamp_min(1e-6)

def objective(head,x,gt):
    b,scale=head(x)
    loss=1-offset_iou(b,gt)+.1*((b-gt).abs()/scale+scale.log()).mean(1)
    return loss,b,scale

def fuse(a,b,scale_b,levels,calibration,foreground):
    # 两个端点独立处理；语义分数只抑制背景，不当作定位不确定性本身。
    scale_a=calibration['a_scale'][levels].clamp_min(.01)
    scale_b=(scale_b*calibration['b_ratio'][levels]).clamp_min(.01)
    disagreement=(a-b).abs()/(scale_a+scale_b)
    weight=scale_a/(scale_a+scale_b)*torch.exp(-disagreement)*foreground[:,None]
    return a+weight*(b-a),weight
