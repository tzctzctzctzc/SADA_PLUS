"""路线二修订：冻结回归塔，源预训练初始化的独立末层及端点尺度。"""
import torch
from torch import nn
from torch.nn import functional as F
SCHEMA='prototype-pretrained-residual-endpoint-head-v1'
SPEC=dict(route=2,revision=1,seed=123,frozen_detector=True,source_only=True,
          input='five-real-frozen-regression-tower-neighbours-plus-level-onehot',
          initialization='original-three-tap-weights-plus-exact-original-distance-skip',
          prediction='original-distance-plus-learned-five-neighbour-weight-delta',
          shared_frozen_regression_tower=True,independent_trainable_final_layer=True,
          epoch_zero_source_validation_candidate=True,
          max_points_per_video=2048,batch=512,lr=.001,max_epochs=50,patience=15,
          source_validation='equal-video-Laplace-NLL-plus-IoU',
          calibration='equal-video-source-validation-endpoint-MAE-and-scale-ratio',
          fusion='per-endpoint-precision-times-agreement-times-foreground',target_gt=False,
          target_training=False,original_prototype_unchanged=True)

@torch.no_grad()
def point_context(features,masks,reg_head=None):
    xs=[];valid=[];levels=[]
    for level,(f,m) in enumerate(zip(features,masks)):
        if reg_head is not None:
            for conv,norm in zip(reg_head.head,reg_head.norm):
                f,_=conv(f,m);f=reg_head.act(norm(f))
        seq=f[0].T;mask=m[0].reshape(-1).bool()
        # 原回归塔LayerNorm在无效格上可能留下bias；保留它才能复制原卷积边界行为。
        if reg_head is None:seq=seq*mask[:,None]
        pad=F.pad(seq.T,(2,2)).T
        z=torch.cat([pad[i:i+len(seq)] for i in range(5)],1)
        one=F.one_hot(torch.full((len(seq),),level,device=f.device,dtype=torch.long),len(features)).to(f.dtype)
        xs.append(torch.cat([z,one],1));valid.append(mask);levels.append(torch.full((len(seq),),level,device=f.device,dtype=torch.long))
    return torch.cat(xs),torch.cat(valid),torch.cat(levels)

class EndpointHead(nn.Module):
    def __init__(self,channels=512,levels=6):
        super().__init__()
        self.channels,self.levels=channels,levels
        self.offset=nn.Linear(channels*5,2)
        self.level_scale=nn.Parameter(torch.ones(levels))
        self.uncertainty=nn.Linear(channels*5,2)
        self.register_buffer('initial_weight',torch.zeros_like(self.offset.weight))
        self.register_buffer('initial_bias',torch.zeros_like(self.offset.bias))
        nn.init.zeros_(self.uncertainty.weight);nn.init.constant_(self.uncertainty.bias,-2.35)
    @torch.no_grad()
    def initialize(self,weight,bias,scales):
        if weight.shape!=(2,self.channels,3):raise ValueError('原定位层不是2通道三邻域回归')
        self.offset.weight.zero_()
        self.offset.weight[:,self.channels:4*self.channels].copy_(weight.permute(0,2,1).reshape(2,-1))
        self.offset.bias.copy_(bias if bias is not None else torch.zeros_like(self.offset.bias))
        self.level_scale.copy_(scales.reshape(-1))
        self.initial_weight.copy_(self.offset.weight);self.initial_bias.copy_(self.offset.bias)
    def forward(self,x,original):
        values=x[:,:self.channels*5];onehot=x[:,-self.levels:]
        scale=onehot@self.level_scale
        # 不用GEMM重算原CuDNN卷积：零参数差分保证初始化逐值等于原预测。
        # 这是以A为锚点的残差专家，不宣称完全独立于A预测。
        delta=F.linear(values,self.offset.weight-self.initial_weight,self.offset.bias-self.initial_bias)
        out=F.relu(original+delta*scale[:,None])
        return out,F.softplus(self.uncertainty(values))+.01

def offset_iou(a,b):
    return torch.minimum(a,b).sum(1)/torch.maximum(a,b).sum(1).clamp_min(1e-6)

def objective(head,x,gt,original):
    b,scale=head(x,original)
    loss=1-offset_iou(b,gt)+.1*((b-gt).abs()/scale+scale.log()).mean(1)
    return loss,b,scale

def fuse(a,b,scale_b,levels,calibration,foreground):
    # 两个端点独立处理；语义分数只抑制背景，不当作定位不确定性本身。
    scale_a=calibration['a_scale'][levels].clamp_min(.01)
    scale_b=(scale_b*calibration['b_ratio'][levels]).clamp_min(.01)
    disagreement=(a-b).abs()/(scale_a+scale_b)
    weight=scale_a/(scale_a+scale_b)*torch.exp(-disagreement)*foreground[:,None]
    return a+weight*(b-a),weight


def audit_history(history):
    import math
    if [r['epoch'] for r in history]!=list(range(len(history))) or not 16<=len(history)<=51:
        raise ValueError('必须保留epoch0初始化和连续完整训练历史')
    best=float('inf');best_epoch=0
    for row in history:
        epoch=row['epoch'];value=row['source_validation']
        if not math.isfinite(value):raise ValueError('源验证非有限')
        if value<best:best=value;best_epoch=epoch
        if epoch< len(history)-1 and epoch-best_epoch>=15:
            raise ValueError('错过首次合法早停')
    if epoch!=50 and epoch-best_epoch<15:raise ValueError('未合法停止')
    return best_epoch,epoch


def audit_entry(entry):
    from pathlib import Path
    from tools.baseline_protocol import read,sha
    e=Path(entry);r=read(e/'complete.json')
    assert r['schema']==SCHEMA and r['status']=='passed'
    for path,digest in r['artifacts'].items():assert sha(path)==digest,path
    gpu=read(e/'gpu_complete.json');accept=read(e/'acceptance.json');init=read(e/'initialization_proof.json')
    assert gpu['frozen_detector_unchanged'] and gpu['independent_head_trained'] and not gpu['target_gt_used']
    assert accept['independent_parameter_storage'] and accept['branch_parameters_updated'] and not accept['target_gt_used']
    assert accept['gradient_norm']>0 and accept['shared_frozen_regression_tower']
    assert init['status']=='passed' and init['original_regression_initialization']
    assert sha(gpu['guard'])==gpu['guard_sha256'] and sha(accept['guard'])==accept['guard_sha256']
    best,stop=audit_history(read(e/'auxiliary_history.json'))
    assert (best,stop)==(gpu['best_epoch'],gpu['stop_epoch'])
    weights=torch.load(e/'best_auxiliary.pt',map_location='cpu')
    latest=torch.load(e/'latest_auxiliary.pt',map_location='cpu')
    assert weights['epoch']==best and weights['specification']==SPEC
    assert latest['history']==read(e/'auxiliary_history.json')
    bound=read(e/'inputs.json');reference=Path(bound['reference'])
    assert sha(reference/'best.pt')==bound['checkpoint_sha256']
    train=read(e/'source_train_gt.json')['database'];val=read(e/'source_val_gt.json')['database']
    assert set(train).isdisjoint(val)
    for domain in ['source','target']:
        manifest=read(e/(domain+'_val_unlabeled.json'))['database']
        assert all(not v.get('annotations') for v in manifest.values())
        expected=read(reference/'evaluation'/(domain+'_val.json'))['map_0.1_to_0.5']
        assert abs(r['results'][domain]['control']-expected)<1e-7
        for mode in ['control','refined']:
            evaluated=read(e/'evaluation'/(domain+'_'+mode+'.json'))
            assert sha(e/(domain+'_'+mode+'.npz'))==evaluated['prediction_sha256']
            assert evaluated['map']==r['results'][domain][mode]
    return dict(status='passed',complete_sha256=sha(e/'complete.json'),best_epoch=best,stop_epoch=stop)
