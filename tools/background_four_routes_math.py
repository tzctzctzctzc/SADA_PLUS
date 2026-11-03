"""四条论文路线的候选过滤适配；官方核心源按固定SHA保存在four_route_sources。"""
import ast
import math
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from tools.hard_background_modules import balanced_bg, quality_loss

SOURCES = Path(__file__).with_name('four_route_sources')


def source_symbols(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def official_pauc(data_len, pos_len, device):
    # 单进程、单GPU；官方loss的u_pos不是buffer，必须显式存盘。
    ns = dict(torch=torch, check_tensor_shape=lambda x, shape: x.reshape(shape), is_distributed=lambda: False,
              get_surrogate_loss=lambda name: lambda margin, delta: torch.maximum(margin-delta, torch.zeros_like(delta)).square())
    source_symbols(SOURCES/'libauc/libauc/losses/auc.py', ['pAUC_CVaR_Loss'], ns)
    # beta=.01为事先固定的非背景误报区间；eta=.001使全激活时单步阈值增量<=.1。
    return ns['pAUC_CVaR_Loss'](data_len=data_len, pos_len=pos_len, beta=.01, eta=.001, margin=1., device=device)


def official_sopa(params, device):
    ns = dict(torch=torch, math=math)
    source_symbols(SOURCES/'libauc/libauc/optimizers/sopa.py', ['SOPA'], ns)
    return ns['SOPA'](params, mode='adam', lr=3e-4, weight_decay=.05, clip_value=1., device=device)


@lru_cache(maxsize=1)
def scan_reference():
    ns = dict(torch=torch, F=F, rearrange=rearrange, repeat=repeat)
    source_symbols(SOURCES/'selective_scan_interface.py', ['selective_scan_ref'], ns)
    return ns['selective_scan_ref']


def mamba_inner_reference(xz, conv_weight, conv_bias, x_weight, dt_weight, A, B=None, C=None,
                          D=None, delta_bias=None, delta_softplus=True):
    """完整input-dependent selective SSM；使用作者PyTorch scan，不以普通卷积冒充Mamba。"""
    assert B is None and C is None
    x, z = xz.chunk(2, dim=1)
    x = F.silu(F.conv1d(x, conv_weight, conv_bias, padding=conv_weight.shape[-1]-1,
                        groups=x.shape[1])[..., :x.shape[-1]])
    projected = F.linear(x.transpose(1, 2), x_weight)
    rank = dt_weight.shape[1]
    dt = F.linear(projected[..., :rank], dt_weight).transpose(1, 2)
    B = projected[..., rank:rank+A.shape[1]].transpose(1, 2)
    C = projected[..., rank+A.shape[1]:].transpose(1, 2)
    return scan_reference()(x, dt, A, B, C, D, z, delta_bias, delta_softplus)


def attention_reference(qkv, deterministic=True, causal=True):
    q, k, v = qkv.unbind(2)  # B,T,H,D
    logits = torch.einsum('bthd,bshd->bhts', q, k) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.ones(logits.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
        logits = logits.masked_fill(mask, -torch.inf)
    return torch.einsum('bhts,bshd->bthd', logits.softmax(-1), v)


def hybrid_block(width=64, causal=True):
    # 原HybridCausalBlock/MixtureCausalBlock类体不改。仅将缺失的融合CUDA算子接到数学参考。
    # 正式新实验明确使用float32参考计算，不宣称等于原FlashAttention混合精度系统。
    def attention(qkv, **kwargs):
        return attention_reference(qkv, causal=causal)
    ns = dict(torch=torch, nn=nn, F=F, math=math, rearrange=rearrange, repeat=repeat,
              mamba_inner_fn_no_out_proj=mamba_inner_reference, flash_attn_qkvpacked_func=attention)
    source_symbols(SOURCES/'opentad/opentad/models/projections/causal_proj.py',
                   ['HybridCausalBlock', 'MixtureCausalBlock'], ns)
    return ns['HybridCausalBlock'](width, stride=1, kernel_size=4, expand=2, num_head=4, drop_path_rate=0.)


def masked_mean(x, valid):
    return (x * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


class ProposalEncoder(nn.Module):
    def __init__(self, kind='asl', causal=True, width=64):
        super().__init__()
        self.project = nn.Linear(512, width)
        self.local = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.norm = nn.LayerNorm(width)
        self.hybrid = hybrid_block(width, causal) if kind == 'causal' else None
        self.head = nn.Sequential(nn.Linear(3*width+15, width), nn.ReLU(), nn.Linear(width, 2))

    def encode(self, samples, valid):
        x = F.relu(self.project(samples)) * valid[..., None]
        x = self.norm(x + self.local(x.transpose(1, 2)).transpose(1, 2)) * valid[..., None]
        if self.hybrid is not None:
            x = self.hybrid(x.transpose(1, 2), valid)[0].transpose(1, 2)
        return x

    def forward(self, samples, valid, metadata):
        x = self.encode(samples, valid)
        left = masked_mean(x[:, :4], valid[:, :4])
        inner = masked_mean(x[:, 4:20], valid[:, 4:20])
        right = masked_mean(x[:, 20:], valid[:, 20:])
        coverage = torch.stack([valid[:, a:b].float().mean(1) for a,b in ((0,4),(4,20),(20,24))], 1)
        return self.head(torch.cat((inner-left, inner, inner-right, coverage, metadata), 1))


class ASLAdapter(ProposalEncoder):
    """ASE与ASCL核心的源GT区间适配；保留原框的BG读出为本任务新增接口。"""
    def __init__(self, contrast=True, width=64):
        super().__init__('asl', width=width)
        self.contrast = contrast
        self.centers = nn.Parameter(torch.tensor([.5, .05, .95]).repeat(10, 1))
        self.log_sigma = nn.Parameter(torch.full((10, 3), math.log(.2)))
        self.sensitivity = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 2))
        self.activity = nn.Linear(width, 10)
        self.offsets = nn.Linear(width, 2)

    def auxiliary(self, samples, valid, bg_context, labels):
        z = self.encode(samples, valid)
        inner, mask = z[:, 4:20], valid[:, 4:20]
        t = (torch.arange(16, device=z.device, dtype=z.dtype)+.5)/16
        center = self.centers[labels].clamp(0, 1)
        sigma = self.log_sigma[labels].exp().clamp(.03, 1.)
        gaussian = torch.exp(-.5*((t[None,None]-center[...,None])/sigma[...,None]).square())
        gaussian = gaussian * mask[:,None]
        p = gaussian / gaussian.mean(-1,keepdim=True).clamp_min(1e-6)
        q = self.sensitivity(inner).sigmoid()
        cls = self.activity(inner)
        cls_target = F.one_hot(labels,10).to(z.dtype)[:,None].expand_as(cls)
        prob = cls.sigmoid()
        focal = F.binary_cross_entropy_with_logits(cls,cls_target,reduction='none') * (prob-cls_target).abs().square()
        h_cls = p[:,0] + q[...,0]
        cls_loss = (focal.mean(-1)*h_cls*mask).sum()/mask.sum().clamp_min(1)
        distances = F.softplus(self.offsets(inner))
        start, end = t[None]-distances[...,0], t[None]+distances[...,1]
        intersection = (torch.minimum(end,torch.ones_like(end))-torch.maximum(start,torch.zeros_like(start))).clamp_min(0)
        union = (end-start)+1-intersection
        iou = intersection/union.clamp_min(1e-6)
        enclosing = (torch.maximum(end,torch.ones_like(end))-torch.minimum(start,torch.zeros_like(start))).clamp_min(1e-6)
        diou = 1-iou+(((start+end)/2-.5)/enclosing).square()
        h_loc = (p[:,1]+p[:,2])/2 + q[...,1]
        loc_loss = (diou*h_loc*mask).sum()/mask.sum().clamp_min(1)
        target_quality = torch.stack((prob.gather(-1,labels[:,None,None].expand(-1,16,1)).squeeze(-1),iou),-1).detach()
        sens_loss = ((q-target_quality).square()*mask[...,None]).sum()/(2*mask.sum().clamp_min(1))
        # 仅将不覆盖任一任务GT、且采样有效的外部token监督为背景。
        outside = torch.cat((z[:,:4],z[:,20:]),1)
        if bg_context.any():
            cls_loss = cls_loss + F.binary_cross_entropy_with_logits(self.activity(outside)[bg_context],torch.zeros_like(self.activity(outside)[bg_context]))
        vectors=[]
        for part in range(3):
            peak = gaussian[:,part].argmax(-1)
            window = (torch.arange(16,device=z.device)[None]-peak[:,None]).abs()<=3
            w = gaussian[:,part]*window*mask
            vectors.append((inner*w[...,None]).sum(1)/w.sum(1,keepdim=True).clamp_min(1e-6))
        features = F.normalize(torch.cat((vectors[0],(vectors[1]+vectors[2])/2),0),dim=-1)
        cats = labels.repeat(2)
        positives = (cats[:,None]==cats[None]) & ~torch.eye(len(cats),dtype=torch.bool,device=z.device)
        logits = features@features.T/.1
        logits = logits.masked_fill(torch.eye(len(cats),dtype=torch.bool,device=z.device),-torch.inf)
        # 背景向量是每实例外部有效纯背景的均值；没有可用背景则不造零向量负例。
        have_bg = bg_context.any(1)
        if have_bg.any():
            bv = F.normalize(masked_mean(outside,bg_context)[have_bg],dim=-1)
            all_logits = torch.cat((logits,features@bv.T/.1),1)
        else:all_logits=logits
        contrast_loss = (torch.logsumexp(all_logits,1)-torch.logsumexp(logits.masked_fill(~positives,-torch.inf),1)).mean()
        total = cls_loss+loc_loss+sens_loss+(contrast_loss if self.contrast else contrast_loss*0)
        return total, dict(cls=cls_loss.detach(),loc=loc_loss.detach(),sensitivity=sens_loss.detach(),contrast=contrast_loss.detach(),BG_context=int(bg_context.sum()))


def head_objective(logits, bg, quality):
    return balanced_bg(logits[:,0],bg)+.25*quality_loss(logits[:,1],quality)


def knn_distance(query, bank, k=50, chunk=128):
    """精确归一化L2近邻；k预先固定，不对验证集搜索。"""
    query = F.normalize(query,dim=1)
    bank = F.normalize(bank,dim=1)
    assert len(bank)>0
    results=[]
    for q in query.split(chunk):
        # 距离排序使用float64累加；防止原环境强制TF32改变最近邻身份。
        distances = (q.double().square().sum(1,keepdim=True)+bank.double().square().sum(1)[None]
                     -2*(q.double()@bank.double().T)).clamp_min(0)
        results.append(distances.topk(min(k,len(bank)),largest=False).values[:,-1])
    return torch.cat(results) if results else query.new_empty(0)
