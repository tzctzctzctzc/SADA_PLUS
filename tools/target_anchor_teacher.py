"""把目标域高置信动作的分类与边界知识接到原检测器，而非再训独立BG头。

参考SoftTeacher的目标域分类及回归伪监督；这是一维dense适配，不是其完整复现。
固定原Prototype教师，学生仅通道mask增强，不改变时间坐标。未选中anchor不当背景。
"""
import torch
from torch.nn import functional as F


def mask_channels(raw, generator, probability=.1):
    # 在CPU专用随机流中抽样，避免不同模块消费原训练/DA的随机数。
    keep = (torch.rand(raw.shape[1], generator=generator) >= probability)
    return raw * keep.to(raw.device, raw.dtype)[None] / (1-probability)


def target_loss(student_cls, student_reg, teacher_cls, teacher_reg, masks, threshold=.9):
    logits = torch.cat(student_cls)
    offsets = torch.cat(student_reg)
    prob = torch.cat(teacher_cls).detach().sigmoid()
    reference = torch.cat(teacher_reg).detach()
    valid = torch.cat(masks)
    confidence = prob.max(-1).values
    selected = valid & (confidence >= threshold)
    count = int(selected.sum())
    zero = logits.sum()*0 + offsets.sum()*0
    if not count:
        return zero, dict(cls=zero.detach(), reg=zero.detach(), selected=0)
    weights = confidence[selected]
    # 同一anchor保留完整类别分布，而非把其余九类强制成伪负例。
    cls = F.binary_cross_entropy_with_logits(logits[selected], prob[selected], reduction='none').sum(-1)
    cls = (cls*weights).sum()/weights.sum()
    # 使用原中心对齐的一维DIoU；教师输出detach，不引入目标标注。
    a, b = offsets[selected], reference[selected]
    inter = torch.minimum(a[:,0],b[:,0])+torch.minimum(a[:,1],b[:,1])
    union = a.sum(-1)+b.sum(-1)-inter
    enclosure = torch.maximum(a[:,0],b[:,0])+torch.maximum(a[:,1],b[:,1])
    centers = ((a[:,1]-a[:,0])-(b[:,1]-b[:,0]))*.5
    diou = 1-inter/union.clamp_min(1e-8)+centers.square()/enclosure.square().clamp_min(1e-8)
    # 正长度只是几何有效性，不声称分类置信度证明边界正确；尚未实现box jitter。
    reliable = (b>0).all(-1)
    reg = (diou[reliable]*weights[reliable]).sum()/weights[reliable].sum().clamp_min(1)
    return cls+reg, dict(cls=cls.detach(), reg=reg.detach(), selected=count)


def gpu_acceptance():
    """新目标损失的CPU/GPU值和梯度参照；不伪称全检测器重建逐bit等价。"""
    results=[]
    for device in ('cpu','cuda'):
        c=torch.tensor([[2.,-2.],[1.,0.],[3.,-1.]],device=device,requires_grad=True)
        r=torch.tensor([[1.,2.],[2.,2.],[1.,3.]],device=device,requires_grad=True)
        tc=torch.tensor([[4.,-3.],[0.,0.],[4.,-3.]],device=device,requires_grad=True)
        tr=torch.tensor([[2.,3.],[1.,1.],[2.,2.]],device=device,requires_grad=True)
        value,stats=target_loss([c],[r],[tc],[tr],[torch.tensor([True,True,False],device=device)])
        value.backward()
        assert stats['selected']==1 and tc.grad is None and tr.grad is None
        assert c.grad[1:].count_nonzero()==0 and r.grad[1:].count_nonzero()==0
        results.append([value.detach().cpu(),c.grad.cpu(),r.grad.cpu()])
    errors=[]
    for a,b in zip(*results):
        assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
        errors.append(float((a-b).abs().max()))
    return dict(passed=True,loss_class_reg_gradient_max_abs_errors=errors,teacher_gradient_free=True,unselected_gradient_zero=True)
