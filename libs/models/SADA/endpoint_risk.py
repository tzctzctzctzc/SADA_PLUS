"""源监督的逐端点误差尺度；target只使用detach门控，不训练误差头。"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'sada-endpoint-relative-risk-v1'
MODES = ('head_only', 'ema', 'random', 'relative')
POLICY = dict(source_loss='detached_residual_gaussian_nll', variance_floor=1e-6,
    detach_source_features=True, detach_source_mean=True, target_updates_risk_head=False,
    separate_optimizer=True, separate_clip=True, independent_teacher=True,
    correspondence='same_crop_original_fpn_real_point', interpolation=False,
    proposal_budget='unchanged_original_selected_40_of_200', max_per_video=256,
    normalization='two_times_original_eligible_proposals',
    target_gt_used=False, prototype_unchanged=True)


def validate_options(options):
    if options.get('schema') != SCHEMA or options.get('mode') not in MODES:
        raise ValueError('不是本方法的明确身份/模式')
    if options.get('policy') != POLICY:
        raise ValueError('端点风险协议改变')
    for key in ('head_learning_rate', 'head_clip', 'target_weight'):
        value=options[key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError('必须显式提供正有限参数: '+key)
    if not isinstance(options['warmup_epochs'],int) or options['warmup_epochs'] < 0:
        raise ValueError('必须显式冻结源域预热轮数')


def risk_nll(variance, mean, target, positive):
    """gt_offset与mean都以当前FPN stride为单位；背景/无效点不参与。"""
    positive=positive.detach().bool()
    if variance.shape!=mean.shape or variance.shape!=target.shape or positive.shape!=mean.shape[:-1]:
        raise ValueError('风险监督形状不符')
    if not positive.any():return variance.sum()*0.
    var=variance[positive]
    residual=mean.detach()[positive]-target.detach()[positive]
    if not torch.isfinite(var).all() or (var<=0).any() or not torch.isfinite(residual).all():
        raise ValueError('风险尺度/源残差非法')
    return .5*(residual.square()/var+var.log()).mean()


def decode_at(observation, level, batch, index):
    point=observation['points'][level][index]
    offset=observation['offsets'][level][batch,index]
    var=observation['variance'][level][batch,index]
    if not torch.isfinite(offset).all() or (offset<0).any() or not torch.isfinite(var).all() or (var<=0).any():
        raise ValueError('端点预测/误差尺度非法')
    # 局部特征时间坐标；只乘一次stride，方差因此乘stride的平方。
    boundary=torch.stack((point[0]-offset[0]*point[3],point[0]+offset[1]*point[3]))
    sigma=var.sqrt()*point[3]
    return boundary,sigma


@torch.no_grad()
def calibrated_gate(student_sigma, teacher_sigma, levels, durations, calibration):
    """比较共同proposal尺度的预测风险；不是直接比较未校准的两个sigma。"""
    if calibration is None or calibration.get('schema')!=SCHEMA or calibration.get('status')!='passed':
        raise ValueError('缺少通过源域教师优势核验的校准')
    if calibration.get('target_gt_used') or calibration.get('policy')!=POLICY:
        raise ValueError('校准来源/方法不符')
    if student_sigma.shape!=teacher_sigma.shape or student_sigma.shape!=(len(levels),2):
        raise ValueError('师生尺度必须是同一批逐端点观测')
    if not torch.isfinite(durations).all() or (durations<=0).any():raise ValueError('proposal时长非法')
    gate=torch.zeros_like(student_sigma,dtype=torch.bool)
    predicted_gain=torch.zeros_like(student_sigma)
    margin=calibration['margin']
    if not math.isfinite(margin) or margin<0:raise ValueError('源域margin非法')
    for i,level in enumerate(levels.tolist()):
        group=calibration['factors'].get(str(level))
        if group is None:continue  # 无源证据的层不借用其他层尺度。
        for side in (0,1):
            if not group['available'][side]:continue
            a,b=group['student'][side],group['teacher'][side]
            if not all(math.isfinite(x) and x>0 for x in (a,b)):
                raise ValueError('校准系数必须为正有限数')
            gain=(a*student_sigma[i,side]-b*teacher_sigma[i,side])/durations[i]
            predicted_gain[i,side]=gain
            gate[i,side]=gain>margin
    return gate.detach(),predicted_gain.detach()


def endpoint_loss(student, teacher, gate, duration, eligible):
    """保持原eligible分母，左右独立；单候选/单端点也不会被归一抵消。"""
    if student.shape!=teacher.shape or gate.shape!=student.shape or student.shape!=(eligible,2):
        raise ValueError('端点损失形状/原候选数不符')
    if eligible==0:return student.sum()*0.
    if not torch.isfinite(duration).all() or (duration<=0).any():raise ValueError('proposal时长非法')
    delta=(student-teacher.detach())/duration.detach()[:,None]
    return (F.smooth_l1_loss(delta,torch.zeros_like(delta),reduction='none')*
            gate.detach().to(delta.dtype)).sum()/(2*eligible)


class EndpointRisk(nn.Module):
    def __init__(self, channels, options):
        super().__init__();validate_options(options)
        self.options=deepcopy(options)
        # 不调用manual_seed：新头只消耗随后恢复的CPU随机流，不污染CUDA播种。
        with torch.random.fork_rng(devices=[]):self.head=nn.Conv1d(channels,2,1)
        self.calibration=None
        self.last_source_loss=None
        self.last_target_loss=None
        self.statistics={}

    def forward(self, regression_features):
        return [F.softplus(self.head(x.detach())).permute(0,2,1)+POLICY['variance_floor']
                for x in regression_features]

    def source_loss(self, variance, mean, masks, gt_classes, gt_offsets):
        var=torch.cat(variance,1);mu=torch.cat(mean,1)
        mask=torch.cat([x.reshape(x.shape[0],-1) for x in masks],1).bool()
        classes=torch.stack(gt_classes);target=torch.stack(gt_offsets)
        positive=mask & (classes.sum(-1)>0)
        loss=risk_nll(var,mu,target,positive)
        self.last_source_loss=loss
        self.statistics=dict(source_positive_points=int(positive.sum()),source_nll=float(loss.detach()))
        return loss

    def install_calibration(self, calibration):
        if calibration['schema']!=SCHEMA or calibration['status']!='passed' or calibration['policy']!=POLICY:
            raise ValueError('源机制证据尚未通过')
        if calibration['target_gt_used'] or set(calibration['fit_videos']) & set(calibration['audit_videos']):
            raise ValueError('校准GT边界或视频隔离失败')
        self.calibration=deepcopy(calibration)

    def target_loss(self, videos, cache, threshold, student, teacher, epoch, random_generator=None):
        from .prototype_capacity import fully_visible_cache,feature_interval,KNOWN_LABELS
        if any(v['segments'].numel() or v['labels'].numel() for v in videos):
            raise ValueError('目标GT不可进入定位辅助')
        mode=self.options['mode'];zero=student['offsets'][0].sum()*0.
        self.last_target_loss=zero
        if mode=='head_only' or epoch<=self.options['warmup_epochs']:
            self.statistics.update(eligible_proposals=0,selected_endpoints=0,pseudo_boundary_loss=0.,
                left_selected=0,right_selected=0,pseudo_gradient_norm=0.)
            return zero
        if teacher is None:raise ValueError('缺少真实独立EMA目标观测')
        kept,_,_=fully_visible_cache(cache,videos,threshold)
        candidates=[]
        for batch,video in enumerate(videos):
            if teacher['metadata'][batch]!=student['metadata'][batch]:raise ValueError('师生crop身份不符')
            meta=video['metadata'];crop=float(meta.get('crop_start',0));length=video['feats'].shape[-1]
            for row in kept[batch][:POLICY['max_per_video']]:
                label=row['semantic_label']
                if label not in KNOWN_LABELS:continue
                cls=KNOWN_LABELS.index(label);level=int(row['source_fpn_levels'][0])
                if not 0<=level<len(student['points']):raise ValueError('原proposal FPN层非法')
                a,b=feature_interval(row,meta);a-=crop;b-=crop
                grid=student['points'][level]
                if not torch.equal(grid,teacher['points'][level]):raise ValueError('师生真实grid不一致')
                common=student['masks'][level][batch].reshape(-1).bool() & teacher['masks'][level][batch].reshape(-1).bool()
                ids=(common & (grid[:,0]>=a) & (grid[:,0]<=b)).nonzero(as_tuple=True)[0]
                if not len(ids):continue
                index=int(ids[(grid[ids,0]-(a+b)/2).abs().argmin()])
                sb,ss=decode_at(student,level,batch,index);tb,ts=decode_at(teacher,level,batch,index)
                # 只使用原proposal的语义支持；不把缓存类别广播给全crop。
                semantic=int(teacher['logits'][level][batch,index].argmax())==cls
                visible=torch.stack((tb>=0,tb<=length)).all(0)  # 左右分别拒绝越界端点。
                candidates.append(dict(student=sb,teacher=tb,s_sigma=ss,t_sigma=ts,
                    duration=b-a,level=level,allowed=visible & semantic,batch=batch,label=cls))
        n=len(candidates)
        if not n:
            self.statistics.update(eligible_proposals=0,selected_endpoints=0,pseudo_boundary_loss=0.,
                left_selected=0,right_selected=0,pseudo_gradient_norm=0.)
            return zero
        sb=torch.stack([x['student'] for x in candidates]);tb=torch.stack([x['teacher'] for x in candidates])
        duration=sb.new_tensor([x['duration'] for x in candidates])
        allowed=torch.stack([x['allowed'] for x in candidates])
        if mode=='ema':gate=allowed
        else:
            gate,_=calibrated_gate(torch.stack([x['s_sigma'] for x in candidates]),
                torch.stack([x['t_sigma'] for x in candidates]),
                torch.tensor([x['level'] for x in candidates],device=sb.device),duration,self.calibration)
            gate &= allowed
            if mode=='random':
                if random_generator is None:raise ValueError('随机门控对照必须使用独立可恢复随机流')
                # 在本批同层/类别/时长箱/左右中等数量抽样；教师坐标保持不变。
                mixed=torch.zeros_like(gate)
                for side in (0,1):
                    groups={}
                    for i,x in enumerate(candidates):
                        groups.setdefault((x['level'],x['label'],math.floor(math.log2(x['duration']))),[]).append(i)
                    for indices in groups.values():
                        pool=[i for i in indices if bool(allowed[i,side])]
                        count=sum(bool(gate[i,side]) for i in indices)
                        order=torch.randperm(len(pool),generator=random_generator).tolist()
                        for j in order[:count]:mixed[pool[j],side]=True
                gate=mixed
        loss=endpoint_loss(sb,tb,gate,duration,n)*self.options['target_weight']
        self.last_target_loss=loss
        observed=[x for x in student['offsets'] if x.requires_grad]
        gradients=(torch.autograd.grad(loss,observed,retain_graph=True,allow_unused=True)
                   if observed and loss.requires_grad else [])
        norm=sum((g.detach().square().sum() for g in gradients if g is not None),loss.detach().new_zeros(())).sqrt()
        self.statistics.update(eligible_proposals=n,selected_endpoints=int(gate.sum()),
            pseudo_boundary_loss=float(loss.detach()),left_selected=int(gate[:,0].sum()),
            right_selected=int(gate[:,1].sum()),pseudo_gradient_norm=float(norm))
        return loss
