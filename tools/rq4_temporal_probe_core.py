"""源域留出机制探针；不是SADA训练器或正式RQ4结果。"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

POLICY = dict(seed=6, half_window=4, source_points_per_instance=5,
              candidate_points=7, participation=.2, hidden=128,
              prototypes_per_role=4, temperature=.1, auxiliary_weight=.1,
              warmup_steps=800, adaptation_steps=400, videos_per_batch=2,
              points_per_video=32, learning_rate=.001)
ARMS = ('none', 'proposal', 'timelens2_8b', 'timeple_8b')


def phase_distribution(times, interval):
    """连续起始/内部/结束角色；区间外为弃权，不标背景。"""
    times = torch.as_tensor(times, dtype=torch.float32)
    if interval is None:
        return torch.zeros(len(times), 3)
    start, end = map(float, interval)
    if not math.isfinite(start + end) or end <= start:
        raise ValueError('invalid phase interval')
    z = (times - start) / (end - start)
    centers = times.new_tensor([0., .5, 1.])
    q = (1 - 2 * (z[:, None] - centers).abs()).clamp_min(0)
    q[(z < 0) | (z > 1)] = 0
    return q


def paired_distribution(times, intervals):
    # 两个上下文等权；仅一个有效时总权重为0.5，不补抽或升权。
    if len(intervals) != 2:
        raise ValueError('two contexts required')
    return sum(phase_distribution(times, box) for box in intervals) / 2


def real_indices(interval, fps, stride, num_frames, length, count):
    lo = max(0, math.ceil((interval[0]*fps - .5*num_frames)/stride))
    hi = min(length-1, math.floor((interval[1]*fps - .5*num_frames)/stride))
    if hi < lo:
        return torch.empty(0, dtype=torch.long)
    return torch.linspace(lo, hi, min(count, hi-lo+1)).round().long().unique()


def local_patches(features, indices):
    # 有序的真实grid邻居；视频边缘重复端点，不插值、不输入GT/角色坐标。
    neighbors = (indices[:, None] + torch.arange(-POLICY['half_window'],
                 POLICY['half_window']+1)[None]).clamp(0, features.shape[1]-1)
    return features[:, neighbors].permute(1, 0, 2).contiguous()


class TemporalProbe(nn.Module):
    def __init__(self, channels):
        super().__init__()
        h = POLICY['hidden']
        self.project = nn.Conv1d(channels, h, 1)
        self.encode = nn.Sequential(nn.GELU(), nn.Flatten(),
            nn.Linear(h*(2*POLICY['half_window']+1), h), nn.GELU(), nn.LayerNorm(h))
        self.role = nn.Linear(h, 3)
        self.offset = nn.Linear(h, 2)

    def forward(self, x):
        z = self.encode(self.project(x))
        return F.normalize(z, dim=-1), self.role(z), F.softplus(self.offset(z))


def supervised_loss(model, x, roles, log_offsets):
    _, logits, prediction = model(x)
    return -(roles*F.log_softmax(logits, -1)).sum(-1).mean() + \
        F.smooth_l1_loss(prediction, log_offsets)


@torch.no_grad()
def source_bank(model, x, roles, device):
    embeddings = torch.cat([model(chunk.to(device))[0].cpu()
                            for chunk in x.split(512)])
    labels = roles.argmax(-1)
    banks = []
    for role in range(3):
        points = embeddings[labels == role]
        k = POLICY['prototypes_per_role']
        if len(points) < k:
            raise ValueError('source role lacks prototype support')
        # 确定性最远点初始化和固定10步球面kmeans；只读源训练表征。
        centers = [points[0]]
        for _ in range(1, k):
            similarity = points @ torch.stack(centers).T
            centers.append(points[similarity.max(1).values.argmin()])
        centers = torch.stack(centers)
        for _ in range(10):
            assigned = (points @ centers.T).argmax(-1)
            centers = torch.stack([F.normalize(points[assigned == j].mean(0), dim=0)
                                   if (assigned == j).any() else centers[j] for j in range(k)])
        banks.append(centers)
    return torch.stack(banks).to(device).detach()


def alignment_loss(embeddings, bank, weights):
    scores = torch.einsum('nd,rkd->nrk', embeddings, bank.detach()) / POLICY['temperature']
    logits = torch.logsumexp(scores, -1) - math.log(bank.shape[1])
    # 共同曝光分母，弃权贡献零；不按接受率倒数放大。
    return -(weights*F.log_softmax(logits, -1)).sum(-1).mean()


def video_batches(video_ids, steps, seed=6):
    generator = torch.Generator().manual_seed(seed)
    groups = [torch.tensor([i for i, v in enumerate(video_ids) if v == vid])
              for vid in sorted(set(video_ids))]
    if not groups:
        raise ValueError('empty training rows')
    result = []
    for _ in range(steps):
        chosen = torch.randperm(len(groups), generator=generator)[:POLICY['videos_per_batch']]
        batch = [groups[j][torch.randint(len(groups[j]), (POLICY['points_per_video'],),
                                        generator=generator)] for j in chosen.tolist()]
        result.append(torch.cat(batch))
    return result


def check_training_payload(source, unlabeled):
    if set(source) != {'x', 'roles', 'log_offsets', 'video_ids'}:
        raise ValueError('source payload schema mismatch')
    if set(unlabeled) != {'x', 'routes', 'video_ids', 'candidate_ids', 'times', 'selected'}:
        raise ValueError('unlabeled payload may contain labels')
    if set(source['video_ids']) & set(unlabeled['video_ids']):
        raise ValueError('training/heldout video overlap')
    if set(unlabeled['routes']) != set(ARMS[1:]):
        raise ValueError('route arms differ')
    for obj in (source, unlabeled):
        if len(obj['x']) != len(obj['video_ids']) or not torch.isfinite(obj['x']).all():
            raise ValueError('invalid visual inputs')
    if source['roles'].shape != (len(source['x']), 3) or not torch.allclose(
            source['roles'].sum(-1), torch.ones(len(source['x'])), atol=1e-5):
        raise ValueError('source roles invalid')
    if source['log_offsets'].shape != (len(source['x']), 2) or not torch.isfinite(source['log_offsets']).all():
        raise ValueError('source offsets invalid')
    if len(unlabeled['candidate_ids'])!=len(unlabeled['x']) or \
            unlabeled['times'].shape!=(len(unlabeled['x']),) or \
            unlabeled['selected'].shape!=(len(unlabeled['x']),) or \
            unlabeled['selected'].dtype!=torch.bool:
        raise ValueError('unlabeled exposure identity mismatch')
    for q in unlabeled['routes'].values():
        if q.shape != (len(unlabeled['x']), 3) or not torch.isfinite(q).all() or \
                (q < 0).any() or (q.sum(-1) > 1.00001).any():
            raise ValueError('invalid unlabeled routing')
    if not unlabeled['selected'].any():
        raise ValueError('empty fixed participation set')


def train_arms(source, unlabeled, device, emit, save_arm):
    """不接收评分标签或评分文件路径；所有臂共享batch索引与源warmup。"""
    check_training_payload(source, unlabeled)
    torch.manual_seed(POLICY['seed'])
    model = TemporalProbe(source['x'].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=POLICY['learning_rate'])
    batches = video_batches(source['video_ids'], POLICY['warmup_steps'])
    model.train()
    for step, idx in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss(model, source['x'][idx].to(device), source['roles'][idx].to(device),
                               source['log_offsets'][idx].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite warmup')
        loss.backward(); optimizer.step()
        if step % 100 == 0:
            emit('warmup', step, float(loss.detach()))
    initial = deepcopy(model.state_dict())
    model.eval()
    bank = source_bank(model, source['x'], source['roles'], device)
    selected = torch.where(unlabeled['selected'])[0]
    adapted_video_ids = [unlabeled['video_ids'][i] for i in selected.tolist()]
    source_batches = video_batches(source['video_ids'], POLICY['adaptation_steps'], seed=POLICY['seed'])
    heldout_batches = video_batches(adapted_video_ids, POLICY['adaptation_steps'], seed=POLICY['seed'])
    records = {}
    for arm in ARMS:
        model.load_state_dict(initial, strict=True)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=POLICY['learning_rate'])
        history = []
        for step, (src, sampled) in enumerate(zip(source_batches, heldout_batches)):
            idx = selected[sampled]
            optimizer.zero_grad(set_to_none=True)
            main = supervised_loss(model, source['x'][src].to(device), source['roles'][src].to(device),
                                   source['log_offsets'][src].to(device))
            embedding, _, _ = model(unlabeled['x'][idx].to(device))
            weights = torch.zeros(len(idx), 3, device=device) if arm == 'none' \
                else unlabeled['routes'][arm][idx].to(device)
            aux = alignment_loss(embedding, bank, weights)
            # 对共享视觉编码参数核验辅助确实产生梯度，不能只查投影输出。
            gradient = torch.autograd.grad(POLICY['auxiliary_weight']*aux,
                model.project.weight, retain_graph=True)[0].norm()
            loss = main + POLICY['auxiliary_weight']*aux
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise RuntimeError('nonfinite probe loss/gradient')
            loss.backward(); optimizer.step()
            history.append(dict(step=step+1, source_loss=float(main.detach()),
                auxiliary_loss=float(aux.detach()), shared_visual_gradient=float(gradient.detach()),
                exposure=len(idx), routed_mass=float(weights.sum()), heldout_gt_loss=0.))
            if step % 100 == 0:
                emit(arm, step, history[-1])
        model.eval()
        with torch.no_grad():
            outputs = [model(x.to(device)) for x in unlabeled['x'].split(512)]
        predictions = dict(roles=torch.cat([out[1].softmax(-1).cpu() for out in outputs]),
                           log_offsets=torch.cat([out[2].cpu() for out in outputs]))
        # 在加载任何评分GT前保存全部模型和预测；不按验证结果选epoch。
        save_arm(arm, model, bank, history, predictions)
        records[arm] = predictions
    return records


def scalar_mean(values):
    return sum(values)/len(values) if values else None


def score_predictions(predictions, rows, truth):
    """评分专用：与原proposal绑定同一GT实例，不按新版预测重新挑GT。"""
    q = predictions['roles']
    distances = predictions['log_offsets'].expm1()
    if not torch.isfinite(distances).all():
        raise ValueError('nonfinite predicted boundary')
    output = []
    for candidate in truth['candidates']:
        ids = torch.tensor(candidate['row_indices'], dtype=torch.long)
        times = rows['times'][ids]
        left = (times-distances[ids, 0]).median().clamp(0, candidate['duration'])
        right = (times+distances[ids, 1]).median().clamp(0, candidate['duration'])
        box = [float(left), float(right)]
        gt = candidate['gt']
        iou, error = 0., None
        if gt is not None:
            overlap = max(0., min(box[1], gt[1])-max(box[0], gt[0]))
            union = max(box[1], gt[1])-min(box[0], gt[0])
            iou = overlap/union if union > 0 else 0.
            error = (abs(box[0]-gt[0])+abs(box[1]-gt[1]))/(2*(gt[1]-gt[0]))
        output.append(dict(candidate_id=candidate['candidate_id'],video_id=candidate['video_id'],
            predicted=box, iou=iou, normalized_endpoint_error=error, matched_gt=gt is not None))
    role_truth = torch.tensor(truth['roles'], dtype=torch.float32)
    role_valid = role_truth.sum(-1) > 0
    accuracy = float((q[role_valid].argmax(-1)==role_truth[role_valid].argmax(-1)).float().mean()) \
        if role_valid.any() else None
    per_video = []
    for video in sorted({r['video_id'] for r in output}):
        values = [r for r in output if r['video_id']==video]
        per_video.append(dict(video_id=video, mean_iou=scalar_mean([r['iou'] for r in values]),
            normalized_endpoint_error=scalar_mean([r['normalized_endpoint_error'] for r in values
                                                  if r['normalized_endpoint_error'] is not None])))
    adapted={c['candidate_id'] for c in truth['candidates'] if c.get('selected',False)}
    subsets={name:dict(count=len(group),mean_iou=scalar_mean([r['iou'] for r in group]))
        for name,group in (('adapted',[r for r in output if r['candidate_id'] in adapted]),
                           ('not_adapted',[r for r in output if r['candidate_id'] not in adapted]))}
    return dict(candidate_count=len(output), matched_gt_count=sum(r['matched_gt'] for r in output),
        mean_iou=scalar_mean([r['iou'] for r in output]),
        video_balanced_mean_iou=scalar_mean([v['mean_iou'] for v in per_video]),
        normalized_endpoint_error=scalar_mean([r['normalized_endpoint_error'] for r in output
                                              if r['normalized_endpoint_error'] is not None]),
        role_accuracy=accuracy, role_evaluated_points=int(role_valid.sum()),
        adaptation_subsets=subsets,per_video=per_video, rows=output)


def score_routes(rows, truth):
    correct = torch.tensor(truth['roles'], dtype=torch.float32)
    foreground = correct.sum(-1)>0
    reports = {}
    for name, q in rows['routes'].items():
        mass=q.sum(-1); accepted=mass>0; valid=foreground & accepted
        normalized=q/mass.clamp_min(1e-8)[:,None]
        reports[name]=dict(exposed_points=len(q), accepted_points=int(accepted.sum()),
            accepted_foreground_points=int(valid.sum()),
            foreground_precision=float(foreground[accepted].float().mean()) if accepted.any() else None,
            role_accuracy=float((q[valid].argmax(-1)==correct[valid].argmax(-1)).float().mean())
                if valid.any() else None,
            phase_brier=float((normalized[valid]-correct[valid]).square().sum(-1).mean())
                if valid.any() else None)
    # 同曝光交集比较，防止弃掉困难点后精度升高被误当新增信息。
    proposal=rows['routes']['proposal']
    reports['paired_common_support']={}
    for teacher in ARMS[2:]:
        q=rows['routes'][teacher]
        keep=foreground & (proposal.sum(-1)>0) & (q.sum(-1)>0)
        scores={name:float((p[keep].argmax(-1)==correct[keep].argmax(-1)).float().mean())
                if keep.any() else None for name,p in (('proposal',proposal),('teacher',q))}
        reports['paired_common_support'][teacher]=dict(points=int(keep.sum()),**scores)
    return reports
