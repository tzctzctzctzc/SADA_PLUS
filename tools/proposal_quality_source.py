"""冻结源模型的完整proposal质量读出；不接触target、不改变NMS或候选集合。"""
from copy import deepcopy
from pathlib import Path
import gc
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SPEC = dict(schema='source-proposal-quality-readout-v1', seed=123,
    features='finest_fpn_left_context_start_interior_end_right_context_score_duration_class',
    context_fraction=.5, hidden=256, epochs=20, batch=2048, lr=.001,
    weight_decay=.0001, supervision='maximum_same_class_tiou', loss='soft_target_bce',
    score='original_post_nms_score_times_quality', target_data_used=False,
    validation='source_validation_once_after_fixed_epoch20', formal_student_training=False)


def same_class_quality(boxes, labels, gt_boxes, gt_labels):
    if len(gt_boxes) == 0:
        return boxes.new_zeros(len(boxes))
    intersection = (torch.minimum(boxes[:, None, 1], gt_boxes[None, :, 1]) -
                    torch.maximum(boxes[:, None, 0], gt_boxes[None, :, 0])).clamp_min(0)
    union = (boxes[:, 1]-boxes[:, 0])[:, None] + (gt_boxes[:, 1]-gt_boxes[:, 0])[None] - intersection
    matches = labels[:, None] == gt_labels[None]
    return (intersection / union.clamp_min(1e-8) * matches).amax(1)


def region_features(fpn, valid, points, boxes, scores, labels, seconds, classes):
    """前缀和批量池化真实最细层；边界nearest，越出视频的上下文裁到有效范围。"""
    count = int(valid.sum())
    if not count or not bool(valid[:count].all()) or bool(valid[count:].any()):
        raise ValueError('源视频有效mask不是连续前缀')
    feature = fpn[:, :count]
    stride, origin = points[0, 3], points[0, 0]
    coordinates = (boxes-origin)/stride
    prefix = torch.cat([feature.new_zeros(feature.shape[0], 1), feature.cumsum(-1)], -1)
    def mean(lo, hi):
        lo = lo.ceil().long().clamp(0, count)
        hi = (hi.floor().long()+1).clamp(0, count)
        length = (hi-lo).clamp_min(0)
        return ((prefix[:, hi]-prefix[:, lo])/length.clamp_min(1)[None] * (length>0)[None]).T
    def nearest(at):
        return feature[:, at.round().long().clamp(0, count-1)].T
    start, end = coordinates.unbind(-1)
    half = (end-start)*SPEC['context_fraction']
    blocks = [mean(start-half, start), nearest(start), mean(start, end), nearest(end), mean(end, end+half)]
    extra = [scores[:, None], torch.log1p(seconds.clamp_min(0))[:, None],
             F.one_hot(labels.long(), classes).to(feature.dtype)]
    return torch.cat(blocks+extra, -1)


class Readout(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, SPEC['hidden']), nn.ReLU(), nn.Linear(SPEC['hidden'], 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_readout(paths, device):
    # 全部源train预测参加；统计量及优化只用train，不用val选择epoch/权重。
    rows = [np.load(path, allow_pickle=False) for path in paths]
    x = np.concatenate([r['features'] for r in rows]); y = np.concatenate([r['quality'] for r in rows])
    for row in rows: row.close()
    if len(x) < SPEC['batch'] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('源train读出样本不足或非有限')
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    std = x.std(0, dtype=np.float64).clip(.001).astype(np.float32)
    generator = torch.Generator().manual_seed(SPEC['seed'])
    torch.manual_seed(SPEC['seed']); torch.cuda.manual_seed(SPEC['seed'])
    model = Readout(x.shape[1]).to(device)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPEC['lr'], weight_decay=SPEC['weight_decay'])
    norm_mean, norm_std = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    history = []; first_gradient_norm = None
    for epoch in range(1, SPEC['epochs']+1):
        order = torch.randperm(len(x), generator=generator).numpy(); total = 0.
        for start in range(0, len(x), SPEC['batch']):
            ids = order[start:start+SPEC['batch']]
            batch = (torch.from_numpy(x[ids]).to(device)-norm_mean)/norm_std
            truth = torch.from_numpy(y[ids]).to(device)
            loss = F.binary_cross_entropy_with_logits(model(batch), truth)
            if not bool(torch.isfinite(loss)): raise ValueError('源train拟合出现非有限loss')
            optimizer.zero_grad(); loss.backward()
            if first_gradient_norm is None:
                first_gradient_norm = float(sum(v.grad.square().sum() for v in model.parameters()).sqrt())
                if not math.isfinite(first_gradient_norm) or first_gradient_norm<=0:raise ValueError('源读出梯度无效')
            optimizer.step()
            total += float(loss.detach())*len(ids)
        history.append(dict(epoch=epoch, loss=total/len(x)))
        print('READOUT_EPOCH', epoch, history[-1]['loss'], flush=True)
    del x, y; gc.collect()
    model.eval()
    updated = any(not torch.equal(initial[k],v.detach().cpu()) for k,v in model.state_dict().items())
    if not updated or not all(bool(torch.isfinite(v).all()) for v in model.parameters()):raise ValueError('源读出参数未正确更新')
    return model, mean, std, history, dict(first_gradient_norm=first_gradient_norm,parameters_updated=updated,
        validation_used_for_fit=False,completed_epochs=SPEC['epochs'])


def score_predictions(model, mean, std, path, device):
    with np.load(path, allow_pickle=False) as rows:
        original = rows['scores'].copy(); prediction = np.empty(len(original), np.float32)
        with torch.no_grad():
            for start in range(0, len(original), SPEC['batch']):
                x = torch.from_numpy((rows['features'][start:start+SPEC['batch']]-mean)/std).to(device)
                prediction[start:start+SPEC['batch']] = model(x).sigmoid().cpu().numpy()
        return rows['segments'].copy(), rows['labels'].copy(), original, original*prediction


def worker(args):
    from libs.models.SADA import make_meta_arch
    from libs.evaluation import ANETdetection
    from tools import baseline_protocol as p
    from tools.baseline_engine import dataset, tensor_digest
    from tools.baseline_gpu import first_cuda_guard
    from tools.calibrate_prototype_geometry import prepare_source_video

    entry = args.output_root/'source'/args.child
    entry.mkdir(parents=True, exist_ok=False)
    identity = p.code_identity(); reference = args.control/'formal/control'/args.child
    cfg = p.read(reference/'config.json'); done = p.read(reference/'complete.json')
    for name in ['config.json', 'best.pt', 'termination.json', 'evaluation/source_val.json']:
        if p.sha(reference/name) != done['artifacts'][str(reference/name)]: raise ValueError('源参考工件改变: '+name)
    manifest = Path(cfg['dataset']['json_file'])
    if p.sha(manifest) != cfg['baseline']['source_manifest_sha256']: raise ValueError('源manifest改变')
    os_data = args.data_root
    import os
    os.chdir(os_data)
    guard = first_cuda_guard(); device = 'cuda:0'
    torch.manual_seed(SPEC['seed']); torch.cuda.manual_seed(SPEC['seed'])
    saved = torch.load(reference/'best.pt', map_location='cpu')
    if saved['config'] != cfg or saved['epoch'] != p.termination(p.read(reference/'termination.json')['history'])['best_epoch']:
        raise ValueError('不是原源选best EMA')
    model = make_meta_arch('sada', deepcopy(cfg), device=device).to(device)
    model.load_state_dict(saved['ema_state_dict'], strict=True); model.eval(); del saved
    before = tensor_digest(model.state_dict()); cache = {}; memberships = {}; feature_sha = {}; null_checked = False
    for split, subsets in [('train', cfg['src_train_split']), ('validation', cfg['src_val_split'])]:
        ds = dataset(cfg, manifest, subsets, False); memberships[split] = []; cache[split] = []
        for i in range(len(ds)):
            raw = ds[i]; video_id = raw['metadata']['video_id']; memberships[split].append(video_id)
            video, _ = prepare_source_video(raw, device); p.assert_no_target_gt([video])
            with torch.no_grad():
                x, mask = model.preprocessing([video], mode='test'); x = model.dropout(x)
                feats, masks, _ = model.backbone_source(x, mask); fpn, masks = model.neck_source(feats, masks)
                points = model.point_generator(fpn); offsets, _ = model.reg_head(fpn, masks); logits, _ = model.cls_head(fpn, masks)
                prediction = model.inference([video], points, [m.squeeze(1) for m in masks],
                    [v.permute(0,2,1) for v in logits], [v.permute(0,2,1) for v in offsets], None, None)[0]
                if not null_checked:
                    full = model(video_list_src=[video], video_list_tgt=[video], mode='test')[0][0]
                    for key in ['segments','scores','labels']: torch.testing.assert_close(prediction[key], full[key], rtol=1e-6, atol=1e-7)
                    null_checked = True
                seconds = prediction['segments'].to(device); labels = prediction['labels'].to(device); scores = prediction['scores'].to(device)
                meta = video['metadata']; boxes = (seconds*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']
                quality = same_class_quality(boxes, labels, raw['segments'].to(device), raw['labels'].to(device))
                features = region_features(fpn[0][0], masks[0][0].reshape(-1), points[0], boxes,
                    scores, labels, seconds[:,1]-seconds[:,0], model.num_classes)
            path = entry/'features'/split/(video_id+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, features=features.cpu().numpy(), quality=quality.cpu().numpy(),
                segments=seconds.cpu().numpy(), labels=labels.cpu().numpy(), scores=scores.cpu().numpy())
            cache[split].append(dict(video_id=video_id, path=str(path), sha256=p.sha(path), predictions=len(scores)))
            feature = Path(cfg['dataset']['feat_folder'])/((cfg['dataset']['file_prefix'] or '')+video_id+cfg['dataset']['file_ext'])
            feature_sha[str(feature)] = p.sha(feature)
            print('SOURCE_FEATURES', args.child, split, i+1, len(ds), len(scores), flush=True)
    if set(memberships['train']) & set(memberships['validation']): raise ValueError('源拟合/核验视频交叉')
    if before != tensor_digest(model.state_dict()): raise ValueError('源诊断修改了冻结模型')
    del model; gc.collect(); torch.cuda.empty_cache()
    readout, mean, std, history, fit_proof = fit_readout([v['path'] for v in cache['train']], device)
    # 这是源诊断读出参数，非student正式checkpoint，也不能用于恢复正式训练。
    torch.save(dict(specification=SPEC, state_dict=readout.cpu().state_dict(), mean=mean, std=std,
        diagnostic_only=True, student_checkpoint=False), entry/'source_readout.pt')
    readout.to(device); columns = {name:{k:[] for k in ['video-id','t-start','t-end','label','score']} for name in ['original','readout']}
    for record in cache['validation']:
        segments, labels, scores, rescored = score_predictions(readout, mean, std, record['path'], device)
        for name, confidence in [('original', scores), ('readout', rescored)]:
            target = columns[name]; target['video-id'].extend([record['video_id']]*len(scores))
            for key, value in [('t-start',segments[:,0]),('t-end',segments[:,1]),('label',labels),('score',confidence)]:target[key].extend(value.tolist())
    results = {}
    for name, values in columns.items():
        evaluator = ANETdetection(str(manifest), cfg['src_val_split'], num_workers=2, action_subsampling='10_majority_classes')
        ap, average, *_ = evaluator.evaluate({k:np.asarray(v) for k,v in values.items()}, verbose=False)
        results[name] = {'map_0.1_to_0.5':float(average), 'map_by_tiou':[float(v) for v in ap], 'predictions':len(values['score'])}
    old = p.read(reference/'evaluation/source_val.json')
    if abs(results['original']['map_0.1_to_0.5']-old['map_0.1_to_0.5']) > 1e-5:
        raise ValueError('源完整预测未复现原正式source mAP')
    p.save(entry/'membership.json', memberships); p.save(entry/'feature_index.json', cache); p.save(entry/'fit_history.json', history)
    result = dict(status='completed_source_only_validation', specification=SPEC, code=identity, shift=args.child,
        target_videos_read=0, target_gt_used=False, frozen_reference=str(reference),
        reference_complete_sha256=p.sha(reference/'complete.json'), checkpoint_sha256=p.sha(reference/'best.pt'),
        source_manifest_sha256=p.sha(manifest), original_model_unchanged=True, full_backend_null_check=null_checked,
        fit_proof=fit_proof,
        unchanged_segments_labels_nms_and_prediction_count=True, results=results,
        delta_map_pp=100*(results['readout']['map_0.1_to_0.5']-results['original']['map_0.1_to_0.5']),
        feature_inputs=feature_sha, diagnostic_only=True, formal_checkpoint_written=False,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        gpu_guard=guard, gpu_guard_sha256=p.sha(guard), config_sha256=p.sha(reference/'config.json'),
        artifacts={str(entry/n):p.sha(entry/n) for n in ['source_readout.pt','membership.json','feature_index.json','fit_history.json']})
    p.save(entry/'result.json', result); print('SOURCE_READOUT_DONE', args.child, result['delta_map_pp'], flush=True)
