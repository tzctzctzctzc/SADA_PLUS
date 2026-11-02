"""SADA+Prototype冷启动：三态密集分类、ARSL一维联合质量与任务分离。"""
from copy import deepcopy
import hashlib
import json
import torch
from torch.nn import functional as F
from tools.dense_assignment_math import (map_time, feature_view, linear_sample, decoded,
    union_contains, class_thresholds, quality_focal, source_quality_targets, task_separation)

MODES = ('tri_state', 'quality_tsa')
SCHEMA = 'sada-prototype-dense-assignment-v1'
SPEC = dict(schema=SCHEMA, initialization='original_seed123_cold_not_mature', seed=123,
    batch=2, source_Prototype_adversarial='unchanged_per_shift', target_GT=False,
    modes=list(MODES), warmup_epochs=10, target_onset=11, ramp_epochs=5,
    target_weight=1., target_reg_weight=1., source_quality_weight=.5, target_quality_weight=.5,
    tri_low=.1, tri_adaptive_high_range=[.25,.35], tri_threshold_ema=.9,
    tri_pseudo_nms=.6, tri_target_regression=False,
    tsa_low=.1, tsa_high='mean_plus_population_std_of_joint_above_.1_cap_.4',
    tsa_empty_positive='no_forced_top10', tsa_mining_iou=.6, tsa_max_level_difference=1,
    tsa_conflicting_instances='reject_if_matching_positive_seeds_iou_below_.6',
    background='raw_class_below_.1_and_outside_all_possible_action_intervals',
    ambiguous_quality='high_raw_class_low_quality_remains_soft_not_background',
    target_classification='all_FPN_valid_points_no_40cache_or_top32_cap',
    tri_foreground='original_center_range_shortest_instance_assignment_of_high_pseudo_boxes',
    strong_view=dict(feature_crop=.8, temporal_lengths=[.9,1.,1.1],
                     map='y=(x-lo+.5)*actual_ratio-.5'),
    selected_checkpoint='original_source_cls_plus_reg_best_EMA',
    secondary_selection='all_saved_every5_source_AP_max_earliest_tie_no_target_selection',
    source_classification='tri_original; quality_tsa_joint_IoU_soft_target_replaces_onehot_smoothing',
    max_epochs=50, validation_every=5, patience=15,
    inference='tri_original_score; quality_tsa_joint_probability; original_NMS_voting_once',
    efficiency='inherited_exact_PL_rng_replay; reused_original_source_outputs; chunked_all_candidates',
    scope='exploratory_formal_UDA_no_oracle_weights_or_labels')


def enable_controls():
    from tools import baseline_engine as base
    if getattr(base, 'dense_controls_enabled', False): return
    previous = base.controls
    def controls(model):
        result = previous(model)
        if hasattr(model, 'dense_state'): result['dense_state'] = deepcopy(model.dense_state)
        return result
    base.controls = controls; base.dense_controls_enabled = True


def wire_model(model, mode):
    """不新增参数、不改offset输出；直接取已有actionness卷积的ReLU前logit。"""
    state = dict(quality=[], cls=[], reg=[], capture=False, inference_calls=0)
    model.reg_head.actionness_head.register_forward_hook(lambda m,i,o:state['quality'].append(o[0]))
    for module, name in [(model.cls_head,'cls'), (model.reg_head,'reg')]:
        module.register_forward_hook(lambda m,i,o,name=name:state[name].append((i,o)) if state['capture'] else None)
    native_inference = model.inference
    def inference(videos, points, masks, logits, offsets, left, right):
        if mode == 'quality_tsa' and getattr(model, 'dense_enabled', True):
            n = len(logits); count = state['inference_calls']; state['inference_calls'] += 1
            # 等价eval缓存复用同一输入时，quality只有一份，两个domain共享它。
            start = min(count*n, len(state['quality'])-n)
            quality = state['quality'][start:start+n]
            logits = [torch.logit((c.sigmoid()*q.permute(0,2,1).sigmoid()).clamp(1e-7,1-1e-7))
                      for c,q in zip(logits,quality)]
        return native_inference(videos,points,masks,logits,offsets,left,right)
    model.inference = inference
    original = model.forward
    def forward(*args, **kwargs):
        state['quality'].clear(); state['inference_calls'] = 0
        if kwargs.get('mode') == 'train' and state['capture']:
            return original(*args, **kwargs)
        try: return original(*args, **kwargs)
        finally: state['quality'].clear()
    model.forward = forward
    model.dense_enabled = True
    model.dense_state = dict(step=0, digest='0'*64, cls_normalizer=250.,
        reg_normalizer=250., class_counts=[0.]*10, class_scores=[0.]*10)
    return state


def native_predictions(model, videos, state, training=True):
    state['quality'].clear()
    x, masks = model.preprocessing(videos, mode='train' if training else 'validation')
    x = model.dropout(x)
    features, masks, _ = model.backbone_source(x,masks)
    features, masks = model.neck_source(features,masks)
    cls = model.cls_head(features,masks)[0]; reg = model.reg_head(features,masks)[0]
    quality = tuple(state['quality']); state['quality'].clear()
    assert len(quality) == len(cls)
    return dict(cls=tuple(c.permute(0,2,1) for c in cls),
                reg=tuple(r.permute(0,2,1) for r in reg),
                quality=tuple(q.permute(0,2,1) for q in quality),
                masks=tuple(m[:,0].bool() for m in masks), points=model.point_generator(features))


def flatten_source(state, model):
    args, cl = state['cls'][0]; _, re = state['reg'][0]
    n = len(cl[0]); masks = args[1]
    points = model.point_generator(args[0])
    return dict(cls=torch.cat([x.permute(0,2,1) for x in cl[0]],1),
        reg=torch.cat([x.permute(0,2,1) for x in re[0]],1),
        quality=torch.cat([q.permute(0,2,1) for q in state['quality'][:n]],1),
        masks=torch.cat([m[:,0].bool() for m in masks],1),points=points)


def source_joint_loss(model, source, videos):
    points = source['points']
    classes, offsets = model.label_points(points,[v['segments'] for v in videos], [v['labels'] for v in videos])
    classes, offsets = torch.stack(classes), torch.stack(offsets)
    logits=source['cls'].flatten(0,1); raw_quality=source['quality'].flatten()
    soft, iou, positive = source_quality_targets(classes.flatten(0,1),source['reg'].flatten(0,1),
                                                offsets.flatten(0,1),source['masks'].flatten())
    norm = model.loss_normalizer
    classification = quality_focal(logits.sigmoid()*raw_quality.sigmoid()[:,None], soft,
                                   source['masks'].flatten(), norm)
    quality = F.binary_cross_entropy_with_logits(raw_quality[positive], iou[positive], reduction='sum')/norm
    return classification, quality


def high_pseudo_assignment(model, points, valid, boxes, labels, scores, high, view):
    """仅前景伪框执行同类NMS，全部候选无top32截断；中分区域另行保护。"""
    import nms_1d_cpu
    selected = []
    for c in range(10):
        ids = (high & (labels==c)).nonzero().flatten()
        if not len(ids): continue
        kept = nms_1d_cpu.nms(boxes[ids].contiguous().cpu(),scores[ids].contiguous().cpu(),
                               iou_threshold=SPEC['tri_pseudo_nms'])
        selected.append(ids[kept.to(ids.device)])
    ids = torch.cat(selected) if selected else labels.new_empty(0)
    mapped = map_time(boxes[ids],view).clamp(0,float(view['length']))
    usable = mapped[:,1]-mapped[:,0]>.05
    classes,_ = model.label_points(points,[mapped[usable]],[labels[ids][usable]])
    return classes[0], int(usable.sum())


def aligned_teacher(teacher, student, vi, view):
    probabilities=[]; qualities=[]; boxes=[]; valid=[]; weak_coordinates=[]; levels=[]
    native_boxes=[]; native_prob=[]; native_mask=[]
    for level,(tp,tr,tq,tm,wp,sp,sm) in enumerate(zip(teacher['cls'],teacher['reg'],teacher['quality'],
            teacher['masks'],teacher['points'],student['points'],student['masks'])):
        stride=float(wp[0,3]); coords=map_time(sp[:,0],view,inverse=True)
        raw=tp[vi].sigmoid(); qb=tq[vi].sigmoid(); decoded_boxes=decoded(wp,tr[vi])
        p,a=linear_sample(raw,tm[vi],coords,stride)
        q,b=linear_sample(qb,tm[vi],coords,stride)
        box,c=linear_sample(decoded_boxes,tm[vi],coords,stride)
        probabilities.append(p);qualities.append(q[:,0]);boxes.append(map_time(box,view))
        valid.append(a&b&c&sm[vi]);weak_coordinates.append(coords)
        levels.append(torch.full_like(sp[:,0],level,dtype=torch.long))
        native_boxes.append(decoded_boxes);native_prob.append(raw);native_mask.append(tm[vi])
    return dict(prob=torch.cat(probabilities),quality=torch.cat(qualities),boxes=torch.cat(boxes),
        valid=torch.cat(valid),weak_points=torch.cat(weak_coordinates),levels=torch.cat(levels),
        native_boxes=torch.cat(native_boxes),native_prob=torch.cat(native_prob),native_mask=torch.cat(native_mask))


def target_losses(model, teacher, student, videos, views, mode, update=True):
    from libs.models.SADA.losses_aux import sigmoid_focal_loss, ctr_diou_loss_1d
    zero = student['cls'][0].sum()*0
    cls_sum=zero; reg_sum=zero; quality_sum=zero
    observations=[]; prepared=[]; cls_mass=0.; reg_mass=0.
    counts=torch.tensor(model.dense_state['class_counts'],device=zero.device)
    sums=torch.tensor(model.dense_state['class_scores'],device=zero.device)
    native_probs=torch.cat([c.sigmoid().flatten(0,1) for c in teacher['cls']])
    native_valid=torch.cat([m.flatten() for m in teacher['masks']])
    threshold,new_stats=class_thresholds(native_probs,native_valid,(counts,sums))
    if update and mode=='tri_state':
        model.dense_state['class_counts']=new_stats[0].cpu().tolist()
        model.dense_state['class_scores']=new_stats[1].cpu().tolist()
    points=torch.cat(student['points'])
    for vi,(video,view) in enumerate(zip(videos,views)):
        with torch.no_grad():
            a=aligned_teacher(teacher,student,vi,view)
            native_score,native_label=a['native_prob'].max(-1)
            legal=(a['native_boxes'][:,1]-a['native_boxes'][:,0]>.05)&torch.isfinite(a['native_boxes']).all(-1)
            possible=a['native_mask'] & (native_score>=.1) & legal
            protected=union_contains(a['weak_points'],a['native_boxes'][possible])
            if mode=='tri_state':
                high=possible & (native_score>=threshold[native_label])
                classes,nboxes=high_pseudo_assignment(model,student['points'],a['valid'],a['native_boxes'],
                                                        native_label,native_score,high,view)
                positive=a['valid']&(classes.sum(-1)>0)
                negative=a['valid']&(a['prob'].max(-1).values<.1)&~protected&~positive
                item=dict(classes=classes,active=positive|negative,positive=positive,negative=negative,
                    ignored=a['valid']&~(positive|negative),soft=torch.zeros_like(positive),
                    regression=torch.zeros_like(positive),mined=torch.zeros_like(positive),
                    conflict=torch.zeros_like(positive),threshold=threshold.mean(),boxes=a['boxes'],quality=a['quality'])
            else:
                item=task_separation(a['prob'],a['quality'],a['boxes'],points,a['levels'],a['valid'],protected,view['length'])
                nboxes=0
            item.update(alignment=a)
        cls_mass += float(item['positive'].sum()) if mode=='tri_state' else float(item['classes'][item['positive']].max(-1).values.sum())
        reg_mass += float(item['classes'][item['regression']].max(-1).values.sum()) if item['regression'].any() else 0.
        prepared.append(item)
        observations.append(dict(video=video['metadata']['video_id'],crop_start=video['metadata'].get('crop_start',0),
            weak_length=video['feats'].shape[-1],view=view,valid=int(a['valid'].sum()),
            positive=int(item['positive'].sum()),negative=int(item['negative'].sum()),
            soft=int(item['soft'].sum()),ignored=int(item['ignored'].sum()),
            regression=int(item['regression'].sum()),mined=int(item['mined'].sum()),
            instance_conflict=int(item['conflict'].sum()),pseudo_boxes=nboxes,threshold=float(item['threshold'])))
    state=model.dense_state
    cls_norm=.9*state['cls_normalizer']+.1*max(cls_mass,1.)
    reg_norm=.9*state['reg_normalizer']+.1*max(reg_mass,1.)
    if update:state.update(cls_normalizer=cls_norm,reg_normalizer=reg_norm)
    for vi,item in enumerate(prepared):
        logits=torch.cat([x[vi] for x in student['cls']]); offsets=torch.cat([x[vi] for x in student['reg']])
        q=torch.cat([x[vi] for x in student['quality']])[:,0]
        if mode=='tri_state':
            loss=sigmoid_focal_loss(logits,item['classes'].detach(),alpha=.25,gamma=2.,reduction='none')
            cls_sum += (loss*item['active'][:,None]).sum()/cls_norm
        else:
            cls_sum += quality_focal(logits.sigmoid()*q.sigmoid()[:,None],item['classes'],item['active'],cls_norm)
            mask=item['regression']; selected=item['boxes'][mask]
            targets=torch.stack([points[mask,0]-selected[:,0],selected[:,1]-points[mask,0]],-1)/points[mask,3,None]
            weights=item['classes'][mask].max(-1).values
            reg_sum += (ctr_diou_loss_1d(offsets[mask],targets.detach(),reduction='none')*weights).sum()/reg_norm
            quality_sum += F.binary_cross_entropy_with_logits(q[mask],item['quality'][mask].detach(),reduction='sum')/max(1.,sum(x['regression'] for x in observations))
    stats={k:sum(x[k] for x in observations) for k in ['valid','positive','negative','soft','ignored','regression','mined','instance_conflict']}
    stats.update(cls=float(cls_sum.detach()),reg=float(reg_sum.detach()),quality=float(quality_sum.detach()))
    return cls_sum+reg_sum+SPEC['target_quality_weight']*quality_sum,stats,observations,prepared


def install(ctx,cfg):
    enable_controls();mode=cfg['dense_assignment']['mode']
    from tools.cold_oracle_fastpath import install as fast
    fast(ctx.model);fast(ctx.model_ema.module)
    student_state=wire_model(ctx.model,mode);teacher_state=wire_model(ctx.model_ema.module,mode)
    ctx.dense_enabled=True;ctx.dense_ledger=None;ctx.dense_gradient_epoch=None;ctx.dense_last={}
    ctx.dense_original_forward=ctx.model.forward
    def forward(*args,**kwargs):
        if kwargs.get('mode')!='train' or not ctx.dense_enabled:
            return ctx.dense_original_forward(*args,**kwargs)
        videos=kwargs['video_list_tgt'];source_videos=kwargs['video_list_src']
        assert all(not v['segments'].numel() and not v['labels'].numel() for v in videos)
        student_state['cls'].clear();student_state['reg'].clear();student_state['capture']=True
        try:
            output=ctx.dense_original_forward(*args,**kwargs)
            source=flatten_source(student_state,ctx.model)
        finally:student_state['capture']=False
        losses=output[0];epoch=kwargs['curr_epoch'];state=ctx.model.dense_state
        source_quality=losses['cls_loss']*0
        if mode=='quality_tsa':
            src_cls,source_quality=source_joint_loss(ctx.model,source,source_videos)
            losses['downstream_loss']=losses['downstream_loss']-losses['cls_loss']+src_cls+.5*source_quality
            losses['dense_source_joint_cls']=src_cls.detach()
        ramp=max(0.,min(1.,(epoch-SPEC['warmup_epochs'])/SPEC['ramp_epochs']))
        stats=dict(valid=0,positive=0,negative=0,soft=0,ignored=0,regression=0,mined=0,instance_conflict=0,
                   cls=0.,reg=0.,quality=0.,ramp=ramp,source_quality=float(source_quality.detach()))
        observations=[]
        if ramp:
            ctx.model_ema.module.eval()
            with torch.no_grad():teacher=native_predictions(ctx.model_ema.module,videos,teacher_state)
            strong=[];views=[]
            for i,video in enumerate(videos):
                v,w=feature_view(video,state['step'],i);strong.append(v);views.append(w)
            student=native_predictions(ctx.model,strong,student_state)
            auxiliary,active_stats,observations,_=target_losses(ctx.model,teacher,student,videos,views,mode)
            stats.update(active_stats);losses['downstream_loss']=losses['downstream_loss']+ramp*auxiliary
            ready=stats['positive']>0 and (mode=='tri_state' or stats['regression']>0)
            if ready and ctx.dense_gradient_epoch!=epoch:
                for key,params in [('target_sgp_gradient',ctx.model.backbone_source.parameters()),
                    ('target_cls_gradient',ctx.model.cls_head.parameters()),
                    ('target_offset_gradient',ctx.model.reg_head.offset_head.parameters())]:
                    grads=torch.autograd.grad(auxiliary,tuple(params),retain_graph=True,allow_unused=True)
                    stats[key]=sum(float(g.detach().square().sum()) for g in grads if g is not None)**.5
                ctx.dense_gradient_epoch=epoch
        state['step']+=1
        event=dict(step=state['step'],epoch=epoch,mode=mode,observations=observations,stats=stats,
                   source_videos=[v['metadata']['video_id'] for v in source_videos])
        state['digest']=hashlib.sha256((state['digest']+json.dumps(event,sort_keys=True)).encode()).hexdigest()
        if ctx.dense_ledger:
            with open(ctx.dense_ledger,'a') as stream:stream.write(json.dumps(dict(event,digest=state['digest']))+'\n')
        ctx.dense_last=stats
        for key,value in stats.items():losses['dense_'+key]=losses['cls_loss'].new_tensor(value).detach()
        student_state['quality'].clear();student_state['cls'].clear();student_state['reg'].clear()
        return output
    ctx.model.forward=forward
    ctx.dense_states=(student_state,teacher_state)
