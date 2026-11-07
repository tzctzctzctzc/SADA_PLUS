"""官方完整检测器＋原SADA类别条件对抗＋未经修改的Prototype。

检测器源码固定在native_sources；仅转换数据接口，保留原分类/定位损失。
"""
from copy import deepcopy
import importlib
import inspect
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F

from libs.models.SADA.meta_archs import CustomSequentialModel, GradReverse
from libs.models.SADA.prototype_capacity import build_capacity

ROOT = Path(__file__).resolve().parent / 'native_sources'


def namespace(detector):
    name = 'generality_native_' + detector
    if name not in sys.modules:
        for module, path in [(name, ROOT/detector), (name+'.libs', ROOT/detector/'libs')]:
            package = types.ModuleType(module)
            package.__path__ = [str(path)]
            sys.modules[module] = package
        # 只载入官方NMS模块，避免原仓库训练入口的绝对libs导入发生冲突。
        package = types.ModuleType(name+'.libs.utils')
        package.__path__ = [str(ROOT/detector/'libs/utils')]
        sys.modules[package.__name__] = package
        package.batched_nms = importlib.import_module(package.__name__+'.nms').batched_nms
    return name


def native_configuration(detector, cfg):
    import yaml
    name = namespace(detector)
    defaults = importlib.import_module(name+'.libs.core.config').DEFAULTS
    result = deepcopy(defaults)
    epic = yaml.safe_load((ROOT/detector/'configs/epic_slowfast_verb.yaml').read_text())
    for key, value in epic.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    result['dataset'].update(input_dim=cfg['dataset']['input_dim'], num_classes=10,
                             max_seq_len=cfg['dataset']['max_seq_len'])
    # 同一原始特征、十类划分和统一推理预算；保留各自原生模型/损失。
    result['test_cfg'].update(cfg['test_cfg'])
    model = result['model']
    model.update(input_dim=result['dataset']['input_dim'], num_classes=10,
                 max_seq_len=result['dataset']['max_seq_len'],
                 train_cfg=result['train_cfg'], test_cfg=result['test_cfg'])
    cls = getattr(importlib.import_module(name+'.libs.modeling.meta_archs'),
                  'PtTransformer' if detector == 'actionformer' else 'TriDet')
    keys = set(inspect.signature(cls.__init__).parameters)-{'self'}
    return cls, {key:model[key] for key in keys}


def native_videos(videos):
    return [dict(v['metadata'], feats=v['feats'], segments=v['segments'], labels=v['labels'])
            for v in videos]


class DetectorWithAdaptation(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = deepcopy(cfg)
        self.detector = cfg['generality']['detector']
        self.condition = cfg['generality']['condition']
        cls, options = native_configuration(self.detector, cfg)
        self.native = cls(**options).to(device)
        self.native_options = options
        self.num_classes = self.native.num_classes
        self.fpn_strides, self.reg_range = self.native.fpn_strides, self.native.reg_range
        channels, levels = options['fpn_dim'], len(self.fpn_strides)
        # 所有实验臂具有相同的检测器/D初始化；新增部件不挤占主随机流。
        devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(11072026)
            model = cfg['model']
            def disc(width):
                result = CustomSequentialModel(width, 1, model.get('num_conditional_adv_layers', 1),
                    activation=model.get('conditional_adv_activation','relu'),
                    drop_rate=model.get('conditional_adv_drop_rate',.5),
                    dim_proj_adv=model.get('dim_proj_adv',256))
                nn.init.constant_(result.final_layer.bias,model.get('conditional_adv_bias',.5))
                return result
            self.disc_dom_adv = nn.ModuleList([disc(channels) for _ in range(levels)])
            self.disc_dom_conditional_adv = nn.ModuleList([disc(2*channels) for _ in range(levels)])
            self.learnable_embeddings = nn.Parameter(torch.randn(11,channels))
        self.rq3_capacity = (build_capacity(cfg['prototype_capacity'], channels, 10,
            self.fpn_strides, self.reg_range, device) if cfg.get('prototype_capacity') else None)
        self.loss_normalizer_pseudo_labels = 1.
        self.cls_thres_prob = torch.zeros(10,device=device)
        self.cls_thres_ti = torch.ones(10,device=device)
        self.ema_src_centroids = [torch.zeros(10,channels,device=device) for _ in range(levels)]
        self.ema_tgt_centroids = [torch.zeros(10,channels,device=device) for _ in range(levels)]
        self.to(device)

    @property
    def device(self):
        return next(self.native.parameters()).device

    @property
    def loss_normalizer(self):
        return self.native.loss_normalizer

    @loss_normalizer.setter
    def loss_normalizer(self, value):
        self.native.loss_normalizer = value

    @property
    def train_label_smoothing(self):
        return self.native.train_label_smoothing

    @train_label_smoothing.setter
    def train_label_smoothing(self, value):
        self.native.train_label_smoothing = value

    def dense(self, videos, losses=False):
        n = self.native
        videos = native_videos(videos)
        inputs, masks = n.preprocessing(videos)
        features, masks = n.backbone(inputs, masks)
        features, masks = n.neck(features, masks)
        points = n.point_generator(features)
        logits = n.cls_head(features, masks)
        if not losses:
            return features, masks, points, logits
        offsets = tuple(x.permute(0,2,1) for x in n.reg_head(features,masks))
        classes, bounds = n.label_points(points,[v['segments'] for v in videos],[v['labels'] for v in videos])
        args = [tuple(x[:,0] for x in masks), tuple(x.permute(0,2,1) for x in logits),
                offsets, classes, bounds]
        if self.detector == 'tridet':
            args.extend([n.start_head(features,masks), n.end_head(features,masks)])
        return n.losses(*args), features, masks, points, logits, classes

    def domain_loss(self, source, target, labels, epoch):
        sf, sm, _, _ = source
        tf, tm, _, tl = target
        cfg = self.cfg['model']
        output = sf[0].sum()*0
        sizes = [x.shape[-1] for x in sf]
        gt = torch.stack(labels).split(sizes,dim=1)
        active_levels = cfg.get('semantic_alignment_levels', list(range(len(sf))))
        for level, (s,t,ms,mt,z,y) in enumerate(zip(sf,tf,sm,tm,tl,gt)):
            s = s.permute(0,2,1)[ms[:,0]]
            t = t.permute(0,2,1)[mt[:,0]]
            y = y[ms[:,0]]
            prob = z.permute(0,2,1).detach().sigmoid()[mt[:,0]]
            confidence, index = prob.max(-1)
            pseudo = F.one_hot(index,10).to(prob.dtype)*(confidence>=cfg.get('pl_threshold',.7))[:,None]
            if cfg.get('use_source_pseudo_labels',False):
                # 本批正式参考配置不启用该选项，不能悄悄改源分组协议。
                raise ValueError('source pseudo-label routing needs explicit paired implementation')
            totals, counters = [output*0,output*0], [0,0]
            for category in range(11 if cfg.get('align_bkg',False) else 10):
                masks = [(y.sum(-1)==0),(pseudo.sum(-1)==0)] if category==10 else [y[:,category]==1,pseudo[:,category]==1]
                weight = cfg.get('weight_no_class_adv',1.) if category==10 else 1.
                for domain,(features,mask) in enumerate(zip([s,t],masks)):
                    # 与SADA相同的类/域归一化，空组不计数、不调用无意义Dropout。
                    if not mask.any():
                        continue
                    values=features[mask]
                    values=torch.cat([values,self.learnable_embeddings[category].expand(len(values),-1)],-1)
                    disc=self.disc_dom_conditional_adv[level if cfg.get('use_level_disc',False) else 0]
                    prediction=disc(GradReverse.apply(values,cfg.get('beta',1.))).squeeze(-1)
                    totals[domain]=totals[domain]+weight*F.binary_cross_entropy_with_logits(
                        prediction,torch.full_like(prediction,float(domain)))
                    counters[domain]+=1
            value=cfg.get('weight_source_cond_adv',1.)*totals[0]/max(counters[0],1)
            value=value+cfg.get('weight_target_cond_adv',1.)*totals[1]/max(counters[1],1)
            if level in active_levels:
                output=output+cfg.get('weight_level_'+str(level),1.)*value/len(active_levels)
        return output

    def forward(self, video_list_src, video_list_tgt, mode=None, domain='src', curr_epoch=0, **unused):
        if mode=='test':
            rows=self.native(native_videos(video_list_src if domain in ('src','source') else video_list_tgt))
            return (rows,rows,None,None,None,None,None)
        if mode=='train':
            assert all(v['segments'].numel()==0 and v['labels'].numel()==0 for v in video_list_tgt)
        losses,sf,sm,sp,sl,labels=self.dense(video_list_src,losses=True)
        result=dict(cls_loss=losses['cls_loss'],reg_loss=losses['reg_loss'],downstream_loss=losses['final_loss'])
        if mode=='train' and self.condition!='so':
            target=self.dense(video_list_tgt)
            if curr_epoch>=self.cfg['train_cfg']['warmup_adv']:
                result['loss_conditional_adv_rev']=self.domain_loss((sf,sm,sp,sl),target,labels,curr_epoch)
            if self.rq3_capacity is not None:
                aux=self.rq3_capacity(video_list_src,video_list_tgt,sf,target[0],curr_epoch)
                result['downstream_loss']=result['downstream_loss']+aux['loss']
                result.update({'rq3_'+k:v if torch.is_tensor(v) else sf[0].new_tensor(v)
                               for k,v in aux.items() if k!='loss'})
                result['rq3_loss']=aux['loss'].detach()
        return (result,None,None,None,None,None,None)


def install_runtime():
    from tools import baseline_engine as base
    base.make_meta_arch=lambda name,cfg,device:DetectorWithAdaptation(cfg,device)
    return base
