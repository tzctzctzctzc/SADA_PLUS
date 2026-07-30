import math
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses_aux import ctr_diou_loss_1d, sigmoid_focal_loss, ctr_giou_loss_1d
from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .stage1_utils import (
    build_dense_stage2_regression_mask,
    build_oracle_target_regression_mask,
    build_oracle_routing,
    build_routing_masks,
    build_same_selection_oracle_mask,
    build_stage2_regression_mask,
    context_view_on_cpu,
    erode_mask,
    interpolate_probabilities,
    select_same_selection_regression_targets,
    shift_features,
    source_action_eligible,
)
from .stage2_utils import (
    aligned_boundary_iou_from_offsets,
    deterministic_gaussian_perturbation,
    match_post_nms_to_pre_nms,
    npl_boundary_ambiguity,
    proposal_loss_terms,
    weighted_anchor_mse,
)
from .stage2_alqa import (
    ALQAQualityHead,
    alqa_joint_score,
    ensemble_quality_maps,
    regression_hidden_and_offsets,
    select_same_budget_alqa,
)
from libs.utils import batched_nms
import einops
import copy


# -------------- Auxiliary classes -----------------
# Function that applies the Gradient Reverse (does nothing on the forward pass, and changes the sign in the backward pass)
from torch.autograd import Function
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None

class CustomSequentialModel(nn.Module):
    def __init__(self, adv_input_dim, output_dim, L, activation='relu', drop_rate=0.5, dim_proj_adv = 256):
        super(CustomSequentialModel, self).__init__()

        # Initial layers
        self.initial_layers = nn.Sequential(
            nn.Linear(adv_input_dim, dim_proj_adv),
            self.get_activation(activation),
            nn.Dropout(drop_rate)
        )

        # Middle layers (repeated L times)
        middle_layers = []
        for _ in range(L):
            middle_layers.extend([
                nn.Linear(dim_proj_adv, dim_proj_adv),
                self.get_activation(activation),
                nn.Dropout(drop_rate)
            ])
        self.middle_layers = nn.Sequential(*middle_layers)

        # Final layer
        self.final_layer = nn.Linear(dim_proj_adv, output_dim)

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'selu':
            return nn.SELU(inplace=True)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.middle_layers(x)
        x = self.final_layer(x)
        return x


class ClsHead(nn.Module):
    """
    1D Conv heads for classification
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            num_classes,
            prior_prob=0.01,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            empty_cls=[],
            detach_feat=False
    ):
        super().__init__()
        self.act = act_layer()
        self.detach_feat = detach_feat

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size // 2
        )


        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        out_feats = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            if self.detach_feat:
                cur_out = cur_feat.detach()
            else:
                cur_out = cur_feat
                
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            
            out_feats += (cur_out,)
            out_logits += (cur_logits,)
            
        # fpn_masks remains the same
        return out_logits, out_feats
    
class RegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            num_bins=16
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # In our case, num_bins = 0
        self.offset_head = MaskedConv1D(
            feat_dim, 2 * (num_bins + 1), kernel_size,
            stride=1, padding=kernel_size // 2
        )
        
        self.actionness_head = MaskedConv1D(
            feat_dim, 1, kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        out_actionness = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)),)
            
            cur_actionness, _ = self.actionness_head(cur_out, cur_mask)
            out_actionness += (F.relu(self.scale[l](cur_actionness)),)

        # fpn_masks remains the same
        return out_offsets, out_actionness


@register_meta_arch("sada")
class SADA(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(self, cfg, device):
        super().__init__()
        
        self.cfg = cfg
        self._device = device
        
        scale_factor = cfg['model']['scale_factor']
        backbone_arch = cfg['model']['backbone_arch']
        iou_weight_power = cfg['model']['iou_weight_power']
        num_classes = cfg['dataset']['num_classes']
        n_sgp_win_size = cfg['model']['n_sgp_win_size']
        
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor ** i for i in range(backbone_arch[-1] + 1)]

        self.input_noise = cfg['model']['input_noise']

        self.reg_range = cfg['model']['regression_range']
        self.n_levels = len(self.fpn_strides)
        self.warmup_adv = cfg['train_cfg'].get('warmup_adv', 0)
        self.oracle_routing_mode = cfg.get('diagnostic', {}).get(
            'oracle_routing', 'off'
        )
        if self.oracle_routing_mode not in ('off', 'o1', 'o2', 'o3'):
            raise ValueError(
                'unknown oracle routing mode: {}'.format(
                    self.oracle_routing_mode
                )
            )
        self.oracle_target_regression_mode = cfg.get('diagnostic', {}).get(
            'oracle_target_regression', 'off'
        )
        if self.oracle_target_regression_mode not in (
                'off', 'selected', 'full', 'clean_pseudo', 'same_selection'):
            raise ValueError(
                'unknown Oracle target regression mode: {}'.format(
                    self.oracle_target_regression_mode
                )
            )
        self.oracle_target_regression_weight = float(
            cfg.get('diagnostic', {}).get('oracle_target_regression_weight', 1.0)
        )
        if self.oracle_target_regression_weight < 0:
            raise ValueError('Oracle target regression weight must be non-negative')
        self.oracle_target_regression_levels = cfg.get(
            'diagnostic', {}
        ).get('target_regression_levels', [0, 1, 2])
        if (
            not self.oracle_target_regression_levels
            or len(set(self.oracle_target_regression_levels))
            != len(self.oracle_target_regression_levels)
            or any(level < 0 or level >= self.n_levels
                   for level in self.oracle_target_regression_levels)
        ):
            raise ValueError('Oracle target regression levels are invalid')
        stage2_cfg = cfg.get('stage2', {})
        self.stage2_enabled = stage2_cfg.get('enabled', False)
        self.stage2_start_epoch = int(stage2_cfg.get('start_epoch', 5))
        self.stage2_weight = float(stage2_cfg.get('weight', 1.0))
        self.stage2_levels = stage2_cfg.get('levels', [0, 1, 2])
        self.stage2_routing = stage2_cfg.get('routing', 'confidence')
        self.stage2_loss_weighting = stage2_cfg.get(
            'loss_weighting', 'anchor_equal'
        )
        self.stage2_stability_weighting = bool(
            stage2_cfg.get('stability_weighting', False)
        )
        self.stage2_stability_noise_std = float(
            stage2_cfg.get('stability_noise_std', 0.01)
        )
        self.stage2_stability_seed = int(
            stage2_cfg.get('stability_seed', 20260730)
        )
        self.stage2_alqa_head_seeds = stage2_cfg.get(
            'alqa_head_seeds', [123, 456, 789]
        )
        self.stage2_alqa_epsilon = float(
            stage2_cfg.get('alqa_epsilon', 1e-6)
        )
        self.stage2_alpha = cfg['model'].get('pl_threshold', 0.7)
        if self.stage2_start_epoch < 1 or self.stage2_weight < 0:
            raise ValueError('Stage 2 start epoch and weight are invalid')
        if (
            self.stage2_stability_weighting
            and (
                self.stage2_routing != 'static_full_dense'
                or self.stage2_stability_noise_std <= 0
            )
        ):
            raise ValueError(
                'Stage 2 stability weighting requires static-full routing '
                'and a positive noise standard deviation'
            )
        if (
            not self.stage2_levels
            or len(set(self.stage2_levels)) != len(self.stage2_levels)
            or any(level < 0 or level >= self.n_levels for level in self.stage2_levels)
        ):
            raise ValueError('Stage 2 levels are invalid')
        if self.stage2_routing not in (
                'confidence', 'alqa_same_budget',
                'static_dense', 'static_full_dense'):
            raise ValueError('unknown Stage 2 routing mode')
        self.stage2_uses_external_teacher = (
            self.stage2_routing
            not in ('static_dense', 'static_full_dense')
        )
        if self.stage2_loss_weighting not in (
                'anchor_equal', 'proposal_equal',
                'proposal_quality', 'proposal_joint'):
            raise ValueError('unknown Stage 2 loss weighting')
        self.stage2_requires_quality = (
            self.stage2_routing == 'alqa_same_budget'
            or self.stage2_loss_weighting
            in ('proposal_quality', 'proposal_joint')
        )
        if (
            self.stage2_requires_quality
            and (
                not self.stage2_enabled
                or not self.stage2_alqa_head_seeds
                or self.stage2_alqa_epsilon <= 0
            )
        ):
            raise ValueError('ALQA Stage 2 configuration is invalid')
        if self.stage2_enabled and self.oracle_target_regression_mode != 'off':
            raise ValueError('Stage 2 and Oracle target regression cannot both be enabled')
        if (
            ((self.stage2_enabled and self.stage2_uses_external_teacher)
             or self.oracle_target_regression_mode
             in ('clean_pseudo', 'same_selection'))
            and not cfg['train_cfg'].get('ema_active', False)
        ):
            raise ValueError('Stage 2 proposal routing requires the external EMA teacher')
        stage1_cfg = cfg.get('stage1', {})
        self.stage1_enabled = stage1_cfg.get('enabled', False)
        self.stage1_perturbation = stage1_cfg.get('perturbation', 'shift')
        self.stage1_seed = stage1_cfg.get('seed', cfg.get('init_rand_seed', 0) + 100003)
        self.stage1_train_rng = torch.Generator(device='cpu')
        self.stage1_calibration_rng = torch.Generator(device='cpu')
        self.stage1_dryrun_rng = torch.Generator(device='cpu')
        self.stage1_train_rng.manual_seed(self.stage1_seed)
        self.stage1_calibration_rng.manual_seed(
            stage1_cfg.get('calibration_seed', self.stage1_seed + 1)
        )
        self.stage1_dryrun_rng.manual_seed(
            stage1_cfg.get('dryrun_seed', self.stage1_seed + 2)
        )
        self.stage1_alpha = cfg['model'].get('pl_threshold', 0.7)
        self.stage1_beta = stage1_cfg.get('beta', None)
        self.stage1_tau_act = stage1_cfg.get('tau_act', None)
        self.stage1_tau_bg = stage1_cfg.get('tau_bg', None)
        self.stage1_calibrated = all(
            value is not None
            for value in (self.stage1_beta, self.stage1_tau_act, self.stage1_tau_bg)
        )
        if self.stage1_enabled:
            assert self.stage1_perturbation in ('shift', 'context')
            if (
                self.oracle_routing_mode != 'off'
                or self.oracle_target_regression_mode != 'off'
            ):
                raise ValueError('Stage 1 and Oracle diagnostics cannot both be enabled')
        
        
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        self.iou_weight_power = iou_weight_power
        
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_class_orig = num_classes
        self.num_classes = num_classes if not self.cfg['model'].get('use_ce_cls', False) else num_classes + 1   # This adds the no-class label, if needed
        
        if self.cfg['dataset'].get('action_subsampling', None) == '10_majority_classes':
            self.num_classes = 10
            self.num_class_orig = 10
        
        
        # check the feature pyramid and local attention window size
        self.max_seq_len = cfg['dataset']['max_seq_len']
        if isinstance(n_sgp_win_size, int):
            self.sgp_win_size = [n_sgp_win_size] * len(self.fpn_strides)
        else:
            assert len(n_sgp_win_size) == len(self.fpn_strides)
            self.sgp_win_size = n_sgp_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.sgp_win_size)):
            stride = s * w if w > 1 else s
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # Model config
        self.train_center_sample = cfg['model']['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = cfg['model']['center_sample_radius']
        self.train_reg_distance = cfg['model'].get('reg_distance', True)
        
        self.train_loss_reg_weight = cfg['model']['loss_weights']['weight_reg']
        self.train_loss_adv_weight = cfg['model']['loss_weights']['weight_adv']
        
        self.train_cls_prior_prob = cfg['model']['cls_prior_prob']
        self.train_dropout = cfg['model']['dropout']
        self.train_droppath = cfg['model']['droppath']
        self.train_label_smoothing = cfg['model']['label_smoothing']
        self.train_head_empty_cls = cfg['model']['head_empty_cls']
        self.train_init_loss_norm = cfg['model']['init_loss_norm']
        

        # test time config
        self.test_pre_nms_thresh = cfg['test_cfg']['pre_nms_thresh']
        self.test_pre_nms_topk = cfg['test_cfg']['pre_nms_topk']
        self.test_iou_threshold = cfg['test_cfg']['iou_threshold']
        self.test_min_score = cfg['test_cfg']['min_score']
        self.test_max_seg_num = cfg['test_cfg']['max_seg_num']
        self.test_nms_method = cfg['test_cfg']['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = cfg['test_cfg']['duration_thresh']
        self.test_multiclass_nms = cfg['test_cfg']['multiclass_nms']
        self.test_nms_sigma = cfg['test_cfg']['nms_sigma']
        self.test_voting_thresh = cfg['test_cfg']['voting_thresh']
        self.num_bins = cfg['model']['num_bins']
        self.use_trident_head = cfg['model']['use_trident_head']
        if self.oracle_target_regression_mode != 'off' and self.use_trident_head:
            raise ValueError(
                'Oracle target regression currently supports the direct '
                'regression head only'
            )
        if self.stage2_enabled and self.use_trident_head:
            raise ValueError(
                'Stage 2 currently supports the direct regression head only'
            )

        backbone_type = cfg['model']['backbone_type']
        input_dim = cfg['model']['input_dim']
        embd_dim = cfg['model']['embd_dim']
        sgp_mlp_dim = cfg['model']['sgp_mlp_dim']
        embd_kernel_size = cfg['model']['embd_kernel_size']
        embd_with_ln = cfg['model']['embd_with_ln']
        downsample_type = cfg['model']['downsample_type']
        use_abs_pe = cfg['model']['use_abs_pe']
        init_conv_vars = cfg['model']['init_conv_vars']
        k = cfg['model']['k']
        fpn_type = cfg['model']['fpn_type']
        fpn_dim = cfg['model']['fpn_dim']
        fpn_with_ln = cfg['model']['fpn_with_ln']
        max_buffer_len_factor = cfg['model']['max_buffer_len_factor']
        head_dim = cfg['model']['head_dim']
        head_kernel_size = cfg['model']['head_kernel_size']
        head_with_ln = cfg['model']['head_with_ln']
        head_num_layers = cfg['model']['head_num_layers']
        use_trident_head = cfg['model']['use_trident_head']
        boudary_kernel_size = cfg['model']['boudary_kernel_size']
        num_bins = cfg['model']['num_bins']
        
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ['SGP', 'conv']
        if self.cfg['model'].get('use_finetune_feature_extractor', False):
            input_dim_backbone = embd_dim
        else:
            input_dim_backbone = input_dim
        if backbone_type == 'SGP':
            self.backbone = make_backbone(
                'SGP',
                **{
                    'n_in': input_dim_backbone,
                    'n_embd': embd_dim,
                    'sgp_mlp_dim': sgp_mlp_dim,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': self.max_seq_len,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln': embd_with_ln,
                    'path_pdrop': self.train_droppath,
                    'downsample_type': downsample_type,
                    'sgp_win_size': self.sgp_win_size,
                    'use_abs_pe': use_abs_pe,
                    'k': k,
                    'init_conv_vars': init_conv_vars
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim_backbone,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln': embd_with_ln
                }
            )
            
        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels': [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel': fpn_dim,
                'scale_factor': scale_factor,
                'with_ln': fpn_with_ln
            }
        )
            
        # Clone the self.backbone for the source and target domain
        self.backbone_source = copy.deepcopy(self.backbone)
        self.neck_source = copy.deepcopy(self.neck)
        self.backbone_target = self.backbone_source
        self.neck_target = self.neck_source
        
        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len': self.max_seq_len * max_buffer_len_factor,
                'fpn_levels': len(self.fpn_strides),
                'scale_factor': scale_factor,
                'regression_range': self.reg_range,
                'strides': self.fpn_strides
            }
        )

        # Initialize the classification and regression heads for each of the two domains (so each of the modules is a tuple of two)
        self.cls_head = ClsHead(
                fpn_dim, head_dim, self.num_classes,
                kernel_size=head_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=head_with_ln,
                num_layers=head_num_layers,
                empty_cls=self.train_head_empty_cls)
        
        if use_trident_head:
            self.start_head = ClsHead(
                fpn_dim, head_dim, self.num_classes,
                kernel_size=boudary_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=head_with_ln,
                num_layers=head_num_layers,
                empty_cls=self.train_head_empty_cls,
                detach_feat=True
            )
            
            self.end_head = ClsHead(
                fpn_dim, head_dim, self.num_classes,
                kernel_size=boudary_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=head_with_ln,
                num_layers=head_num_layers,
                empty_cls=self.train_head_empty_cls,
                detach_feat=True
            )

            self.reg_head = RegHead(
                fpn_dim, head_dim, len(self.fpn_strides),
                kernel_size=head_kernel_size,
                num_layers=head_num_layers,
                with_ln=head_with_ln,
                num_bins=num_bins
            )
        else:
            self.reg_head = RegHead(
                    fpn_dim, head_dim, len(self.fpn_strides),
                    kernel_size=head_kernel_size,
                    num_layers=head_num_layers,
                    with_ln=head_with_ln,
                    num_bins=0
            )

        self.cls_head_target = self.cls_head
        self.reg_head_target = self.reg_head
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3407)
            self.alqa_quality_heads = nn.ModuleList([
                ALQAQualityHead(head_dim)
                for _ in self.stage2_alqa_head_seeds
            ]) if (
                self.stage2_requires_quality
            ) else nn.ModuleList()

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = self.train_init_loss_norm
        self.loss_normalizer_pseudo_labels = self.train_init_loss_norm
        self.loss_normalizer_momentum = 0.9
        
        self.dropout = nn.Dropout(self.train_dropout)
        
        # -------------
        # Domain adaptation modules
        self.criterion_domain = nn.BCEWithLogitsLoss().cuda()
        
        # Binary classification module for domain adaptation (frame level)
        adv_input_dim = input_dim if self.cfg['model'].get('use_inter_emb_backbone', False) else embd_dim
        
        # Create the conditional adv disc
        L = self.cfg['model'].get('num_conditional_adv_layers', 1)
        adv_act = self.cfg['model'].get('conditional_adv_activation', 'relu')
        adv_bias = self.cfg['model'].get('conditional_adv_bias', 0.5)
        drop_rate = self.cfg['model'].get('conditional_adv_drop_rate', 0.5)
        dim_proj_adv = self.cfg['model'].get('dim_proj_adv', 256)
        
        self.disc_dom_adv = nn.ModuleList([copy.deepcopy(CustomSequentialModel(adv_input_dim=adv_input_dim,
                                                                                           dim_proj_adv=dim_proj_adv,
                                                                                           output_dim=1,
                                                                                           L=L,
                                                                                           activation=adv_act,
                                                                                           drop_rate=drop_rate)) for _ in range(self.n_levels)])
        # Initialize the bias of the last layer to 0.5
        for disc in self.disc_dom_adv:
            torch.nn.init.constant_(disc.final_layer.bias, adv_bias)
            
        self.disc_dom_conditional_adv = nn.ModuleList([copy.deepcopy(CustomSequentialModel(adv_input_dim=2*adv_input_dim,
                                                                                           dim_proj_adv=dim_proj_adv,
                                                                                           output_dim=1,
                                                                                           L=L,
                                                                                           activation=adv_act,
                                                                                           drop_rate=drop_rate)) for _ in range(self.n_levels)])
        # Initialize the bias of the last layer to 0.5
        for disc in self.disc_dom_conditional_adv:
            torch.nn.init.constant_(disc.final_layer.bias, adv_bias)
        

        # Initialize the list of EMA centroids for each of the classes for both source and target domains
        self.ema_src_centroids = [torch.nn.Parameter(torch.zeros(self.num_class_orig, fpn_dim), requires_grad=False).to(self.device) for _ in range(6)]
        self.ema_tgt_centroids = [torch.nn.Parameter(torch.zeros(self.num_class_orig, fpn_dim), requires_grad=False).to(self.device) for _ in range(6)]
        
        
        self.momentum = self.cfg['model'].get('da_ema_momentum', 0.7) 
        self.semantic_align_loss = self.cfg['model'].get('semantic_align_loss', 'mse')
        if self.semantic_align_loss == 'mse':
            self.criterion_alignment = nn.MSELoss() 
        else:
            raise ValueError('Semantic alignment loss not implemented')
        
        
        # Initialize the tensors for the dynamic thresholding
        self.cls_thres_prob = torch.zeros(self.num_classes).to(self.device) # shape: C (not using the no-class)
        self.cls_thres_ti = torch.ones(self.num_classes).to(self.device) # shape: C # To avoid NaNs, initialize to 1
        
        self.cls_head_pl = self.cls_head
        if (
            self.stage2_enabled
            and self.stage2_routing == 'static_full_dense'
        ):
            self.backbone_pl = copy.deepcopy(self.backbone_source)
            self.neck_pl = copy.deepcopy(self.neck_source)
            self.reg_head_pl = copy.deepcopy(self.reg_head)
            for module in (
                self.backbone_pl, self.neck_pl, self.reg_head_pl
            ):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        elif self.stage2_enabled and self.stage2_routing == 'static_dense':
            self.backbone_pl = self.backbone_source
            self.neck_pl = self.neck_source
            self.reg_head_pl = copy.deepcopy(self.reg_head)
            self.reg_head_pl.eval()
            for parameter in self.reg_head_pl.parameters():
                parameter.requires_grad_(False)
        else:
            self.backbone_pl = self.backbone_source
            self.neck_pl = self.neck_source
            self.reg_head_pl = self.reg_head
        
        
        # Related to the classification loss
        self.cls_loss_type = self.cfg['model'].get('cls_loss_type', 'sigmoid_focal_loss')

        # Create a set of C learnable embeddings
        self.align_bkg = self.cfg['model'].get('align_bkg', False)
        n_classes = self.num_classes if not self.align_bkg else self.num_classes + 1
        
        self.learnable_embeddings = nn.Parameter(torch.randn(n_classes, embd_dim), requires_grad=True)
        
    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        #return list(set(p.device for p in self.parameters()))[0]
        return self._device
    

    # This function implements the binary domain classifier at frame level
    def domain_classifier_frame(self, feat, mask, level=0, curr_epoch=0, beta=1):
        # Apply gradient Reverse to ensure adversarial training
        if self.cfg['model']['use_rev_grad']:
            feat_fc_domain_frame = GradReverse.apply(feat, beta)
        else:
            feat_fc_domain_frame = feat  
        
        # Project the input feature embedding of the frame into a binary r.v
        if not self.cfg['model'].get('use_level_disc', False):
            # For MLP based
            pred_fc_domain_frame = self.disc_dom_adv[0](feat_fc_domain_frame) # Always use the same discriminator regardless of the level
        else:
            # For MLP based
            pred_fc_domain_frame = self.disc_dom_adv[level](feat_fc_domain_frame)
            
        return pred_fc_domain_frame


    def decode_offset(self, out_offsets, pred_start_neighbours, pred_end_neighbours, mode):
        # decode the offset value from the network output
        # If a normal regression head is used, the offsets is predicted directly in the out_offsets.
        # If the Trident-head is used, the predicted offset is calculated using the value from
        # center offset head (out_offsets), start boundary head (pred_left) and end boundary head (pred_right)

        if not self.use_trident_head:
            if mode != 'test':
                out_offsets = torch.cat(out_offsets, dim=1)
            return out_offsets
        else:
            # Make an adaption for train and validation, when training, the out_offsets is a list with feature outputs
            # from each FPN level. Each feature with shape [batchsize, T_level, (Num_bin+1)x2].
            # For validation, the out_offsets is a feature with shape [T_level, (Num_bin+1)x2]
            if mode != 'test':
                out_offsets = torch.cat(out_offsets, dim=1)
                out_offsets = out_offsets.view(out_offsets.shape[:2] + (2, -1))
                pred_start_neighbours = torch.cat(pred_start_neighbours, dim=1)
                pred_end_neighbours = torch.cat(pred_end_neighbours, dim=1)

                pred_left_dis = torch.softmax(pred_start_neighbours + out_offsets[:, :, :1, :], dim=-1)
                pred_right_dis = torch.softmax(pred_end_neighbours + out_offsets[:, :, 1:, :], dim=-1)

            else:
                out_offsets = out_offsets.view(out_offsets.shape[0], 2, -1)
                pred_left_dis = torch.softmax(pred_start_neighbours + out_offsets[None, :, 0, :], dim=-1)
                pred_right_dis = torch.softmax(pred_end_neighbours + out_offsets[None, :, 1, :], dim=-1)

            max_range_num = pred_left_dis.shape[-1]

            left_range_idx = torch.arange(max_range_num - 1, -1, -1, device=self.device,
                                          dtype=torch.float).unsqueeze(-1)
            right_range_idx = torch.arange(max_range_num, device=self.device,
                                           dtype=torch.float).unsqueeze(-1)

            pred_left_dis = pred_left_dis.masked_fill(torch.isnan(pred_right_dis), 0)
            pred_right_dis = pred_right_dis.masked_fill(torch.isnan(pred_right_dis), 0)

            # calculate the value of expectation for the offset:
            decoded_offset_left = torch.matmul(pred_left_dis, left_range_idx)
            decoded_offset_right = torch.matmul(pred_right_dis, right_range_idx)
            return torch.cat([decoded_offset_left, decoded_offset_right], dim=-1)


    def compute_dom_preds(self, fpn_feats, fpn_masks, curr_epoch, beta=1):
        # Apply the domain adaptation module for each of the levels
        pred_domain_all = []
        
        for l_idx, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            # For MLP based
            d_feat = einops.rearrange(cur_feat, 'b c t -> b t c')
            d_mask = cur_mask
            
            pred_fc_domain_frame = self.domain_classifier_frame(d_feat, mask=d_mask, level = l_idx, curr_epoch=curr_epoch, beta=beta)   # shape: 2B x T x 2 or 2B x T x 1 (for wgan)
            pred_domain_all.append(pred_fc_domain_frame)
        
        return pred_domain_all
    
    
    def compute_level_dom_preds(self, cur_feat, reverse, level, curr_epoch, beta=1):
        # Apply the domain adaptation module for a given level. Assume all the embeddings that should be masked have already been masked
        # cur_feat: shape N x F
        
        if reverse:
            feat_fc_domain_frame = GradReverse.apply(cur_feat, beta)
        else:
            feat_fc_domain_frame = cur_feat
            
        if not self.cfg['model'].get('use_level_disc', False):
            pred_fc_domain_frame = self.disc_dom_conditional_adv[0](feat_fc_domain_frame) # Always use the same discriminator regardless of the level
        else:
            pred_fc_domain_frame = self.disc_dom_conditional_adv[level](feat_fc_domain_frame)
                
        return pred_fc_domain_frame

    @torch.no_grad()
    def initialize_static_stage2_head(self):
        """Snapshot the localization head when dense Stage 2 becomes active."""
        if not (
            self.stage2_enabled and self.stage2_routing == 'static_dense'
        ):
            return
        self.reg_head_pl.load_state_dict(self.reg_head.state_dict())
        self.reg_head_pl.eval()
        for parameter in self.reg_head_pl.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def initialize_static_stage2_path(self):
        """Snapshot the complete target feature-to-localization path."""
        if not (
            self.stage2_enabled
            and self.stage2_routing == 'static_full_dense'
        ):
            return
        self.backbone_pl.load_state_dict(self.backbone_target.state_dict())
        self.neck_pl.load_state_dict(self.neck_target.state_dict())
        self.reg_head_pl.load_state_dict(self.reg_head_target.state_dict())
        for module in (
            self.backbone_pl, self.neck_pl, self.reg_head_pl
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def _stage2_perturbed_inputs(
            self, inputs, masks, video_list_tgt, curr_epoch):
        """Create deterministic Gaussian-noise target features for V1."""
        sample_keys = []
        for index, item in enumerate(video_list_tgt):
            metadata = item.get('metadata', {})
            sample_keys.append('{}:{}'.format(
                metadata.get('video_id', index),
                metadata.get('crop_start', 0),
            ))
        return deterministic_gaussian_perturbation(
            inputs,
            masks,
            sample_keys,
            self.stage2_stability_noise_std,
            self.stage2_stability_seed,
            curr_epoch,
        )

    @torch.no_grad()
    def _stage1_shift_routing(self, video_list_tgt, points, rng=None):
        """Deterministic two-view routing for the feature-shift experiment."""
        if not self.stage1_calibrated:
            raise RuntimeError("Stage 1 thresholds must be calibrated before routing")

        modules = (self.backbone_target, self.neck_target, self.cls_head_target)
        training_states = [module.training for module in modules]
        for module in modules:
            module.eval()
        try:
            inputs, masks = self.preprocessing(video_list_tgt, mode='validation')
            rng = self.stage1_train_rng if rng is None else rng
            delta = -2 if torch.randint(
                0, 2, (1,), generator=rng
            ).item() == 0 else 2
            self._stage1_last_perturbation = [
                {'perturbation_direction': int(delta), 'offset': abs(int(delta))}
                for _ in video_list_tgt
            ]
            shifted_inputs, shifted_masks = shift_features(inputs, masks, delta)

            strict0 = self._stage1_strict_masks(masks)
            strict1 = self._stage1_strict_masks(shifted_masks)
            feats0, masks0, _ = self.backbone_target(inputs, masks)
            fpn0, fpn_masks0 = self.neck_target(feats0, masks0)
            logits0, _ = self.cls_head_target(fpn0, fpn_masks0)

            feats1, masks1, _ = self.backbone_target(shifted_inputs, shifted_masks)
            fpn1, fpn_masks1 = self.neck_target(feats1, masks1)
            logits1, _ = self.cls_head_target(fpn1, fpn_masks1)

            routes = []
            for level, point_grid in enumerate(points):
                prob0 = logits0[level].permute(0, 2, 1).sigmoid()
                prob1 = logits1[level].permute(0, 2, 1).sigmoid()
                valid0 = strict0[level].squeeze(1)
                valid1 = strict1[level].squeeze(1)
                query = point_grid[:, 0] + delta
                interpolated, interpolation_valid = interpolate_probabilities(
                    prob1, valid1, point_grid, query
                )
                valid = valid0 & interpolation_valid
                level_route = build_routing_masks(
                    prob0,
                    interpolated,
                    valid,
                    self.stage1_alpha,
                    self.stage1_beta,
                    self.stage1_tau_act,
                    self.stage1_tau_bg,
                )
                level_route["delta"] = delta
                routes.append(level_route)
            return routes
        finally:
            for module, was_training in zip(modules, training_states):
                module.train(was_training)

    def _stage1_strict_masks(self, input_mask):
        """Propagate local-support validity through SGP and classification head."""
        cfg_model = self.cfg['model']
        mask = input_mask.bool()
        embd_kernel = cfg_model['embd_kernel_size']
        for _ in range(cfg_model['backbone_arch'][0]):
            mask = erode_mask(mask, embd_kernel, stride=1, padding=embd_kernel // 2)

        k_scale = cfg_model['k']
        stem_kernel = 1
        stem_up = round((stem_kernel + 1) * k_scale)
        stem_up += 1 if stem_up % 2 == 0 else 0
        for _ in range(cfg_model['backbone_arch'][1]):
            mask = erode_mask(mask, stem_up, stride=1, padding=stem_up // 2)

        masks = [mask]
        windows = self.sgp_win_size[1:]
        for window in windows:
            stride = self.scale_factor
            mask = erode_mask(
                mask, stride + 1, stride=stride, padding=(stride + 1) // 2
            )
            local_kernel = max(int(window), 1)
            local_up = round((local_kernel + 1) * k_scale)
            local_up += 1 if local_up % 2 == 0 else 0
            mask = erode_mask(
                mask, local_up, stride=1, padding=local_up // 2
            )
            masks.append(mask)

        head_kernel = cfg_model['head_kernel_size']
        strict_outputs = []
        for level_mask in masks:
            current = level_mask
            for _ in range(cfg_model['head_num_layers']):
                current = erode_mask(
                    current, head_kernel, stride=1, padding=head_kernel // 2
                )
            strict_outputs.append(current.bool())
        return strict_outputs

    @torch.no_grad()
    def _stage1_context_routing(self, video_list_tgt, points, rng=None):
        """Deterministic routing for baseline-crop-preserving context views."""
        if not self.stage1_calibrated:
            raise RuntimeError("Stage 1 thresholds must be calibrated before routing")
        context_videos, directions, offsets, context_valid = [], [], [], []
        rho = self.cfg.get('stage1', {}).get('context_rho', 64.0 / 2304.0)
        rho_max = self.cfg.get('stage1', {}).get('context_rho_max', 2.0 * rho)
        rng = self.stage1_train_rng if rng is None else rng
        for video in video_list_tgt:
            original = video['feats']
            full = video.get('full_feats')
            start = int(video['metadata'].get('crop_start', 0))
            length = original.shape[-1]
            offset = max(1, int(round(rho * length)))
            legal = []
            if full is not None and offset / float(length) <= rho_max:
                if start - offset >= 0:
                    legal.append(-1)
                if start + length + offset <= full.shape[-1]:
                    legal.append(1)
            if legal:
                pick = torch.randint(0, len(legal), (1,), generator=rng).item()
                direction = legal[pick]
                context_start = start + direction * offset
                is_valid = True
            else:
                direction, context_start, is_valid = 0, start, False
            context = context_view_on_cpu(
                original, full, context_start, length
            )
            context_video = dict(video)
            context_video['feats'] = context
            context_videos.append(context_video)
            directions.append(direction)
            offsets.append(offset)
            context_valid.append(is_valid)

        modules = (self.backbone_target, self.neck_target, self.cls_head_target)
        training_states = [module.training for module in modules]
        for module in modules:
            module.eval()
        self._stage1_last_perturbation = [
            {'perturbation_direction': int(direction), 'offset': int(offset)}
            for direction, offset in zip(directions, offsets)
        ]
        self._stage1_last_context_stats = {
            'video_total': len(context_valid),
            'video_valid': sum(context_valid),
        }
        try:
            inputs0, masks0 = self.preprocessing(video_list_tgt, mode='validation')
            inputs1, masks1 = self.preprocessing(context_videos, mode='validation')
            strict0 = self._stage1_strict_masks(masks0)
            strict1 = self._stage1_strict_masks(masks1)
            feats0, backbone_masks0, _ = self.backbone_target(inputs0, masks0)
            fpn0, fpn_masks0 = self.neck_target(feats0, backbone_masks0)
            logits0, _ = self.cls_head_target(fpn0, fpn_masks0)
            feats1, backbone_masks1, _ = self.backbone_target(inputs1, masks1)
            fpn1, fpn_masks1 = self.neck_target(feats1, backbone_masks1)
            logits1, _ = self.cls_head_target(fpn1, fpn_masks1)

            routes = []
            for level, point_grid in enumerate(points):
                prob0 = logits0[level].permute(0, 2, 1).sigmoid()
                prob1 = logits1[level].permute(0, 2, 1).sigmoid()
                valid0 = strict0[level].squeeze(1)
                valid1 = strict1[level].squeeze(1)
                interpolated_rows, valid_rows = [], []
                for batch_idx, (direction, offset, is_valid) in enumerate(
                    zip(directions, offsets, context_valid)
                ):
                    query = point_grid[:, 0] - direction * offset
                    interpolated, valid = interpolate_probabilities(
                        prob1[batch_idx:batch_idx + 1],
                        valid1[batch_idx:batch_idx + 1],
                        point_grid,
                        query,
                    )
                    interpolated_rows.append(interpolated)
                    valid_rows.append(valid & is_valid)
                interpolated = torch.cat(interpolated_rows, dim=0)
                valid = valid0 & torch.cat(valid_rows, dim=0)
                route = build_routing_masks(
                    prob0, interpolated, valid, self.stage1_alpha,
                    self.stage1_beta, self.stage1_tau_act, self.stage1_tau_bg,
                )
                route['direction'] = directions
                route['offset'] = offsets
                if level == 0:
                    route['context_video_total'] = len(context_valid)
                    route['context_video_valid'] = sum(context_valid)
                routes.append(route)
            return routes
        finally:
            for module, was_training in zip(modules, training_states):
                module.train(was_training)

    @torch.no_grad()
    def collect_stage1_calibration_records(self, video_list):
        """Return detached source-validation anchor records for calibration."""
        inputs, masks = self.preprocessing(video_list, mode='validation')
        feats, backbone_masks, _ = self.backbone_target(inputs, masks)
        fpn_feats, _ = self.neck_target(feats, backbone_masks)
        points = self.point_generator(fpn_feats)

        saved = (
            self.stage1_beta, self.stage1_tau_act,
            self.stage1_tau_bg, self.stage1_calibrated,
        )
        self.stage1_beta = self.stage1_tau_act = self.stage1_tau_bg = 1.0
        self.stage1_calibrated = True
        try:
            if self.stage1_perturbation == 'shift':
                routes = self._stage1_shift_routing(
                    video_list, points, rng=self.stage1_calibration_rng
                )
            else:
                routes = self._stage1_context_routing(
                    video_list, points, rng=self.stage1_calibration_rng
                )
        finally:
            (
                self.stage1_beta, self.stage1_tau_act,
                self.stage1_tau_bg, self.stage1_calibrated,
            ) = saved

        gt_segments = [item['segments'].to(self.device) for item in video_list]
        gt_labels = [item['labels'].to(self.device) for item in video_list]
        gt_cls, _ = self.label_points(points, gt_segments, gt_labels)
        gt_cls = torch.stack(gt_cls)
        records, start = [], 0
        for level, point_grid in enumerate(points):
            length = point_grid.shape[0]
            labels = gt_cls[:, start:start + length]
            route = routes[level]
            is_action = labels.sum(dim=-1) > 0
            valid = route['valid']
            action_eligible = source_action_eligible(
                labels,
                route['pseudo_class'],
                route['perturbed_class'],
                route['score_original'],
                valid,
                self.stage1_alpha,
            )
            records.append({
                'level': torch.full(
                    (labels.numel() // labels.shape[-1],), level,
                    dtype=torch.long, device=labels.device,
                ),
                'is_action': is_action.flatten(),
                'r': torch.maximum(
                    route['score_original'], route['score_perturbed']
                ).flatten(),
                'action_eligible': action_eligible.flatten(),
                'action_std': route['action_std'].flatten(),
                'background_valid': valid.flatten(),
                'background_std': route['background_std'].flatten(),
            })
            start += length
        return records

    @torch.no_grad()
    def stage1_route_only(self, video_list):
        """Route target videos without reading segments or labels."""
        inputs, masks = self.preprocessing(video_list, mode='validation')
        feats, backbone_masks, _ = self.backbone_target(inputs, masks)
        fpn_feats, _ = self.neck_target(feats, backbone_masks)
        points = self.point_generator(fpn_feats)
        if self.stage1_perturbation == 'shift':
            return self._stage1_shift_routing(
                video_list, points, rng=self.stage1_dryrun_rng
            )
        return self._stage1_context_routing(
            video_list, points, rng=self.stage1_dryrun_rng
        )
    
    

    @torch.no_grad()
    def generate_stage2_pseudo_targets(
            self, video_list, return_pre_nms=False,
            return_npl_details=False, npl_neighbor_radii=(1, 2, 4)):
        """Generate detached, class-agnostic NMS proposals on feature grids."""
        if self.training:
            raise RuntimeError('Stage 2 EMA teacher must be in eval mode')
        inputs, masks = self.preprocessing(video_list, mode='train')
        feats, backbone_masks, _ = self.backbone_target(inputs, masks)
        fpn_feats, fpn_masks = self.neck_target(feats, backbone_masks)
        cls_logits, _ = self.cls_head_target(fpn_feats, fpn_masks)
        if self.stage2_requires_quality:
            hidden, offsets = regression_hidden_and_offsets(
                self.reg_head_target, fpn_feats, fpn_masks
            )
            quality_maps = ensemble_quality_maps(
                self.alqa_quality_heads, hidden
            )
        else:
            offsets, _ = self.reg_head_target(fpn_feats, fpn_masks)
            quality_maps = None
        points = self.point_generator(fpn_feats)
        targets, pre_nms_targets = [], []
        for batch_idx in range(len(video_list)):
            all_segments, all_scores, all_labels = [], [], []
            all_classification_scores = []
            all_quality_scores = []
            all_levels, all_indices = [], []
            all_ambiguity, all_neighbor_counts = [], []
            dense_budget = 0
            rescued_pre_nms = 0
            for level in self.stage2_levels:
                probabilities = cls_logits[level][batch_idx].transpose(0, 1).sigmoid()
                scores, labels = probabilities.max(dim=-1)
                valid = fpn_masks[level][batch_idx, 0]
                point_grid = points[level]
                level_offsets = offsets[level][batch_idx].transpose(0, 1)
                segments = torch.stack((
                    point_grid[:, 0] - level_offsets[:, 0] * point_grid[:, 3],
                    point_grid[:, 0] + level_offsets[:, 1] * point_grid[:, 3],
                ), dim=-1)
                valid_length = float(video_list[batch_idx]['feats'].shape[-1])
                segments[:, 0].clamp_(min=0.0, max=valid_length)
                segments[:, 1].clamp_(min=0.0, max=valid_length)
                duration = segments[:, 1] - segments[:, 0]
                usable = (
                    valid
                    & torch.isfinite(segments).all(dim=-1)
                    & (duration > self.test_duration_thresh)
                )
                if self.stage2_routing == 'alqa_same_budget':
                    predicted = quality_maps[level][batch_idx]
                    joint_score = alqa_joint_score(
                        scores, predicted[0], predicted[1],
                        self.stage2_alqa_epsilon,
                    )
                    selected_indices, level_budget = select_same_budget_alqa(
                        scores, joint_score, usable, self.stage2_alpha
                    )
                else:
                    selected_indices = torch.where(
                        usable & (scores >= self.stage2_alpha)
                    )[0]
                    level_budget = int(selected_indices.numel())
                if not selected_indices.numel():
                    continue
                dense_budget += level_budget
                selected_scores = scores[selected_indices]
                rescued_pre_nms += int(
                    (selected_scores < self.stage2_alpha).sum()
                )
                all_segments.append(segments[selected_indices])
                all_scores.append(selected_scores)
                all_classification_scores.append(selected_scores)
                if self.stage2_requires_quality:
                    predicted = quality_maps[level][batch_idx]
                    all_quality_scores.append(
                        (predicted[0] - predicted[1])
                        .clamp_min(self.stage2_alqa_epsilon)[selected_indices]
                    )
                all_labels.append(labels[selected_indices])
                if return_npl_details:
                    dense_points = points[level]
                    dense_offsets = offsets[level][
                        batch_idx
                    ].transpose(0, 1)
                    dense_segments = torch.stack((
                        dense_points[:, 0]
                        - dense_offsets[:, 0] * dense_points[:, 3],
                        dense_points[:, 0]
                        + dense_offsets[:, 1] * dense_points[:, 3],
                    ), dim=-1)
                    # NPL measures dense predictions before post-processing.
                    # Keep raw boundaries here; clipping can create artificial
                    # zero variance at crop edges.
                    dense_duration = (
                        dense_segments[:, 1] - dense_segments[:, 0]
                    )
                    dense_valid = (
                        valid
                        & torch.isfinite(dense_segments).all(dim=-1)
                        & (dense_duration > self.test_duration_thresh)
                    )
                    ambiguity, neighbor_counts = npl_boundary_ambiguity(
                        dense_segments, dense_valid, npl_neighbor_radii
                    )
                    all_levels.append(torch.full_like(selected_indices, level))
                    all_indices.append(selected_indices)
                    all_ambiguity.append(ambiguity[selected_indices])
                    all_neighbor_counts.append(neighbor_counts[selected_indices])

            if not all_segments or not any(segment.numel() for segment in all_segments):
                empty_target = {
                    'segments': torch.empty((0, 2), device=self.device),
                    'labels': torch.empty((0,), dtype=torch.long, device=self.device),
                    'scores': torch.empty((0,), device=self.device),
                    'classification_scores': torch.empty(
                        (0,), device=self.device
                    ),
                    'quality_scores': torch.empty((0,), device=self.device),
                    'pre_nms_budget': 0,
                    'rescued_pre_nms': 0,
                    'rescued_post_nms': 0,
                }
                if return_npl_details:
                    empty_target.update({
                        'levels': torch.empty(
                            (0,), dtype=torch.long, device=self.device
                        ),
                        'indices': torch.empty(
                            (0,), dtype=torch.long, device=self.device
                        ),
                        'npl_ambiguity': torch.empty(
                            (0, len(npl_neighbor_radii)), device=self.device
                        ),
                        'npl_neighbor_counts': torch.empty(
                            (0, len(npl_neighbor_radii)),
                            dtype=torch.long, device=self.device,
                        ),
                    })
                targets.append(empty_target)
                if return_pre_nms:
                    pre_nms_targets.append(empty_target)
                continue
            segments = torch.cat(all_segments).detach().cpu()
            scores = torch.cat(all_scores).detach().cpu()
            classification_scores = torch.cat(
                all_classification_scores
            ).detach().cpu()
            quality_scores = (
                torch.cat(all_quality_scores).detach().cpu()
                if all_quality_scores
                else torch.empty_like(classification_scores)
            )
            labels = torch.cat(all_labels).detach().cpu()
            details = {}
            if return_npl_details:
                details = {
                    'levels': torch.cat(all_levels).detach().cpu(),
                    'indices': torch.cat(all_indices).detach().cpu(),
                    'npl_ambiguity': torch.cat(all_ambiguity).detach().cpu(),
                    'npl_neighbor_counts': torch.cat(
                        all_neighbor_counts
                    ).detach().cpu(),
                }
            if return_pre_nms:
                pre_target = {
                    'segments': segments.to(self.device),
                    'labels': labels.to(self.device),
                    'scores': scores.to(self.device),
                }
                pre_target.update({
                    key: value.to(self.device) for key, value in details.items()
                })
                pre_nms_targets.append(pre_target)
            pre_segments, pre_scores, pre_labels = segments, scores, labels
            segments, scores, labels = batched_nms(
                segments, scores, labels,
                self.test_iou_threshold,
                (
                    0.0
                    if self.stage2_routing == 'alqa_same_budget'
                    else self.test_min_score
                ),
                self.test_max_seg_num,
                use_soft_nms=False,
                multiclass=False,
                sigma=self.test_nms_sigma,
                voting_thresh=0.0,
            )
            keep = (
                torch.ones_like(scores, dtype=torch.bool)
                if self.stage2_routing == 'alqa_same_budget'
                else scores >= self.stage2_alpha
            )
            post_source_indices = match_post_nms_to_pre_nms(
                segments[keep], scores[keep], labels[keep],
                pre_segments, pre_scores, pre_labels,
            )
            post_classification_scores = classification_scores[
                post_source_indices
            ]
            target = {
                'segments': segments[keep].to(self.device),
                'labels': labels[keep].to(self.device),
                'scores': scores[keep].to(self.device),
                'classification_scores': post_classification_scores.to(
                    self.device
                ),
                'quality_scores': quality_scores[
                    post_source_indices
                ].to(self.device),
                'pre_nms_budget': dense_budget,
                'rescued_pre_nms': rescued_pre_nms,
                'rescued_post_nms': int(
                    (post_classification_scores < self.stage2_alpha).sum()
                ),
            }
            if return_npl_details:
                target.update({
                    key: value[post_source_indices].to(self.device)
                    for key, value in details.items()
                })
            targets.append(target)
        return (targets, pre_nms_targets) if return_pre_nms else targets

    def forward(self, video_list_src, video_list_tgt, mode=None, domain='src', curr_epoch=0, cls_static_anchors=None, mean_class_conf=None, stage2_pseudo_targets=None):
        
        # Note: The domain is by default src, which indicates in the case of training that we care about applying supervision on the source domain (upper part of the architecture)
        # While during validation, we may want to evaluate the losses on the target domain only (lower part of the architecture)
        bs = len(video_list_src)
        
        # Concatenate the lists of both source and target data
        video_list = video_list_src + video_list_tgt

        # batch the video list into feats (2B, C, T) and masks (2B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list, mode=mode)
        
        batched_inputs = self.dropout(batched_inputs)
        
        # Get the source and target data separately
        batched_inputs_src = batched_inputs[:bs]
        batched_masks_src = batched_masks[:bs]
        
        batched_inputs_tgt = batched_inputs[bs:]
        batched_masks_tgt = batched_masks[bs:]
        
        # FOR NOW, backbone_source = backbone_target
        # feats returns a tuple of L features levels, each of which with shape [B x F x T]
        feats_src, masks_src, _ = self.backbone_source(batched_inputs_src, batched_masks_src)
        fpn_feats_src, fpn_masks_src = self.neck_source(feats_src, masks_src)  # Maintains the same shape as its input
        
        feats_tgt, masks_tgt, _ = self.backbone_target(batched_inputs_tgt, batched_masks_tgt)
        fpn_feats_tgt, fpn_masks_tgt = self.neck_target(feats_tgt, masks_tgt)  # Maintains the same shape as its input

        # Run the complete warm-up pseudo path without gradients.  The older
        # static mode keeps using the current features and freezes only the
        # localization head.
        if (
            self.stage2_enabled
            and self.stage2_routing == 'static_full_dense'
        ):
            for module in (
                self.backbone_pl, self.neck_pl, self.reg_head_pl
            ):
                module.eval()
            with torch.no_grad():
                feats_tgt_pl, masks_tgt_pl, _ = self.backbone_pl(
                    batched_inputs_tgt, batched_masks_tgt
                )
                fpn_feats_tgt_pl, fpn_masks_tgt_pl = self.neck_pl(
                    feats_tgt_pl, masks_tgt_pl
                )
                fpn_feats_tgt_pl_perturbed = None
                if (
                    self.stage2_stability_weighting
                    and mode == 'train'
                    and curr_epoch >= self.stage2_start_epoch
                ):
                    perturbed_inputs = self._stage2_perturbed_inputs(
                        batched_inputs_tgt,
                        batched_masks_tgt,
                        video_list_tgt,
                        curr_epoch,
                    )
                    perturbed_feats, perturbed_masks, _ = self.backbone_pl(
                        perturbed_inputs, batched_masks_tgt
                    )
                    (
                        fpn_feats_tgt_pl_perturbed,
                        _,
                    ) = self.neck_pl(perturbed_feats, perturbed_masks)
        else:
            feats_tgt_pl, masks_tgt_pl, _ = self.backbone_pl(
                batched_inputs_tgt, batched_masks_tgt
            )
            fpn_feats_tgt_pl, fpn_masks_tgt_pl = self.neck_pl(
                feats_tgt_pl, masks_tgt_pl
            )
            fpn_feats_tgt_pl_perturbed = None
        

        # Concatenate the two lists L features levels, each of which with shape [B x F x T] into L feature levels of shape [2B x F x T]
        fpn_feats = [torch.cat([fpn_feats_src[i], fpn_feats_tgt[i]], dim=0) for i in range(len(fpn_feats_src))]
        fpn_masks = [torch.cat([fpn_masks_src[i], fpn_masks_tgt[i]], dim=0) for i in range(len(fpn_masks_src))]

        fpn_feats_tgt_pl = [torch.cat([fpn_feats_tgt_pl[i]], dim=0) for i in range(len(fpn_feats_tgt_pl))]
        fpn_masks_tgt_pl = [torch.cat([fpn_masks_tgt_pl[i]], dim=0) for i in range(len(fpn_masks_tgt_pl))]
        
        fpn_feats_src = [cur_feat[:bs] for cur_feat in fpn_feats]
        fpn_feats_tgt = [cur_feat[bs:] for cur_feat in fpn_feats]   # List of L elements of shape  [B x F x T]
        
        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T_i x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)    
        points_per_level = [p.shape[0] for p in points]
        stage1_routes = None
        if (
            self.stage1_enabled
            and curr_epoch >= self.warmup_adv
            and self.stage1_calibrated
            and mode == 'train'
        ):
            if self.stage1_perturbation == 'shift':
                stage1_routes = self._stage1_shift_routing(video_list_tgt, points)
            else:
                stage1_routes = self._stage1_context_routing(video_list_tgt, points)

        # ------------------------------------------------------------
        # Apply the classification and regression head for the source and target domains, separately
        
        # Separate the dom_invariant features and masks into source and target
        fpn_masks_src = [cur_mask[:bs] for cur_mask in fpn_masks]
        fpn_masks_tgt = [cur_mask[bs:] for cur_mask in fpn_masks]
        
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits_src, _ = self.cls_head(fpn_feats_src, fpn_masks_src)
        out_cls_logits_tgt, _ = self.cls_head_target(fpn_feats_tgt, fpn_masks_tgt)
        
        out_cls_logits = [torch.cat((out_cls_logits_src[i], out_cls_logits_tgt[i]), dim=0) for i in range(len(out_cls_logits_src))]

        # Generate the logits using the pseudo-label network (frozen in the case of static, or changing in the case of dynamic)
        if self.stage2_routing == 'static_full_dense':
            out_cls_logits_tgt_pl = [
                logits.detach() for logits in out_cls_logits_tgt
            ]
        else:
            out_cls_logits_tgt_pl, _ = self.cls_head_pl(
                fpn_feats_tgt_pl, fpn_masks_tgt_pl
            )
        
        if self.use_trident_head:
            out_lb_logits_src = self.start_head(fpn_feats_src, fpn_masks_src)
            out_lb_logits_tgt = self.start_head(fpn_feats_tgt, fpn_masks_tgt)
            out_lb_logits = [torch.cat((out_lb_logits_src[i], out_lb_logits_tgt[i]), dim=0) for i in range(len(out_lb_logits_src))]
            
            out_rb_logits_src = self.end_head(fpn_feats_src, fpn_masks_src)
            out_rb_logits_tgt = self.end_head(fpn_feats_tgt, fpn_masks_tgt)
            out_rb_logits = [torch.cat((out_rb_logits_src[i], out_rb_logits_tgt[i]), dim=0) for i in range(len(out_rb_logits_src))]
        else:
            out_lb_logits_src = None
            out_lb_logits_tgt = None
            out_lb_logits = None
            
            out_rb_logits_src = None
            out_rb_logits_tgt = None
            out_rb_logits = None

        # out_offset: List[B, 2, T_i]
        out_offsets_src, out_actionness_src = self.reg_head(fpn_feats_src, fpn_masks_src)
        out_offsets_tgt, out_actionness_tgt= self.reg_head_target(fpn_feats_tgt, fpn_masks_tgt)
        
        out_offsets = [torch.cat((out_offsets_src[i], out_offsets_tgt[i]), dim=0) for i in range(len(out_offsets_src))]
        out_actionness = [torch.cat((out_actionness_src[i], out_actionness_tgt[i]), dim=0) for i in range(len(out_offsets_src))]

        # Generate the regression offsets of the frozen pseudo-label network (frozen in the case of static, or changing in the case of dynamic)
        if (
            self.stage2_enabled
            and self.stage2_routing
            in ('static_dense', 'static_full_dense')
        ):
            self.reg_head_pl.eval()
            with torch.no_grad():
                pseudo_features = (
                    fpn_feats_tgt_pl
                    if self.stage2_routing == 'static_full_dense'
                    else fpn_feats_tgt
                )
                pseudo_masks = (
                    fpn_masks_tgt_pl
                    if self.stage2_routing == 'static_full_dense'
                    else fpn_masks_tgt
                )
                out_offsets_tgt_pl, out_actionness_tgt_pl = self.reg_head_pl(
                    pseudo_features, pseudo_masks
                )
                out_offsets_tgt_pl_perturbed = None
                if fpn_feats_tgt_pl_perturbed is not None:
                    (
                        out_offsets_tgt_pl_perturbed,
                        _,
                    ) = self.reg_head_pl(
                        fpn_feats_tgt_pl_perturbed, fpn_masks_tgt_pl
                    )
        else:
            out_offsets_tgt_pl, out_actionness_tgt_pl = self.reg_head_pl(
                fpn_feats_tgt, fpn_masks_tgt
            )
            out_offsets_tgt_pl_perturbed = None

        # ------------------------------------------------------------
        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        out_cls_logits_tgt_pl = [x.permute(0, 2, 1) for x in out_cls_logits_tgt_pl]
        
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        out_offsets_tgt_pl = [x.permute(0, 2, 1) for x in out_offsets_tgt_pl]
        if out_offsets_tgt_pl_perturbed is not None:
            out_offsets_tgt_pl_perturbed = [
                x.permute(0, 2, 1)
                for x in out_offsets_tgt_pl_perturbed
            ]
        
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 1 (xC)]
        out_actionness = [x.permute(0, 2, 1) for x in out_actionness]
        out_actionness_tgt_pl = [x.permute(0, 2, 1) for x in out_actionness_tgt_pl]
        
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        fpn_masks_tgt_pl = [x.squeeze(1) for x in fpn_masks_tgt_pl]
        
        valid_mask_tgt_pl = torch.cat(fpn_masks_tgt_pl, dim=1)    # List of B elements of shape [T_i * L]
        # ------------------------------------------------------------      
        
        # return loss during training
        if mode != 'test':
            # generate segment/lable List[N x 2] / List[N] with length = B_src
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels)
            stage2_pseudo_cls, stage2_pseudo_offsets = None, None
            stage2_pseudo_assignments = None
            stage2_proposal_weights = None
            stage2_dense_pseudo_offsets = None
            stage2_dense_perturbed_offsets = None
            stage2_proposal_count = 0
            stage2_pre_nms_budget = 0
            stage2_rescued_pre_nms = 0
            stage2_rescued_post_nms = 0
            if stage2_pseudo_targets is not None:
                if len(stage2_pseudo_targets) != bs:
                    raise ValueError('Stage 2 target batch does not match source batch')
                assignment_result = self.label_points(
                    points,
                    [item['segments'].to(self.device) for item in stage2_pseudo_targets],
                    [item['labels'].to(self.device) for item in stage2_pseudo_targets],
                    return_assignments=(
                        self.stage2_loss_weighting != 'anchor_equal'
                    ),
                )
                if self.stage2_loss_weighting == 'anchor_equal':
                    stage2_pseudo_cls, stage2_pseudo_offsets = assignment_result
                else:
                    (
                        stage2_pseudo_cls,
                        stage2_pseudo_offsets,
                        stage2_pseudo_assignments,
                    ) = assignment_result
                    stage2_proposal_weights = []
                    for item in stage2_pseudo_targets:
                        if self.stage2_loss_weighting == 'proposal_equal':
                            weights = torch.ones_like(item['scores'])
                        elif self.stage2_loss_weighting == 'proposal_quality':
                            weights = item['quality_scores']
                        else:
                            weights = (
                                item['classification_scores']
                                * item['quality_scores']
                            )
                        stage2_proposal_weights.append(
                            weights.to(self.device).detach()
                        )
                stage2_proposal_count = sum(
                    int(item['scores'].numel()) for item in stage2_pseudo_targets
                )
                stage2_pre_nms_budget = sum(
                    int(item.get('pre_nms_budget', item['scores'].numel()))
                    for item in stage2_pseudo_targets
                )
                stage2_rescued_pre_nms = sum(
                    int(item.get('rescued_pre_nms', 0))
                    for item in stage2_pseudo_targets
                )
                stage2_rescued_post_nms = sum(
                    int(item.get('rescued_post_nms', 0))
                    for item in stage2_pseudo_targets
                )
            elif (
                self.stage2_enabled
                and self.stage2_routing
                in ('static_dense', 'static_full_dense')
                and mode == 'train'
                and curr_epoch >= self.stage2_start_epoch
            ):
                stage2_dense_pseudo_offsets = out_offsets_tgt_pl
                stage2_dense_perturbed_offsets = (
                    out_offsets_tgt_pl_perturbed
                )
            
            
            # For debugging ...
            gt_cls_labels_src = gt_cls_labels[:bs]  # list of B elements, each of shape (T_1 + ... + T_L) x C
            gt_cls_labels_tgt = gt_cls_labels[bs:]  # list of B elements, each of shape (T_1 + ... + T_L) x C
            gt_offsets_src = gt_offsets[:bs]    # list of B elements, each of shape (T_1 + ... + T_L) x 2
            gt_offsets_tgt = gt_offsets[bs:]    # list of B elements, each of shape (T_1 + ... + T_L) x 2
            start_idx = 0

            gts_cls_per_level_src, gts_cls_per_level_tgt, gts_offset_per_level_src, gts_offset_per_level_tgt = [], [], [], []
            for l_idx in range(len(fpn_feats)):
                # Save the GT of each of the levels (according to points_per_level)
                gt_cls_labels_src_level = [x[start_idx:start_idx+points_per_level[l_idx], :] for x in gt_cls_labels_src]
                gt_cls_labels_tgt_level = [x[start_idx:start_idx+points_per_level[l_idx], :] for x in gt_cls_labels_tgt]
                gt_offsets_src_level = [x[start_idx:start_idx+points_per_level[l_idx], :] for x in gt_offsets_src]
                gt_offsets_tgt_level = [x[start_idx:start_idx+points_per_level[l_idx], :] for x in gt_offsets_tgt]
                
                
                gts_cls_per_level_src.append(torch.stack(gt_cls_labels_src_level,0))    # B x T_i x C
                gts_cls_per_level_tgt.append(torch.stack(gt_cls_labels_tgt_level,0))    # B x T_i x C
                gts_offset_per_level_src.append(torch.stack(gt_offsets_src_level,0))    # B x T_i x C
                gts_offset_per_level_tgt.append(torch.stack(gt_offsets_tgt_level,0))    # B x T_i x C
                
                # Update the start_idx
                start_idx += points_per_level[l_idx]
            
            
            dom_inv_feats = fpn_feats
                
            beta = self.cfg['model'].get('beta', 1)
            pred_domain_all = self.compute_dom_preds(dom_inv_feats, fpn_masks, curr_epoch, beta=beta)
            
            
            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, 
                out_offsets,
                out_actionness,
                gt_cls_labels, gt_offsets,
                out_lb_logits, out_rb_logits,
                pred_domain_all, 
                dom_inv_feats,  # These are the domain invariant features that we want to apply supervision on for the semantic loss
                gts_cls_per_level_src,
                gts_cls_per_level_tgt,
                mode=mode,
                domain=domain,
                curr_epoch=curr_epoch,
                stage1_routes=stage1_routes,
                stage2_pseudo_cls=stage2_pseudo_cls,
                stage2_pseudo_offsets=stage2_pseudo_offsets,
                stage2_pseudo_assignments=stage2_pseudo_assignments,
                stage2_proposal_weights=stage2_proposal_weights,
                stage2_proposal_count=stage2_proposal_count,
                stage2_dense_pseudo_offsets=stage2_dense_pseudo_offsets,
                stage2_dense_perturbed_offsets=(
                    stage2_dense_perturbed_offsets
                ),
            )
            losses.update({
                'stage2_pre_nms_budget': torch.tensor(
                    float(stage2_pre_nms_budget), device=self.device
                ),
                'stage2_rescued_pre_nms': torch.tensor(
                    float(stage2_rescued_pre_nms), device=self.device
                ),
                'stage2_rescued_post_nms': torch.tensor(
                    float(stage2_rescued_post_nms), device=self.device
                ),
            })
            
            # Return the GTs of the target domain as well, to be used to compute the anchors
            gt_tgt_cls = torch.stack(gt_cls_labels[bs:])   # shape: List of B elements B x FT x C
            gt_tgt_cls = gt_tgt_cls[valid_mask_tgt_pl]  # T' x C
            
            gt_tgt_offsets = torch.stack(gt_offsets[bs:])   # shape: List of B elements B x FT x 2
            gt_tgt_offsets = gt_tgt_offsets[valid_mask_tgt_pl]  # T' x 2

            points = torch.cat(points, dim=0)   # T x 4
            points = points.unsqueeze(0).repeat(bs, 1, 1) # shape: B x T_i x 4
            masked_points = points[valid_mask_tgt_pl] # shape: B(T_i)F x 4

            return (losses, 
                   torch.cat(out_cls_logits_tgt_pl, dim=1)[valid_mask_tgt_pl], 
                   gt_tgt_cls, 
                   torch.cat(out_offsets_tgt_pl, dim=1)[valid_mask_tgt_pl], 
                   gt_tgt_offsets, 
                   masked_points,    # T' x 4
                   torch.cat(out_actionness_tgt_pl, dim=1)[valid_mask_tgt_pl]    # T' x 1
            )

        else:
            # decode the actions (sigmoid / stride, etc)
            points_src = self.point_generator(fpn_feats_src)   
            points_tgt = self.point_generator(fpn_feats_tgt)  
            
            # Separate all the tensors into source and target
            out_cls_logits_src = [x[:bs] for x in out_cls_logits]
            out_cls_logits_tgt = [x[bs:] for x in out_cls_logits]
            
            out_offsets_src = [x[:bs] for x in out_offsets]
            out_offsets_tgt = [x[bs:] for x in out_offsets]
            
            fpn_masks_src = [x[:bs] for x in fpn_masks]
            fpn_masks_tgt = [x[bs:] for x in fpn_masks]
            
            results_src = self.inference(
                video_list_src, points_src, fpn_masks_src,
                out_cls_logits_src, out_offsets_src,
                out_lb_logits_src, out_rb_logits_src,
            )
            
            results_tgt = self.inference(
                video_list_tgt, points_tgt, fpn_masks_tgt,
                out_cls_logits_tgt, out_offsets_tgt,
                out_lb_logits_tgt, out_rb_logits_tgt,
            )
            
            return results_src, results_tgt
        
        
    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0, mode=None):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats], device=self.device)
        max_len = feats_lens.max(0).values.item()

        if mode == 'train':
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training, got {} vs {}".format(max_len, self.max_seq_len)
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
            if self.input_noise > 0:
                # trick, adding noise slightly increases the variability between input features.
                noise = torch.randn_like(batched_inputs) * self.input_noise
                batched_inputs += noise
        else:
            batched_inputs_final = []
            for feat_video, feat_len in zip(feats, feats_lens):
                #assert len(video_list) == 2, "Only support batch_size = 1 during inference (so batch size 1 for source and 1 for target)"
                # input length < self.max_seq_len, pad to max_seq_len
                if max_len <= self.max_seq_len:
                    max_len = self.max_seq_len
                else:
                    # pad the input to the next divisible size
                    stride = self.max_div_factor
                    max_len = (max_len + (stride - 1)) // stride * stride
                padding_size = [0, max_len - int(feat_len)]
                batched_inputs = F.pad(feat_video, padding_size, value=padding_val).unsqueeze(0)
                batched_inputs_final.append(batched_inputs)
            batched_inputs = torch.cat(batched_inputs_final, dim=0)
            
        # generate the mask
        batched_masks = torch.arange(max_len)[None, :].to(self.device) < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(
            self, points, gt_segments, gt_labels, return_assignments=False):
        # concat points on all fpn levels List[T_i x 2B] -> (T_1 + T_2 + ... + T_6) x 4
        # This is shared for all samples in the mini-batch
                
        # Concat points:
        # Dim 0: Point
        # Dim 1: Regression range left (how close to be depending on the level?)
        # Dim 2: Regression range right (how close to be depending on the level?)
        # Dim 3: Stride
        concat_points = torch.cat(points, dim=0)    
        gt_cls, gt_offset, assignments = [], [], []
        
        # loop over each of the video samples (2B)
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            result = self.label_points_single_video(
                concat_points, gt_segment, gt_label,
                return_assignment=return_assignments,
            )
            if return_assignments:
                cls_targets, reg_targets, assignment = result
                assignments.append(assignment)
            else:
                cls_targets, reg_targets = result
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        if return_assignments:
            return gt_cls, gt_offset, assignments
        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(
            self, concat_points, gt_segment, gt_label, return_assignment=False):
        # Important! Only keep the points that are inside the gt segment and inside the regression range

        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_class_orig), 0)   
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            if return_assignment:
                return cls_targets, reg_targets, torch.full(
                    (num_pts,), -1, dtype=torch.long, device=gt_segment.device
                )
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]  # Distance of every GT for every point/anchor
        lens = lens[None, :].repeat(num_pts, 1) 

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0] # Distance of every point (embedding) to the left boundaries of each GT
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]    # Distance of every point (embedding) to the right boundaries of each GT
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])    # Compute the center of every GT
            
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = center_pts - concat_points[:, 3, None] * self.train_center_sample_radius   # Determine the left radius from each GT center
            t_maxs = center_pts + concat_points[:, 3, None] * self.train_center_sample_radius   # Determine the right radius from each GT center
            
            # If the radius is bigger than the action boundary, then prevent it from overrunning it
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0]) # Distance from the point to the beginning of the left center radius (negative if not inside the radius)
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None] # Distance from the point to the beginning of the right center radius  (negative if not inside the radius)
            
            # F T x N x 2
            center_seg = torch.stack((cb_dist_left, cb_dist_right), -1) # New left and right distance
            
            # F T x N (for every anchor, wheather it is inside the center radius of the GT)
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0  # Discard if the point is not inside the center radius
        else:
            # Guarantee that it is inside a GT action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # Take the maximum distance of the embedding to either the left or the right boundary of the GT
        max_regress_distance = reg_targets.max(-1)[0]
        
        # F T x N (wether the point is inside the regression range of the GT)
        # Mark the ones that are outside of the regression range as negatives
        # Discard the embeddings (no-class) that one of the boundaries is further that regression_range from the given boundaries
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # If due to the downsampling, two GT are assigned to 1 action, then keep the smallest only
        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens = lens.to(torch.float32)  # or torch.float64   # TODO: Added by me!
        
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))    # Set the length of each of the GT segments to infinity if they are not inside the GT segment
        
        if self.train_reg_distance:
            # Apply the filtering according to their distance to one of the boarders of the actions
            lens.masked_fill_(inside_regress_range == 0, float('inf'))  # Set the lens of the GT segments to infinity if 
            
        # F T x N -> F T
        # Get GT with smallest length for each of the points (satisfying the conditions)
        # Actually it makes sense that I cannot just relax this condition, because o/w I always just take the smallest one?
        min_len, min_len_inds = lens.min(dim=1) 

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_class_orig
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        if return_assignment:
            assignment = min_len_inds.masked_fill(
                min_len == float('inf'), -1
            )
            return cls_targets, reg_targets, assignment
        return cls_targets, reg_targets


    def losses(
            self, fpn_masks,
            out_cls_logits, 
            out_offsets,
            out_actionness,
            gt_cls_labels, 
            gt_offsets,
            out_start, out_end,
            pred_domain_all,
            dom_inv_feats, 
            gts_per_level_src,
            gts_per_level_tgt,
            mode, 
            domain, 
            curr_epoch,
            stage1_routes=None,
            stage2_pseudo_cls=None,
            stage2_pseudo_offsets=None,
            stage2_pseudo_assignments=None,
            stage2_proposal_weights=None,
            stage2_proposal_count=0,
            stage2_dense_pseudo_offsets=None,
            stage2_dense_perturbed_offsets=None,):
        bs = int(fpn_masks[0].shape[0] / 2)
        
        # gt_cls_labels is a list of 2B elements, each of shape FT x C
        
        # ----------------- #
        # Separate the outputs between source and target domain (first and second half)
        fpn_masks_src = [mask[:bs] for mask in fpn_masks]
        fpn_masks_tgt = [mask[bs:] for mask in fpn_masks]
        
        out_cls_logits_src = [out_cls[:bs] for out_cls in out_cls_logits]   # List of L elements each with B x T_i x C
        out_cls_logits_tgt = [out_cls[bs:] for out_cls in out_cls_logits]
        
        out_offsets_src = [out_off[:bs] for out_off in out_offsets]
        out_offsets_tgt = [out_off[bs:] for out_off in out_offsets]
        
        out_actionness_src = [out_off[:bs] for out_off in out_actionness]
        out_actionness_tgt = [out_off[bs:] for out_off in out_actionness]
        
        # Compute the two following only if we are using the Trident loss
        if out_start is not None:
            out_start_src = [os[:bs] if os is not None else None for os in out_start]
        
        if out_end is not None:
            out_end_src = [oe[:bs] if oe is not None else None for oe in out_end]
        
        # Now separate the GTs
        gt_cls_labels_src = gt_cls_labels[:bs]
        gt_cls_labels_tgt = gt_cls_labels[bs:]
        
        gt_offsets_src = gt_offsets[:bs]
        gt_offsets_tgt = gt_offsets[bs:]
        # ----------------- #
        
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask_src = torch.cat(fpn_masks_src, dim=1)    # shape: B x FT_i
        valid_mask_tgt = torch.cat(fpn_masks_tgt, dim=1)

        if self.use_trident_head:
            out_start_logits = []
            out_end_logits = []
            for i in range(len(out_start_src)):
                x = (F.pad(out_start_src[i], (self.num_bins, 0), mode='constant', value=0)).unsqueeze(-1)  # pad left
                x_size = list(x.size())  # bz, cls_num, T+num_bins, 1
                x_size[-1] = self.num_bins + 1  # bz, cls_num, T+num_bins, num_bins + 1
                x_size[-2] = x_size[-2] - self.num_bins  # bz, cls_num, T+num_bins, num_bins + 1
                x_stride = list(x.stride())
                x_stride[-2] = x_stride[-1]

                x = x.as_strided(size=x_size, stride=x_stride)
                out_start_logits.append(x.permute(0, 2, 1, 3))

                x = (F.pad(out_end_src[i], (0, self.num_bins), mode='constant', value=0)).unsqueeze(-1)  # pad right
                x = x.as_strided(size=x_size, stride=x_stride)
                out_end_logits.append(x.permute(0, 2, 1, 3))
        else:
            out_start_logits = None
            out_end_logits = None

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls_src = torch.stack(gt_cls_labels_src) # shape: B x (T_1 + ... + T_L) x C  -> B x FT x C
        gt_cls_tgt = torch.stack(gt_cls_labels_tgt) # shape: B x (T_1 + ... + T_L) x C  -> B x FT x C
        gt_cls_tgt_full = gt_cls_tgt
        stage2_cls_tgt = (
            torch.stack(stage2_pseudo_cls)
            if stage2_pseudo_cls is not None else None
        )
        stage2_offsets_tgt = (
            torch.stack(stage2_pseudo_offsets)
            if stage2_pseudo_offsets is not None else None
        )
        stage2_assignments_tgt = (
            torch.stack(stage2_pseudo_assignments)
            if stage2_pseudo_assignments is not None else None
        )
        
        # NOTE: We want to be able to specify if we want to apply the supervision over the source or the target domain
        # This is necessary when each of them uses a different backbone (i.e., not shared weights)
        if domain == 'src':
            gt_offsets = gt_offsets_src
            gt_cls = gt_cls_src
            out_offsets = out_offsets_src
            out_actionness = out_actionness_src
            valid_mask = valid_mask_src
        else:
            gt_offsets = gt_offsets_tgt
            gt_cls = gt_cls_tgt
            out_offsets = out_offsets_tgt
            out_actionness = out_actionness_tgt
            valid_mask = valid_mask_tgt
            
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)  # Apply regression only on the ones that we have not discard (so not on the background ones, this is why they are so)
        pos_mask_tgt = torch.logical_and((gt_cls_tgt.sum(-1) > 0), valid_mask_tgt)

        decoded_offsets = self.decode_offset(out_offsets, out_start_logits, out_end_logits, mode=mode)  # shape: B x FT x 2
        decoded_offsets = decoded_offsets[pos_mask]

        if self.use_trident_head:
            # the boundary head predicts the classification score for each categories.
            pred_offsets = decoded_offsets[gt_cls[pos_mask].bool()]
            # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
            vid = torch.where(gt_cls[pos_mask])[0]
            gt_offsets = torch.stack(gt_offsets)[pos_mask][vid]
        else:
            pred_offsets = decoded_offsets
            gt_offsets = torch.stack(gt_offsets)[pos_mask]

        num_pos = pos_mask.sum().item()
        num_pos_tgt = pos_mask_tgt.sum().item()
        if mode == 'train':
            # update the loss normalizer. Do this during training only. During validation, don't update this value
            # Problem: During inference the batch is smaller, so the sum is smaller, and thus the losses are super small
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                    1 - self.loss_normalizer_momentum) * max(num_pos, 1)
            self.loss_normalizer_pseudo_labels = self.loss_normalizer_momentum * self.loss_normalizer + (
                    1 - self.loss_normalizer_momentum) * max(num_pos_tgt, 1)

        # gt_cls_src is already one hot encoded now, simply masking out
        gt_cls_src = gt_cls_src[valid_mask_src]
        gt_cls_tgt = gt_cls_tgt[valid_mask_tgt]

        # Mask out the predictions as well
        if domain == 'src':
            concat_cls_logits = torch.cat(out_cls_logits_src, dim=1)[valid_mask_src]    # shape: B(T_i)F x C
            gt_cls = gt_cls_src # shape: B(T_i)F x C
        else:
            concat_cls_logits = torch.cat(out_cls_logits_tgt, dim=1)[valid_mask_tgt]    # shape: B(T_i)F x C
            gt_cls = gt_cls_tgt # shape: B(T_i)F x C
        
        if mode == 'train':
            train_label_smoothing = self.train_label_smoothing
        else:
            train_label_smoothing = 0.0
        
        # optinal label smoothing (not used by default)  
        gt_cls *= 1 - train_label_smoothing     
        gt_cls += train_label_smoothing / (self.num_class_orig + 1)    
    
        if self.cls_loss_type == 'sigmoid_focal_loss':
            focal_loss_gamma = self.cfg['model'].get('focal_loss_gamma', 2.0)
            cls_loss = sigmoid_focal_loss(
                concat_cls_logits,
                gt_cls,
                reduction='none',
                alpha=0.25, #0.5, #0.75
                gamma=focal_loss_gamma
            )
        else:
            raise NotImplementedError('Unknown classification loss type')
            
            
        if self.use_trident_head: 
            # couple the classification loss with iou score
            iou_rate = ctr_giou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='none'
            )
            rated_mask = gt_cls > self.train_label_smoothing / (self.num_class_orig + 1)
            cls_loss[rated_mask] *= (1 - iou_rate) ** self.iou_weight_power

        cls_loss = cls_loss.sum()
        cls_loss = (
            cls_loss / self.loss_normalizer
            if mode == 'train' else cls_loss / max(num_pos, 1)
        )

        
        # 2. regression using IoU/GIoU loss (defined on positive samples only!)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss = reg_loss/self.loss_normalizer if mode == 'train' else reg_loss/num_pos

        if self.train_loss_reg_weight > 0:
            loss_reg_weight = self.train_loss_reg_weight
        else:
            loss_reg_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        downstream_loss = cls_loss + reg_loss * loss_reg_weight 

        target_reg_loss = 0 * decoded_offsets.sum()
        target_reg_positive = 0 * decoded_offsets.sum()
        if self.oracle_target_regression_mode != 'off':
            target_predictions = self.decode_offset(
                out_offsets_tgt, None, None, mode=mode
            )
            target_offsets = torch.stack(gt_offsets_tgt)
            target_level_mask = torch.cat([
                torch.full_like(
                    fpn_masks_tgt[level],
                    level in self.oracle_target_regression_levels,
                    dtype=torch.bool,
                )
                for level in range(self.n_levels)
            ], dim=1)
            target_regression_offsets = target_offsets
            if self.oracle_target_regression_mode in (
                    'clean_pseudo', 'same_selection'):
                if stage2_cls_tgt is None:
                    target_reg_mask = torch.zeros_like(
                        valid_mask_tgt, dtype=torch.bool
                    )
                else:
                    target_reg_mask = build_same_selection_oracle_mask(
                        stage2_cls_tgt,
                        gt_cls_tgt_full,
                        valid_mask_tgt,
                        target_level_mask,
                    )
                    target_regression_offsets = (
                        select_same_selection_regression_targets(
                            stage2_offsets_tgt,
                            target_offsets,
                            self.oracle_target_regression_mode,
                        )
                    )
            else:
                target_scores = torch.cat(
                    out_cls_logits_tgt, dim=1
                ).sigmoid().max(dim=-1).values
                target_reg_mask = build_oracle_target_regression_mask(
                    target_scores,
                    gt_cls_tgt_full,
                    valid_mask_tgt,
                    target_level_mask,
                    self.cfg['model'].get('pl_threshold', 0.7),
                    self.oracle_target_regression_mode,
                )
            target_reg_positive = target_reg_mask.sum().float()
            if target_reg_mask.any():
                target_reg_loss = ctr_diou_loss_1d(
                    target_predictions[target_reg_mask],
                    target_regression_offsets[target_reg_mask],
                    reduction='sum',
                ) / target_reg_positive
            else:
                target_reg_loss = 0 * target_predictions.sum()
            downstream_loss = (
                downstream_loss
                + self.oracle_target_regression_weight
                * (
                    loss_reg_weight
                    if self.oracle_target_regression_mode
                    in ('clean_pseudo', 'same_selection')
                    else 1.0
                )
                * target_reg_loss
            )

        stage2_reg_loss = 0 * decoded_offsets.sum()
        stage2_reg_positive = 0 * decoded_offsets.sum()
        stage2_reg_ratio = 0 * decoded_offsets.sum()
        stage2_active_batch = 0 * decoded_offsets.sum()
        stage2_effective_proposals = 0 * decoded_offsets.sum()
        stage2_anchor_amplification = 0 * decoded_offsets.sum()
        stage2_weight_ess = 0 * decoded_offsets.sum()
        stage2_weight_mean = 0 * decoded_offsets.sum()
        stage2_offset_l1 = 0 * decoded_offsets.sum()
        stage2_stability_weight_sum = 0 * decoded_offsets.sum()
        stage2_stability_mean = 0 * decoded_offsets.sum()
        stage2_stability_low = 0 * decoded_offsets.sum()
        stage2_stability_mid = 0 * decoded_offsets.sum()
        stage2_stability_high = 0 * decoded_offsets.sum()
        stage2_level_positive = [
            0 * decoded_offsets.sum() for _ in range(self.n_levels)
        ]
        if (
            self.stage2_enabled
            and self.stage2_routing
            in ('static_dense', 'static_full_dense')
            and stage2_dense_pseudo_offsets is not None
        ):
            stage2_predictions = self.decode_offset(
                out_offsets_tgt, None, None, mode=mode
            )
            stage2_targets = self.decode_offset(
                stage2_dense_pseudo_offsets, None, None, mode=mode
            ).detach()
            stage2_level_masks = [
                torch.full_like(
                    fpn_masks_tgt[level],
                    level in self.stage2_levels,
                    dtype=torch.bool,
                )
                for level in range(self.n_levels)
            ]
            stage2_level_mask = torch.cat(stage2_level_masks, dim=1)
            stage2_reg_mask = build_dense_stage2_regression_mask(
                torch.cat(out_cls_logits_tgt, dim=1),
                valid_mask_tgt,
                stage2_level_mask,
                self.stage2_alpha,
            )
            stage2_reg_positive = stage2_reg_mask.sum().float()
            stage2_active_batch = torch.ones_like(stage2_reg_positive)
            offset = 0
            for level, level_mask in enumerate(stage2_level_masks):
                next_offset = offset + level_mask.shape[1]
                stage2_level_positive[level] = stage2_reg_mask[
                    :, offset:next_offset
                ].sum().float()
                offset = next_offset
            if stage2_reg_mask.any():
                if (
                    self.stage2_routing == 'static_full_dense'
                    and self.stage2_stability_weighting
                ):
                    if stage2_dense_perturbed_offsets is None:
                        raise RuntimeError(
                            'Stage 2 stability offsets were not generated'
                        )
                    perturbed_targets = self.decode_offset(
                        stage2_dense_perturbed_offsets,
                        None,
                        None,
                        mode=mode,
                    ).detach()
                    stability = aligned_boundary_iou_from_offsets(
                        stage2_targets, perturbed_targets
                    ).detach()
                    selected_stability = stability[stage2_reg_mask]
                    stage2_reg_loss = weighted_anchor_mse(
                        stage2_predictions,
                        stage2_targets,
                        stability,
                        stage2_reg_mask,
                    )
                    stage2_stability_weight_sum = (
                        selected_stability.sum()
                    )
                    stage2_stability_mean = (
                        selected_stability.mean()
                    )
                    stage2_stability_low = (
                        selected_stability < 0.2
                    ).float().mean()
                    stage2_stability_mid = (
                        (selected_stability >= 0.2)
                        & (selected_stability < 0.5)
                    ).float().mean()
                    stage2_stability_high = (
                        selected_stability >= 0.5
                    ).float().mean()
                elif self.stage2_routing == 'static_full_dense':
                    stage2_reg_loss = F.mse_loss(
                        stage2_predictions[stage2_reg_mask],
                        stage2_targets[stage2_reg_mask],
                        reduction='mean',
                    )
                else:
                    stage2_reg_loss = ctr_diou_loss_1d(
                        stage2_predictions[stage2_reg_mask],
                        stage2_targets[stage2_reg_mask],
                        reduction='sum',
                    ) / stage2_reg_positive
                stage2_offset_l1 = (
                    stage2_predictions[stage2_reg_mask]
                    - stage2_targets[stage2_reg_mask]
                ).abs().mean()
            else:
                stage2_reg_loss = 0 * stage2_predictions.sum()
        elif self.stage2_enabled and stage2_cls_tgt is not None:
            stage2_predictions = self.decode_offset(
                out_offsets_tgt, None, None, mode=mode
            )
            stage2_level_masks = [
                torch.full_like(
                    fpn_masks_tgt[level],
                    level in self.stage2_levels,
                    dtype=torch.bool,
                )
                for level in range(self.n_levels)
            ]
            stage2_level_mask = torch.cat(stage2_level_masks, dim=1)
            stage2_reg_mask = build_stage2_regression_mask(
                stage2_cls_tgt,
                valid_mask_tgt,
                stage2_level_mask,
            )
            stage2_reg_positive = stage2_reg_mask.sum().float()
            stage2_active_batch = torch.ones_like(stage2_reg_positive)
            offset = 0
            for level, level_mask in enumerate(stage2_level_masks):
                next_offset = offset + level_mask.shape[1]
                stage2_level_positive[level] = stage2_reg_mask[
                    :, offset:next_offset
                ].sum().float()
                offset = next_offset
            if stage2_reg_mask.any():
                if self.stage2_loss_weighting == 'anchor_equal':
                    stage2_reg_loss = ctr_diou_loss_1d(
                        stage2_predictions[stage2_reg_mask],
                        stage2_offsets_tgt[stage2_reg_mask],
                        reduction='sum',
                    ) / stage2_reg_positive
                else:
                    numerator = 0 * stage2_predictions.sum()
                    denominator = 0 * stage2_predictions.sum()
                    weight_square_sum = 0 * stage2_predictions.sum()
                    proposal_count = 0 * stage2_predictions.sum()
                    for batch_idx in range(stage2_reg_mask.shape[0]):
                        batch_mask = stage2_reg_mask[batch_idx]
                        anchor_losses = ctr_diou_loss_1d(
                            stage2_predictions[batch_idx][batch_mask],
                            stage2_offsets_tgt[batch_idx][batch_mask],
                        )
                        terms = proposal_loss_terms(
                            anchor_losses,
                            stage2_assignments_tgt[batch_idx][batch_mask],
                            stage2_proposal_weights[batch_idx],
                        )
                        numerator = numerator + terms[0]
                        denominator = denominator + terms[1]
                        weight_square_sum = weight_square_sum + terms[2]
                        proposal_count = proposal_count + terms[3]
                    stage2_reg_loss = (
                        numerator / denominator.clamp_min(1e-8)
                    )
                    stage2_effective_proposals = proposal_count
                    stage2_anchor_amplification = (
                        stage2_reg_positive
                        / proposal_count.clamp_min(1.0)
                    )
                    stage2_weight_ess = (
                        denominator.square()
                        / weight_square_sum.clamp_min(1e-8)
                    )
                    stage2_weight_mean = (
                        denominator / proposal_count.clamp_min(1.0)
                    )
            else:
                stage2_reg_loss = 0 * stage2_predictions.sum()
        if (
            self.stage2_enabled
            and (
                stage2_dense_pseudo_offsets is not None
                or stage2_cls_tgt is not None
            )
        ):
            stage2_reg_ratio = (
                stage2_reg_loss.detach() / reg_loss.detach().clamp_min(1e-8)
            )
            downstream_loss = (
                downstream_loss
                + self.stage2_weight * loss_reg_weight * stage2_reg_loss
            )
        
        losses =  {'cls_loss': cls_loss,
                   'reg_loss': reg_loss,
                   'oracle_target_reg_loss': target_reg_loss,
                   'oracle_target_reg_positive': target_reg_positive,
                   'stage2_pseudo_reg_loss': stage2_reg_loss,
                   'stage2_pseudo_reg_positive': stage2_reg_positive,
                   'stage2_pseudo_active_batch': stage2_active_batch,
                   'stage2_pseudo_to_source_reg_ratio': stage2_reg_ratio,
                   'stage2_pseudo_proposals': torch.tensor(
                       float(stage2_proposal_count), device=self.device
                   ),
                   'stage2_effective_proposals': stage2_effective_proposals,
                   'stage2_anchor_amplification': stage2_anchor_amplification,
                   'stage2_weight_ess': stage2_weight_ess,
                   'stage2_weight_mean': stage2_weight_mean,
                   'stage2_pseudo_offset_l1': stage2_offset_l1,
                   'stage2_stability_weight_sum': stage2_stability_weight_sum,
                   'stage2_stability_mean': stage2_stability_mean,
                   'stage2_stability_low_fraction': stage2_stability_low,
                   'stage2_stability_mid_fraction': stage2_stability_mid,
                   'stage2_stability_high_fraction': stage2_stability_high,
                   'downstream_loss': downstream_loss}
        for level, count in enumerate(stage2_level_positive):
            losses['stage2_pseudo_l{}_positive'.format(level)] = count
        
        
        # Set a threshold for the confidence of the pseudo-labels
        static_threshold = self.cfg['model'].get('pl_threshold', 0.7)             
        threshold = torch.ones(self.num_classes).to(self.device) * static_threshold   
            
            
        # ----------------------------------------------------
        # Implement the adversarial loss
        loss_reverse = 0
        
        semantic_alignment_levels = self.cfg['model'].get('semantic_alignment_levels', [0,1,2,3,4,5]) # By default apply it to all the 6 levels
        
        
        if mode == 'train':
            if 'reverse' in self.cfg['model']['adv_method']:
                # Don't compute this during validation, to now screw the centroids
                for l in range(len(pred_domain_all)):
                    # This iterates over all the levels (or just one representation if we use one from before the downsampling)
                    
                    # Apply the adversarial loss over each of the feature levels
                    da_pred_source = pred_domain_all[l][:bs, :]  # shape: B x T x output_dim (1 or 2)
                    da_pred_target = pred_domain_all[l][bs:, :]  # shape: B x T x output_dim (1 or 2)
                    
                    # fpn_masks_src: shape (B x T x output_dim)
                    
                    # Consider only the unmasked tokens
                    da_pred_source = da_pred_source[fpn_masks_src[l]]    # shape: T' x output_dim (1 or 2)
                    da_pred_target = da_pred_target[fpn_masks_tgt[l]]   # shape: T'' x output_dim (1 or 2)
                    
                    
                    if 'reverse' in self.cfg['model']['adv_method']:
                        # Prepare the domain annotations
                        source_domain_label = torch.zeros(da_pred_source.size(0), dtype=torch.float32).to(self.device) # shape: T'
                        target_domain_label = torch.ones(da_pred_target.size(0), dtype=torch.float32).to(self.device)  # shape: T''
                        
                        # Compute the adversarial domain loss
                        domain_loss_src = self.criterion_domain(torch.squeeze(da_pred_source, -1), source_domain_label)
                        domain_loss_tgt = self.criterion_domain(torch.squeeze(da_pred_target, -1), target_domain_label)


                        weight_source = self.cfg['model'].get('weight_source_cond_adv', 1.0)
                        weight_target = self.cfg['model'].get('weight_target_cond_adv', 1.0)

                        loss_adversarial_single = weight_source * domain_loss_src + weight_target * domain_loss_tgt 
                        
                        losses['loss_adv_rev_' + str(l)] = loss_adversarial_single

                        if l in semantic_alignment_levels:
                            weight_level = self.cfg['model'].get('weight_level_' + str(l), 1.0)
                            loss_reverse += weight_level * (loss_adversarial_single / min(len(semantic_alignment_levels), len(pred_domain_all)))
                
                if 'reverse' in self.cfg['model']['adv_method']:
                    losses['loss_adv_rev'] = loss_reverse
                if 'None' in self.cfg['model']['adv_method']:
                    pass
            
            
            # -------------------------------
            # Implementation of the conditional adversarial loss, where we try to fool the discriminator while it knows to which action class this belong (source), or the action class that we infer (target)
            if 'conditional_rev' in self.cfg['model']['adv_method'] and curr_epoch >= self.cfg['train_cfg'].get('warmup_adv', 0):
                # For this loss, apply the discriminator to the source and target embeddings, but including the information of the class that it belongs (or that we infer)
                # This will condition the prediction, and "tell" the discriminator that what we want is to fool it class-wise (e.g., class 1 of source and target,...)
                
                # TODO: For now we can apply this only on the selected anchors, which will need to end up being all (or nearly all) in a more automatic way
                
                
                # Start computing this only after the warmup period and if semantic_alignment is True
                loss_cond_reverse = 0
                
                for l in range(self.n_levels):
                    # Compute it over all the levels for debugging purposes, but use only the selected ones to add them to the final loss
                    
                    valid_mask_l_src = fpn_masks_src[l] # shape: B x T
                    valid_mask_l_tgt = fpn_masks_tgt[l] # shape: B x T
                    
                    # 1. Optional filtering of the anchors
                    out_cls_logits_tgt2 = out_cls_logits_tgt[l][valid_mask_l_tgt].sigmoid()  # shape: B(T_i) x C
                    out_cls_logits_src2 = out_cls_logits_src[l][valid_mask_l_src].sigmoid()  # shape: B(T_i) x C
                    
                    # 2) Compute the pseudo-labels of the target domain
                    # c* = argmax_c \hat{p}_c
                    max_cls_prob_tgt, max_cls_idx_tgt = torch.max(out_cls_logits_tgt2, dim=1)  # shape: B(T_i)
                    
                    # Apply thresholding to the maximum class probability
                    pseudo_gt_cls_target = torch.zeros_like(out_cls_logits_tgt2).detach()  # shape: B(T_l) x C or B(T_l) x C+1
                    if self.oracle_routing_mode != 'off':
                        oracle_route = build_oracle_routing(
                            out_cls_logits_tgt2,
                            gts_per_level_tgt[l][valid_mask_l_tgt],
                            static_threshold,
                            self.oracle_routing_mode,
                        )
                        selected_anchors_mask = oracle_route['action']
                        pseudo_gt_cls_target = oracle_route['action_labels']
                        stage1_background_mask = oracle_route['background']
                        metric_prefix = 'oracle_l{}_'.format(l)
                        losses[metric_prefix + 'action'] = (
                            oracle_route['action'].sum().float()
                        )
                        losses[metric_prefix + 'background'] = (
                            oracle_route['background'].sum().float()
                        )
                        losses[metric_prefix + 'ignore'] = (
                            oracle_route['ignore'].sum().float()
                        )
                        losses[metric_prefix + 'gt_foreground'] = (
                            oracle_route['gt_foreground'].sum().float()
                        )
                        losses[metric_prefix + 'high_confidence'] = (
                            oracle_route['high_confidence'].sum().float()
                        )
                    elif stage1_routes is None:
                        selected_anchors_mask = max_cls_prob_tgt >= threshold[max_cls_idx_tgt]
                        pseudo_gt_cls_target[selected_anchors_mask] = F.one_hot(max_cls_idx_tgt[selected_anchors_mask], num_classes=self.num_classes).float()
                        stage1_background_mask = None
                    else:
                        route_l = stage1_routes[l]
                        route_valid = fpn_masks_tgt[l]
                        selected_anchors_mask = route_l['reliable_action'][route_valid]
                        selected_classes = route_l['pseudo_class'][route_valid]
                        pseudo_gt_cls_target[selected_anchors_mask] = F.one_hot(
                            selected_classes[selected_anchors_mask],
                            num_classes=self.num_classes,
                        ).float()
                        stage1_background_mask = route_l['stable_background'][route_valid]
                        metric_prefix = 'stage1_l{}_'.format(l)
                        losses[metric_prefix + 'valid'] = route_l['valid'].sum().float()
                        losses[metric_prefix + 'action_candidate'] = route_l['action_candidate'].sum().float()
                        losses[metric_prefix + 'class_consistent'] = route_l['class_consistent'].sum().float()
                        losses[metric_prefix + 'action'] = route_l['reliable_action'].sum().float()
                        losses[metric_prefix + 'background_candidate'] = route_l['background_candidate'].sum().float()
                        losses[metric_prefix + 'all_view_low'] = route_l['all_view_low'].sum().float()
                        losses[metric_prefix + 'background'] = route_l['stable_background'].sum().float()
                        losses[metric_prefix + 'ignore'] = route_l['ignore'].sum().float()
                        if l == 0 and 'context_video_total' in route_l:
                            losses['stage1_context_video_total'] = torch.tensor(
                                float(route_l['context_video_total']),
                                device=self.device,
                            )
                            losses['stage1_context_video_valid'] = torch.tensor(
                                float(route_l['context_video_valid']),
                                device=self.device,
                            )
                        losses[metric_prefix + 'action_std'] = (
                            route_l['action_std'][route_l['valid']].mean()
                            if route_l['valid'].any()
                            else route_l['action_std'].sum() * 0
                        )
                        losses[metric_prefix + 'background_std'] = (
                            route_l['background_std'][route_l['valid']].mean()
                            if route_l['valid'].any()
                            else route_l['background_std'].sum() * 0
                        )
            
                    # 2) Compute the pseudo-labels of the source domain
                    # c* = argmax_c \hat{p}_c
                    max_cls_prob_src, max_cls_idx_src = torch.max(out_cls_logits_src2, dim=1)  # shape: B(T_i)
                    
                    # Apply thresholding to the maximum class probability
                    pseudo_gt_cls_source = torch.zeros_like(out_cls_logits_src2).detach()  # shape: B(T_l) x C or B(T_l) x C+1
                    selected_anchors_mask = max_cls_prob_src >= threshold[max_cls_idx_src]
                    pseudo_gt_cls_source[selected_anchors_mask] = F.one_hot(max_cls_idx_src[selected_anchors_mask], num_classes=self.num_classes).float()

                    if not self.cfg['model'].get('use_source_pseudo_labels', False):
                        source_labels = gts_per_level_src[l][valid_mask_l_src]
                    else:
                        # Both the selected pseudo-labels and the GT labels have the same shape. Multiply to keep only the 1s on both of them
                        source_labels = gts_per_level_src[l][valid_mask_l_src] * pseudo_gt_cls_source
                     
                    
                    # 2) Iterate over each of the classes. Update the centroids and align both of them
                    # 2.0) Separate the source and target domain invariant representations
                    dom_inv_feats_src = dom_inv_feats[l][:bs]   # shape: [B x F x T_i]
                    dom_inv_feats_tgt = dom_inv_feats[l][bs:]   # shape: [B x F x T_i]
                    
                    # Transpose to have shapes: B x (T_1 + ... + T_L) x C, and filter out padding
                    dom_inv_feats_src = einops.rearrange(dom_inv_feats_src, 'b c t -> b t c')[valid_mask_l_src] # shape: T' x F where T' is B(T_i)
                    dom_inv_feats_tgt = einops.rearrange(dom_inv_feats_tgt, 'b c t -> b t c')[valid_mask_l_tgt] # shape: T'' x F
                    
                    loss_cond_reverse_level_source = torch.tensor(0.0, requires_grad=True).to(self.device)
                    loss_cond_reverse_level_target = torch.tensor(0.0, requires_grad=True).to(self.device)
                    source_class_losses = []
                    target_class_losses = []
                    source_sample_count = 0
                    target_sample_count = 0
                    min_source_class_samples = None
                    min_target_class_samples = None
                    
                    # Iterate over the true class labels (and not the no-class)
                    n_classes = self.num_class_orig if not self.align_bkg else self.num_class_orig + 1
                    counter_src, counter_tgt = 0, 0
                    
                    # Define the hyperparameters for the influence of the source and the target domain, and also the effect of every level
                    weight_source = self.cfg['model'].get('weight_source_cond_adv', 1.0)
                    weight_target = self.cfg['model'].get('weight_target_cond_adv', 1.0)
                    
                    weight_level = {}
                    weight_level['weight_level_0'] = self.cfg['model'].get('weight_level_0', 1.0)
                    weight_level['weight_level_1'] = self.cfg['model'].get('weight_level_1', 1.0)
                    weight_level['weight_level_2'] = self.cfg['model'].get('weight_level_2', 1.0)
                    weight_level['weight_level_3'] = self.cfg['model'].get('weight_level_3', 1.0)
                    weight_level['weight_level_4'] = self.cfg['model'].get('weight_level_4', 1.0)
                    weight_level['weight_level_5'] = self.cfg['model'].get('weight_level_5', 1.0)
                    
                    
                    for c in range(n_classes):
                        class_c_weight = 1.0
                        if c == self.num_class_orig:
                            # Select all the background embeddings (all zeros in the source), and all the ones below all the thresholds in the pseudo-labels
                            class_c_mask_src = torch.all(source_labels == 0, dim=-1)  # shape: T'
                            if stage1_background_mask is None:
                                class_c_mask_tgt = torch.all(pseudo_gt_cls_target == 0, dim=-1)  # shape: T''
                            else:
                                class_c_mask_tgt = stage1_background_mask
                            
                            # Set the influence of the background class to the predefined tunnable parameter
                            class_c_weight = self.cfg['model'].get('weight_no_class_adv', 1.0)
                        else:
                            # NOTE: We are not aligning here the background class (or no-class really)
                            # This might be needed to grab all the rest and align togehter. NOT SURE
                            
                            # 2.1) Select all the embeddings that belong to this class (source) and that we infer belong to it as well (target)
                            class_c_mask_src = source_labels[:, c] == 1   # shape: T' (subtract start_idx to to adapt to the no-class)   
                            class_c_mask_tgt = pseudo_gt_cls_target[:, c] == 1   # shape: T''
                            
                        # 2.2) Select their corresponding embeddings
                        out_features_src = dom_inv_feats_src[class_c_mask_src]  # shape: N' x F
                        out_features_tgt = dom_inv_feats_tgt[class_c_mask_tgt]
                        
                        # 2.3) Concatenate the learnable embedding of this class
                        out_features_src = torch.cat((out_features_src, self.learnable_embeddings[c].unsqueeze(0).expand(out_features_src.size(0), -1)), dim=-1)  # shape: N' x 2F
                        out_features_tgt = torch.cat((out_features_tgt, self.learnable_embeddings[c].unsqueeze(0).expand(out_features_tgt.size(0), -1)), dim=-1)  # shape: N'' x 2F
                        
                        if 'conditional_rev' in self.cfg['model']['adv_method']:
                            # 2.4) Predict the domain of each of the embeddings (forward pass)
                            beta = self.cfg['model'].get('beta', 1.0)
                            da_cond_pred_src = self.compute_level_dom_preds(out_features_src, reverse=True, level=l, curr_epoch=curr_epoch, beta=beta)  # shape: N' x 1
                            da_cond_pred_tgt = self.compute_level_dom_preds(out_features_tgt, reverse=True, level=l, curr_epoch=curr_epoch, beta=beta)  # shape: N'' x 1
                            
                            # 2.5) Apply the reverse gradient strategy
                            # Prepare the domain annotations
                            source_domain_label = torch.zeros(da_cond_pred_src.size(0), dtype=torch.float32).to(self.device) # shape: T'
                            target_domain_label = torch.ones(da_cond_pred_tgt.size(0), dtype=torch.float32).to(self.device)  # shape: T''
                            
                            # Compute the adversarial domain loss
                            domain_loss_src = self.criterion_domain(torch.squeeze(da_cond_pred_src, -1), source_domain_label)
                            domain_loss_tgt = self.criterion_domain(torch.squeeze(da_cond_pred_tgt, -1), target_domain_label)

                            # Check for nan values and replace with 0
                            if torch.isnan(domain_loss_src).any():
                                domain_loss_src = torch.where(torch.isnan(domain_loss_src), torch.zeros_like(domain_loss_src), domain_loss_src).to(self.device)
                            
                            if torch.isnan(domain_loss_tgt).any():
                                domain_loss_tgt = torch.where(torch.isnan(domain_loss_tgt), torch.zeros_like(domain_loss_tgt), domain_loss_tgt).to(self.device)
                            
                            if torch.sum(class_c_mask_src) > 0:
                                loss_cond_reverse_level_source += class_c_weight * domain_loss_src
                                counter_src += 1
                                class_samples = int(class_c_mask_src.sum().item())
                                source_sample_count += class_samples
                                min_source_class_samples = (
                                    class_samples if min_source_class_samples is None
                                    else min(min_source_class_samples, class_samples)
                                )
                                source_class_losses.append(domain_loss_src.detach())
                            if torch.sum(class_c_mask_tgt) > 0:
                                loss_cond_reverse_level_target += class_c_weight * domain_loss_tgt
                                counter_tgt += 1
                                class_samples = int(class_c_mask_tgt.sum().item())
                                target_sample_count += class_samples
                                min_target_class_samples = (
                                    class_samples if min_target_class_samples is None
                                    else min(min_target_class_samples, class_samples)
                                )
                                target_class_losses.append(domain_loss_tgt.detach())

                    if 'conditional_rev' in self.cfg['model']['adv_method']:
                        loss_cond_reverse_level_source = loss_cond_reverse_level_source / (max(counter_src,1))
                        loss_cond_reverse_level_target = loss_cond_reverse_level_target / (max(counter_tgt,1))

                        diagnostic_prefix = 'conditional_l{}_'.format(l)
                        losses[diagnostic_prefix + 'source_loss'] = (
                            loss_cond_reverse_level_source.detach()
                        )
                        losses[diagnostic_prefix + 'target_loss'] = (
                            loss_cond_reverse_level_target.detach()
                        )
                        losses[diagnostic_prefix + 'source_classes'] = torch.tensor(
                            float(counter_src), device=self.device
                        )
                        losses[diagnostic_prefix + 'target_classes'] = torch.tensor(
                            float(counter_tgt), device=self.device
                        )
                        losses[diagnostic_prefix + 'source_samples'] = torch.tensor(
                            float(source_sample_count), device=self.device
                        )
                        losses[diagnostic_prefix + 'target_samples'] = torch.tensor(
                            float(target_sample_count), device=self.device
                        )
                        losses[diagnostic_prefix + 'source_min_samples'] = torch.tensor(
                            float(min_source_class_samples or 0), device=self.device
                        )
                        losses[diagnostic_prefix + 'target_min_samples'] = torch.tensor(
                            float(min_target_class_samples or 0), device=self.device
                        )
                        losses[diagnostic_prefix + 'source_max_class_loss'] = (
                            torch.stack(source_class_losses).max()
                            if source_class_losses else torch.tensor(0.0, device=self.device)
                        )
                        losses[diagnostic_prefix + 'target_max_class_loss'] = (
                            torch.stack(target_class_losses).max()
                            if target_class_losses else torch.tensor(0.0, device=self.device)
                        )
                        
                        loss_cond_reverse_level = weight_source * loss_cond_reverse_level_source + weight_target * loss_cond_reverse_level_target
                        losses['loss_conditional_adv_rev_' + str(l)] = loss_cond_reverse_level

                        if l in semantic_alignment_levels:
                            loss_cond_reverse += weight_level['weight_level_' + str(l)] * (loss_cond_reverse_level / len(semantic_alignment_levels))
                    
                
                if 'conditional_rev' in self.cfg['model']['adv_method']:
                    losses['loss_conditional_adv_rev'] = loss_cond_reverse
            
        return losses

    @torch.no_grad()
    def inference(
            self,
            video_list,
            points, fpn_masks,
            out_cls_logits, out_offsets,
            out_lb_logits, out_rb_logits,
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []
        
        # 1: gather video meta information
        vid_idxs = [x['metadata']['video_id'] for x in video_list]
        vid_fps = [x['metadata']['fps'] for x in video_list]
        vid_duration = [x['metadata']['duration'] for x in video_list]
        vid_ft_stride = [x['metadata']['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['metadata']['feat_num_frames'] for x in video_list]
        vid_ft_utf = [x['metadata']['use_temporal_grid_feats'] for x in video_list]
        vid_ft_vlen = [x['metadata']['video_length'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vduration, stride, nframes, utf, vlen) in enumerate(
                zip(vid_idxs, vid_fps, vid_duration, vid_ft_stride, vid_ft_nframes, vid_ft_utf, vid_ft_vlen)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]

            if self.use_trident_head:
                lb_logits_per_vid = [x[idx] for x in out_lb_logits]
                rb_logits_per_vid = [x[idx] for x in out_rb_logits]
            else:
                lb_logits_per_vid = [None for x in range(len(out_cls_logits))]
                rb_logits_per_vid = [None for x in range(len(out_cls_logits))]

            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid,
                lb_logits_per_vid, rb_logits_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vduration
            results_per_vid['video_length'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results_per_vid['use_temporal_grid_feats'] = utf
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
            self,
            points,
            fpn_masks,
            out_cls_logits,
            out_offsets,
            lb_logits_per_vid, rb_logits_per_vid
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i, sb_cls_i, eb_cls_i in zip(out_cls_logits, out_offsets, points, fpn_masks, lb_logits_per_vid, rb_logits_per_vid):
            if self.cfg['model'].get('use_ce_cls', False):
                pred_prob = cls_i.softmax(-1) * mask_i.unsqueeze(-1)
                
                # Now filter out the background class, to keep only the foreground classes (just like in the sigmoid case)
                # This allow for the maximum to be the background class, and yet choose a class label if the confidence is high enough
                pred_prob = pred_prob[:, 1:]   # TODO: Uncomment to filter out the background and apply the maximum over the rest.
            else:
                pred_prob = cls_i.sigmoid() * mask_i.unsqueeze(-1)
            pred_prob = pred_prob.flatten() # [T_i * C]
            
            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs = torch.div(topk_idxs, self.num_class_orig, rounding_mode='floor') 
            cls_idxs = torch.fmod(topk_idxs, self.num_class_orig)

            # 3. For efficiency, pad the boarder head with num_bins zeros (Pad left for start branch and Pad right
            # for end branch). Then we re-arrange the output of boundary branch to [class_num, T, num_bins + 1 (the
            # neighbour bin for each instant)]. In this way, the output can be directly added to the center offset
            # later.
            if self.use_trident_head:
                # pad the boarder
                x = (F.pad(sb_cls_i, (self.num_bins, 0), mode='constant', value=0)).unsqueeze(-1)  # pad left
                x_size = list(x.size())  # cls_num, T+num_bins, 1
                x_size[-1] = self.num_bins + 1
                x_size[-2] = x_size[-2] - self.num_bins  # cls_num, T, num_bins + 1
                x_stride = list(x.stride())
                x_stride[-2] = x_stride[-1]

                pred_start_neighbours = x.as_strided(size=x_size, stride=x_stride)

                x = (F.pad(eb_cls_i, (0, self.num_bins), mode='constant', value=0)).unsqueeze(-1)  # pad right
                pred_end_neighbours = x.as_strided(size=x_size, stride=x_stride)
            else:
                pred_start_neighbours = None
                pred_end_neighbours = None

            decoded_offsets = self.decode_offset(offsets_i, pred_start_neighbours, pred_end_neighbours, mode='test')

            # pick topk output from the prediction
            if self.use_trident_head:
                offsets = decoded_offsets[cls_idxs, pt_idxs]
            else:
                offsets = decoded_offsets[pt_idxs]

            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]

            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments': segs_all,
                   'scores': scores_all,
                   'labels': cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            duration = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            utgf = results_per_vid['use_temporal_grid_feats']
            vlen = results_per_vid['video_length']
            
            # 1: unpack the results and move to CPU
            segs_pre_nms = results_per_vid['segments'].detach().cpu()
            scores_pre_nms = results_per_vid['scores'].detach().cpu()
            labels_pre_nms = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs_pre_nms, scores_pre_nms, labels_pre_nms,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms=(self.test_nms_method == 'soft'),
                    multiclass=self.test_multiclass_nms,
                    sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh
                )
                
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                if utgf:
                    # Transform from temporal grid to seconds. Otherwise, we just directly make the predictions in seconds/frames
                    segs = (segs * stride + 0.5 * nframes) / fps
                    segs_pre_nms = (segs_pre_nms * stride + 0.5 * nframes) / fps

                    vlen = duration # The video length is the duration of the video in these datasets

                # truncate all boundaries within [0, duration]
                segs[segs <= 0.0] *= 0.0
                segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen
                
                # Right now, the pre-nms segments are simply truncated between [0, duration] (but we don't transform it from grid to seconds for visualization purposes)
                segs_pre_nms[segs_pre_nms <= 0.0] *= 0.0
                segs_pre_nms[segs_pre_nms >= vlen] = segs_pre_nms[segs_pre_nms >= vlen] * 0.0 + vlen
                    
            # 4: repack the results
            processed_results.append(
                {'video_id': vidx,
                 'segments': segs,
                 'scores': scores,
                 'labels': labels,
                 'segments_pre_nms': segs_pre_nms,
                 'scores_pre_nms': scores_pre_nms,
                 'labels_pre_nms': labels_pre_nms}
            )

        return processed_results

    
