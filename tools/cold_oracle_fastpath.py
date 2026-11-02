"""仅省去原trainer丢弃的PL计算；重放随机流，保留所用loss及完整优化状态。"""
import torch
from torch import nn


def replay_backbone_rng(backbone, x):
    # SGP当前唯一训练随机算子是各branch的AffineDropPath。
    for block in list(backbone.stem) + list(backbone.branch):
        for path in (block.drop_path_out, block.drop_path_mlp):
            if isinstance(path, nn.Identity):
                continue
            assert type(path).__name__ == 'AffineDropPath'
            if path.training and path.drop_prob > 0:
                torch.rand((x.shape[0], 1, 1), dtype=x.dtype, device=x.device)


def install(model):
    if hasattr(model, 'equivalent_fast_enabled'):
        return
    assert model.backbone_source is model.backbone_target is model.backbone_pl
    assert model.neck_source is model.neck_target is model.neck_pl
    assert model.cls_head is model.cls_head_target is model.cls_head_pl
    assert model.reg_head is model.reg_head_target is model.reg_head_pl
    backbone = model.backbone_source
    assert type(backbone).__name__ == 'SGPBackbone'
    for module in (backbone, model.neck_source, model.cls_head, model.reg_head):
        assert not any(isinstance(x, nn.modules.batchnorm._BatchNorm) for x in module.modules())
        assert not any(isinstance(x, nn.Dropout) and x.p > 0 for x in module.modules())
    model.equivalent_fast_enabled = True
    state = dict(active=False, same_eval=False, counts={}, cached={})
    modules = [('backbone', backbone), ('neck', model.neck_source),
               ('cls', model.cls_head), ('reg', model.reg_head)]
    for name, module in modules:
        native = module.forward
        def forward(*args, _name=name, _native=native, **kwargs):
            if not state['active']:
                return _native(*args, **kwargs)
            count = state['counts'].get(_name, 0) + 1
            state['counts'][_name] = count
            assert count <= 3
            reuse = count == 3 or (count == 2 and state['same_eval'])
            if reuse:
                if _name == 'backbone' and module_training(model) and count == 3:
                    replay_backbone_rng(backbone, args[0])
                return state['cached'][_name]
            result = _native(*args, **kwargs)
            state['cached'][_name] = result
            return result
        module.forward = forward
    native_model = model.forward
    def outer(*args, **kwargs):
        if not model.equivalent_fast_enabled:
            return native_model(*args, **kwargs)
        assert not torch.is_autocast_enabled()
        state['active'] = True
        state['same_eval'] = not model.training and kwargs.get('video_list_src') is kwargs.get('video_list_tgt')
        state['counts'].clear(); state['cached'].clear()
        try:
            result = native_model(*args, **kwargs)
            assert all(state['counts'].get(k) == 3 for k,_ in modules)
            return result
        finally:
            state['active'] = False
            state['cached'].clear()
    model.forward = outer


def module_training(model):
    return model.backbone_source.training
