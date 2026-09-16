from contextlib import contextmanager
import torch

@contextmanager
def delimiter_intervention(model, input_ids, mode, scale=8.0, layers=(0, 1, 2, 3), token_kind="both"):
    if mode == "baseline":
        yield
        return
    assert input_ids.shape[0] == 1
    cfg = model.config
    start = input_ids == cfg.vision_start_token_id
    end = input_ids == cfg.vision_end_token_id
    mask = start if token_kind == "start" else end if token_kind == "end" else start | end
    positions = mask[0].nonzero(as_tuple=True)[0]
    prompt_len = input_ids.shape[1]
    handles = []
    def multiply(tensor, factor):
        if tensor.shape[1] != prompt_len:
            return tensor
        tensor = tensor.clone()
        tensor[:, positions, :] *= factor
        return tensor
    def output_hook(factor, is_tuple):
        def hook(module, args, output):
            if is_tuple:
                return (multiply(output[0], factor),) + output[1:]
            return multiply(output, factor)
        return hook
    cumulative = 1.0
    if int(start.sum()) > 1:
        for i, block in enumerate(model.model.layers):
            if mode == "scale" and i in layers:
                handles.append(block.register_forward_hook(output_hook(scale, True)))
            if mode == "gate":
                handles.append(block.self_attn.register_forward_hook(output_hook(1.0 / cumulative, True)))
                handles.append(block.mlp.register_forward_hook(output_hook(1.0 / cumulative, False)))
            if i in layers:
                cumulative *= scale
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
