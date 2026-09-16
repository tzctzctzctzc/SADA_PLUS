from contextlib import contextmanager
from interventions import delimiter_intervention

@contextmanager
def boundary_intervention(model, input_ids, mode):
    if mode in ("baseline", "scale", "gate"):
        with delimiter_intervention(model, input_ids, mode):
            yield
        return
    kind = mode.removeprefix("canonical_")
    assert kind in ("start", "end")
    token_id = getattr(model.config, "vision_" + kind + "_token_id")
    positions = (input_ids[0] == token_id).nonzero(as_tuple=True)[0]
    n = input_ids.shape[1]
    handles = []
    def canonicalize(module, args, output):
        if output.shape[1] != n or positions.numel() < 2:
            return output
        result = output.clone()
        # Preserve original rotary positions; only replace the normalized content.
        result[:, positions[1:], :] = output[:, positions[:1], :]
        return result
    for layer in model.model.layers:
        handles.append(layer.input_layernorm.register_forward_hook(canonicalize))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
