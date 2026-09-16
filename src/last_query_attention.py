"""Reconstruct only the last-query attention row; never retain an N x N map."""
import math
import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import rotate_half, repeat_kv

class LastQueryAttention:
    def __init__(self, model, input_ids, layers=None, keep_rows=False):
        assert input_ids.shape[0] == 1
        self.model = model
        self.length = input_ids.shape[1]
        self.layers = list(range(len(model.model.layers))) if layers is None else list(layers)
        starts = (input_ids[0] == model.config.vision_start_token_id).nonzero(as_tuple=True)[0].tolist()
        ends = (input_ids[0] == model.config.vision_end_token_id).nonzero(as_tuple=True)[0].tolist()
        assert len(starts) == len(ends) and starts
        self.ranges = list(zip(starts, [x+1 for x in ends]))
        self.keep_rows = keep_rows
        self.rows = {}
        self.scores = {}
        self.handles = []

    def hook(self, layer_id):
        def capture(module, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            if h.shape[1] != self.length:
                return
            cp = kwargs.get("cache_position")
            if cp is not None:
                assert int(cp[0]) == 0, "Only a complete prefill is supported"
            q = module.q_proj(h[:, -1:, :]).view(1,1,-1,module.head_dim).transpose(1,2)
            k = module.k_proj(h).view(1,self.length,-1,module.head_dim).transpose(1,2)
            cos,sin = kwargs["position_embeddings"]
            sections = module.rope_scaling["mrope_section"]*2
            cos = torch.cat([part[i%3] for i,part in enumerate(cos.split(sections,dim=-1))],dim=-1).unsqueeze(1)
            sin = torch.cat([part[i%3] for i,part in enumerate(sin.split(sections,dim=-1))],dim=-1).unsqueeze(1)
            q = q*cos[:,:,-1:,:] + rotate_half(q)*sin[:,:,-1:,:]
            k = k*cos + rotate_half(k)*sin
            k = repeat_kv(k,module.num_key_value_groups)
            scores = torch.matmul(q,k.transpose(2,3))/math.sqrt(module.head_dim)
            mask = kwargs.get("attention_mask")
            if mask is not None:
                scores = scores + mask[:,:,-1:,:self.length]
            weights = scores.softmax(-1,dtype=torch.float32).to(q.dtype)[0,:,0,:]
            # Include both boundary tokens to match the official LAD range parser.
            self.scores[layer_id] = torch.stack([weights[:,a:b].sum(-1).mean() for a,b in self.ranges]).float().cpu()
            if self.keep_rows:
                self.rows[layer_id] = weights.float().cpu()
        return capture

    def __enter__(self):
        for i in self.layers:
            self.handles.append(self.model.model.layers[i].self_attn.register_forward_pre_hook(self.hook(i),with_kwargs=True))
        return self

    def __exit__(self,*exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
