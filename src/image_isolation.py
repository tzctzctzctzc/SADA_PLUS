"""Block cross-image content paths while keeping question readout intact."""
from contextlib import contextmanager
import torch

@contextmanager
def isolate_images(model,input_ids,skeleton=False,block_cross_image=True):
    ids=input_ids[0];length=input_ids.shape[1]
    starts=(ids==model.config.vision_start_token_id).nonzero(as_tuple=True)[0].tolist()
    ends=(ids==model.config.vision_end_token_id).nonzero(as_tuple=True)[0].tolist()
    assert len(starts)==len(ends)>1
    # Include labels/gap text preceding later images, so they cannot relay prior pixels.
    groups=[(starts[0],ends[0]+1)]+[(ends[i-1]+1,ends[i]+1) for i in range(1,len(starts))]
    edited=None;handles=[]
    def hook(module,args,kwargs):
        nonlocal edited
        h=kwargs.get('hidden_states',args[0] if args else None)
        if h.shape[1]!=length:return
        cp=kwargs.get('cache_position')
        assert cp is None or int(cp[0])==0
        if edited is None:
            mask=kwargs.get('attention_mask')
            if mask is None:
                mask=torch.full((length,length),torch.finfo(h.dtype).min,device=h.device,dtype=h.dtype).triu(1)[None,None]
            assert mask.ndim==4
            edited=mask.clone()
            for j,(a,b) in (enumerate(groups) if block_cross_image else []):
                for i in range(j):
                    c,d=groups[i]
                    edited[:,:,a:b,c:d]=torch.finfo(h.dtype).min
                    if skeleton:
                        edited[:,:,starts[j],starts[i]]=mask[:,:,starts[j],starts[i]]
        kwargs=dict(kwargs);kwargs['attention_mask']=edited
        return args,kwargs
    try:
        for layer in model.model.layers:
            handles.append(layer.self_attn.register_forward_pre_hook(hook,with_kwargs=True))
        yield
    finally:
        for handle in handles:handle.remove()
