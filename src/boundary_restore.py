"""Capture or restore normalized image boundaries before each decoder attention."""
import torch
from contextlib import contextmanager

def boundary_positions(model,input_ids):
    start=(input_ids[0]==model.config.vision_start_token_id).nonzero(as_tuple=True)[0]
    end=(input_ids[0]==model.config.vision_end_token_id).nonzero(as_tuple=True)[0]
    assert input_ids.shape[0]==1 and len(start)==len(end) and len(start)>1
    return torch.stack([start,end])

class CaptureBoundaries:
    def __init__(self,model,input_ids):
        self.model=model;self.positions=boundary_positions(model,input_ids)
        self.length=input_ids.shape[1];self.states={};self.handles=[]
    def hook(self,i):
        def capture(module,args,output):
            if output.shape[1]==self.length:
                self.states[i]=output[0,self.positions,:].detach().clone()
        return capture
    def __enter__(self):
        for i,layer in enumerate(self.model.model.layers):
            self.handles.append(layer.input_layernorm.register_forward_hook(self.hook(i)))
        return self
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.handles.clear()

@contextmanager
def restore_boundaries(model,input_ids,states,kind='both',shift=0):
    positions=boundary_positions(model,input_ids);length=input_ids.shape[1]
    kinds=[0,1] if kind=='both' else [0] if kind=='start' else [1]
    assert kind in ('both','start','end')
    n=positions.shape[1];order=(torch.arange(n,device=input_ids.device)+shift)%n
    handles=[]
    def make_hook(i):
        def restore(module,args,output):
            if output.shape[1]!=length:return output
            result=output.clone()
            source=states[i].to(device=output.device,dtype=output.dtype)
            assert source.shape==(2,n,output.shape[-1])
            for k in kinds:result[0,positions[k],:]=source[k,order,:]
            return result
        return restore
    try:
        for i,layer in enumerate(model.model.layers):
            handles.append(layer.input_layernorm.register_forward_hook(make_hook(i)))
        yield
    finally:
        for h in handles:h.remove()
