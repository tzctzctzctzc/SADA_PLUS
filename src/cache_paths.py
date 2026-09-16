"""Readout-path interventions on immutable prefill KV snapshots."""
import torch
from transformers.cache_utils import DynamicCache

def snapshot(cache):
    return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache.to_legacy_cache())

def hybrid_cache(clean, altered, start_positions, part):
    assert part in ('baseline','full','start_only','other_prefix')
    source=altered if part in ('full','other_prefix') else clean
    result=[]
    for i,(k,v) in enumerate(source):
        kk,vv=k.clone(),v.clone()
        if part in ('start_only','other_prefix'):
            other=altered[i] if part=='start_only' else clean[i]
            assert other[0].shape==kk.shape and other[1].shape==vv.shape
            kk[:,:,start_positions,:]=other[0][:,:,start_positions,:]
            vv[:,:,start_positions,:]=other[1][:,:,start_positions,:]
        result.append((kk,vv))
    return DynamicCache.from_legacy_cache(tuple(result))
