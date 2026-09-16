import sys,json
from pathlib import Path
import torch
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
sys.path.insert(0,'src')
from cache_paths import snapshot,hybrid_cache
torch.manual_seed(61);torch.set_num_threads(4)
cfg=Qwen2_5_VLConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=3,
 num_attention_heads=4,num_key_value_heads=2,rope_scaling={'type':'default','mrope_section':[2,3,3]})
cfg._attn_implementation='sdpa';m=Qwen2_5_VLModel(cfg).eval()
ids=torch.randint(2,100,(1,31));n=24;starts=torch.tensor([3,14])
pos=torch.stack([torch.arange(31),torch.arange(31)//2,torch.arange(31)%7]).unsqueeze(1)
with torch.inference_mode():
 full=m(input_ids=ids,position_ids=pos,use_cache=False).last_hidden_state
 p=m(input_ids=ids[:,:n],position_ids=pos[:,:,:n],use_cache=True)
 clean=snapshot(p.past_key_values)
 alt=tuple((k+.25,v-.25) for k,v in clean)
 original=tuple((k.clone(),v.clone()) for k,v in clean)
 for part in ['baseline','full','start_only','other_prefix']:
  cache=hybrid_cache(clean,alt,starts,part)
  mask=torch.zeros(n,dtype=torch.bool);mask[starts]=True
  for i,(k,v) in enumerate(cache.to_legacy_cache()):
   for out,base,changed in [(k,clean[i][0],alt[i][0]),(v,clean[i][1],alt[i][1])]:
    expected=base.clone() if part in ['baseline','start_only'] else changed.clone()
    if part=='start_only':expected[:,:,mask,:]=changed[:,:,mask,:]
    if part=='other_prefix':expected[:,:,mask,:]=base[:,:,mask,:]
    assert torch.equal(out,expected)
  if part=='baseline':
   suffix=m(input_ids=ids[:,n:],position_ids=pos[:,:,n:],attention_mask=torch.ones_like(ids),
    cache_position=torch.arange(n,31),past_key_values=cache,use_cache=True).last_hidden_state
 for a,b in zip(clean,original):
  assert all(torch.equal(x,y) for x,y in zip(a,b))
 err=float((full[:,n:]-suffix).abs().max())
report={'passed':err<1e-6,'fp32_split_prefill_max_error':err,'hybrid_slots_exact':True,'source_snapshots_unchanged':True}
Path('results/P013_cache_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));assert report['passed']
