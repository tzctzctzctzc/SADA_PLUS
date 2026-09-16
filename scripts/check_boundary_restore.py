import json,sys,types
from pathlib import Path
import torch
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
sys.path.insert(0,'src')
from boundary_restore import CaptureBoundaries,restore_boundaries,boundary_positions
from interventions import delimiter_intervention
torch.manual_seed(43);torch.set_num_threads(4)
cfg=Qwen2_5_VLConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=6,
 num_attention_heads=4,num_key_value_heads=2,vision_start_token_id=126,vision_end_token_id=127,
 image_token_id=125,rope_scaling={'type':'default','mrope_section':[2,3,3]})
cfg._attn_implementation='sdpa'
m=Qwen2_5_VLModel(cfg).eval();wrapper=types.SimpleNamespace(config=cfg,model=m)
ids=torch.randint(2,100,(1,29));ids[0,3]=ids[0,16]=126;ids[0,10]=ids[0,23]=127
positions=torch.stack([torch.arange(29),torch.arange(29)//2,torch.arange(29)%7]).unsqueeze(1)
kwargs={'input_ids':ids,'position_ids':positions,'attention_mask':torch.ones_like(ids),'use_cache':False}
with torch.inference_mode():
 with CaptureBoundaries(wrapper,ids) as cap:base=m(**kwargs).last_hidden_state
 with delimiter_intervention(wrapper,ids,'scale'):scaled=m(**kwargs).last_hidden_state
 with delimiter_intervention(wrapper,ids,'scale'),restore_boundaries(wrapper,ids,cap.states):restored=m(**kwargs).last_hidden_state
 mask=torch.ones(ids.shape[1],dtype=torch.bool);mask[boundary_positions(wrapper,ids).flatten()]=False
 full_error=float((base[:,mask]-restored[:,mask]).abs().max())
 change=float((base[:,mask]-scaled[:,mask]).abs().max())
 with restore_boundaries(wrapper,ids,cap.states,shift=1):rotated=m(**kwargs).last_hidden_state
 after=m(**kwargs).last_hidden_state
 report={'scope':'CPU fp32 tiny decoder, mixed rotary axes; restore both normalized boundaries should recover every non-boundary state',
  'clean_both_nonboundary_max_error':full_error,'scale_nonboundary_max_change':change,
  'rotated_state_effect':float((base[:,-1]-rotated[:,-1]).abs().max()),
  'after_context_max_error':float((base-after).abs().max()),
  'hooks_remaining':sum(len(layer.input_layernorm._forward_hooks) for layer in m.layers),
  'passed':full_error<1e-6 and change>1e-6 and torch.equal(base,after)}
Path('results/P009_restore_hook_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report));assert report['passed']
