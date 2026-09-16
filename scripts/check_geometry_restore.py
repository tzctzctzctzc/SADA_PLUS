import json,sys,types
from pathlib import Path
import torch
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
sys.path.insert(0,'src')
from boundary_restore import CaptureBoundaries,restore_boundaries,boundary_positions
torch.manual_seed(53);torch.set_num_threads(4)
cfg=Qwen2_5_VLConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=3,
 num_attention_heads=4,num_key_value_heads=2,vision_start_token_id=126,vision_end_token_id=127,
 image_token_id=125,rope_scaling={'type':'default','mrope_section':[2,3,3]})
cfg._attn_implementation='sdpa'
m=Qwen2_5_VLModel(cfg).eval();wrapper=types.SimpleNamespace(config=cfg,model=m)
def make(length,starts,ends):
 ids=torch.randint(2,100,(1,length));ids[0,starts]=126;ids[0,ends]=127
 pos=torch.stack([torch.arange(length),torch.arange(length)//2,torch.arange(length)%7]).unsqueeze(1)
 return {'input_ids':ids,'position_ids':pos,'attention_mask':torch.ones_like(ids),'use_cache':False}
a=make(29,[3,16],[10,23]);b=make(41,[4,25],[14,37])
with torch.inference_mode():
 with CaptureBoundaries(wrapper,b['input_ids']) as donor:m(**b)
 before=m(**a).last_hidden_state
 with restore_boundaries(wrapper,a['input_ids'],donor.states,'start',1):
  with CaptureBoundaries(wrapper,a['input_ids']) as observed:m(**a)
 errors=[float((observed.states[i][0]-v[0,[1,0]]).abs().max()) for i,v in donor.states.items()]
 after=m(**a).last_hidden_state
report={'different_sequence_lengths':[29,41],'shifted_start_capture_max_error':max(errors),
 'after_context_max_error':float((before-after).abs().max()),
 'hooks_remaining':sum(len(l.input_layernorm._forward_hooks) for l in m.layers)}
report['passed']=report['shifted_start_capture_max_error']==0 and report['after_context_max_error']==0 and report['hooks_remaining']==0
Path('results/P011_restore_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report));assert report['passed']
