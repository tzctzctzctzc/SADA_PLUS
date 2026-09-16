import json,sys,types
from pathlib import Path
import torch
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
sys.path.insert(0,'src')
from image_isolation import isolate_images
torch.manual_seed(71);torch.set_num_threads(4)
cfg=Qwen2_5_VLConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=5,
 num_attention_heads=4,num_key_value_heads=2,vision_start_token_id=126,vision_end_token_id=127,
 image_token_id=125,rope_scaling={'type':'default','mrope_section':[2,3,3]})
cfg._attn_implementation='sdpa'
m=Qwen2_5_VLModel(cfg).eval();wrapper=types.SimpleNamespace(config=cfg,model=m)
ids=torch.randint(2,100,(1,35));ids[0,[3,20]]=126;ids[0,[12,29]]=127
other=ids.clone();other[0,4:12]=torch.randint(2,100,(8,))
pos=torch.arange(35).expand(3,1,35)
def run(tokens):return m(input_ids=tokens,position_ids=pos,use_cache=False).last_hidden_state
report={}
with torch.inference_mode():
 a=run(ids);b=run(other)
 report['full_later_image_change']=float((a[:,13:30]-b[:,13:30]).abs().max())
 with isolate_images(wrapper,ids,block_cross_image=False):
  report['explicit_causal_mask_max_error']=float((run(ids)-a).abs().max())
 for skeleton in [False,True]:
  with isolate_images(wrapper,ids,skeleton):
   a=run(ids);b=run(other)
  report['skeleton' if skeleton else 'isolated']={
   'later_label_and_image_max_change':float((a[:,13:30]-b[:,13:30]).abs().max()),
   'question_readout_change':float((a[:,30:]-b[:,30:]).abs().max())}
report['hooks_remaining']=sum(len(l.self_attn._forward_pre_hooks) for l in m.layers)
report['passed']=report['explicit_causal_mask_max_error']<1e-5 and report['full_later_image_change']>1e-6 and all(report[k]['later_label_and_image_max_change']==0 and report[k]['question_readout_change']>1e-6 for k in ['isolated','skeleton']) and report['hooks_remaining']==0
Path('results/P014_isolation_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report));assert report['passed']
