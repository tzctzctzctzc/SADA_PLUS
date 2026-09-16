import json,sys,types
from pathlib import Path
import torch
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
sys.path.insert(0,str(Path("/vepfs/vlms/1/src")))
from last_query_attention import LastQueryAttention
torch.manual_seed(21)
torch.set_num_threads(4)
cfg=Qwen2_5_VLConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=4,
    num_attention_heads=4,num_key_value_heads=2,vision_start_token_id=126,vision_end_token_id=127,
    image_token_id=125,rope_scaling={"type":"default","mrope_section":[2,3,3]})
cfg._attn_implementation="eager"
m=Qwen2_5_VLModel(cfg).eval()
wrapper=types.SimpleNamespace(config=cfg,model=m)
ids=torch.randint(2,100,(1,29))
ids[0,3]=ids[0,16]=126
ids[0,10]=ids[0,23]=127
# Nonidentical temporal/height/width positions exercise multimodal rotary splitting.
positions=torch.stack([torch.arange(29),torch.arange(29)//2,torch.arange(29)%7]).unsqueeze(1)
records=[]
with torch.inference_mode():
    for masked in (False,True):
        mask=torch.ones_like(ids)
        if masked: mask[0,6:8]=0
        baseline=m(input_ids=ids,position_ids=positions,attention_mask=mask,use_cache=False,output_attentions=True)
        with LastQueryAttention(wrapper,ids,keep_rows=True) as h:
            captured=m(input_ids=ids,position_ids=positions,attention_mask=mask,use_cache=False,output_attentions=True)
        errors=[float((h.rows[i]-captured.attentions[i][0,:,-1,:]).abs().max()) for i in h.layers]
        error=float((baseline.last_hidden_state-captured.last_hidden_state).abs().max())
        records.append({"masked":masked,"row_max_errors":errors,"model_output_error":error})
result={"scope":"CPU fp32 tiny decoder, grouped-query attention, multimodal rotary positions and masked keys",
    "records":records,"passed":all(max(x["row_max_errors"])<1e-6 and x["model_output_error"]==0 for x in records)}
Path("runs/last_query_attention_check.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
assert result["passed"]
