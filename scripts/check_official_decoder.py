"""CPU regression: compare the author's decoder with the hook port, including KV decoding."""
import ast
import copy
import hashlib
import json
import sys
import types
from pathlib import Path
import torch
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
sys.path.insert(0, str(Path("/vepfs/vlms/1/src")))
from interventions import delimiter_intervention

path=Path("third_party/delim_modeling_qwen2_5_vl.py")
tree=ast.parse(path.read_text())
# Plotting imports are unused by the decoder and are omitted from this CPU check.
filtered=[]
for node in tree.body:
    if isinstance(node,ast.Import) and any(n.name.split(".")[0] in ("matplotlib","seaborn") for n in node.names):
        continue
    if isinstance(node,ast.ImportFrom) and (node.module or "").split(".")[0] in ("matplotlib","seaborn"):
        continue
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=="matplotlib.use":
        continue
    filtered.append(node)
tree.body=filtered
name="transformers.models.qwen2_5_vl.official_dts_audit"
module=types.ModuleType(name)
module.__package__="transformers.models.qwen2_5_vl"
module.__file__=str(path.resolve())
sys.modules[name]=module
exec(compile(tree,str(path),"exec"),module.__dict__)
torch.manual_seed(19)
torch.set_num_threads(4)
cfg=Qwen2_5_VLConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=6,
    num_attention_heads=4,num_key_value_heads=2,vision_start_token_id=126,vision_end_token_id=127,
    image_token_id=125,rope_scaling={"type":"default","mrope_section":[2,3,3]})
cfg._attn_implementation="sdpa"
gold_cfg=copy.deepcopy(cfg)
gold_cfg._attn_implementation="eager"
gold=module.Qwen2_5_VLModel(gold_cfg).eval()
port=Qwen2_5_VLModel(copy.deepcopy(cfg)).eval()
port.load_state_dict(gold.state_dict())
wrapper=types.SimpleNamespace(model=port,config=cfg)
records=[]
with torch.inference_mode():
    for n_images in (0,1,2):
        ids=torch.randint(2,100,(1,24))
        for i in range(n_images):
            ids[0,3+i*9]=126;ids[0,8+i*9]=127
        for mode in ("baseline","scale"):
            kwargs={"temp_input_ids":ids,"delim_scaling":mode=="scale","scale":8.0,"select_layer":[0,1,2,3]}
            a=gold(input_ids=ids,use_cache=True,**kwargs)
            with delimiter_intervention(wrapper,ids,mode):
                b=port(input_ids=ids,use_cache=True)
                errors=[(a.last_hidden_state-b.last_hidden_state).abs().max().item()]
                for token in (7,23,16):
                    nxt=torch.tensor([[token]])
                    a=gold(input_ids=nxt,past_key_values=a.past_key_values,use_cache=True,
                        **(kwargs|{"temp_input_ids":nxt}))
                    b=port(input_ids=nxt,past_key_values=b.past_key_values,use_cache=True)
                    errors.append((a.last_hidden_state-b.last_hidden_state).abs().max().item())
            records.append({"n_images":n_images,"mode":mode,"prefill_and_decode_errors":errors})
report={"official_source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    "official_repository_commit":json.loads(Path("third_party/delim_tree.json").read_text())["sha"],
    "scope":"Tiny random decoder, CPU fp32, official eager versus port SDPA, 0/1/2 images, prefill and 3 KV-cache decode steps. Vision encoder and FlashAttention not covered.",
    "records":records,"passed":all(max(r["prefill_and_decode_errors"])<1e-5 for r in records)}
Path("runs/official_decoder_audit.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report))
assert report["passed"]
