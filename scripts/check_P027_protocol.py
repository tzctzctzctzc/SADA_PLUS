"""CPU preflight for official reranker chat template and actual ImageCoDe image inputs."""
import hashlib, importlib.util, json
from pathlib import Path
from transformers import AutoConfig, AutoProcessor

root=Path("/vepfs/vlms/1")
cfg=json.loads((root/"configs/P027_qwen3_imagecode.json").read_text())
source_meta=json.loads((root/"literature/Qwen3-VL-Reranker_source.json").read_text())
source=root/"third_party/Qwen3-VL-Embedding"/source_meta["file"]
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_meta["sha256"]
spec=importlib.util.spec_from_file_location("official",source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
obj=module.Qwen3VLReranker.__new__(module.Qwen3VLReranker)
obj.processor=AutoProcessor.from_pretrained(root/"models/Qwen3-VL-Reranker-2B",local_files_only=True,padding_side="left")
for k,v in {"max_length":10240,"total_pixels":7864320,"fps":1,"max_frames":64,"default_instruction":cfg["instruction"]}.items():
    setattr(obj,k,v)
c=AutoConfig.from_pretrained(root/"models/Qwen3-VL-Reranker-2B",local_files_only=True)
data=[r for r in json.loads((root/"runs/P023_data/records.json").read_text()) if r["split"]=="evaluation"]
checks=[]
for setting in cfg["settings"]:
    obj.min_pixels=setting["min_pixels"];obj.max_pixels=setting["max_pixels"]
    for row in data[:2]:
        pair=obj.format_mm_instruction(query_text=row["caption"],doc_image=str(root/row["images"][0]["path"]))
        inputs=obj.tokenize([pair]);grid=inputs["image_grid_thw"]
        actual=int((inputs["input_ids"]==c.image_token_id).sum());expected=int(grid.prod())//4
        assert actual==expected>0
        checks.append({"id":row["id"],"setting":setting["name"],"grid":grid.tolist(),
                       "image_tokens":actual,"total_tokens":inputs["input_ids"].shape[1]})
report={"complete":True,"checks":checks,
        "scope":"Official formatting and image tokenization only; no model scores or GPU compatibility tested.",
        "template_sha256":hashlib.sha256((root/"models/Qwen3-VL-Reranker-2B/chat_template.jinja").read_bytes()).hexdigest()}
(root/"results/P027_protocol_cpu.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
