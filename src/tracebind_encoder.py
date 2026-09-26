"""TraceBind 共用的真实编码、训练参数和检查点路径。"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils.vision_process import process_vision_info
from torch.nn import functional as F

from tracebind import TraceBindHead, compile_role_keys


def last_token(hidden, mask):
    last = mask.shape[1] - mask.flip(1).argmax(1) - 1
    return hidden[torch.arange(len(hidden), device=hidden.device), last].float()


class TraceBindEncoder:
    def __init__(self, config, image_entries):
        self.config, self.images = config, image_entries
        mc = config["model"]
        model_path = Path(mc["path"])
        assert (model_path / ".cache/huggingface/download/config.json.metadata").read_text().splitlines()[0] == mc["revision"]
        source = model_path / "scripts/qwen3_vl_embedding.py"
        spec = importlib.util.spec_from_file_location("tracebind_official_embedding", source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.official_source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        self.processor = module.Qwen3VLProcessor.from_pretrained(model_path, local_files_only=True, padding_side="right")
        self.model = module.Qwen3VLForEmbedding.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16,
                                                               low_cpu_mem_usage=True, attn_implementation=mc["attn_implementation"]).to("cuda")
        self.model.config.use_cache = False
        self.head = TraceBindHead(self.model.config.text_config.hidden_size, mc).to("cuda")
        self.adapted = False

    def adapt(self):
        assert not self.adapted
        tc = self.config["training"]
        self.model = get_peft_model(self.model, LoraConfig(r=tc["lora_rank"], lora_alpha=tc["lora_alpha"], lora_dropout=0.0,
                                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type=TaskType.FEATURE_EXTRACTION))
        for name, param in self.model.named_parameters():
            if ".visual.merger." in name:
                param.requires_grad = tc["train_visual_merger"]
                if param.requires_grad:
                    # 小学习率不能直接写回 BF16 主参数，否则 LayerNorm 等权重可能有梯度却不更新。
                    param.data = param.data.float()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.adapted = True

    def forward_hidden(self, inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**inputs, use_cache=False).last_hidden_state

    def optimizer(self):
        assert self.adapted
        tc = self.config["training"]
        lora, merger = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora.append(param)
            elif ".visual.merger." in name:
                merger.append(param)
            else:
                raise AssertionError("未经方案声明的可训练参数: " + name)
        assert lora and merger
        return torch.optim.AdamW([{"params": lora, "lr": tc["learning_rate_lora"]},
                                 {"params": merger, "lr": tc["learning_rate_visual_merger"]},
                                 {"params": self.head.parameters(), "lr": tc["learning_rate_head"]}], weight_decay=0.0)

    def trainable_parameters(self):
        values = {"model." + name: param for name, param in self.model.named_parameters() if param.requires_grad}
        values.update({"head." + name: param for name, param in self.head.named_parameters()})
        return values

    def image(self, product):
        entry = self.images[product]
        return {"type": "image", "image": "file://" + str(Path(entry["path"]).resolve()),
                "min_pixels": self.config["model"]["min_pixels"], "max_pixels": self.config["model"]["max_pixels"]}

    @staticmethod
    def conversation(content, system):
        return [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": content}]

    def prepare(self, conversations):
        text = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        pics, videos, kwargs = process_vision_info(conversations, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True)
        assert videos is None
        inputs = self.processor(text=text, images=pics, padding=True, truncation=False, do_resize=False, return_tensors="pt", **kwargs)
        assert inputs["input_ids"].shape[1] <= self.config["model"]["max_length"], "禁止静默截断参考或角色"
        return {key: value.to("cuda") for key, value in inputs.items()}

    def encode_images(self, products, local=True):
        conversations = [self.conversation([self.image(product)], "Represent the candidate product image for retrieval.") for product in products]
        inputs = self.prepare(conversations)
        hidden = self.forward_hidden(inputs)
        globals = F.normalize(last_token(hidden, inputs["attention_mask"]), dim=-1)
        if not local:
            return globals
        assert len(inputs["image_grid_thw"]) == len(products)
        patches, banks, shapes = [], [], []
        merge = self.model.config.vision_config.spatial_merge_size
        for i, grid in enumerate(inputs["image_grid_thw"]):
            t, height, width = grid.tolist()
            assert t == 1 and height % merge == width % merge == 0
            values = hidden[i, inputs["input_ids"][i] == self.model.config.image_token_id]
            height, width = height // merge, width // merge
            assert len(values) == height * width
            patches.append(values.float()); shapes.append((height, width))
            banks.append(self.head.evidence(values, height, width))
        return globals, torch.stack(banks), patches, shapes

    def encode_query(self, row, variant="original", order=None):
        n = len(row["source_ids"])
        order = list(range(n)) if order is None else list(order)
        assert sorted(order) == list(range(n))
        instruction = row["instructions"][variant]
        # 先替换成不会撞名的占位符，再按新序号展开。
        for old in range(n):
            instruction = instruction.replace(f"Product {old+1}", f"SOURCE_SLOT_{old}")
        for new, old in enumerate(order):
            instruction = instruction.replace(f"SOURCE_SLOT_{old}", f"Product {new+1}")
        content = []
        for new, old in enumerate(order):
            content += [{"type": "text", "text": f"Product {new+1}:\n"}, self.image(row["source_ids"][old])]
        content.append({"type": "text", "text": instruction})
        inputs = self.prepare([self.conversation(content, "Represent the multi-reference product retrieval query. Respect the attribute assigned to each numbered product.")])
        output = self.forward_hidden(inputs)
        return F.normalize(last_token(output, inputs["attention_mask"]), dim=-1)[0]

    def encode_probes(self, row):
        roles = compile_role_keys(row["instructions"]["original"], len(row["source_ids"]))
        if "role_keys" in row:
            assert roles == row["role_keys"], "文本与监督标签的来源角色映射不一致"
        conversations = [self.conversation([self.image(source), {"type": "text", "text": f"Represent only the visible {role} of this reference product for matching a candidate."}], "Extract the specified visual role from the reference image.")
                         for source in row["source_ids"] for role in roles]
        inputs = self.prepare(conversations)
        output = self.forward_hidden(inputs)
        n = len(row["source_ids"])
        return self.head.probes(last_token(output, inputs["attention_mask"])).reshape(n, n, -1)

    def score_row(self, row, arm="tracebind", candidates=None):
        products = [row["positive_target_id"], row["negative_target_id"]] if candidates is None else candidates
        g, banks, _, _ = self.encode_images(products)
        probes = self.encode_probes(row)
        order = list(reversed(range(len(row["source_ids"]))))
        queries = [self.encode_query(row), self.encode_query(row, "paraphrase"), self.encode_query(row, order=order)]
        scores = [self.head.score(q, g, probes, banks, arm=arm) for q in queries]
        return scores, (queries, g, probes, banks)

    def save(self, path, optimizer=None, provenance=None):
        assert self.adapted
        torch.save({"config": self.config, "official_source_sha": self.official_source_sha,
                    "model_state": {name: param.detach().cpu() for name, param in self.model.named_parameters() if param.requires_grad},
                    "head_state": self.head.state_dict(), "optimizer": None if optimizer is None else optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "provenance": provenance or {}}, path)

    def load(self, path, optimizer=None):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["config"]["model"] == self.config["model"] and saved["official_source_sha"] == self.official_source_sha
        assert saved["config"]["training"]["lora_rank"] == self.config["training"]["lora_rank"]
        assert saved["config"]["training"]["lora_alpha"] == self.config["training"]["lora_alpha"]
        if not self.adapted:
            self.adapt()
        expected = {name for name, param in self.model.named_parameters() if param.requires_grad}
        assert set(saved["model_state"]) == expected
        result = self.model.load_state_dict(saved["model_state"], strict=False)
        assert not result.unexpected_keys
        self.head.load_state_dict(saved["head_state"], strict=True)
        if optimizer is not None:
            assert saved["config"]["training"] == self.config["training"], "恢复优化器时不得静默变更训练目标或配置"
            assert saved["optimizer"] is not None
            optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"])
        return saved["provenance"]
