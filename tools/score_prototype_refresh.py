"""四卡分片承载同一Qwen8B；沿用原native-video十类单token读出。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.prototype_refresh import MODEL,POLICY,SHIFTS


def checkpoint_identity():
    old=read('/vepfs/vlms/rq3_teachers_434f630_0928/qwen3_vl_8b/S1/interface_audit.json')
    actual={name:sha(MODEL/name) for name in old['checkpoint_sha256']}
    if actual!=old['checkpoint_sha256']:raise ValueError('Qwen8B权重/processor与原教师不同')
    return dict(path=str(MODEL),checkpoint_sha256=actual,
        original_interface_sha256=sha('/vepfs/vlms/rq3_teachers_434f630_0928/qwen3_vl_8b/S1/interface_audit.json'),
        scorer_sha256=sha(CODE/'tools/qwen_source_interface.py'),
        runtime={name:importlib.metadata.version(name) for name in ('torch','transformers','qwen-vl-utils')})


def worker(root):
    from tools.qwen_shard_gpu import first_cuda_guard_four,WEIGHT_BUDGET_MIB
    from tools.qwen_source_interface import infer_one,resolve_digit_token_ids
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from tools.refresh_frames import pixel_hash
    if code_identity()!=read(root/'plan.json')['code']:raise ValueError('评分代码身份改变')
    audit=read(root/'candidate_audit.json')
    identity=checkpoint_identity();teacher=root/'teacher';save(teacher/'model_identity.json',identity)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    tokens=resolve_digit_token_ids(processor.tokenizer)
    torch.set_num_threads(8)
    proof=first_cuda_guard_four(teacher)
    model,info=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True,output_loading_info=True,
        device_map='balanced',max_memory={i:f'{WEIGHT_BUDGET_MIB}MiB' for i in range(4)})
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Qwen模型加载不完整')
    used={str(v).replace('cuda:','') for v in model.hf_device_map.values()}
    if used!={'0','1','2','3'}:raise ValueError('必须真实四卡分片且禁止CPU/disk卸载')
    model.eval().requires_grad_(False)
    save(teacher/'device_map.json',dict(map={k:str(v) for k,v in model.hf_device_map.items()},
        uuids=os.environ['CUDA_VISIBLE_DEVICES']))
    runtime=dict(torch=torch,model=model,processor=processor,process_vision_info=process_vision_info,
        digit_token_ids=tokens,device=model.device)
    config=SimpleNamespace(frame_count=8,prompt_mode='baseline',input_mode='native_video',
        min_pixels=POLICY['min_pixels'],max_pixels=POLICY['max_pixels'],score_temperature=1.,
        uniform_mixed_frame_grid=True)
    started=time.monotonic();count=0
    with (teacher/'predictions.jsonl').open('x') as stream:
        for s in SHIFTS:
            path=Path(audit['shifts'][s]['manifest'])
            if sha(path)!=audit['shifts'][s]['sha256']:raise ValueError('评分清单SHA改变')
            for row in read(path)['candidates']:
                paths=[path.parent/'frames'/row['candidate_id']/(str(i)+'.png') for i in row['frames']]
                hashes=[]
                for p in paths:
                    with Image.open(p) as img:hashes.append(pixel_hash(np.asarray(img.convert('RGB'))))
                if hashes!=row['frame_sha256']:raise ValueError('真实输入帧像素不符')
                result=infer_one(runtime,paths,row['candidate_id'],config,
                    sample_fps=8/(row['proposal_end_sec']-row['proposal_start_sec']))
                # 原接口只返回按类别重排的logits；还原选项顺序，供CPU独立精确复算。
                from tools.qwen_source_interface import LABEL_TO_ID
                digits=[result['class_logits'][LABEL_TO_ID[label]] for label in result['digit_to_label']]
                out=dict(shift=s,candidate_id=row['candidate_id'],frame_sha256=hashes,
                    class_probabilities=result['class_probabilities'],class_id=result['class_id'],
                    digit_logits=digits,prompt_sha256=result['prompt_sha256'],candidate_mass=result['candidate_mass'])
                stream.write(json.dumps(out,allow_nan=False)+'\n');stream.flush();count+=1
                if count==1 or count%20==0:
                    save(teacher/'progress.json',dict(scored=count,total=1200,seconds=time.monotonic()-started,
                        peak_reserved_mib={str(i):torch.cuda.max_memory_reserved(i)/2**20 for i in range(4)}))
                    print('TEACHER_SCORED',count,1200,flush=True)
    if count!=1200:raise ValueError('评分数量不是1200')
    save(teacher/'inference_complete.json',dict(status='passed',count=count,
        code=code_identity(),candidate_audit_sha256=sha(root/'candidate_audit.json'),
        predictions_sha256=sha(teacher/'predictions.jsonl'),model_identity_sha256=sha(teacher/'model_identity.json'),
        device_map_sha256=sha(teacher/'device_map.json'),guard_sha256=sha(proof),seconds=time.monotonic()-started,
        target_gt_used=False,policy=POLICY))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    worker(p.parse_args().root.resolve())
