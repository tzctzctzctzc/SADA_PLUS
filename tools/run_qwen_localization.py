"""单卡源验证定位能力诊断；复用8B类别缓存，不修改存活Oracle或学生。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.qwen_localization import (SCHEMA,POLICY,OPTIONS,prepare,build_prompt,option_scores,
                                     combine,audit_predictions,summarize,reuse_preparation)


def choice_tokens(tokenizer):
    ids=[tokenizer.encode(x,add_special_tokens=False) for x in OPTIONS]
    if any(len(x)!=1 for x in ids) or len({x[0] for x in ids})!=len(OPTIONS):
        raise ValueError('固定定位选项必须对应独立的单token')
    return [x[0] for x in ids]


def infer_boundary(runtime,row,boundary):
    torch=runtime['torch'];prompt=build_prompt(row,boundary)
    content=[dict(type='text',text='Chronological context frames with timestamps:')]
    for i,(path,t) in enumerate(zip(row['frame_paths'],row['frame_times_sec'])):
        content += [dict(type='text',text=f'FRAME {i}: {t:.6f} seconds'),
                    dict(type='image',image=Path(path).resolve().as_uri(),
                         min_pixels=POLICY['min_pixels'],max_pixels=POLICY['max_pixels'])]
    content.append(dict(type='text',text=prompt))
    messages=[dict(role='user',content=content)]
    processor=runtime['processor']
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images,videos=runtime['process_vision_info'](messages,image_patch_size=16)
    inputs=processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt').to(runtime['model'].device)
    with torch.inference_mode():
        output=runtime['model'](**inputs,use_cache=False,logits_to_keep=1,return_dict=True)
    logits=output.logits[0,-1].float()
    selected=logits[runtime['token_ids']].cpu().tolist()
    result=option_scores(selected,float(torch.logsumexp(logits,dim=0).cpu()))
    result['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
    if result['prompt_sha256']!=row['prompt_sha256'][boundary]:
        raise ValueError('真实提示与冻结提示不符')
    return result


def worker(args):
    # 推理过程只读无GT的manifest；评价标注仅由父进程在全部评分后读取。
    from tools.baseline_gpu import first_cuda_guard
    from tools.compare_qwen_source_holdout import model_path,model_identity
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    identity=code_identity()
    if identity!=read(args.output/'plan.json')['code']:
        raise ValueError('推理代码身份改变')
    inputs=read(args.output/'inputs.json')
    if sha(args.output/'manifest.json')!=inputs['manifest_sha256']:
        raise ValueError('固定定位输入改变')
    weights=model_identity('8b');save(args.output/'model_identity.json',weights)
    processor=AutoProcessor.from_pretrained(model_path('8b'),local_files_only=True)
    token_ids=choice_tokens(processor.tokenizer)
    torch.set_num_threads(8)
    if args.shard_four:
        from tools.qwen_shard_gpu import first_cuda_guard_four,WEIGHT_BUDGET_MIB
        proof=first_cuda_guard_four(args.output)
        placement=dict(device_map='balanced',max_memory={i:f'{WEIGHT_BUDGET_MIB}MiB' for i in range(4)})
    else:
        proof=first_cuda_guard();placement=dict(device_map={'':0})
    model,loading=Qwen3VLForConditionalGeneration.from_pretrained(model_path('8b'),dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True,output_loading_info=True,**placement)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('VLM完整加载检查失败: '+str(loading))
    if any(str(x) in ('cpu','disk') for x in model.hf_device_map.values()):
        raise ValueError('不允许静默CPU/disk卸载')
    if args.shard_four and {int(str(x).replace('cuda:','')) for x in model.hf_device_map.values()}!={0,1,2,3}:
        raise ValueError('模型未实际分到四张卡')
    model.eval().requires_grad_(False)
    runtime=dict(torch=torch,processor=processor,model=model,token_ids=token_ids,process_vision_info=process_vision_info)
    rows=read(args.output/'manifest.json')['candidates'];started=time.monotonic()
    save(args.output/'device_map.json',dict(visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],device_map=model.hf_device_map,
        first_cuda_guard=proof,guard_sha256=sha(proof),choice_token_ids=token_ids))
    with (args.output/'predictions.jsonl').open('x') as stream:
        for i,row in enumerate(rows):
            for path,h in zip(row['frame_paths'],row['frame_sha256']):
                if sha(path)!=h:
                    raise ValueError('真实定位帧SHA改变')
            if row['input_status']!='ready':
                result=dict(candidate_id=row['candidate_id'],shift=row['shift'],video_id=row['video_id'],
                    class_id=row['class_id'],status='input_unavailable',reason=row['input_status'],
                    confidence=-1.,interval=None,start=None,end=None)
            else:
                start=infer_boundary(runtime,row,'START');end=infer_boundary(runtime,row,'END')
                result=combine(row,start,end)
            stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
            if i==0 or (i+1)%10==0 or i+1==len(rows):
                print('LOCALIZED',i+1,len(rows),result['status'],flush=True)
                save(args.output/'inference_progress.json',dict(scored=i+1,total=len(rows),
                    seconds=time.monotonic()-started,
                    peak_reserved_mib_by_logical_gpu={str(j):torch.cuda.max_memory_reserved(j)/2**20
                        for j in range(4 if args.shard_four else 1)}))
    save(args.output/'inference_complete.json',dict(status='passed',count=len(rows),
        manifest_sha256=sha(args.output/'manifest.json'),predictions_sha256=sha(args.output/'predictions.jsonl'),
        model_identity_sha256=sha(args.output/'model_identity.json'),guard_sha256=sha(proof)))


def supervisor(args):
    from tools.baseline_gpu import slot
    args.output=args.output.resolve();args.previous=args.previous.resolve();args.baseline=args.baseline.resolve()
    identity=code_identity();args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'plan.json',dict(schema=SCHEMA,policy=POLICY,code=identity,gpus=args.gpus,
        minimum_free_mib=12288 if args.shard_four else 24576,
        shard_four=args.shard_four,teacher_only=True,student_training=False))
    active=None
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        save(args.output/'state.json',dict(phase='preparing_source_context_frames',pid=os.getpid()))
        if args.prepared_root:
            prepared_identity=read(args.prepared_root/'plan.json')['code']
            # 仅资源调度变化允许复用原clean代码的CPU准备；提示/评分/GT处理实现逐字节不变。
            unchanged=['tools/qwen_localization.py','tools/qwen_source_interface.py','tools/compare_qwen_source_holdout.py']
            changed=subprocess.check_output(['git','diff','--name-only',prepared_identity['commit'],identity['commit'],'--',*unchanged],cwd=CODE,text=True)
            if changed.strip():raise ValueError('分片部署改变了冻结的数据/提示/评分实现')
            manifest=reuse_preparation(args.output,args.prepared_root.resolve(),prepared_identity)
            save(args.output/'sharding_provenance.json',dict(prepared_code=prepared_identity,current_code=identity,
                unchanged_files={p:sha(CODE/p) for p in unchanged},same_method_different_gpu_placement=True))
        else:manifest=prepare(args.output,args.previous,args.baseline)
        print('CPU_INPUTS_READY',len(manifest['candidates']),manifest['input_status_counts'],flush=True)
        if args.prepare_only:
            save(args.output/'state.json',dict(phase='prepared_only',pid=os.getpid()));return
        held=None
        while held is None:
            if args.shard_four:
                from tools.qwen_shard_gpu import acquire_four
                found=acquire_four(args.gpus,args.data_root,args.output,identity['commit'])
                if found:held,cards=found
            else:
                for index in args.gpus:
                    ctx=slot(index,args.data_root,args.output,24576,identity['commit'])
                    try:card=ctx.__enter__()
                    except BlockingIOError:continue
                    held=ctx;cards=[card];break
            if held is None:
                save(args.output/'state.json',dict(phase='waiting_four_shared_gpus' if args.shard_four else 'waiting_one_shared_gpu',pid=os.getpid()))
                time.sleep(15)
        try:
            save(args.output/'active_leases.json',cards)
            env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                CUDA_VISIBLE_DEVICES=','.join(c['uuid'] for c in cards),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            if args.shard_four:
                env.update(QWEN_SHARD_LEASES=str(args.output/'active_leases.json'),
                           QWEN_SHARD_LEASES_SHA256=sha(args.output/'active_leases.json'))
            else:env.update(SADA_BASELINE_LEASE=cards[0]['lease'],SADA_BASELINE_LEASE_SHA256=cards[0]['lease_sha256'])
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--output',str(args.output)]
            if args.shard_four:command+=['--shard-four']
            with (args.output/'inference.log').open('x') as stream:
                active=subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                save(args.output/'state.json',dict(phase='teacher_localization',pid=os.getpid(),child_pid=active.pid,
                    gpus=[c['index'] for c in cards],uuids=[c['uuid'] for c in cards]))
                if active.wait():
                    raise RuntimeError('源域定位推理失败，保留全部已评分记录')
            active=None
        finally:
            # 用户停止时先退出自有CUDA子组，之后才能释放物理卡锁。
            if active is not None and active.poll() is None:
                os.killpg(active.pid,signal.SIGTERM)
                try:active.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
            held.__exit__(None,None,None)
        proof=audit_predictions(args.output);save(args.output/'prediction_audit.json',proof)
        summarize(args.output)
        save(args.output/'final_audit.json',dict(status='passed',schema=SCHEMA,count=len(manifest['candidates']),
            artifacts={n:sha(args.output/n) for n in ['plan.json','inputs.json','manifest.json','evaluation_labels.json',
                'model_identity.json','device_map.json','inference_complete.json','prediction_audit.json',
                'predictions.jsonl','top20_selection.json','comparison.json']},
            preparation_reuse_sha256=sha(args.output/'preparation_reuse.json') if args.prepared_root else None,
            sharding_provenance_sha256=sha(args.output/'sharding_provenance.json') if args.prepared_root else None))
        save(args.output/'state.json',dict(phase='complete',pid=os.getpid()))
    except BaseException as error:
        save(args.output/'error.json',dict(type=type(error).__name__,message=str(error)))
        save(args.output/'state.json',dict(phase='failed',pid=os.getpid()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path,default=Path('/vepfs/vlms/qwen8b32b_source_0adb413_1030'))
    p.add_argument('--baseline',type=Path,default=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029'))
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--prepared-root',type=Path)
    p.add_argument('--shard-four',action='store_true')
    p.add_argument('--gpus',type=lambda x:[int(g) for g in x.split(',')],default=[0,1,2,3])
    p.add_argument('--worker',action='store_true');p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or not set(args.gpus)<={0,1,2,3}:
        p.error('物理GPU集合非法')
    if args.shard_four and set(args.gpus)!={0,1,2,3}:p.error('四卡分片必须使用物理0/1/2/3')
    worker(args) if args.worker else supervisor(args)
