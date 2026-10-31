"""固定GT片段上的B定义/C源示例配对诊断；复用A，不训练学生。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_qwen_gt_oracle import MODEL,VLM_PY,LABELS,OPTIONS,LETTERS,safe_row,option_order,metrics
from tools.qwen_prompt_examples import messages,prompt


def task_key(arm,shift,candidate):return '%s:%s:%s'%(arm,shift,candidate)


def shard(key):return int(hashlib.sha256(key.encode()).hexdigest()[:16],16)%4


def jobs(manifest,labels):
    lookup={x['candidate_id']:safe_row(x) for x in manifest['candidates']}
    result=[dict(arm='B',shift='ALL',row=x) for x in lookup.values()]
    seen=set()
    for label in labels:
        key=(label['shift'],label['candidate_id'])
        if key in seen:raise ValueError('方向内重复候选')
        seen.add(key);result.append(dict(arm='C',shift=key[0],row=lookup[key[1]]))
    for row in result:row['job_id']=task_key(row['arm'],row['shift'],row['row']['candidate_id'])
    return sorted(result,key=lambda x:(x['arm'],x['shift'],x['job_id']))


def infer(runtime,row,bank):
    import torch
    processor,model,vision=runtime;conversation=messages(row,bank)
    text=processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
    images,videos,kwargs=vision(conversation,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
    videos,metadata=zip(*videos)
    if len(videos)!=len(bank)+1:raise ValueError('源参考视频与query数量不符')
    inputs=processor(text=[text],images=images,videos=list(videos),video_metadata=list(metadata),padding=True,
        return_tensors='pt',do_resize=False,**kwargs).to(model.device)
    grid=inputs['video_grid_thw']
    if grid.shape[0]!=len(bank)+1:raise ValueError('processor丢失/重排视频数量')
    ids=[processor.tokenizer.encode(x,add_special_tokens=False) for x in LETTERS]
    if any(len(x)!=1 for x in ids) or len({x[0] for x in ids})!=14:raise ValueError('选项非单token')
    with torch.inference_mode():logits=model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True).logits[0,-1].float()
    values=logits[torch.tensor([x[0] for x in ids],device=logits.device)];probability=values.softmax(0).cpu().tolist()
    if not all(math.isfinite(x) for x in probability):raise ValueError('预测非有限')
    order=option_order(row['candidate_id']);result={label:probability[order.index(label)] for label in OPTIONS};guess=max(result,key=result.get)
    return dict(candidate_id=row['candidate_id'],prediction=guess,confidence=result[guess],probabilities=result,
        candidate_mass=float((values.logsumexp(0)-logits.logsumexp(0)).exp().cpu()),
        prompt_sha256=hashlib.sha256(prompt(row['candidate_id']).encode()).hexdigest(),
        video_count=len(videos),video_grids=grid.cpu().tolist(),input_tokens=inputs['input_ids'].shape[-1])


def verify_parent_inputs(root):
    inputs=read(root/'inputs.json')
    for p,h in inputs['artifacts'].items():
        if sha(p)!=h:raise ValueError('冻结输入改变: '+p)
    return inputs


def worker(args):
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from tools.baseline_gpu import first_cuda_guard
    root=args.output;plan=read(root/'plan.json')
    if plan['code']!=code_identity():raise ValueError('代码改变')
    inputs=verify_parent_inputs(root);banks=read(inputs['banks'])['banks']
    for file,stat in inputs['model_files'].items():
        current=(MODEL/file).stat()
        if [current.st_size,current.st_mtime_ns]!=stat:raise ValueError('已校验模型文件改变')
    selected=[x for x in read(root/'jobs.json') if shard(x['job_id'])==args.shard]
    for job in selected:safe_row(job['row'])
    guard=first_cuda_guard()
    model,loading=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',
        device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('模型加载不完整')
    model.eval().requires_grad_(False);processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    verified=set();accepted={};path=root/('predictions_shard%d.jsonl'%args.shard)
    with path.open('x') as stream:
        for i,job in enumerate(selected):
            row=job['row'];bank=[] if job['arm']=='B' else banks[job['shift']]
            if bank and len(bank)!=10:raise ValueError('C须每类一个示例')
            for item in [row]+bank:
                for p,h in zip(item['frame_paths'],item['frame_sha256']):
                    if p not in verified:
                        if sha(p)!=h:raise ValueError('视频帧改变')
                        verified.add(p)
            started=time.monotonic();result=infer((processor,model,process_vision_info),row,bank)
            result.update(job_id=job['job_id'],arm=job['arm'],shift=job['shift'],seconds=time.monotonic()-started,
                bank_sha256=None if not bank else hashlib.sha256(json.dumps(bank,sort_keys=True).encode()).hexdigest())
            if job['arm'] not in accepted:
                accepted[job['arm']]=dict(status='passed',job_id=job['job_id'],video_count=result['video_count'],
                    grids=result['video_grids'],input_tokens=result['input_tokens'],peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    target_class_in_input=False,no_model_update=True)
                save(root/('acceptance_shard%d.json'%args.shard),dict(guard=str(guard),guard_sha256=sha(guard),arms=accepted))
            stream.write(json.dumps(result)+'\n');stream.flush()
            if (i+1)%10==0:print('SCORED',i+1,len(selected),job['arm'],job['shift'],flush=True)
    save(root/('complete_shard%d.json'%args.shard),dict(status='passed',count=len(selected),prediction_sha256=sha(path),
        jobs_sha256=sha(root/'jobs.json'),inputs_sha256=sha(root/'inputs.json'),guard=str(guard),guard_sha256=sha(guard),
        acceptance_sha256=sha(root/('acceptance_shard%d.json'%args.shard))))


def summarize(root):
    inputs=verify_parent_inputs(root);old=Path(inputs['previous_root']);labels=read(old/'evaluation_labels.json')
    previous=[json.loads(x) for x in (old/'predictions.jsonl').read_text().splitlines()]
    a={x['candidate_id']:x for x in previous};rows=[json.loads(x) for x in (root/'predictions.jsonl').read_text().splitlines()]
    if len(rows)!=len({x['job_id'] for x in rows}):raise ValueError('重复评分')
    expected={x['job_id'] for x in read(root/'jobs.json')}
    if {x['job_id'] for x in rows}!=expected:raise ValueError('B/C评分不完整')
    results={}
    for shift in ('S1','S2','S3','S4','S5','S6'):
        chosen=[x for x in labels if x['shift']==shift];arms={}
        for arm in ('A','B','C'):
            p=a if arm=='A' else {x['candidate_id']:x for x in rows if x['arm']==arm and (arm=='B' or x['shift']==shift)}
            better=worse=0
            for x in chosen:
                key=x['candidate_id'];before=a[key]['prediction']==LABELS[x['label']];after=p[key]['prediction']==LABELS[x['label']]
                better+=after and not before;worse+=before and not after
            arms[arm]=dict(all=metrics(chosen,p),no_other_known_overlap=metrics([x for x in chosen if not x['other_known_overlap']],p),
                per_class={name:metrics([x for x in chosen if x['label']==k],p) for k,name in enumerate(LABELS)},
                corrected_vs_A=better,regressed_vs_A=worse)
        results[shift]=arms
    save(root/'final_audit.json',dict(status='passed',schema='qwen8-definition-source-examples-BC-v1',results=results,
        macro_accuracy={arm:sum(r[arm]['all']['accuracy'] for r in results.values())/6 for arm in ('A','B','C')},
        unique_B=645,C_direction_queries=1200,A_reused=True,target_gt_in_prompts=False,no_student_training=True,
        exploratory_after_target_error_analysis=True,independent_across_shifts=False,
        artifacts={n:sha(root/n) for n in ('plan.json','inputs.json','jobs.json','predictions.jsonl','inference_complete.json')}))


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);identity=code_identity();previous=args.previous_root.resolve();banks=args.banks.resolve()
    old=read(previous/'final_audit.json')
    if old['status']!='passed':raise ValueError('A组未完整')
    for name,digest in old['artifacts'].items():
        if sha(previous/name)!=digest:raise ValueError('A组工件改变')
    selected=jobs(read(previous/'manifest.json'),read(previous/'evaluation_labels.json'))
    if len([x for x in selected if x['arm']=='B'])!=645 or len([x for x in selected if x['arm']=='C'])!=1200:raise ValueError('不是原六方向精确集合')
    bankdata=read(banks)
    if set(bankdata['banks'])!={'S%d'%s for s in range(1,7)} or any([x['label'] for x in bank]!=list(LABELS) for bank in bankdata['banks'].values()):raise ValueError('示例未冻结/缺类别')
    weights=read(previous/'inference_complete.json')['shards']['0']['model_weights']
    for name,digest in weights.items():
        if sha(MODEL/name)!=digest:raise ValueError('8B模型版本改变')
    save(root/'inputs.json',dict(previous_root=str(previous),banks=str(banks),
        artifacts={str(p):sha(p) for p in [previous/'final_audit.json',previous/'manifest.json',previous/'evaluation_labels.json',previous/'predictions.jsonl',banks]},
        model_weights=weights,model_files={n:[(MODEL/n).stat().st_size,(MODEL/n).stat().st_mtime_ns] for n in weights}))
    save(root/'jobs.json',selected)
    save(root/'plan.json',dict(code=identity,schema='qwen8-definition-source-examples-BC-v1',arms=['B','C'],reuse_A=True,
        physical_gpus=[0,1,2,3],minimum_free_mib=22528,batch=1,precision='bf16',query_frames=8,query_pixels=[224*224,448*448],
        source_examples_per_class=1,source_frames=8,source_pixels=[224*224,224*224],
        prompt_example=prompt('0'*64),inputs_sha256=sha(root/'inputs.json'),jobs_sha256=sha(root/'jobs.json')))
    from tools.baseline_gpu import slot
    active={};pending=list(range(4));completed=[]
    try:
        while pending or active:
            for s,t in list(active.items()):
                if t['process'].poll() is None:continue
                if t['process'].returncode:raise RuntimeError('B/C分片失败: '+str(s))
                t['context'].__exit__(None,None,None);completed.append(s);del active[s]
            for s in list(pending):
                for gpu in range(4):
                    context=slot(gpu,args.data_root,root,22528,identity['commit'])
                    try:lease=context.__enter__()
                    except BlockingIOError:continue
                    break
                else:break
                env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                    SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                try:
                    with (root/('shard%d.log'%s)).open('x') as log:
                        process=subprocess.Popen([VLM_PY,'-u','-B',str(Path(__file__).resolve()),'--worker','--shard',str(s),'--output',str(root)],
                            cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:context.__exit__(None,None,None);raise
                active[s]=dict(process=process,context=context,gpu=gpu);pending.remove(s)
            save(root/'state.json',dict(pid=os.getpid(),phase='inference_or_waiting_gpu',pending=pending,completed=completed,
                active={str(s):dict(pid=t['process'].pid,gpu=t['gpu']) for s,t in active.items()}));time.sleep(3)
        lines=[];records={}
        for s in range(4):
            record=read(root/('complete_shard%d.json'%s));path=root/('predictions_shard%d.jsonl'%s)
            if record['status']!='passed' or sha(path)!=record['prediction_sha256'] or record['jobs_sha256']!=sha(root/'jobs.json') or record['inputs_sha256']!=sha(root/'inputs.json'):raise ValueError('分片SHA不符')
            records[str(s)]=record;lines+=path.read_text().splitlines()
        (root/'predictions.jsonl').write_text('\n'.join(lines)+'\n');save(root/'inference_complete.json',records);summarize(root)
        save(root/'state.json',dict(pid=os.getpid(),phase='complete'))
    except BaseException as error:
        save(root/'error.json',dict(type=type(error).__name__,message=str(error)));save(root/'state.json',dict(pid=os.getpid(),phase='failed'));raise
    finally:
        for t in active.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in active.values():t['process'].wait();t['context'].__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--banks',type=Path)
    p.add_argument('--previous-root',type=Path,default=Path('/vepfs/vlms/qwen8_gt_oracle6_d0591c6_1031'))
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--worker',action='store_true');p.add_argument('--shard',type=int)
    args=p.parse_args()
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.worker:worker(args)
    else:main(args)
