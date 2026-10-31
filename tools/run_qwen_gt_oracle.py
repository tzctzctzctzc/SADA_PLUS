"""六方向目标训练GT区间的8B语义oracle；标签只用于选样和离线评价，不训练学生。"""
import argparse
from bisect import bisect_left
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.qwen_source_interface import LABELS,ACTION_PHRASES,build_native_video_messages,mixed_frame_grid
from tools.refresh_frames import exact_rgb_frames,pixel_hash,video_paths_by_id

OPTIONS=LABELS+('OTHER_ACTION','MIXED_ACTIONS','NO_ACTION','UNCERTAIN')
LETTERS=tuple('ABCDEFGHIJKLMN')
MODEL=Path('/vepfs/vlms/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
BASE=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/formal')
VLM_PY='/vepfs/vlms/envs/rq1-cu121/bin/python'


def shard_id(key):
    return int(key[:16],16)%4


def cid(video,start,end):
    return hashlib.sha256(('gt-oracle-v1:%s:%.9f:%.9f'%(video,start,end)).encode()).hexdigest()


def balanced(rows,maximum=200):
    buckets={k:sorted([r for r in rows if r['label']==k],key=lambda r:r['sample_id']) for k in range(10)}
    chosen=[]
    while len(chosen)<maximum and any(buckets.values()):
        for k in range(10):
            if buckets[k] and len(chosen)<maximum:chosen.append(buckets[k].pop(0))
    return chosen


def option_order(key):
    labels=list(OPTIONS);random.Random(int(key[:16],16)).shuffle(labels);return labels


def prompt(key):
    definitions=dict(zip(LABELS,ACTION_PHRASES))
    definitions.update(OTHER_ACTION='a hand-object action outside the listed ten categories',
        MIXED_ACTIONS='multiple different actions are visible without one clearly dominant action',
        NO_ACTION='no visible hand-object action; objects or persistent states alone do not establish an action',
        UNCERTAIN='the visible frames do not provide enough evidence to identify the action')
    return ('The video contains eight chronological frames from one interval. Judge only the visible evidence. '
        'Identify the dominant hand-object action, following the main object over time. '
        'Distinguish picking up from placing down, opening from closing, and turning on from turning off. '
        'Do not infer an unseen intention from an object alone. If no single listed action is supported, '
        'use the appropriate rejection option.\n'+
        '\n'.join('%s: %s (%s)'%(a,b,definitions[b]) for a,b in zip(LETTERS,option_order(key)))+
        '\nReply immediately with exactly one letter from A to N, without explanation.')


def safe_row(row):
    keys=('candidate_id','video_id','start','end','frame_paths','frame_sha256','actual_times','unique_frames')
    if set(row)!=set(keys):raise ValueError('推理清单不允许GT类别等额外字段')
    if len(row['frame_paths'])!=8 or len(row['frame_sha256'])!=8:raise ValueError('必须固定8帧')
    return row


def inside_indices(times,start,end):
    lo=bisect_left(times,start);hi=bisect_left(times,end)
    if lo>=hi:return None
    chosen=[]
    for k in range(8):
        t=start+(end-start)*(k+.5)/8;i=bisect_left(times,t)
        candidates={min(max(i,lo),hi-1),min(max(i-1,lo),hi-1)}
        chosen.append(min(candidates,key=lambda j:(abs(times[j]-t),j)))
    return chosen


def prepare_video(video,items,args):
    import av
    from PIL import Image
    path=args.video_paths[video]
    with av.open(str(path)) as container:
        stream=container.streams.video[0];timestamps=sorted(x.pts for x in container.demux(stream) if x.pts is not None)
        if len(timestamps)!=stream.frames or len(set(timestamps))!=len(timestamps):raise ValueError('视频显示索引不完整: '+video)
        origin=stream.start_time if stream.start_time is not None else timestamps[0]
        times=[float((x-origin)*stream.time_base) for x in timestamps]
    indices={};unavailable=[]
    for row in items:
        start,end=row['start'],row['end'];chosen=inside_indices(times,start,end)
        if chosen is None:unavailable.append(row['candidate_id']);continue
        indices[row['candidate_id']]=chosen
    wanted=sorted({i for rr in indices.values() for i in rr})
    frames=exact_rgb_frames(path,wanted,args.recovery_root) if wanted else {}
    results=[]
    for row in items:
        key=row['candidate_id']
        if key not in indices:continue
        folder=args.output/'frames'/key;folder.mkdir(parents=True,exist_ok=False)
        paths=[];hashes=[]
        for k,i in enumerate(indices[key]):
            p=folder/('%02d.png'%k);Image.fromarray(frames[i]).save(p);paths.append(str(p));hashes.append(sha(p))
        payload=dict(row,frame_paths=paths,frame_sha256=hashes,actual_times=[times[i] for i in indices[key]],unique_frames=len(set(indices[key])))
        results.append(safe_row(payload))
    return results,unavailable,dict(video=video,path=str(path),raw_sha256=sha(path))


def prepare(args):
    evaluation=[];queries={};sources={}
    for s in range(1,7):
        shift='S%d'%s;cfg=read(BASE/shift/'uda/config.json');annotation=Path(cfg['baseline']['annotation_for_offline_evaluation']);db=read(annotation)['database']
        subsets=set(cfg['tgt_train_split']);rows=[]
        for video,record in db.items():
            if record['subset'].lower() not in subsets:continue
            annotations=record.get('annotations',[])
            for index,a in enumerate(annotations):
                label=int(a['label_id']);start,end=map(float,a['segment'])
                if not 0<=label<10:continue
                if not math.isfinite(start+end) or not 0<=start<end:raise ValueError('GT时间区间非法')
                key=cid(video,start,end);sample_id=hashlib.sha256(('seed123:'+video+':'+str(index)).encode()).hexdigest()
                overlap=any(0<=int(b['label_id'])<10 and int(b['label_id'])!=label and
                    min(end,b['segment'][1])>max(start,b['segment'][0]) for b in annotations)
                rows.append(dict(sample_id=sample_id,candidate_id=key,video_id=video,start=start,end=end,
                    label=label,other_known_overlap=overlap,shift=shift))
        chosen=balanced(rows);evaluation+=chosen
        for row in chosen:queries[row['candidate_id']]={k:row[k] for k in ('candidate_id','video_id','start','end')}
        sources[shift]=dict(annotation=str(annotation),annotation_sha256=sha(annotation),target_subsets=sorted(subsets),
            eligible=len(rows),selected=len(chosen),classes=dict(Counter(x['label'] for x in chosen)))
    save(args.output/'evaluation_labels.json',evaluation)
    save(args.output/'selection.json',dict(seed=123,maximum_per_shift=200,balancing='round-robin-ten-known-classes-fixed-SHA-order',sources=sources,
        gt_used_for_candidate_intervals=True,gt_class_in_prompt=False,no_student_training=True))
    args.video_paths=video_paths_by_id(args.video_root);grouped=defaultdict(list)
    for row in queries.values():grouped[row['video_id']].append(row)
    missing=set(grouped)-set(args.video_paths)
    if missing:raise ValueError('缺少原始视频: '+str(sorted(missing)))
    results=[];unavailable=[];videos=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(prepare_video,v,rr,args) for v,rr in sorted(grouped.items())]
        for f in futures:
            rows,bad,source=f.result();results+=rows;unavailable+=bad;videos.append(source)
            print('VIDEOS_READY',len(videos),len(grouped),flush=True)
    results.sort(key=lambda x:x['candidate_id'])
    save(args.output/'manifest.json',dict(schema='target-GT-interval-oracle-v1',candidates=results,unavailable_no_in_interval_frame=unavailable,
        input_mode='native_video',frames=8,min_pixels=224*224,max_pixels=448*448,precision='bf16',temperature=1.,
        timing='nearest-real-PTS-inside-exact-GT-interval; repeated-frames-recorded',videos=videos))


def infer(runtime,row):
    import torch
    row=safe_row(row);key=row['candidate_id'];text_prompt=prompt(key)
    messages=build_native_video_messages(row['frame_paths'],text_prompt,224*224,448*448,8/(row['end']-row['start']))
    messages[0]['content'][0].update(mixed_frame_grid(row['frame_paths'],224*224,448*448))
    processor,model,vision=runtime
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images,videos,kwargs=vision(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
    videos,metadata=zip(*videos)
    inputs=processor(text=[text],images=images,videos=list(videos),video_metadata=list(metadata),padding=True,
        return_tensors='pt',do_resize=False,**kwargs).to(model.device)
    ids=[processor.tokenizer.encode(x,add_special_tokens=False) for x in LETTERS]
    if any(len(x)!=1 for x in ids) or len({x[0] for x in ids})!=14:raise ValueError('选项不能单token读出')
    with torch.inference_mode():logits=model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True).logits[0,-1].float()
    values=logits[torch.tensor([x[0] for x in ids],device=logits.device)];probabilities=values.softmax(0).cpu().tolist();order=option_order(key)
    result={label:probabilities[order.index(label)] for label in OPTIONS};guess=max(result,key=result.get)
    return dict(candidate_id=key,prediction=guess,probabilities=result,confidence=result[guess],
        candidate_mass=float((values.logsumexp(0)-logits.logsumexp(0)).exp().cpu()),prompt_sha256=hashlib.sha256(text_prompt.encode()).hexdigest())


def worker(args):
    from tools.baseline_gpu import first_cuda_guard
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    if read(args.output/'plan.json')['code']!=code_identity():raise ValueError('不是冻结代码')
    index=read(MODEL/'model.safetensors.index.json');identity={f:sha(MODEL/f) for f in sorted(set(index['weight_map'].values()))}
    proof=first_cuda_guard()
    model,loading=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',
        device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('8B加载不完整')
    model.eval().requires_grad_(False);processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    rows=[x for x in read(args.output/'manifest.json')['candidates'] if shard_id(x['candidate_id'])==args.shard]
    with (args.output/('predictions_shard%d.jsonl'%args.shard)).open('x') as stream:
        for i,row in enumerate(rows):
            for p,h in zip(row['frame_paths'],row['frame_sha256']):
                if sha(p)!=h:raise ValueError('输入帧改变')
            result=infer((processor,model,process_vision_info),row);stream.write(json.dumps(result)+'\n');stream.flush()
            if (i+1)%20==0:print('SCORED',i+1,len(rows),flush=True)
    save(args.output/('inference_complete_shard%d.json'%args.shard),dict(status='passed',shard=args.shard,count=len(rows),guard=str(proof),guard_sha256=sha(proof),
        model_weights=identity,manifest_sha256=sha(args.output/'manifest.json'),predictions_sha256=sha(args.output/('predictions_shard%d.jsonl'%args.shard))))


def metrics(rows,predictions):
    available=[x for x in rows if x['candidate_id'] in predictions];accepted=[];correct=0;bins=defaultdict(list)
    for row in available:
        p=predictions[row['candidate_id']];ok=p['prediction']==LABELS[row['label']];correct+=ok
        if p['prediction'] in LABELS:accepted.append(ok)
        bins[min(9,int(p['confidence']*10))].append((p['confidence'],ok))
    n=len(available)
    return dict(selected=len(rows),scored=n,unavailable=len(rows)-n,correct=correct,accuracy=correct/n if n else None,
        known_prediction_coverage=len(accepted)/n if n else None,accepted_accuracy=sum(accepted)/len(accepted) if accepted else None,
        ece=sum(abs(sum(x for x,y in v)/len(v)-sum(y for x,y in v)/len(v))*len(v) for v in bins.values())/n if n else None,
        predictions=dict(Counter(predictions[x['candidate_id']]['prediction'] for x in available)))


def summarize(root):
    rows=read(root/'evaluation_labels.json');predictions={x['candidate_id']:x for x in map(json.loads,(root/'predictions.jsonl').read_text().splitlines())}
    expected={x['candidate_id'] for x in read(root/'manifest.json')['candidates']}
    if set(predictions)!=expected:raise ValueError('评分集合不完整')
    results={}
    for s in range(1,7):
        selected=[x for x in rows if x['shift']=='S%d'%s]
        results['S%d'%s]=dict(all=metrics(selected,predictions),no_other_known_overlap=metrics([x for x in selected if not x['other_known_overlap']],predictions),
            per_class={LABELS[k]:metrics([x for x in selected if x['label']==k],predictions) for k in range(10)})
    save(root/'final_audit.json',dict(status='passed',schema='qwen8-target-GT-interval-oracle-v1',results=results,
        unique_scored=len(predictions),sample_count=len(rows),independent_across_shifts=False,no_student_training=True,
        interpretation='GT-interval recognition diagnostic, not student-mAP upper bound; new abstention prompt differs from old forced-ten-way protocol',
        artifacts={n:sha(root/n) for n in ('plan.json','selection.json','evaluation_labels.json','manifest.json','predictions.jsonl','inference_complete.json')}))


def main(args):
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    identity=code_identity();save(args.output/'plan.json',dict(code=identity,model='Qwen3-VL-8B-Instruct',model_revision=MODEL.name,
        scope='S1-S6-target-training-GT-intervals-only',maximum_per_shift=200,student_training=False,options=OPTIONS,
        prompt_example=prompt('0'*64),physical_gpus=[0,1,2,3],minimum_free_mib=22528,
        memory_basis='previous single-card 8B max_memory 17408MiB plus 5120MiB inference headroom'))
    active={}
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        save(args.output/'state.json',dict(phase='CPU_preparation',pid=os.getpid()));prepare(args)
        from tools.baseline_gpu import slot
        save(args.output/'state.json',dict(phase='waiting_gpu',pid=os.getpid()))
        pending=list(range(4));completed=[]
        while pending or active:
            for shard,t in list(active.items()):
                if t['process'].poll() is None:continue
                if t['process'].returncode:raise RuntimeError('8B oracle分片失败: '+str(shard))
                record=read(args.output/('inference_complete_shard%d.json'%shard))
                if record['status']!='passed':raise ValueError('分片未全审')
                t['context'].__exit__(None,None,None);del active[shard];completed.append(shard)
            for shard in list(pending):
                found=None
                for index in (0,1,2,3):
                    candidate=slot(index,args.data_root,args.output,22528,identity['commit'])
                    try:lease=candidate.__enter__()
                    except BlockingIOError:continue
                    found=candidate,lease;break
                if found is None:break
                context,lease=found
                env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                    SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                try:
                    with (args.output/('inference_shard%d.log'%shard)).open('x') as log:
                        child=subprocess.Popen([VLM_PY,'-u','-B',str(Path(__file__).resolve()),'--worker','--shard',str(shard),'--output',str(args.output)],
                            cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:context.__exit__(None,None,None);raise
                active[shard]=dict(process=child,context=context,gpu=lease['index']);pending.remove(shard)
            save(args.output/'state.json',dict(phase='inference_or_waiting_gpu',pid=os.getpid(),pending=pending,completed=completed,
                active={str(k):dict(pid=t['process'].pid,gpu=t['gpu']) for k,t in active.items()}));time.sleep(3)
        records={};lines=[]
        for shard in range(4):
            record=read(args.output/('inference_complete_shard%d.json'%shard));p=args.output/('predictions_shard%d.jsonl'%shard)
            if record['manifest_sha256']!=sha(args.output/'manifest.json') or record['predictions_sha256']!=sha(p):raise ValueError('分片SHA改变')
            lines+=p.read_text().splitlines();records[str(shard)]=record
        keys=[json.loads(x)['candidate_id'] for x in lines]
        if len(keys)!=len(set(keys)):raise ValueError('分片重复评分')
        (args.output/'predictions.jsonl').write_text('\n'.join(lines)+'\n')
        save(args.output/'inference_complete.json',dict(status='passed',shards=records,count=len(lines)));summarize(args.output)
        save(args.output/'state.json',dict(phase='complete',pid=os.getpid()))
    except BaseException as error:
        save(args.output/'error.json',dict(type=type(error).__name__,message=str(error)));save(args.output/'state.json',dict(phase='failed',pid=os.getpid()));raise
    finally:
        for t in active.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in active.values():t['process'].wait();t['context'].__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--shard',type=int,choices=range(4))
    p.add_argument('--video-root',type=Path,default=Path('/vepfs/vlms/data/EPIC_KITCHENS/videos'))
    p.add_argument('--recovery-root',type=Path,default=Path('/vepfs/vlms/data/EPIC_KITCHENS/rq2_frame_recovery_qwen_0924'))
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));args=p.parse_args()
    if args.worker:worker(args)
    else:main(args)
