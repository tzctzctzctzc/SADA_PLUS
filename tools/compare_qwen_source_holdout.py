"""固定源验证候选：8B/32B同帧同提示BF16比较，不查询目标数据，不启动学生训练。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
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

MODELS={'8b':('models--Qwen--Qwen3-VL-8B-Instruct','0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
        '32b':('models--Qwen--Qwen3-VL-32B-Instruct','0cfaf48183f594c314753d30a4c4974bc75f3ccb')}


def model_path(key):
    family,revision=MODELS[key]
    return Path('/vepfs/vlms/cache/huggingface/hub')/family/'snapshots'/revision


def temporal_iou(a,b):
    intersection=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    return intersection/max(1e-12,max(a[1],b[1])-min(a[0],b[0]))


def same_source_splits(left,right):
    # 子集顺序不改变视频成员；不能把同一集合的不同排列当作换了划分。
    return sorted(left)==sorted(right)


def evaluation_label(proposal,annotations):
    scores=[temporal_iou(proposal,a['segment']) for a in annotations]
    if not scores or max(scores)<.5:return None,max(scores,default=0.)
    best=max(scores);labels={int(a['label_id']) for a,s in zip(annotations,scores) if abs(s-best)<1e-9}
    return (next(iter(labels)) if len(labels)==1 and next(iter(labels))<10 else None),best


def frames_for_candidate(item,root):
    import cv2
    from PIL import Image
    clip=Path(item['clip_path'])
    if sha(clip)!=item['clip_sha256']:raise ValueError('源clip SHA改变: '+str(clip))
    cap=cv2.VideoCapture(str(clip))
    fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps<=0 or count<=0:raise ValueError('无法读取源clip: '+str(clip))
    start,end=item['proposal'];offset=item['clip_start_sec']
    timestamps=[start+(end-start)*(k+.5)/8 for k in range(8)]
    indices=[int(math.floor((t-offset)*fps)) for t in timestamps]
    if len(set(indices))!=8 or min(indices)<0 or max(indices)>=count:raise ValueError('8个真实独立帧不足: '+item['candidate_id'])
    folder=root/'frames'/item['candidate_id'];folder.mkdir(parents=True,exist_ok=False)
    paths=[];digests=[]
    for k,index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
        if not ok:raise ValueError('真实源帧解码失败: '+item['candidate_id'])
        p=folder/('%02d.png'%k);Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(p)
        paths.append(str(p));digests.append(sha(p))
    cap.release()
    return dict(candidate_id=item['candidate_id'],shift=item['shift'],video_id=item['video_id'],
        proposal_start_sec=start,proposal_end_sec=end,frame_paths=paths,frame_sha256=digests,
        clip_sha256=item['clip_sha256'],frame_indices=indices,clip_fps=fps,
        fixed_20pct=item['fixed_20pct'])


def prepare(root,baseline):
    inputs=[];labels={};sources={};membership={}
    for i in range(1,7):
        shift='S'+str(i)
        p=Path('/vepfs/vlms/rq4_source_seed6_ec7882a_1011/inputs/S1/requests.json') if i==1 else Path('/vepfs/vlms/rq4_source_seed6_b08c147_1011/inputs')/shift/'requests.json'
        old=read(p);cfg=read(baseline/'formal'/shift/'uda/config.json');db=read(cfg['dataset']['json_file'])['database']
        if old['split']!='source_validation' or not same_source_splits(old['source_subsets'],cfg['src_val_split']):raise ValueError('不是当前源验证split')
        train={v for v,d in db.items() if d['subset'].lower() in cfg['src_train_split']}
        val={v for v,d in db.items() if d['subset'].lower() in cfg['src_val_split']}
        if train&val:raise ValueError('源训练/留出视频交叉')
        candidates=[]
        for row in old['requests']:
            c=row['candidate']
            if not c['candidate_id'].endswith('-context2'):continue
            if c['video_id'] not in val:raise ValueError('候选不在源域留出集')
            cid=c['candidate_id'].rsplit('-context',1)[0]
            candidates.append(dict(candidate_id=cid,shift=shift,video_id=c['video_id'],proposal=c['proposal'],
                clip_path=row['clip_path'],clip_sha256=row['clip_sha256'],clip_start_sec=row['clip_start_sec']))
            label,overlap=evaluation_label(c['proposal'],db[c['video_id']].get('annotations',[]))
            labels[cid]=dict(label=label,maximum_gt_tiou=overlap)
        if len({x['candidate_id'] for x in candidates})!=len(candidates):raise ValueError('候选身份重复')
        chosen={x['candidate_id'] for x in sorted(candidates,key=lambda x:hashlib.sha256(('seed6:'+x['candidate_id']).encode()).hexdigest())[:len(candidates)//5]}
        for row in candidates:row['fixed_20pct']=row['candidate_id'] in chosen
        inputs+=candidates;sources[shift]=dict(requests=str(p),sha256=sha(p),source_json_sha256=sha(cfg['dataset']['json_file']))
        membership[shift]=dict(train_videos=sorted(train),validation_videos=sorted(val),candidates=len(candidates))
    save(root/'frozen_candidates.json',dict(inputs=inputs,sources=sources,membership=membership,
        selection='all existing source-validation context2 proposal identities; ignore proposal predicted class',
        target_data_used=False))
    save(root/'evaluation_labels.json',labels)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=[]
        for i,row in enumerate(pool.map(lambda x:frames_for_candidate(x,root),inputs)):
            rows.append(row)
            if (i+1)%50==0:print('FRAMES_READY',i+1,len(inputs),flush=True)
    save(root/'manifest.json',dict(schema='qwen-source-holdout-comparison-v1',candidates=rows,
        frames=8,sampling='eight proposal interior temporal midpoints from SHA-verified source clips',
        labels_not_in_inference_manifest=True,target_data_used=False))


def model_identity(key):
    path=model_path(key);idx=read(path/'model.safetensors.index.json')
    files=sorted(set(idx['weight_map'].values()))
    weights={}
    for name in files:
        p=path/name;digest=sha(p)
        if len(p.resolve().name)==64 and digest!=p.resolve().name:raise ValueError('模型权重SHA不符: '+name)
        weights[name]=digest
    return dict(path=str(path),snapshot=MODELS[key][1],weight_sha256=weights,
        processor_sha256={f:sha(path/f) for f in ['tokenizer.json','tokenizer_config.json','chat_template.json','preprocessor_config.json','video_preprocessor_config.json']})


def worker(args):
    # 必须先在首CUDA前验证父持卡锁、实际UUID、显存；不采用外部PID白名单。
    from tools.baseline_gpu import snapshot
    import fcntl
    leases=read(args.output/'active_leases.json')
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from tools.qwen_source_interface import infer_one,resolve_digit_token_ids
    from types import SimpleNamespace
    torch.set_num_threads(8)
    identity=model_identity(args.model);save(args.output/(args.model+'_model_identity.json'),identity)
    # 大权重SHA核验与CPU导入完成后，再紧邻from_pretrained复核；显存不足只等待。
    while True:
        uuids=[];enough=True;observed=[]
        for card in leases:
            record=read(card['lease'])
            if sha(card['lease'])!=card['lease_sha256'] or record['owner_pid']!=os.getppid():raise ValueError('GPU租约/父进程不符')
            with Path(record['lock']).open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:raise ValueError('父进程没有持卡锁')
            now=snapshot(card['index'])
            if now['uuid']!=card['uuid']:raise ValueError('首CUDA UUID改变')
            enough &= now['free_mib']>=record['minimum_free_mib']
            uuids.append(now['uuid']);observed.append(now)
        if os.environ['CUDA_VISIBLE_DEVICES']!=','.join(uuids):raise ValueError('可见GPU不符')
        if enough:break
        print('WAIT_FIRST_CUDA',flush=True);time.sleep(15)
    save(args.output/(args.model+'_first_cuda.json'),dict(status='passed',leases=leases,observed=observed,child_pid=os.getpid()))
    options=dict(dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    options.update(device_map='balanced',max_memory={i:'17408MiB' for i in range(len(leases))})
    model,loading=Qwen3VLForConditionalGeneration.from_pretrained(model_path(args.model),**options)
    if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('模型没有完整strict加载: '+str(loading))
    if any(str(v) in ['cpu','disk'] for v in model.hf_device_map.values()):raise ValueError('不能静默CPU/disk卸载来完成GPU比较')
    model.eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(model_path(args.model),local_files_only=True)
    runtime=dict(torch=torch,model=model,processor=processor,process_vision_info=process_vision_info,
        digit_token_ids=resolve_digit_token_ids(processor.tokenizer),device=model.device)
    config=SimpleNamespace(frame_count=8,prompt_mode='baseline',input_mode='native_video',
        min_pixels=224*224,max_pixels=448*448,score_temperature=1.,uniform_mixed_frame_grid=True)
    rows=read(args.output/'manifest.json')['candidates']
    save(args.output/(args.model+'_device_map.json'),dict(device_map=model.hf_device_map,
        visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],manifest_sha256=sha(args.output/'manifest.json')))
    with (args.output/(args.model+'_predictions.jsonl')).open('x') as stream:
        for i,row in enumerate(rows):
            for path,digest in zip(row['frame_paths'],row['frame_sha256']):
                if sha(path)!=digest:raise ValueError('固定输入帧改变')
            result=infer_one(runtime,row['frame_paths'],row['candidate_id'],config,
                sample_fps=8./(row['proposal_end_sec']-row['proposal_start_sec']))
            values=result['class_probabilities']
            if len(values)!=10 or not all(math.isfinite(x) and 0<=x<=1 for x in values) or abs(sum(values)-1)>1e-5:raise ValueError('类别概率非法')
            stream.write(json.dumps(dict(candidate_id=row['candidate_id'],class_probabilities=values,
                candidate_mass=result['candidate_mass'],prompt_sha256=result['prompt_sha256'],frame_sha256=row['frame_sha256']))+'\n');stream.flush()
            if (i+1)%20==0:print('SCORED',args.model,i+1,len(rows),flush=True)
    save(args.output/(args.model+'_inference_complete.json'),dict(status='passed',count=len(rows),
        predictions_sha256=sha(args.output/(args.model+'_predictions.jsonl')),manifest_sha256=sha(args.output/'manifest.json'),
        model_identity_sha256=sha(args.output/(args.model+'_model_identity.json'))))


def score_predictions(rows,labels,predictions):
    selected=[r for r in rows if labels[r['candidate_id']]['label'] is not None]
    cm=[[0]*10 for _ in range(10)];correct=0;nll=0.
    for r in selected:
        label=labels[r['candidate_id']]['label'];p=predictions[r['candidate_id']]['class_probabilities']
        guess=max(range(10),key=lambda c:p[c]);cm[label][guess]+=1;correct+=guess==label;nll-=math.log(max(p[label],1e-12))
    perclass=[cm[c][c]/sum(cm[c]) if sum(cm[c]) else None for c in range(10)]
    return dict(total_candidates=len(rows),gt_match_tiou_at_least_half=len(selected),
        matched_accuracy=correct/len(selected) if selected else None,
        matched_macro_accuracy=sum(x for x in perclass if x is not None)/sum(x is not None for x in perclass) if selected else None,
        matched_nll=nll/len(selected) if selected else None,per_class_accuracy=perclass,confusion_matrix=cm)


def summarize(root):
    rows=read(root/'manifest.json')['candidates'];labels=read(root/'evaluation_labels.json')
    predictions={k:{d['candidate_id']:d for d in [json.loads(x) for x in (root/(k+'_predictions.jsonl')).read_text().splitlines()]} for k in MODELS}
    expected={r['candidate_id'] for r in rows}
    for p in predictions.values():
        if set(p)!=expected:raise ValueError('模型候选集合不完整')
    for cid in expected:
        a,b=[predictions[k][cid] for k in MODELS]
        if a['prompt_sha256']!=b['prompt_sha256'] or a['frame_sha256']!=b['frame_sha256']:raise ValueError('两教师输入/提示不一致')
    result={}
    for scope in ['pooled_candidates_not_independent']+['S'+str(i) for i in range(1,7)]:
        subset=rows if scope.startswith('pooled') else [r for r in rows if r['shift']==scope]
        result[scope]={k:dict(all=score_predictions(subset,labels,p),
            fixed_20pct=score_predictions([r for r in subset if r['fixed_20pct']],labels,p)) for k,p in predictions.items()}
    save(root/'comparison.json',dict(status='passed',scope='existing finite SO-proposal source-validation pool; no target data',
        no_student_training=True,no_target_model_selection=True,results=result,
        manifest_sha256=sha(root/'manifest.json'),labels_sha256=sha(root/'evaluation_labels.json'),
        predictions_sha256={k:sha(root/(k+'_predictions.jsonl')) for k in MODELS}))


def supervisor(args):
    os.chdir(args.data_root);args.output.mkdir(parents=True,exist_ok=False)
    identity=code_identity();save(args.output/'state.json',dict(phase='preparing',pid=os.getpid(),code=identity))
    try:
        prepare(args.output,args.baseline)
    except BaseException as error:
        save(args.output/'error.json',dict(type=type(error).__name__,message=str(error),cuda_started=False))
        save(args.output/'state.json',dict(phase='failed_preparation',pid=os.getpid()))
        raise
    # 处理器完全相同才将差异归因到教师；模型架构维度允许不同。
    for name in ['tokenizer.json','tokenizer_config.json','chat_template.json','preprocessor_config.json','video_preprocessor_config.json']:
        if sha(model_path('8b')/name)!=sha(model_path('32b')/name):raise ValueError('两教师处理器协议不同: '+name)
    from tools.baseline_gpu import slot,snapshot
    active=None
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if active is not None and active.poll() is None:os.killpg(active.pid,signal.SIGTERM)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for key in ['8b','32b']:
            indices=[1] if key=='8b' else [0,1,2,3]
            with ExitStack() as held:
                while True:
                    attempt=ExitStack();cards=[]
                    try:
                        for index in indices:cards.append(attempt.enter_context(slot(index,args.data_root,args.output,20480,identity['commit'])))
                    except BlockingIOError:
                        attempt.close();save(args.output/'state.json',dict(phase='waiting_gpu',model=key,pid=os.getpid()));time.sleep(15);continue
                    held.enter_context(attempt);break
                save(args.output/'active_leases.json',cards)
                env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                    CUDA_VISIBLE_DEVICES=','.join(c['uuid'] for c in cards),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                with (args.output/(key+'_inference.log')).open('x') as log:
                    active=subprocess.Popen([sys.executable,'-B',str(Path(__file__)), '--worker','--model',key,
                        '--output',str(args.output),'--data-root',str(args.data_root)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    save(args.output/'state.json',dict(phase='inference',model=key,pid=os.getpid(),child_pid=active.pid))
                    if active.wait():raise RuntimeError('教师推理失败: '+key)
                active=None
        summarize(args.output);save(args.output/'state.json',dict(phase='complete',pid=os.getpid()))
    except BaseException as error:
        save(args.output/'error.json',dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path,default=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029'))
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--worker',action='store_true');p.add_argument('--model',choices=MODELS)
    args=p.parse_args();worker(args) if args.worker else supervisor(args)
