"""单GPU原生区间定位8/16fps有限诊断；不训练、不自动扩实验。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import sys

from tools.rq4_protocol import digest,file_sha,validate_candidate
from tools.rq4_source_acceptance import save,weights,guard_teacher_cuda,timeple_prompt_variant
from tools.run_rq4_teacher_queries import BASE,MODELS,read,clean_head,media_info,frame_times
from tools.rq4_teacher_queries import match_gt
from tools.rq4_native_density import POLICY,select_requests,actual_times,evaluate,decision

CODE=Path(__file__).resolve().parents[1]


def code_files():
    return {p:file_sha(CODE/'tools'/p) for p in ('rq4_native_density.py','run_rq4_native_density.py',
        'rq4_teachers.py','rq4_protocol.py','rq4_source_acceptance.py')}


def prepare(args):
    clean_head(args.expected_head)
    root=args.root;root.mkdir(parents=True,exist_ok=False)
    old=BASE/'rq4_source_seed6_ec7882a_1011/inputs/S1/requests.json'
    excluded={r['candidate']['video_id'] for r in read(old)['requests']}
    paths=[BASE/'rq4_source_seed6_b08c147_1011/inputs'/s/'requests.json' for s in POLICY['shifts']]
    manifests=[read(p) for p in paths]
    chosen=select_requests(manifests,excluded)
    annotations=BASE/'SADA_PLUS/data/ek100_slowfast/annotations/epic_kitchens_100_verb_detailed_split.json'
    database=read(annotations)['database']
    baseline_paths={s:BASE/('sada_unified_f9ac276_1009_gpu3' if s=='S2' else 'sada_unified_f9ac276_1009_gpu2')/s/'uda_preflight.json' for s in POLICY['shifts']}
    cfgs={s:read(p) for s,p in baseline_paths.items()}
    requests=[];audit={}
    for request in chosen:
        c=validate_candidate(request['candidate']);shift=c['candidate_id'].split('-')[0];cfg=cfgs[shift]
        subset=database[c['video_id']]['subset'].lower()
        assert subset in {s.lower() for s in cfg['src_val_split']}
        assert subset not in {s.lower() for key in ('src_train_split','tgt_train_split') for s in cfg[key]}
        assert request['clip_start_sec']==c['window'][0] and request['clip_end_sec']==c['window'][1]
        assert file_sha(request['clip_path'])==request['clip_sha256']
        info=media_info(request['clip_path']);ts=frame_times(request['clip_path'])
        assert len(ts)==int(info['streams'][0]['nb_read_frames'])
        assert abs(float(info['format']['duration'])-(c['window'][1]-c['window'][0]))<.1
        times=[float(t) for t in ts];assert times==sorted(times)
        audit[c['candidate_id']]=dict(media=info,frame_timestamps_sha256=digest(ts),
            first_frame_time=times[0],last_frame_time=times[-1],proposal_length=c['proposal'][1]-c['proposal'][0])
        conditions=list(POLICY['conditions'])
        if int(digest(c['candidate_id'])[:4],16)%2:conditions.reverse()
        for condition in conditions:
            requests.append(dict(request_id=c['candidate_id']+'-'+condition,condition=condition,request=request))
    save(root/'requests.json',dict(schema='rq4-native-density-diagnostic-v1',policy=POLICY,
        requests=requests,code_files=code_files(),formal_rq4=False))
    # 成员归属可用于隔离审计；精确边界仅在请求冻结后写入独立评分文件。
    truth={c['candidate']['candidate_id']:match_gt(c['candidate'],database[c['candidate']['video_id']]['annotations']) for c in chosen}
    save(root/'scoring_only.json',truth)
    save(root/'preflight.json',dict(status='passed',code_commit=args.expected_head,code_files=code_files(),
        request_sha256=file_sha(root/'requests.json'),scoring_sha256=file_sha(root/'scoring_only.json'),
        source_manifests={str(p):file_sha(p) for p in paths},excluded_S1=sorted(excluded),
        exclusion_manifest_sha256=file_sha(old),annotations_sha256=file_sha(annotations),
        baseline_configs={str(p):file_sha(p) for p in baseline_paths.values()},media=audit,
        candidates=len(chosen),videos=len({r['candidate']['video_id'] for r in chosen}),
        independence='S1 mechanism videos excluded; S2/S3 previously used in source calibration, exploratory not untouched test',
        selection='SHA20 videos with >=2 proposals, then SHA2/video; no GT/teacher outcome selection'))
    print('NATIVE_DENSITY_PREPARED',len(chosen),len(requests),flush=True)


def check(root,head):
    clean_head(head);m=read(root/'requests.json');p=read(root/'preflight.json')
    assert p['status']=='passed' and p['code_commit']==head and m['policy']==POLICY
    assert p['code_files']==m['code_files']==code_files()
    assert file_sha(root/'requests.json')==p['request_sha256']
    assert len(m['requests'])==80 and len({r['request_id'] for r in m['requests']})==80
    seen=set()
    for r in m['requests']:
        validate_candidate(r['request']['candidate'])
        assert set(r)=={'request_id','condition','request'} and r['condition'] in POLICY['conditions']
        path=r['request']['clip_path']
        if path not in seen:assert file_sha(path)==r['request']['clip_sha256'];seen.add(path)
    return m


def video_options(condition):
    return dict(POLICY['conditions'][condition],min_pixels=32*32,max_pixels=480*480,total_pixels=400*480*480)


def audit_inputs(args):
    """实际官方CPU采样；预览来自模型采样帧，不用GT挑帧。"""
    from qwen_vl_utils import process_vision_info
    import numpy as np
    from PIL import Image,ImageDraw
    m=check(args.root,args.expected_head);rows={};preview={};preview_videos=[];spatial={}
    for r in m['requests']:
        q=r['request'];c=q['candidate'];cid=c['candidate_id']
        if c['video_id'] not in preview_videos and len(preview_videos)<3:preview_videos.append(c['video_id'])
        messages=[dict(role='user',content=[dict(type='video',video=Path(q['clip_path']).as_uri(),**video_options(r['condition']))])]
        _,videos,_=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
        tensor,metadata=videos[0];times=actual_times(dict(video_metadata=[metadata]))
        shape=list(tensor.shape)
        if cid in spatial:assert spatial[cid]==shape[-2:]
        spatial[cid]=shape[-2:]
        assert len(times)==shape[0] and times[-1]<=c['window'][1]-c['window'][0]+.1
        rows[r['request_id']]=dict(times=times,shape=shape,metadata=metadata,request_sha256=digest(r))
        # 每个固定预览视频仅取首候选，六列为proposal起止附近；两行分别8/16fps。
        if c['video_id'] in preview_videos:
            entry=preview.setdefault(c['video_id'],dict(candidate_id=cid,conditions={}))
            if entry['candidate_id']==cid:
                selected=[]
                for endpoint in c['proposal']:
                    for offset in (-.25,0,.25):
                        t=endpoint-c['window'][0]+offset;i=min(range(len(times)),key=lambda i:abs(times[i]-t))
                        pixels=tensor[i].permute(1,2,0).numpy().clip(0,255).astype(np.uint8)
                        im=Image.fromarray(pixels).resize((256,144));selected.append((times[i],im))
                entry['conditions'][r['condition']]=selected
        del tensor,videos
    for vid,entry in preview.items():
        canvas=Image.new('RGB',(6*256,2*170+25),'white');draw=ImageDraw.Draw(canvas)
        draw.text((5,4),entry['candidate_id']+' proposal-neighborhood / no GT',(0,0,0))
        for y,condition in enumerate(POLICY['conditions']):
            for x,(t,im) in enumerate(entry['conditions'][condition]):
                canvas.paste(im,(x*256,25+y*170));draw.text((x*256+3,25+y*170+145),f'{condition} clip t={t:.3f}',(0,0,0))
        canvas.save(args.root/('input_preview_'+vid+'.png'))
    save(args.root/'input_sampling_audit.json',dict(status='passed',code_commit=args.expected_head,
        requests_sha256=file_sha(args.root/'requests.json'),rows=rows,preview_videos=preview_videos,
        note='Actual TimeLens2 CPU input frames; TimePLE runtime independently records native frame metadata; no GT seen'))
    print('ACTUAL_INPUT_AUDIT_PASS',len(rows),flush=True)


def timelens_predict(model,processor,request,condition):
    import torch
    from qwen_vl_utils import process_vision_info
    from tools.rq4_teachers import TIMELENS_PROMPT
    from tools.run_clip_target_adapter import ACTION_PHRASES
    options=video_options(condition)
    query=ACTION_PHRASES[request['candidate']['class_id']]
    messages=[dict(role='user',content=[dict(type='video',video=Path(request['clip_path']).as_uri(),**options),
        dict(type='text',text=TIMELENS_PROMPT.format(query=query))])]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images,videos,kwargs=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
    videos,meta=zip(*videos);shape=list(videos[0].shape)
    inputs=processor(text=text,images=images,videos=list(videos),video_metadata=list(meta),do_resize=False,return_tensors='pt',**kwargs).to(model.device)
    with torch.inference_mode():out=model.generate(**inputs,do_sample=False,max_new_tokens=POLICY['max_new_tokens'])
    raw=processor.batch_decode([o[len(i):] for i,o in zip(inputs.input_ids,out)],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    return dict(raw_text=raw,error='',metadata=dict(video_metadata=list(meta),tensor_shape=shape,video_options=options))


def worker(args):
    from tools.rq3_gpu import gpu_slot
    m=check(args.root,args.expected_head);out=args.root/args.teacher;out.mkdir(exist_ok=False)
    input_audit=read(args.root/'input_sampling_audit.json')
    assert input_audit['status']=='passed' and input_audit['requests_sha256']==file_sha(args.root/'requests.json')
    model_path=MODELS[args.teacher]['model']
    save(out/'identity.json',dict(code_commit=args.expected_head,teacher=args.teacher,weights=weights(Path(model_path)),
        requests_sha256=file_sha(args.root/'requests.json'),policy=POLICY))
    def stop(*_):raise RuntimeError('explicit stop, preserve responses')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with gpu_slot(str(POLICY['gpu']),BASE/'SADA_PLUS',wait=True) as card:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(card['index']) if args.teacher=='timeple_8b' else card['uuid'],
            RQ3_GPU_INDEX=str(card['index']),RQ3_GPU_UUID=card['uuid'],TOKENIZERS_PARALLELISM='false')
        if args.teacher=='timelens2_8b':
            import torch
            from transformers import AutoModelForImageTextToText,AutoProcessor
            processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
            guard_teacher_cuda(card,args.teacher);torch.manual_seed(6)
            model,loading=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,
                torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',output_loading_info=True)
            if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise RuntimeError(str(loading))
            model.eval().requires_grad_(False);save(out/'loading.json',loading)
            predict=lambda r:timelens_predict(model,processor,r['request'],r['condition'])
        else:
            import yaml
            from tools.rq4_teachers import TimePLEAdapter,NativeIntervalMissing
            from tools.run_clip_target_adapter import ACTION_PHRASES
            repo=BASE/'repos/TimePLE';os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
            sys.path[:0]=[str(repo/'evaluation/src'),str(repo/'src')]
            from runners.qwen_family import QwenTimePLERunner
            from benchmark_loader import BenchmarkSample
            inference=yaml.safe_load((repo/'evaluation/configs/models/timeple_8b.yaml').read_text())['model']['eval']['config']['inference']
            inference.update(model_path=model_path,allowed_local_media_path=str(BASE))
            # 两档共用较长上下文容量，保证400帧不被原20480长度截断；不改变精度/生成规则。
            inference['engine'].update(max_num_seqs=1,enforce_eager=True,seed=6,
                max_model_len=32768,kv_cache_memory_bytes=8*1024**3)
            inference['generation']['max_tokens']=POLICY['max_new_tokens']
            prompt=yaml.safe_load((repo/'evaluation/configs/prompts/common.yaml').read_text())['prompt']['by_runner_family']['qwen3_vl_timeple']
            prompt=timeple_prompt_variant(prompt,'native-only')
            config=dict(model=dict(name=args.teacher),inference=inference,prompt=prompt)
            save(out/'official_config.json',dict(config=config,repo_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),runner_sha256=file_sha(repo/'evaluation/src/runners/qwen_family.py')))
            runner=QwenTimePLERunner(config,config_path=out/'official_config.json');adapter=TimePLEAdapter(runner,BenchmarkSample)
            guard_teacher_cuda(card,args.teacher)
            def predict(r):
                runner.inference_cfg['video']['message_kwargs'].update(POLICY['conditions'][r['condition']])
                q=r['request']
                try:return adapter.predict(r['request_id'],q['clip_path'],ACTION_PHRASES[q['candidate']['class_id']])
                except NativeIntervalMissing as exc:
                    if '<|TIMESPAN|>' in exc.response['raw_text']:raise RuntimeError('codec failed after native token') from exc
                    return dict(raw_text=exc.response['raw_text'],metadata=exc.response['metadata'],error='',output_status='native_missing')
        responses=[];previous={}
        for index,r in enumerate(m['requests']):
            started=time.monotonic();reply=predict(r);times=actual_times(reply['metadata'])
            if args.teacher=='timelens2_8b':assert times==input_audit['rows'][r['request_id']]['times']
            c=r['request']['candidate'];duration=c['window'][1]-c['window'][0]
            assert times[0]>=0 and times[-1]<=duration+.1
            assert len(times)<=POLICY['conditions'][r['condition']]['max_frames']
            meta=reply['metadata']
            spatial=meta['tensor_shape'][-2:] if 'tensor_shape' in meta else [meta['resized_height'],meta['resized_width']]
            if c['candidate_id'] in previous:assert previous[c['candidate_id']]==spatial
            previous[c['candidate_id']]=spatial
            response=dict(request_id=r['request_id'],request_sha256=digest(r),teacher=args.teacher,
                seconds=time.monotonic()-started,**reply)
            save(out/(r['request_id']+'.json'),response);responses.append(response)
            print('NATIVE_QUERY',args.teacher,index+1,len(m['requests']),r['condition'],'frames',len(times),'seconds',round(response['seconds'],2),flush=True)
        save(out/'responses.json',responses)
        save(out/'complete.json',dict(status='complete',count=len(responses),card=card,
            responses_sha256=file_sha(out/'responses.json'),identity_sha256=file_sha(out/'identity.json')))


def score(root,head):
    m=check(root,head);p=read(root/'preflight.json');truth=read(root/'scoring_only.json')
    assert file_sha(root/'scoring_only.json')==p['scoring_sha256']
    reports={};bound={}
    for name in MODELS:
        done=read(root/name/'complete.json')
        assert done['status']=='complete' and done['count']==80
        assert file_sha(root/name/'responses.json')==done['responses_sha256']
        assert file_sha(root/name/'identity.json')==done['identity_sha256']
        reports[name]=evaluate(m,read(root/name/'responses.json'),truth,name);bound[name]=done
    return dict(status='complete',diagnostic_only=True,policy=POLICY,code_commit=head,teachers=reports,
        decision=decision(reports),completed=bound,limitations=p['independence'],automatic_training_started=False)


def run(args):
    check(args.root,args.expected_head);save(args.root/'started.json',dict(pid=os.getpid(),code_commit=args.expected_head,time=time.time()))
    child=None
    def stop(*_):
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise RuntimeError('explicit stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for name,config in MODELS.items():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',
            LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1')
        with (args.root/(name+'.log')).open('x') as log:
            child=subprocess.Popen([config['python'],'-B','-u','-m','tools.run_rq4_native_density','worker',
                '--root',str(args.root),'--teacher',name,'--expected-head',args.expected_head],cwd=CODE,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('TEACHER_CHILD',name,child.pid,flush=True)
            rc=child.wait()
            if rc:raise RuntimeError(f'{name} failed {rc}; preserve all artifacts')
        child=None
    save(args.root/'responses_frozen.json',{n:file_sha(args.root/n/'responses.json') for n in MODELS})
    save(args.root/'result.json',score(args.root,args.expected_head))
    print('NATIVE_DENSITY_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','audit-inputs','run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--expected-head',required=True);p.add_argument('--teacher',choices=list(MODELS))
    args=p.parse_args();args.root=args.root.resolve()
    {'prepare':prepare,'audit-inputs':audit_inputs,'run':run,'worker':worker}[args.mode](args)


if __name__=='__main__':main()
