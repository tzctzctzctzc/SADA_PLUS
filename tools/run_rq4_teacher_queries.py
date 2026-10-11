"""新视频证据的有限诊断：准备、单卡教师、最后评分相互隔离。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.rq4_protocol import digest,file_sha
from tools.rq4_source_acceptance import save,weights,guard_teacher_cuda
from tools.rq4_teacher_queries import POLICY,choose,tasks,match_gt,answer_for,parse_answer,evaluate,validate_completion

CODE=Path(__file__).resolve().parents[1]
BASE=Path('/vepfs/vlms')
MODELS={
 'timelens2_8b':dict(python=str(BASE/'envs/rq1-cu121/bin/python'),model=str(BASE/'cache/huggingface/hub/models--MCG-NJU--TimeLens2-8B/snapshots/afb728d176c1ab919b3ee12e581c1620cbdfa5c0')),
 'timeple_8b':dict(python=str(BASE/'envs/rq4-timeple-1010/bin/python'),model=str(BASE/'cache/huggingface/hub/models--KlingTeam--TimePLE/snapshots/f583ef50ea9fc3498d537ff14438af25d5613d38'))}


def read(path):return json.loads(Path(path).read_text())


def clean_head(expected):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
    if head!=expected or dirty:raise ValueError('audited clean worktree required')
    return head


def code_files():
    return {name:file_sha(CODE/'tools'/name) for name in ('rq4_teacher_queries.py','run_rq4_teacher_queries.py')}


def media_info(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate,nb_read_frames:format=duration','-of','json',str(path)],text=True))


def prepare(args):
    from tools.run_clip_target_adapter import ACTION_PHRASES
    clean_head(args.expected_head)
    root=args.root.resolve();root.mkdir(exist_ok=False,parents=True)
    old=read(args.probe_root/'plan.json'); reference=old['source_evidence']['timelens2_8b']['requests']
    assert file_sha(reference['path'])==reference['sha256']
    manifest=read(reference['path'])
    assert manifest['split']=='source_validation' and manifest['shift']=='S1' and manifest['seed']==6
    selected=choose(manifest['requests']); requests=[]; media={}; source=[]
    (root/'black').mkdir()
    for request in selected:
        c=request['candidate'];vid=c['video_id']
        assert vid in old['membership']['heldout'] and vid not in old['membership']['train']+old['membership']['target']
        original=Path(request['clip_path']);assert file_sha(original)==request['clip_sha256']
        assert [request['clip_start_sec'],request['clip_end_sec']]==c['window']
        black=root/'black'/(digest([vid,c['candidate_id']])+'.mkv')
        subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-i',str(original),'-map','0:v:0','-an',
            '-vf','lutyuv=y=16:u=128:v=128','-c:v','ffv1','-threads','2',str(black)],check=True)
        infos={key:media_info(path) for key,path in (('video',original),('black',black))}
        assert infos['video']['streams']==infos['black']['streams']
        assert abs(float(infos['video']['format']['duration'])-float(infos['black']['format']['duration']))<.001
        duration=float(infos['video']['format']['duration'])
        assert abs(duration-(c['window'][1]-c['window'][0]))<.1
        # 对每帧缩小后实际解码RGB，核实遮黑条件，而非仅相信滤镜参数。
        pixels=subprocess.check_output(['ffmpeg','-v','error','-i',str(black),'-vf','scale=8:8',
            '-f','rawvideo','-pix_fmt','rgb24','-'])
        assert pixels and max(pixels)<=2
        infos['black_rgb_max']=max(pixels)
        media[c['candidate_id']]=infos
        built=tasks(c,ACTION_PHRASES[c['class_id']]);source.append(dict(candidate=c,tasks=built))
        for task in built:
            conditions=['video','black']
            if int(digest(task['task_id'])[:4],16)%2:conditions.reverse()
            for condition in conditions:
                clip=original if condition=='video' else black
                requests.append(dict(request_id=task['task_id']+'-'+condition,task=task,
                    condition=condition,clip_path=str(clip),clip_sha256=file_sha(clip)))
    # 教师请求先冻结；GT仅生成另外的评分工件，worker不会加载它。
    save(root/'requests.json',dict(schema='rq4-direct-teacher-capability-v1',policy=POLICY,requests=requests,
        code_files=code_files(),source_request=reference,diagnostic_only=True))
    annotation=old['annotations'];assert file_sha(annotation['path'])==annotation['sha256']
    database=read(annotation['path'])['database'];truth={}
    for item in source:
        c=item['candidate'];label=match_gt(c,database[c['video_id']]['annotations'])
        for task in item['tasks']:
            truth[task['task_id']]=dict(**label,answer=answer_for(task,label['gt']))
    save(root/'scoring_only.json',truth)
    save(root/'preflight.json',dict(status='passed',code_commit=args.expected_head,code_files=code_files(),
        policy=POLICY,request_sha256=file_sha(root/'requests.json'),scoring_sha256=file_sha(root/'scoring_only.json'),
        candidates=len(selected),requests_per_teacher=len(requests),media=media,annotations=annotation,
        target_videos_loaded=False,score_before_all_responses=False,
        protocol='S1 fixed2 per video by seed6 SHA; actual video vs matched black clip; ranking AB reversal; no fit or prompt sweep'))
    print('QUERY_CPU_PREFLIGHT_PASS',len(selected),len(requests),flush=True)


def check_requests(root,head):
    clean_head(head);manifest=read(root/'requests.json');proof=read(root/'preflight.json')
    assert proof['status']=='passed' and proof['request_sha256']==file_sha(root/'requests.json')
    assert manifest['code_files']==proof['code_files']==code_files() and manifest['policy']==POLICY
    checked=set()
    for r in manifest['requests']:
        assert set(r)=={'request_id','task','condition','clip_path','clip_sha256'}
        assert not any(x in r['task'] for x in ('gt','answer','scorable','label'))
        if r['clip_path'] not in checked:
            assert file_sha(r['clip_path'])==r['clip_sha256'];checked.add(r['clip_path'])
    return manifest


def tl_predict(model,processor,request):
    import torch
    from qwen_vl_utils import process_vision_info
    messages=[dict(role='user',content=[dict(type='video',video=Path(request['clip_path']).as_uri(),
        fps=POLICY['fps'],min_pixels=32*32,max_pixels=480*480,max_frames=POLICY['max_frames']),
        dict(type='text',text=request['task']['prompt'])])]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images,videos,kwargs=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
    videos,metadata=zip(*videos)
    inputs=processor(text=text,images=images,videos=list(videos),video_metadata=list(metadata),
        do_resize=False,return_tensors='pt',**kwargs).to(model.device)
    with torch.inference_mode():
        out=model.generate(**inputs,do_sample=False,max_new_tokens=POLICY['max_new_tokens'])
    raw=processor.batch_decode([o[len(i):] for i,o in zip(inputs.input_ids,out)],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    return dict(raw_text=raw,metadata=dict(video_metadata=list(metadata)),error='')


def teacher(args):
    from tools.rq3_gpu import gpu_slot
    manifest=check_requests(args.root,args.expected_head);out=args.root/args.teacher;out.mkdir(exist_ok=False)
    model_path=MODELS[args.teacher]['model']
    identity=dict(teacher=args.teacher,weights=weights(Path(model_path)),code_commit=args.expected_head,
        requests_sha256=file_sha(args.root/'requests.json'),policy=POLICY,task='direct text answers, not cached timestamp conversion')
    if args.teacher=='timeple_8b':
        repo=BASE/'repos/TimePLE'
        identity['official_repo_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        identity['official_runner_sha256']=file_sha(repo/'evaluation/src/runners/qwen_family.py')
    save(out/'identity.json',identity)
    responses=[]
    def stopped(*_):raise RuntimeError('user stopped diagnostic; preserve outputs')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    with gpu_slot('0',BASE/'SADA_PLUS',wait=True) as card:
        visible='0' if args.teacher=='timeple_8b' else card['uuid']
        os.environ.update(CUDA_VISIBLE_DEVICES=visible,RQ3_GPU_INDEX='0',RQ3_GPU_UUID=card['uuid'],TOKENIZERS_PARALLELISM='false')
        if args.teacher=='timelens2_8b':
            import torch
            from transformers import AutoModelForImageTextToText,AutoProcessor
            processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
            guard_teacher_cuda(card,args.teacher);torch.manual_seed(6)
            model,loading=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,
                torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',output_loading_info=True)
            assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
            model.eval().requires_grad_(False);save(out/'loading.json',loading)
            predict=lambda r:tl_predict(model,processor,r)
        else:
            import yaml
            os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
            sys.path[:0]=[str(repo/'evaluation/src'),str(repo/'src')]
            from runners.qwen_family import QwenTimePLERunner
            from benchmark_loader import BenchmarkSample
            inference=yaml.safe_load((repo/'evaluation/configs/models/timeple_8b.yaml').read_text())['model']['eval']['config']['inference']
            inference.update(model_path=model_path,allowed_local_media_path=str(BASE))
            inference['engine'].update(max_num_seqs=1,enforce_eager=True,seed=6,kv_cache_memory_bytes=4*1024**3)
            inference['generation']['max_tokens']=POLICY['max_new_tokens']
            inference['video']['message_kwargs'].update(fps=POLICY['fps'],max_frames=POLICY['max_frames'])
            config=dict(model=dict(name=args.teacher),inference=inference,prompt=dict(enabled_sys_prompt=True,
                system_prompt='You are a video understanding assistant. Answer the requested multiple-choice question from visual evidence.',
                user_prompt='{query}',enabled_format_prompt=False,format_prompt=''))
            save(out/'official_config.json',config)
            runner=QwenTimePLERunner(config,config_path=out/'official_config.json')
            guard_teacher_cuda(card,args.teacher)
            def predict(r):
                sample=BenchmarkSample(global_index=0,sample_id=r['request_id'],query=r['task']['prompt'],
                    video_path=r['clip_path'],gt_timestamps=[],raw_sample={})
                result=runner.predict_batch([sample])
                assert len(result)==1 and result[0].sample_id==r['request_id']
                v=result[0]
                if v.error:raise RuntimeError(v.error)
                if '<|TIMESPAN|>' in v.raw_text and v.metadata.get('codec_predictions') is None:
                    raise RuntimeError('native TIMESPAN emitted but codec failed')
                return dict(raw_text=v.raw_text,metadata=v.metadata,error='')
        status='complete'
        for index,request in enumerate(manifest['requests']):
            started=time.monotonic();reply=predict(request)
            response=dict(request_id=request['request_id'],request_sha256=digest(request),
                teacher=args.teacher,seconds=time.monotonic()-started,**reply)
            save(out/(request['request_id']+'.json'),response);responses.append(response)
            print('QUERY_RESPONSE',args.teacher,index+1,len(manifest['requests']),
                request['task']['kind'],request['condition'],repr(reply['raw_text']),flush=True)
            # 固定第一候选14条仅检验接口，不能读取GT后决定是否继续。
            if index==13 and not any(parse_answer(r['task']['kind'],v['raw_text']) is not None
                                    for r,v in zip(manifest['requests'][:14],responses)):
                status='unsupported_answer_interface';break
        save(out/'responses.json',responses)
        save(out/'complete.json',dict(status=status,count=len(responses),card=card,
            responses_sha256=file_sha(out/'responses.json'),identity_sha256=file_sha(out/'identity.json'),formal_rq4=False))


def run(args):
    check_requests(args.root,args.expected_head)
    save(args.root/'started.json',dict(pid=os.getpid(),code_commit=args.expected_head,time=time.time()))
    child=None
    def stop(*_):
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise RuntimeError('explicit diagnostic stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for teacher_name,config in MODELS.items():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
            LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1')
        with (args.root/(teacher_name+'.log')).open('x') as log:
            child=subprocess.Popen([config['python'],'-B','-u','-m','tools.run_rq4_teacher_queries','teacher',
                '--root',str(args.root),'--teacher',teacher_name,'--expected-head',args.expected_head],
                env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('TEACHER_CHILD',teacher_name,child.pid,flush=True)
            rc=child.wait()
            if rc:raise RuntimeError(f'{teacher_name} exited {rc}; preserve all outputs')
        child=None
    # 所有教师响应已冻结，才进入评分。
    completions={name:read(args.root/name/'complete.json') for name in MODELS}
    save(args.root/'responses_frozen.json',completions)
    truth=read(args.root/'scoring_only.json');proof=read(args.root/'preflight.json')
    assert file_sha(args.root/'scoring_only.json')==proof['scoring_sha256']
    requests=read(args.root/'requests.json')['requests'];results={}
    for name,done in completions.items():
        assert file_sha(args.root/name/'responses.json')==done['responses_sha256']
        responses=read(args.root/name/'responses.json')
        validate_completion(requests,responses,done)
        results[name]=dict(status=done['status'],**evaluate(requests,responses,truth))
    save(args.root/'result.json',dict(status='complete',diagnostic_only=True,formal_rq4=False,
        code_commit=args.expected_head,teachers=results,policy=POLICY,query_gt_used=False,
        limitations='fixed14 source proposals, no fit; repeated actions marked; unmatched same-class events unscored; black control may be out-of-distribution; new prompts, not official grounding benchmark'))
    print('TEACHER_CAPABILITY_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run','teacher']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--expected-head',required=True);p.add_argument('--probe-root',type=Path)
    p.add_argument('--teacher',choices=list(MODELS));args=p.parse_args();args.root=args.root.resolve()
    {'prepare':prepare,'run':run,'teacher':teacher}[args.mode](args)


if __name__=='__main__':main()
