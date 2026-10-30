"""两个源域验证：复用类别缓存+18个学生只推理；32B四卡判断候选有效性与完整性。"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1];sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha, code_identity, assert_no_target_gt
from tools.qwen_information import SCHEMA, POLICY, OPTIONS, prompt, scores, inference_row, offline_truth, analyze

PREVIOUS = Path('/vepfs/vlms/qwen8b32b_source_0adb413_1030')
LOCALIZATION = Path('/vepfs/vlms/qwen_source_localization_shard4_fd9d887_1030')
BASE = Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029')
PROTO = Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
VLM_PY = '/vepfs/vlms/envs/rq1-cu121/bin/python'
MIN_FREE = 20480
WEIGHT_BUDGET = 17408
EXPECTED = {'manifest.json':'5dac740273be46fa3339ff6d93c97471a0e0618e0cbcaa1ae3c5a96063f1a344',
 '8b_predictions.jsonl':'03f2123fa093835ff073dd8018231bcf2ef2af51e5c3b26e1bbfe76671c792ed',
 '32b_predictions.jsonl':'f4f1812d9fd37cba1f61e75cbf4a08d0b1ebfd12fd338a85582a4a2e888700bf'}


def extract_context(item, root):
    # 仅为旧类别平票而未抽帧的候选补相同16帧，不更换候选。
    from bisect import bisect_left
    import cv2
    from PIL import Image
    if sha(item['clip_path']) != item['clip_sha256']:raise ValueError('原视频片段SHA改变')
    records = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',item['clip_path']],text=True))['frames']
    pts = [float(r['best_effort_timestamp_time']) for r in records]
    if len(pts)<16 or any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('真实PTS不足或无序')
    indices=[]
    for i in range(16):
        t=pts[0]+(pts[-1]-pts[0])*(i+.5)/16;r=min(bisect_left(pts,t),len(pts)-1)
        indices.append(min({r,max(0,r-1)},key=lambda j:(abs(pts[j]-t),j)))
    if len(set(indices))!=16:raise ValueError('不足16个不同帧；不伪装成正常输入')
    folder=root/'extra_frames'/item['candidate_id'];folder.mkdir(parents=True,exist_ok=False)
    cap=cv2.VideoCapture(item['clip_path']);paths=[]
    try:
        for i,index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(index+1))>.1:raise ValueError('视频解码帧号不符')
            path=folder/f'{i:02d}.png';Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(path);paths.append(str(path))
    finally:cap.release()
    return dict(item,frame_paths=paths,frame_sha256=[sha(p) for p in paths],
        frame_times_sec=[item['clip_start_sec']+pts[i] for i in indices],frame_indices=indices)


def prepare(root):
    for name,digest in EXPECTED.items():
        if sha(PREVIOUS/name)!=digest:raise ValueError('原类别实验身份改变: '+name)
    frozen=read(PREVIOUS/'frozen_candidates.json');manifest=read(PREVIOUS/'manifest.json')['candidates']
    old={r['candidate_id']:r for r in read(LOCALIZATION/'manifest.json')['candidates']}
    classes={t:{r['candidate_id']:r for r in map(json.loads,(PREVIOUS/(t+'_predictions.jsonl')).read_text().splitlines())}
             for t in ['8b','32b']}
    ids={r['candidate_id'] for r in manifest}
    if len(manifest)!=1095 or len(ids)!=1095 or set(old)!=ids or any(set(x)!=ids for x in classes.values()):
        raise ValueError('固定1095候选身份不完整')
    items={r['candidate_id']:r for r in frozen['inputs']};sources={};jobs=[];rows=[];evaluation=[]
    for shift in [f'S{i}' for i in range(1,7)]:
        cfg=read(BASE/'formal'/shift/'uda/config.json');source=Path(cfg['dataset']['json_file'])
        if sha(source)!=frozen['sources'][shift]['source_json_sha256']:raise ValueError('源GT版本不符')
        db=read(source)['database'];val={v for v,x in db.items() if x['subset'].lower() in cfg['src_val_split']}
        train={v for v,x in db.items() if x['subset'].lower() in cfg['src_train_split']}
        if train&val or val!=set(frozen['membership'][shift]['validation_videos']):raise ValueError('源留出成员不符')
        source_manifest=root/'source_inputs'/shift/'unlabeled.json'
        save(source_manifest,dict(database={v:{k:x[k] for k in ['subset','duration','fps'] if k in x}
                                             for v,x in db.items() if v in val}))
        request_path=Path(frozen['sources'][shift]['requests'])
        if sha(request_path)!=frozen['sources'][shift]['sha256']:raise ValueError('原候选请求SHA改变')
        requests={r['candidate']['candidate_id'].removesuffix('-context2'):r for r in read(request_path)['requests']
                  if r['candidate']['candidate_id'].endswith('-context2')}
        origins=[]
        for item in [r for r in manifest if r['shift']==shift]:
            cid=item['candidate_id'];context=old[cid]
            if item['video_id'] not in val:raise ValueError('候选不在源验证')
            if context['input_status']!='ready':context=extract_context(items[cid],root)
            row=inference_row(context)
            if row['proposal']!=items[cid]['proposal']:raise ValueError('上下文proposal与类别缓存不符')
            for p,h in zip(row['frame_paths'],row['frame_sha256']):
                if sha(p)!=h:raise ValueError('上下文帧SHA改变')
            row['prompt_sha256']=hashlib.sha256(prompt(row).encode()).hexdigest();rows.append(row)
            evaluated=dict(candidate_id=cid,shift=shift,video_id=row['video_id'],fixed_20pct=item['fixed_20pct'],
                truth=offline_truth(row['proposal'],db[row['video_id']].get('annotations',[]),row['frame_times_sec']))
            for t in classes:
                p=classes[t][cid]
                if p['frame_sha256']!=item['frame_sha256']:raise ValueError('类别缓存帧不符')
                probs=p['class_probabilities'];guess=max(range(10),key=lambda k:probs[k])
                tied=probs.count(max(probs))>1
                evaluated[t]=dict(class_id=None if tied else guess,confidence=max(probs),tied=tied)
            evaluation.append(evaluated)
            candidate=requests[cid]['candidate'];coordinate=int(cid.rsplit('-p',1)[1])
            origins.append(dict(candidate_id=cid,video_id=row['video_id'],absolute_feature_index=coordinate,
                level=candidate['fpn_level'],anchor_sec=candidate['anchor_sec']))
        for regime,reference in [('so',BASE/'formal'/shift/'so'),('uda',BASE/'formal'/shift/'uda'),('prototype',PROTO/shift)]:
            complete=read(reference/'complete.json');rcfg=read(reference/'config.json')
            artifacts={n:sha(reference/n) for n in ['config.json','complete.json','best.pt','termination.json']}
            if complete['artifacts'][str(reference/'best.pt')]!=artifacts['best.pt']:raise ValueError('已审EMA checkpoint SHA不符')
            if rcfg['seed']!=123 or rcfg['loader']['batch_size']!=2 or rcfg['src_val_split']!=cfg['src_val_split']:
                raise ValueError('学生协议不配对')
            entry=root/'students'/(shift+'_'+regime)
            job=dict(shift=shift,regime=regime,reference=str(reference),artifacts=artifacts,
                manifest=str(source_manifest),manifest_sha256=sha(source_manifest),origins=origins)
            save(entry/'job.json',job);jobs.append(str(entry))
        sources[shift]=dict(annotation=str(source),annotation_sha256=sha(source),validation_videos=sorted(val),
            request_sha256=sha(request_path),unlabeled_manifest=str(source_manifest),unlabeled_sha256=sha(source_manifest))
    rows.sort(key=lambda r:r['candidate_id']);evaluation.sort(key=lambda r:r['candidate_id'])
    save(root/'manifest.json',dict(schema=SCHEMA,candidates=rows,ground_truth_in_input=False))
    save(root/'evaluation_rows.json',evaluation)
    save(root/'inputs.json',dict(schema=SCHEMA,sources=sources,previous=str(PREVIOUS),
        previous_sha256={n:sha(PREVIOUS/n) for n in EXPECTED},manifest_sha256=sha(root/'manifest.json'),
        evaluation_rows_sha256=sha(root/'evaluation_rows.json'),jobs=jobs,
        extra_frames_candidates=sum(x['input_status']!='ready' for x in old.values())))
    save(root/'preparation_complete.json',dict(status='passed',count=len(rows),jobs=len(jobs),
        inputs_sha256=sha(root/'inputs.json'),class_queries_reused=2190,student_training=False))
    print('PREPARED',len(rows),'students',len(jobs),flush=True)


def student_worker(root,entry,data):
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.trainer.da_trainer import move_to_device
    from tools.baseline_engine import dataset,tensor_digest
    from tools.audit_prototype_gradients import restore_controls
    from tools.baseline_gpu import first_cuda_guard
    os.chdir(data);job=read(entry/'job.json');ref=Path(job['reference']);cfg=read(ref/'config.json')
    if code_identity()!=read(root/'plan.json')['code']:raise ValueError('学生推理代码不符')
    for name,h in job['artifacts'].items():
        if sha(ref/name)!=h:raise ValueError('学生参考工件改变')
    if sha(job['manifest'])!=job['manifest_sha256']:raise ValueError('源无GT输入改变')
    saved=torch.load(ref/'best.pt',map_location='cpu')
    if saved['epoch']!=read(ref/'termination.json')['decision']['best_epoch']:raise ValueError('不是源选best EMA')
    proof=first_cuda_guard();torch.manual_seed(cfg['seed']);torch.cuda.manual_seed(cfg['seed'])
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').to('cuda:0')
    model.load_state_dict(saved['ema_state_dict'],strict=True);restore_controls(model,saved['ema_controls'])
    model.eval().requires_grad_(False);before=tensor_digest(dict(model.named_parameters()))
    ds=dataset(cfg,job['manifest'],cfg['src_val_split'],False,True)
    by=defaultdict(list)
    for r in job['origins']:by[r['video_id']].append(r)
    predictions=[];feature_sha={}
    with torch.inference_mode():
        for video in ds:
            vid=video['metadata']['video_id']
            if vid not in by:continue
            assert_no_target_gt([video]);feature_sha[vid]=sha(Path(cfg['dataset']['feat_folder'])/(vid+cfg['dataset']['file_ext']))
            video=move_to_device([video],'cuda:0')[0]
            x,m=model.preprocessing([video],mode='validation');f,mask,_=model.backbone_source(x,m)
            f,mask=model.neck_source(f,mask);logits,_=model.cls_head(f,mask);points=model.point_generator(f)
            for origin in by[vid]:
                level=origin['level'];coordinate=origin['absolute_feature_index'];metadata=video['metadata']
                expected=(origin['anchor_sec']*metadata['fps']-.5*metadata['feat_num_frames'])/metadata['feat_stride']
                if abs(expected-coordinate)>1e-4:raise ValueError('原anchor时间/feature坐标不一致')
                match=torch.nonzero((points[level][:,0]-coordinate).abs()<1e-5).flatten()
                if len(match)!=1 or not mask[level][0,0,int(match[0])]:raise ValueError('原FPN anchor不可用；不按GT换层')
                values=logits[level][0,:,int(match[0])].sigmoid().cpu().tolist();cls=max(range(len(values)),key=lambda i:values[i])
                predictions.append(dict(candidate_id=origin['candidate_id'],class_id=cls,score=values[cls],
                    known_class_scores=values[:10],level=level,absolute_feature_index=coordinate))
    if {x['candidate_id'] for x in predictions}!={x['candidate_id'] for x in job['origins']}:raise ValueError('学生候选预测不完整')
    if tensor_digest(dict(model.named_parameters()))!=before:raise ValueError('只推理修改了模型')
    save(entry/'predictions.json',predictions)
    save(entry/'inference_complete.json',dict(status='passed',regime=job['regime'],shift=job['shift'],count=len(predictions),
        job_sha256=sha(entry/'job.json'),predictions_sha256=sha(entry/'predictions.json'),guard=proof,guard_sha256=sha(proof),
        checkpoint_sha256=sha(ref/'best.pt'),parameter_sha256=before,source_feature_sha256=feature_sha,
        source_gt_in_input=False,source_selected_epoch=saved['epoch'],no_training=True))
    print('STUDENT_COMPLETE',job['shift'],job['regime'],len(predictions),flush=True)


def guard_four(root):
    import fcntl
    from tools.baseline_gpu import snapshot
    cards=read(root/'active_leases.json')
    if os.environ['QWEN_INFORMATION_LEASE_SHA']!=sha(root/'active_leases.json'):raise ValueError('四卡租约SHA不符')
    if len(cards)!=4 or [x['index'] for x in cards]!=list(range(4)):raise ValueError('必须精确四卡')
    if os.environ['CUDA_VISIBLE_DEVICES']!=','.join(x['uuid'] for x in cards):raise ValueError('可见GPU不符')
    while True:
        observed=[]
        for card in cards:
            lease=read(card['lease'])
            if sha(card['lease'])!=card['lease_sha256'] or lease['owner_pid']!=os.getppid():raise ValueError('GPU租约父进程/SHA不符')
            if lease['minimum_free_mib']!=MIN_FREE:raise ValueError('32B显存准入不足')
            with Path(lease['lock']).open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:raise ValueError('父进程未持物理卡锁')
            now=snapshot(card['index'])
            if now['uuid']!=card['uuid']:raise ValueError('实际GPU UUID改变')
            observed.append(now)
        if all(x['free_mib']>=MIN_FREE for x in observed):break
        print('WAIT_FIRST_CUDA_32B',[(x['index'],x['free_mib']) for x in observed],flush=True);time.sleep(15)
    save(root/'validity/first_cuda.json',dict(status='passed',cards=cards,observed=observed,child_pid=os.getpid(),
         owner_pid=os.getppid(),leases_sha256=sha(root/'active_leases.json'),weight_budget_mib=WEIGHT_BUDGET))


def validity_worker(root):
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    if code_identity()!=read(root/'plan.json')['code']:raise ValueError('教师代码身份改变')
    inputs=read(root/'inputs.json')
    if sha(root/'manifest.json')!=inputs['manifest_sha256']:raise ValueError('固定教师输入改变')
    identity=read(PREVIOUS/'32b_model_identity.json');path=Path(identity['path'])
    for group in ['weight_sha256','processor_sha256']:
        for name,h in identity[group].items():
            if sha(path/name)!=h:raise ValueError('32B模型/处理器SHA改变: '+name)
    save(root/'validity/model_identity.json',identity)
    processor=AutoProcessor.from_pretrained(path,local_files_only=True)
    tokens=[processor.tokenizer.encode(x,add_special_tokens=False) for x in OPTIONS]
    if any(len(x)!=1 for x in tokens) or len({x[0] for x in tokens})!=len(OPTIONS):raise ValueError('选项不是唯一单token')
    token_ids=[x[0] for x in tokens];torch.set_num_threads(8)
    guard_four(root)
    model,loading=Qwen3VLForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa',
        local_files_only=True,output_loading_info=True,device_map='balanced',max_memory={i:f'{WEIGHT_BUDGET}MiB' for i in range(4)})
    if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('32B未完整加载')
    if {str(x).replace('cuda:','') for x in model.hf_device_map.values()}!={'0','1','2','3'}:raise ValueError('32B必须实际四卡且无CPU/disk卸载')
    model.eval().requires_grad_(False)
    save(root/'validity/device_map.json',dict(device_map=model.hf_device_map,option_token_ids=token_ids,
        visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
    rows=read(root/'manifest.json')['candidates'];output=root/'validity/predictions.jsonl'
    prefix=[json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if len(prefix)>len(rows):raise ValueError('已有评分超出固定集合')
    for old,row in zip(prefix,rows):
        if (old['candidate_id']!=row['candidate_id'] or old['prompt_sha256']!=row['prompt_sha256']
                or old['frame_sha256']!=row['frame_sha256']):raise ValueError('接续评分不是相同输入的连续前缀')
        recomputed=scores(row['candidate_id'],old['option_logits'],old['full_vocab_logsumexp'])
        if any(old[k]!=v for k,v in recomputed.items()):raise ValueError('旧评分不能由同解释器精确复算')
    started=time.monotonic()
    with output.open('a' if output.exists() else 'x') as stream:
        for i,row in enumerate(rows[len(prefix):],len(prefix)):
            for p,h in zip(row['frame_paths'],row['frame_sha256']):
                if sha(p)!=h:raise ValueError('推理真实帧SHA改变')
            content=[dict(type='text',text='Chronological context frames with actual timestamps:')]
            for j,(p,t) in enumerate(zip(row['frame_paths'],row['frame_times_sec'])):
                content += [dict(type='text',text=f'FRAME {j}: {t:.6f} seconds'),
                    dict(type='image',image=Path(p).resolve().as_uri(),min_pixels=224**2,max_pixels=448**2)]
            text_prompt=prompt(row)
            if hashlib.sha256(text_prompt.encode()).hexdigest()!=row['prompt_sha256']:raise ValueError('冻结提示SHA不符')
            content.append(dict(type='text',text=text_prompt));messages=[dict(role='user',content=content)]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            images,videos=process_vision_info(messages,image_patch_size=16)
            batch=processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt').to(model.device)
            with torch.inference_mode():result=model(**batch,use_cache=False,logits_to_keep=1,return_dict=True)
            logits=result.logits[0,-1].float();values=logits[token_ids].cpu().tolist()
            prediction=dict(candidate_id=row['candidate_id'],**scores(row['candidate_id'],values,float(torch.logsumexp(logits,0).cpu())),
                prompt_sha256=row['prompt_sha256'],frame_sha256=row['frame_sha256'])
            stream.write(json.dumps(prediction,allow_nan=False)+'\n');stream.flush()
            if i==len(prefix) or (i+1)%10==0:
                print('VALIDITY_SCORED',i+1,len(rows),flush=True)
                save(root/'validity/progress.json',dict(scored=i+1,total=len(rows),elapsed_sec=time.monotonic()-started,
                    peak_reserved_mib={str(j):torch.cuda.max_memory_reserved(j)/2**20 for j in range(4)}))
    save(root/'validity/inference_complete.json',dict(status='passed',count=len(rows),predictions_sha256=sha(output),
        manifest_sha256=sha(root/'manifest.json'),model_identity_sha256=sha(root/'validity/model_identity.json'),
        first_cuda_sha256=sha(root/'validity/first_cuda.json'),device_map_sha256=sha(root/'validity/device_map.json')))


def supervisor(args):
    from tools.run_sada_baseline import acquire
    from tools.baseline_gpu import slot,snapshot
    root=args.output.resolve();data=args.data_root.resolve();identity=code_identity();active={}
    if not args.resume:
        root.mkdir(parents=True,exist_ok=False)
        save(root/'plan.json',dict(schema=SCHEMA,policy=POLICY,code=identity,physical_gpus=[0,1,2,3],
            minimum_free_per_gpu_mib=MIN_FREE,vlm_weight_budget_per_gpu_mib=WEIGHT_BUDGET))
    elif read(root/'plan.json')['code']!=identity:raise ValueError('恢复必须相同clean代码，不覆盖旧版本')
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
    def spawn(key,exe,mode,extra=(),child_env=None):
        log=(root/(key+'.log')).open('a' if args.resume else 'x')
        p=subprocess.Popen([exe,'-B',str(Path(__file__).resolve()),'--mode',mode,'--output',str(root),'--data-root',str(data),*extra],
            cwd=CODE,env=child_env or env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
        active[key]=dict(p=p);return p
    try:
        if not (root/'preparation_complete.json').exists():
            save(root/'state.json',dict(phase='source_preparation',pid=os.getpid()))
            p=spawn('prepare',VLM_PY,'prepare')
            if p.wait():raise RuntimeError('源输入准备失败，查看prepare.log')
            del active['prepare']
        inputs=read(root/'inputs.json')
        if sha(root/'manifest.json')!=inputs['manifest_sha256']:raise ValueError('准备输入改变')
        pending=[Path(p) for p in inputs['jobs'] if not (Path(p)/'inference_complete.json').exists()]
        while pending or active:
            for key,task in list(active.items()):
                if task['p'].poll() is None:continue
                if task['p'].returncode:raise RuntimeError('学生只推理失败: '+key)
                task['ctx'].__exit__(None,None,None);del active[key]
            for entry in list(pending):
                used={t['card']['index'] for t in active.values()}
                found=acquire([g for g in range(4) if g not in used],data,root,12288,identity['commit'])
                if not found:break
                ctx,card=found
                try:
                    child_env=dict(env,CUDA_VISIBLE_DEVICES=card['uuid'],SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
                    spawn(entry.name,sys.executable,'student',['--entry',str(entry)],child_env)
                except BaseException:ctx.__exit__(None,None,None);raise
                active[entry.name].update(ctx=ctx,card=card);pending.remove(entry)
            save(root/'state.json',dict(phase='student_inference',pid=os.getpid(),pending=len(pending),
                active={k:dict(pid=t['p'].pid,gpu=t['card']['index']) for k,t in active.items()}))
            time.sleep(3)
        analyze(root)
        if not (root/'validity/inference_complete.json').exists():
            while True:
                held=ExitStack();cards=[]
                try:
                    for i in range(4):cards.append(held.enter_context(slot(i,data,root,MIN_FREE,identity['commit'])))
                except BlockingIOError:
                    held.close();save(root/'state.json',dict(phase='waiting_four_gpu_for_32b',pid=os.getpid(),
                        required_free_mib=MIN_FREE,observed=[snapshot(i) for i in range(4)]));time.sleep(15);continue
                break
            with held:
                save(root/'active_leases.json',cards)
                child_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(x['uuid'] for x in cards),
                    QWEN_INFORMATION_LEASE_SHA=sha(root/'active_leases.json'))
                p=spawn('validity',VLM_PY,'validity',child_env=child_env)
                save(root/'state.json',dict(phase='32b_validity_inference',pid=os.getpid(),child_pid=p.pid))
                if p.wait():raise RuntimeError('32B有效性推理失败，已保留评分前缀')
                del active['validity']
        analyze(root)
        completes={p:sha(Path(p)/'inference_complete.json') for p in inputs['jobs']}
        save(root/'final_audit.json',dict(status='passed',schema=SCHEMA,students=18,teacher_candidates=1095,
            student_completes=completes,complementarity_sha256=sha(root/'complementarity.json'),
            validity_sha256=sha(root/'validity_comparison.json'),no_student_training=True))
        save(root/'state.json',dict(phase='complete',pid=os.getpid()))
    except BaseException as e:
        save(root/'error.json',dict(type=type(e).__name__,message=str(e)))
        save(root/'state.json',dict(phase='stopped' if isinstance(e,SystemExit) else 'failed',pid=os.getpid()))
        raise
    finally:
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        for t in active.values():
            try:t['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['p'].pid,signal.SIGKILL);t['p'].wait()
            if 'ctx' in t:t['ctx'].__exit__(None,None,None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--entry',type=Path)
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--mode',choices=['supervisor','prepare','student','validity'],default='supervisor')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    {'supervisor':lambda:supervisor(args),'prepare':lambda:prepare(args.output),
     'student':lambda:student_worker(args.output,args.entry,args.data_root),'validity':lambda:validity_worker(args.output)}[args.mode]()
