"""共享四卡32B的三线背景预检；请求级保存，完整视频就绪即离线评估。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.vlm_background_protocol import SPEC,MODES,digest,prompt,option_order,score,decisions,frame_indices,verify_response
PILOT=Path('/vepfs/vlms/hard_background_pilots24_8e80d74_1103')
SOURCE=Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
VIDEO=Path('/vepfs/vlms/data/EPIC_KITCHENS/videos')
IDENTITY=Path('/vepfs/vlms/cott_quality_pilot_85b6d10_1102/model_identity.json')
VLM_PY='/vepfs/vlms/envs/rq1-cu121/bin/python'
CPU_PY='/vepfs/vlms/envs/sada_py39/bin/python'
MIN_FREE=20480  # 每卡权重预算17408MiB；每个请求另外检查至少3072MiB空余。


def cards():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total',
                                  '--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(x[0]),uuid=x[1].strip(),free_mib=int(x[2]),total_mib=int(x[3]))
            for row in text.strip().splitlines() if (x:=row.split(',')) and int(x[0]) in range(4)]


def model_identity(root):
    old=read(IDENTITY);assert old['verified_all_sha256'];identity=old['identity'];base=Path(identity['path'])
    for name,stat in old['file_stat'].items():
        now=(base/name).stat();assert now.st_size==stat['size'] and now.st_mtime_ns==stat['mtime_ns'],name
    assert len(identity['weight_sha256'])==14
    for name,h in identity['processor_sha256'].items():assert sha(base/name)==h
    save(root/'model_identity.json',dict(identity=identity,file_stat=old['file_stat'],
        prior_full_SHA_verified_stat_unchanged=True,origin=str(IDENTITY),origin_sha256=sha(IDENTITY)))


def video_frames(video,rows,root):
    import av
    import numpy as np
    from PIL import Image
    path=VIDEO/(video+'.MP4');assert path.exists(),path
    with av.open(str(path)) as cap:
        stream=cap.streams.video[0];tb=float(stream.time_base)
        pts=sorted(p.pts for p in cap.demux(stream) if p.pts is not None)
        assert pts and len(pts)==len(set(pts))
        if stream.frames:assert len(pts)==stream.frames
    times=[p*tb for p in pts];wanted=set()
    for row in rows:
        plan=frame_indices(times,*row['proposal']);row.update(plan)
        if plan['observable']:wanted.update(plan['indices'])
    groups=[]
    for i in sorted(wanted):
        if not groups or times[i]-times[groups[-1][-1]]>2:groups.append([])
        groups[-1].append(i)
    folder=root/'frames'/video;folder.mkdir(parents=True,exist_ok=True)
    positions={p:i for i,p in enumerate(pts)};records={}
    with av.open(str(path)) as cap:
        stream=cap.streams.video[0]
        for group in groups:
            cap.seek(pts[group[0]],stream=stream,backward=True,any_frame=False);requested=set(group)
            for frame in cap.decode(stream):
                assert frame.pts in positions
                i=positions[frame.pts]
                if i in requested and i not in records:
                    rgb=frame.to_ndarray(format='rgb24');f=folder/(str(i)+'.jpg')
                    Image.fromarray(rgb).save(f,quality=95)
                    records[i]=dict(path=str(f),sha256=sha(f),pts=int(frame.pts),time_sec=frame.pts*tb,
                        raw_pixel_sha256=__import__('hashlib').sha256(rgb.tobytes()).hexdigest())
                if i>=group[-1]:break
            assert requested<=set(records),'missing exact requested frame'
    stat=path.stat();save(root/'video_indices'/(video+'.json'),dict(path=str(path),size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,time_base=tb,frame_count=len(pts),pts_sha256=digest(pts),sampled_frames=records))
    for row in rows:
        row['frame_records']=[records[i] for i in row['indices']] if row['observable'] else []
    print('FRAMES',video,len(records),len(rows),flush=True)


def prepare(root):
    import torch
    assert read(root/'plan.json')['spec']==SPEC
    jobs={};cases={};inputs={}
    for s in ['S%d'%i for i in range(1,7)]:
        for domain in ['source_val','target_train']:
            p=PILOT/s/(domain+'_predictions.pt');done=read(PILOT/s/'gpu_complete.json')
            assert sha(p)==done['artifacts'][str(p)]
            rows=torch.load(p,map_location='cpu',weights_only=False)['conditional']
            eligible=[r for r in rows if bool((r['scores']>=SPEC['cutoff']).any())]
            selected=min(eligible,key=lambda r:digest(['vlm-bg-three-v1',s,domain,r['video']]))
            group=s+'_'+domain;e=root/'groups'/group;e.mkdir(parents=True)
            torch.save(selected,e/'original.pt');cfg=read(SOURCE/s/'config.json');save(e/'config.json',cfg)
            mapping={}
            for i in (selected['scores']>=SPEC['cutoff']).nonzero().flatten().tolist():
                interval=[float(v) for v in selected['segments'][i]];label=int(selected['labels'][i])
                view=digest([selected['video'],interval]);cid=digest([view,label])
                mapping[str(i)]=cid
                cases.setdefault(cid,dict(candidate_id=cid,view_id=view,video_id=selected['video'],proposal=interval,label=label))
            jobs[group]=dict(shift=s,domain=domain,video=selected['video'],candidate_map=mapping,
                original=str(e/'original.pt'),original_sha256=sha(e/'original.pt'),config_sha256=sha(e/'config.json'))
            inputs[str(p)]=done['artifacts'][str(p)]
    by_video={}
    for row in cases.values():by_video.setdefault(row['video_id'],[]).append(row)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda x:video_frames(x[0],x[1],root),by_video.items()))
    manifest=dict(spec=SPEC,cases=list(cases.values()),GT_in_inference_input=False)
    assert any(r['observable'] for r in cases.values()),'all candidates lack sufficient physical observations'
    save(root/'inference_manifest.json',manifest)
    save(root/'groups.json',jobs)
    model_identity(root)
    save(root/'prepare_complete.json',dict(cases=len(cases),groups=len(jobs),
        observable=sum(r['observable'] for r in cases.values()),unique_physical_videos=len(by_video),
        artifacts={str(p):sha(p) for p in [root/'inference_manifest.json',root/'groups.json',root/'model_identity.json']},
        original_prediction_inputs=inputs,no_detector_forward_or_training=True))


def infer(root):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    ready=read(root/'prepare_complete.json')
    for p,h in ready['artifacts'].items():assert sha(p)==h
    lease_path=Path(os.environ['VLM_BG_LEASE']);lease=read(lease_path)
    assert sha(lease_path)==os.environ['VLM_BG_LEASE_SHA256']
    assert lease['owner_pid']==os.getppid() and os.environ['CUDA_VISIBLE_DEVICES']==','.join(c['uuid'] for c in lease['cards'])
    observed=cards();assert all(c['free_mib']>=MIN_FREE for c in observed)
    assert [c['uuid'] for c in observed]==[c['uuid'] for c in lease['cards']]
    guard=root/'guards'/(str(os.getpid())+'.json');save(guard,dict(owner_pid=os.getppid(),child_pid=os.getpid(),
        lease=str(lease_path),lease_sha256=sha(lease_path),observed=observed,minimum_free_mib=MIN_FREE))
    model_identity(root)
    assert sha(root/'model_identity.json')==ready['artifacts'][str(root/'model_identity.json')]
    identity=read(root/'model_identity.json')['identity'];path=identity['path']
    processor=AutoProcessor.from_pretrained(path,local_files_only=True)
    ids=[processor.tokenizer.encode(v,add_special_tokens=False) for v in 'ABC']
    assert all(len(x)==1 for x in ids) and len(set(x[0] for x in ids))==3;ids=[x[0] for x in ids]
    torch.set_num_threads(8)
    model,loading=Qwen3VLForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True,output_loading_info=True,device_map='balanced',
        max_memory={i:'17408MiB' for i in range(4)})
    assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    assert {str(v).replace('cuda:','') for v in model.hf_device_map.values()}=={'0','1','2','3'}
    model.eval().requires_grad_(False)
    save(root/'runtime.json',dict(device_map=model.hf_device_map,option_token_ids=ids,guard=str(guard),
        guard_sha256=sha(guard),torch=torch.__version__,no_quantization=True,no_CPU_or_disk_offload=True))
    rows=read(root/'inference_manifest.json')['cases'];calls=0;started=time.monotonic();cache={};active_key=None
    native_visual=model.model.visual.forward
    def visual_forward(*args,**kwargs):
        if active_key in cache:return cache[active_key]
        out=native_visual(*args,**kwargs);cache[active_key]=out;return out
    model.model.visual.forward=visual_forward
    proved_questions=set();proofs={};pixels={}
    def query(row,question,static=False):
        nonlocal calls,active_key
        frames=row['frame_records'];frame_paths=[x['path'] for x in frames]
        if static:frame_paths=[frame_paths[8]]*len(frames)
        text_prompt=prompt(row,question)
        request=dict(model_identity_sha256=ready['artifacts'][str(root/'model_identity.json')],
            prompt=text_prompt,option_order=option_order(row['view_id'],question),
            frame_sha256=[sha(p) for p in frame_paths],times=row['times'],
            view='static_counterfactual' if static else 'real_chronological',min_pixels=SPEC['min_pixels'],max_pixels=SPEC['max_pixels'])
        rid=digest(request);dest=root/'responses'/(rid+'.json')
        if dest.exists():return verify_response(request,read(dest))
        now=cards()
        if min(c['free_mib'] for c in now)<3072:
            for i in range(4):
                with torch.cuda.device(i):torch.cuda.empty_cache()
            now=cards()
        while min(c['free_mib'] for c in now)<3072:
            save(root/'progress.json',dict(stage='waiting_request_memory',candidate=row['candidate_id'],cards=now));time.sleep(10);now=cards()
        content=[]
        for i,(f,t) in enumerate(zip(frame_paths,row['times'])):
            region='BEFORE' if i<2 else 'INSIDE' if i<14 else 'AFTER'
            content.extend([dict(type='text',text=f'{region} frame at {t:.6f} seconds:'),
                dict(type='image',image=Path(f).as_uri(),min_pixels=SPEC['min_pixels'],max_pixels=SPEC['max_pixels'])])
        content.append(dict(type='text',text=text_prompt));messages=[dict(role='user',content=content)]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images,videos=process_vision_info(messages,image_patch_size=16)
        batch=processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt')
        active_key=(row['view_id'],static)
        pair=(batch['pixel_values'],batch['image_grid_thw'])
        if active_key in pixels:assert all(torch.equal(a,b) for a,b in zip(pixels[active_key],pair))
        else:pixels[active_key]=pair
        batch=batch.to(model.device);assert len(batch['input_ids'][0])<32768
        attempts=root/'attempts'/rid;attempts.mkdir(parents=True,exist_ok=True)
        attempt=attempts/(str(len(list(attempts.glob('*.json')))+1)+'.json')
        save(attempt,dict(status='running',request_id=rid,request=request,guard=str(guard),start_time=time.time()))
        t0=time.monotonic()
        with torch.inference_mode():out=model(**batch,use_cache=False,logits_to_keep=1,return_dict=True).logits[0,-1].float()
        primary_seconds=time.monotonic()-t0;forward_count=1
        tag='static' if static else question
        if tag not in proved_questions:
            model.model.visual.forward=native_visual
            for i in range(4):torch.cuda.synchronize(i)
            native_start=time.monotonic()
            with torch.inference_mode():native=model(**batch,use_cache=False,logits_to_keep=1,return_dict=True).logits[0,-1].float()
            for i in range(4):torch.cuda.synchronize(i)
            native_seconds=time.monotonic()-native_start
            model.model.visual.forward=visual_forward
            cache_start=time.monotonic()
            with torch.inference_mode():reused=model(**batch,use_cache=False,logits_to_keep=1,return_dict=True).logits[0,-1].float()
            for i in range(4):torch.cuda.synchronize(i)
            cached_seconds=time.monotonic()-cache_start
            assert torch.equal(out,native) and torch.equal(reused,native),'visual cache must preserve complete logits'
            proved_questions.add(tag);forward_count=3
            proofs[tag]=dict(request_id=rid,native_seconds=native_seconds,cached_seconds=cached_seconds,
                             complete_vocab_logits_exact=True)
            save(root/'GPU_acceptance.json',dict(passed=len(proved_questions)==4,all_four_real_devices=True,
                proofs=proofs,visual_feature_reuse_full_vocab_logits_exact=True,guard=str(guard),guard_sha256=sha(guard),
                model_identity_sha256=request['model_identity_sha256'],not_mAP_evidence=True))
        elapsed=time.monotonic()-t0
        values=out[ids].cpu().tolist();lse=float(torch.logsumexp(out,0).cpu());score(values,request['option_order'],lse)
        result=dict(request_id=rid,request=request,logits=values,logsumexp=lse,input_tokens=int(batch['input_ids'].shape[1]),
            seconds=elapsed,primary_forward_seconds=primary_seconds,forward_count=forward_count,
            actual_prefill_tokens=int(batch['input_ids'].shape[1])*forward_count,
            guard=str(guard),guard_sha256=sha(guard),attempt=str(attempt))
        save(dest,result);save(attempt,dict(status='complete',request_id=rid,result_sha256=sha(dest),
            seconds=elapsed,input_tokens=result['input_tokens'],actual_prefill_tokens=result['actual_prefill_tokens'],
            forward_count=forward_count,guard=str(guard)))
        calls+=1;return result
    # 四种请求都真实验收，包括首例不需要路由到的other查询；额外验收成本完整记录。
    first=next(r for r in rows if r['observable'])
    for question,static in [('presence',False),('predicted',False),('others',False),('presence',True)]:
        query(first,question,static)
    assert read(root/'GPU_acceptance.json')['passed']
    for n,row in enumerate(rows):
        dest=root/'cases'/(row['candidate_id']+'.json');case_key=digest(row)
        if dest.exists():assert read(dest)['input_sha256']==case_key;continue
        cache.clear();pixels.clear();responses={}
        if row['observable']:
            for f in row['frame_records']:assert sha(f['path'])==f['sha256']
            responses['presence']=query(row,'presence')
            responses['predicted']=query(row,'predicted')
            if score(responses['predicted']['logits'],option_order(row['view_id'],'predicted'))['state']=='absent':
                responses['others']=query(row,'others')
            responses['static']=query(row,'presence',True)
        save(dest,dict(candidate_id=row['candidate_id'],input_sha256=case_key,results=decisions(row,responses),
            response_files={k:dict(path=str(root/'responses'/(r['request_id']+'.json')),sha256=sha(root/'responses'/(r['request_id']+'.json'))) for k,r in responses.items()}))
        save(root/'progress.json',dict(stage='inference',completed_cases=n+1,total_cases=len(rows),new_calls=calls,
            elapsed_seconds=time.monotonic()-started,peak_reserved_mib={str(i):torch.cuda.max_memory_reserved(i)/2**20 for i in range(4)}))
        print('CASE',n+1,len(rows),'CALLS',calls,flush=True)
    save(root/'inference_complete.json',dict(cases=len(rows),new_calls_this_engine=calls,
        responses=len(list((root/'responses').glob('*.json'))),guard=str(guard),guard_sha256=sha(guard)))


def evaluate(root,group):
    import numpy as np,torch
    from tools.foreground_oracle_ap import ClassAP,prediction_frame,background_mask
    from tools.run_foreground_calibration_oracle import load_truth
    job=read(root/'groups.json')[group];entry=root/'groups'/group;assert sha(entry/'original.pt')==job['original_sha256']
    original=torch.load(entry/'original.pt',map_location='cpu');cfg=read(entry/'config.json')
    assert sha(entry/'config.json')==job['config_sha256']
    domain=job['domain'];splits=cfg['src_val_split'] if domain=='source_val' else cfg['tgt_train_split']
    path=cfg['dataset']['json_file'] if domain=='source_val' else cfg['baseline']['annotation_for_offline_evaluation']
    truth=load_truth(path,splits);truth=truth[truth['video-id']==job['video']].reset_index(drop=True)
    frame=prediction_frame([original]);bg=background_mask(truth,frame);engines={}
    for c in range(10):engines[c]=ClassAP(truth[truth.label==c],frame[frame.label==c].reset_index(drop=True))
    def metric(keep):
        rows=[engines[c].evaluate(keep[frame.label.to_numpy()==c]) for c in range(10)]
        return dict(map=float(np.mean([r['map'] for r in rows])),AP_by_tiou=np.mean([r['ap'] for r in rows],axis=0).tolist(),TP=np.sum([r['tp'] for r in rows],axis=0).tolist())
    baseline=metric(np.ones(len(frame),dtype=bool));results={};artifacts={str(entry/'original.pt'):sha(entry/'original.pt')}
    cases={cid:read(root/'cases'/(cid+'.json')) for cid in set(job['candidate_map'].values())}
    for cid in cases:artifacts[str(root/'cases'/(cid+'.json'))]=sha(root/'cases'/(cid+'.json'))
    for mode in MODES:
        keep=np.ones(len(frame),dtype=bool)
        for index,cid in job['candidate_map'].items():keep[int(index)]=not cases[cid]['results'][mode]['delete']
        assert keep[frame.score.to_numpy()<SPEC['cutoff']].all()
        control=np.ones(len(frame),dtype=bool)
        for c in range(10):
            idx=np.flatnonzero((frame.label.to_numpy()==c)&(frame.score.to_numpy()>=SPEC['cutoff']))
            n=int((~keep[idx]).sum());order=np.lexsort((idx,frame.score.to_numpy()[idx]));control[idx[order[:n]]]=False
        assert int((~keep).sum())==int((~control).sum())
        actual=metric(keep);confidence=metric(control)
        results[mode]=dict(actual=actual,delta_pp=100*(actual['map']-baseline['map']),
            matched_confidence_delta_pp=100*(confidence['map']-baseline['map']),
            removed=int((~keep).sum()),BG_removed=int((~keep&bg).sum()),nonBG_removed=int((~keep&~bg).sum()),
            CPU_BG_only_counterfactual=metric(keep|~bg),CPU_nonBG_only_counterfactual=metric(keep|bg))
    save(root/'evaluations'/(group+'.json'),dict(group=group,video=job['video'],baseline=baseline,results=results,
        scope='one complete selected video; all ten task classes; not full domain validation mAP',
        target_val_read=False,GT_only_CPU=True,artifacts=artifacts))


def audit(root):
    from tools.vlm_background_protocol import decisions
    plan=read(root/'plan.json');assert plan['spec']==SPEC
    manifest=read(root/'inference_manifest.json');assert not manifest['GT_in_inference_input']
    for row in manifest['cases']:
        case=read(root/'cases'/(row['candidate_id']+'.json'));assert case['input_sha256']==digest(row)
        responses={}
        for name,item in case['response_files'].items():
            assert sha(item['path'])==item['sha256'];r=read(item['path']);verify_response(r['request'],r)
            assert r['request']['prompt']==prompt(row,'presence' if name=='static' else name)
            expected=[f['sha256'] for f in row['frame_records']]
            if name=='static':expected=[expected[8]]*len(expected)
            assert r['request']['frame_sha256']==expected and r['request']['times']==row['times']
            assert r['request']['option_order']==option_order(row['view_id'],'presence' if name=='static' else name)
            assert r['request']['model_identity_sha256']==sha(root/'model_identity.json')
            assert sha(r['guard'])==r['guard_sha256']
            g=read(r['guard']);assert sha(g['lease'])==g['lease_sha256']
            lease=read(g['lease']);assert g['owner_pid']==lease['owner_pid']
            assert [x['uuid'] for x in g['observed']]==[x['uuid'] for x in lease['cards']]
            assert min(x['free_mib'] for x in g['observed'])>=MIN_FREE
            responses[name]=r
        assert case['results']==decisions(row,responses)
    results={}
    for group,job in read(root/'groups.json').items():
        path=root/'evaluations'/(group+'.json');r=read(path)
        for p,h in r['artifacts'].items():assert sha(p)==h
        results[group]=dict(sha256=sha(path),results=r['results'])
    save(root/'final.json',dict(passed=True,groups=results,methods=list(MODES),scope=SPEC['scope'],
        exact_cases=len(manifest['cases']),exact_groups=len(results),formal_training=False,
        inference_complete_sha256=sha(root/'inference_complete.json')))


def run(root,resume=False):
    import fcntl
    if not resume:
        root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(code=code_identity(),spec=SPEC,owner_pid=os.getpid()))
    else:
        assert read(root/'plan.json')['code']==code_identity()
        assert (root/'prepare_complete.json').exists(),'partial preparation needs an explicit independent repair'
        assert (root/'GPU_acceptance.json').exists() and read(root/'GPU_acceptance.json')['passed'],'incomplete first GPU acceptance needs independent repair'
        assert not (root/'stop.json').exists(),'user stop revokes automatic restart'
        previous=read(root/'state.json') if (root/'state.json').exists() else {}
        assert all(not Path('/proc/'+str(pid)).exists() for pid in previous.get('active',{}).values()),'old owned workers still alive'
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);active={}
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,owner_pid=os.getpid(),automatic_restart_authorized=False))
        for p in active.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def spawn(stage,group=None,lease=None):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        if lease:
            env.update(CUDA_VISIBLE_DEVICES=','.join(c['uuid'] for c in read(lease)['cards']),VLM_BG_LEASE=str(lease),VLM_BG_LEASE_SHA256=sha(lease))
        name=group or stage;log=(root/(name+'.log')).open('a')
        args=[VLM_PY if stage in ['prepare','infer'] else CPU_PY,'-u','-B',str(Path(__file__)),'--root',str(root),'--stage',stage]
        if group:args+=['--group',group]
        p=subprocess.Popen(args,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();active[name]=p;return p
    try:
        if not (root/'prepare_complete.json').exists():
            p=spawn('prepare');assert p.wait()==0,'CPU preparation failed';del active['prepare']
        groups=read(root/'groups.json');evaluated=set();engine=None;cpu=None;cpu_group=None
        while len(evaluated)<len(groups) or not (root/'inference_complete.json').exists() or engine is not None or cpu is not None:
            if engine is None and not (root/'inference_complete.json').exists():
                snapshot=cards()
                if len(snapshot)==4 and min(c['free_mib'] for c in snapshot)>=MIN_FREE:
                    lease=root/'leases'/(str(os.getpid())+'.json');save(lease,dict(owner_pid=os.getpid(),cards=snapshot,minimum_free_mib=MIN_FREE,code=code_identity()))
                    engine=spawn('infer',lease=lease)
            if engine is not None and engine.poll() is not None:
                assert engine.returncode==0,'VLM engine failed; preserve all request outputs';del active['infer'];engine=None
            if cpu is not None and cpu.poll() is not None:
                assert cpu.returncode==0,'CPU evaluation failed';evaluated.add(cpu_group);del active[cpu_group];cpu=None
            for name,job in groups.items():
                if (root/'evaluations'/(name+'.json')).exists():evaluated.add(name)
                elif cpu is None and all((root/'cases'/(cid+'.json')).exists() for cid in job['candidate_map'].values()):
                    cpu=spawn('evaluate',name);cpu_group=name
            save(root/'state.json',dict(owner_pid=os.getpid(),stage='running' if engine else 'waiting_GPU_or_CPU',
                active={k:p.pid for k,p in active.items()},complete_groups=sorted(evaluated),total_groups=len(groups),cards=cards()))
            if len(evaluated)<len(groups) or not (root/'inference_complete.json').exists():time.sleep(10)
        audit(root)
    except Exception as e:
        save(root/'error.json',dict(error=repr(e),owner_pid=os.getpid(),time=time.time()));raise
    finally:
        for p in active.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['run','prepare','infer','evaluate','audit'],default='run')
    p.add_argument('--group');p.add_argument('--resume',action='store_true');a=p.parse_args()
    os.chdir('/vepfs/vlms/SADA_PLUS')
    if a.stage=='run':run(a.root,a.resume)
    elif a.stage=='evaluate':evaluate(a.root,a.group)
    else:dict(prepare=prepare,infer=infer,audit=audit)[a.stage](a.root)
