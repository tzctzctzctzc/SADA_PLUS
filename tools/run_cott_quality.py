"""Qwen3-VL-32B完整CoTT Labeler；请求级原子保存/接续，不读取任何GT文件。"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.cott_protocol import *
from tools.cott_frames import refined_stream

MODEL_IDENTITY=Path('/vepfs/vlms/qwen8b32b_source_0adb413_1030/32b_model_identity.json')
UUIDS=['GPU-4a240306-a05c-ccd6-dade-76204303c5b2','GPU-81af5f84-c16b-c951-f536-37d32a7cc269',
    'GPU-36980d73-6348-6626-2104-2f8b2480eb27','GPU-9b8189b9-ec48-f373-1201-5b1e1d2eb1b5']

def gpu_snapshot():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(a),uuid=b.strip(),free_mib=int(c),util=int(d))
        for a,b,c,d in (line.split(',') for line in out.strip().splitlines())]

class Generator:
    def __init__(self,root):
        import torch
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        self.root=root; torch.set_num_threads(8)
        identity=read(MODEL_IDENTITY); modelpath=Path(identity['path'])
        current_stat={name:dict(size=(modelpath/name).stat().st_size,mtime_ns=(modelpath/name).stat().st_mtime_ns)
            for group in ['weight_sha256','processor_sha256'] for name in identity[group]}
        receipt=root/'model_identity.json'
        if receipt.exists():
            old=read(receipt);assert old['identity']==identity and old['file_stat']==current_stat
        else:
            for group in ['weight_sha256','processor_sha256']:
                for name,h in identity[group].items(): assert sha(modelpath/name)==h, name
            save(receipt,dict(identity=identity,file_stat=current_stat,verified_all_sha256=True))
        self.model_sha=sha(receipt)
        cards=gpu_snapshot();assert [x['uuid'] for x in cards]==UUIDS
        assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(UUIDS)
        while any(x['free_mib']<24576 for x in cards):
            save(root/'state.json',dict(phase='waiting_memory',cards=cards,minimum_free_mib=24576))
            time.sleep(15);cards=gpu_snapshot()
        save(root/'first_cuda.json',dict(cards=cards,pid=os.getpid(),parent=os.getppid(),
            minimum_free_mib=24576,shared_gpus=True,public_lock=False,
            policy='user explicitly allows shared four-GPU jobs; no other processes inspected or signalled'))
        self.processor=AutoProcessor.from_pretrained(modelpath,local_files_only=True)
        self.model,loading=Qwen3VLForConditionalGeneration.from_pretrained(modelpath,
            dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,
            output_loading_info=True,device_map='balanced',max_memory={i:'18000MiB' for i in range(4)})
        assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        assert {str(x).replace('cuda:','') for x in self.model.hf_device_map.values()}=={'0','1','2','3'}
        self.model.eval().requires_grad_(False)
        save(root/'device_map.json',dict(device_map=self.model.hf_device_map,loading=loading,
            model_sha256=self.model_sha,trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        print('MODEL_READY',flush=True)

    def call(self,video_id,call_id,stage,prompt,frames,lo,hi,pair_count=0):
        entry=self.root/'calls'/video_id/call_id; entry.mkdir(parents=True,exist_ok=True)
        for f in frames: assert sha(f['path'])==f['sha256']
        req=dict(stage=stage,system=SYSTEM,prompt=prompt,frames=frames,lo=lo,hi=hi,pair_count=pair_count,
            model_sha256=self.model_sha,spec_sha256=digest(SPEC))
        h=digest(req)
        if (entry/'result.json').exists():
            old=read(entry/'result.json')
            if old['request_sha256']!=h:
                # 仅明确绑定的、未改变提示/帧/模型的成功transition/direct前缀可复用。
                assert stage in ['transition','direct']
                receipt=self.root/'accepted_prefix_reuse.json'
                assert receipt.exists(), 'changed request requires explicit immutable-prefix binding'
                binding=read(receipt)['calls'][video_id+'/'+call_id]
                assert sha(entry/'result.json')==binding['result_sha256']
                prior=read(entry/'request.json')
                assert digest(prior)==old['request_sha256']==binding['request_sha256']
                assert dict(prior,spec_sha256=req['spec_sha256'])==req,'reused call changed beyond grouping-only SPEC'
            assert sha(entry/'request.json')==old['request_file_sha256']
            for name,v in old['attempt_sha256'].items(): assert sha(entry/name)==v
            return validate(old['parsed'],stage,lo,hi,pair_count)
        if (entry/'request.json').exists(): assert read(entry/'request.json')==req
        else: save(entry/'request.json',req)
        error=None; previous=None; attempts={}
        for attempt in range(SPEC['schema_retries']+1):
            ap=entry/f'attempt_{attempt}.json'
            if ap.exists():
                old=read(ap);assert old['request_sha256']==h
                if old['status']=='valid':
                    attempts[ap.name]=sha(ap)
                    result=validate(old['parsed'],stage,lo,hi,pair_count)
                    save(entry/'result.json',dict(request_sha256=h,request_file_sha256=sha(entry/'request.json'),
                        attempt_sha256=attempts,parsed=result));return result
                error=old['error'];previous=old['raw'];attempts[ap.name]=sha(ap);continue
            import torch
            from qwen_vl_utils import process_vision_info
            content=[]
            for i,f in enumerate(frames):
                content += [dict(type='text',text=f"FRAME {i}, absolute time {f['time_sec']:.6f} seconds"),
                    dict(type='image',image=Path(f['path']).resolve().as_uri(),
                        min_pixels=SPEC['min_pixels'],max_pixels=SPEC['max_pixels'])]
            content.append(dict(type='text',text=prompt))
            if attempt:
                content.append(dict(type='text',text='A prior attempt failed the output contract: '+error+
                    ' Recheck the same evidence and fill the declared output slots correctly. '
                    'Do not invent actions or copy an invalid per-frame caption list.'))
            messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)]
            formatted=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            if frames:
                images,videos=process_vision_info(messages,image_patch_size=16)
                batch=self.processor(text=[formatted],images=images,videos=videos,padding=True,return_tensors='pt')
            else:
                batch=self.processor(text=[formatted],padding=True,return_tensors='pt')
            n=batch['input_ids'].shape[1]
            assert n<=SPEC['max_input_tokens'], 'context overflow: full evidence cannot be silently truncated'
            cards=gpu_snapshot()
            assert all(x['free_mib']>=3072 for x in cards), 'insufficient generation headroom, no precision/batch fallback'
            batch=batch.to(self.model.device); started=time.monotonic()
            with torch.inference_mode():
                output=self.model.generate(**batch,do_sample=False,max_new_tokens=SPEC['max_new_tokens'][stage],
                    use_cache=True,pad_token_id=self.processor.tokenizer.pad_token_id)
            ids=output[0,n:]; raw=self.processor.tokenizer.decode(ids,skip_special_tokens=True)
            elapsed=time.monotonic()-started
            eos=self.model.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else eos
            truncated=len(ids)>=SPEC['max_new_tokens'][stage] and int(ids[-1]) not in (eos or [])
            parsed=None
            try:
                if truncated: raise ValueError('response truncated at token budget')
                parsed=validate(parse_object(raw),stage,lo,hi,pair_count);status='valid';error=None
            except (ValueError,TypeError,KeyError,AttributeError) as exc:
                status='invalid';error=str(exc)
            save(ap,dict(status=status,request_sha256=h,raw=raw,parsed=parsed,error=error,
                truncated=truncated,input_tokens=n,output_tokens=len(ids),elapsed_sec=elapsed,
                peak_reserved_mib=[torch.cuda.max_memory_reserved(i)/2**20 for i in range(4)]))
            attempts[ap.name]=sha(ap)
            print('CALL',video_id,call_id,attempt,status,round(elapsed,2),flush=True)
            if status=='valid':
                save(entry/'result.json',dict(request_sha256=h,request_file_sha256=sha(entry/'request.json'),
                    attempt_sha256=attempts,parsed=parsed));return parsed
            previous=raw
        raise RuntimeError(f'{video_id}/{call_id}: schema failure after recorded retries: {error}')

def run_video(gen,row):
    root=gen.root; v=row['video_id']; entry=root/'videos'/v
    if (entry/'complete.json').exists():
        done=read(entry/'complete.json')
        assert done['video_sha256']==row['video_sha256']
        for name,h in done['artifacts'].items(): assert sha(entry/name)==h
        return
    frames=row['frames']; transition_rows={}; direct=[]; requests=[]
    for wi,indices in enumerate(windows(len(frames))):
        subset=[frames[i] for i in indices]; lo=subset[0]['time_sec'];hi=subset[-1]['time_sec']
        tc=f'transition_{wi:04d}';dc=f'direct_{wi:04d}'
        result=gen.call(v,tc,'transition',transition_prompt(subset),subset,lo,hi,len(subset)-1)
        requests.append(tc)
        for j,r in enumerate(result['transitions']):
            global_pair=indices[j]
            transition_rows.setdefault(global_pair,dict(start=subset[j]['time_sec'],end=subset[j+1]['time_sec'],observations=[]))['observations'].append(r)
        result=gen.call(v,dc,'direct',direct_prompt(lo,hi),subset,lo,hi)
        requests.append(dc)
        direct.extend(dict(r,video_id=v,origin_call=dc) for r in result['intervals'])
    assert set(transition_rows)==set(range(len(frames)-1))
    transitions=[dict(pair=i,**r) for i,r in sorted(transition_rows.items())]
    save(entry/'transitions.json',transitions)
    grouped=gen.call(v,'grouping','grouping',grouping_prompt(transitions,row['duration']),[],0.,row['duration'])
    requests.append('grouping');save(entry/'primitive_intervals.json',grouped)
    refined=[]; rejected=[]
    for i,r in enumerate(grouped['intervals']):
        pad=max(1.,.25*(r['end']-r['start']));lo=max(0.,r['start']-pad);hi=min(row['duration']-.001,r['end']+pad)
        subset=refined_stream(row['path'],lo,hi,root/'frames'/v/f'refine_{i:04d}')
        # The legal context uses requested bounds; actual PTS may differ by <= one frame.
        lo=min(lo,subset[0]['time_sec']);hi=max(hi,subset[-1]['time_sec'])
        rid=f'refine_{i:04d}'
        result=gen.call(v,rid,'refine',refine_prompt(r,lo,hi),subset,lo,hi)
        requests.append(rid)
        if result['valid']: refined.append(dict(result,video_id=v,origin_call=rid,primitive=r))
        else: rejected.append(dict(primitive=r,result=result))
    direct_kept,direct_removed=deduplicate([p for p in direct if p['label']!='OTHER'])
    cott_kept,cott_removed=deduplicate([p for p in refined if p['label']!='OTHER'])
    save(entry/'predictions.json',dict(direct=direct_kept,cott=cott_kept,
        raw_direct=direct,raw_refined=refined,rejected=rejected,
        dedup_direct=direct_removed,dedup_cott=cott_removed))
    calls={rid:sha(root/'calls'/v/rid/'result.json') for rid in requests}
    cost={}
    for stage in ['transition','grouping','refine','direct']:
        attempts=[read(p) for rid in requests if rid.startswith(stage) for p in (root/'calls'/v/rid).glob('attempt_*.json')]
        cost[stage]=dict(calls=sum(rid.startswith(stage) for rid in requests),attempts=len(attempts),
            input_tokens=sum(x['input_tokens'] for x in attempts),output_tokens=sum(x['output_tokens'] for x in attempts),
            elapsed_sec=sum(x['elapsed_sec'] for x in attempts),invalid_attempts=sum(x['status']!='valid' for x in attempts))
    save(entry/'cost.json',cost)
    save(entry/'complete.json',dict(schema=SCHEMA,video_id=v,video_sha256=row['video_sha256'],
        frames=len(frames),transitions=len(transitions),call_result_sha256=calls,
        artifacts={name:sha(entry/name) for name in ['transitions.json','primitive_intervals.json','predictions.json','cost.json']},
        target_gt_used=False,training=False))
    print('VIDEO_COMPLETE',v,len(cott_kept),len(direct_kept),flush=True)

def main(root):
    root=Path(root);manifest=read(root/'manifest.json');plan=read(root/'plan.json')
    assert sha(root/'manifest.json')==plan['manifest_sha256']
    assert manifest['spec']==SPEC;no_annotations(manifest)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    assert code==plan['code']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
    save(root/'runtime.json',dict(code=code,pid=os.getpid(),parent=os.getppid(),manifest_sha256=sha(root/'manifest.json')))
    gen=Generator(root)
    rows=sorted(manifest['videos'],key=lambda x:(x['duration'],x['video_id']))
    for i,row in enumerate(rows):
        save(root/'state.json',dict(phase='generating',video_id=row['video_id'],complete_videos=i,total_videos=len(rows)))
        run_video(gen,row)
    save(root/'inference_complete.json',dict(schema=SCHEMA,count=len(rows),manifest_sha256=sha(root/'manifest.json'),
        videos={r['video_id']:sha(root/'videos'/r['video_id']/'complete.json') for r in rows}))
    save(root/'state.json',dict(phase='inference_complete',complete_videos=len(rows),total_videos=len(rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    try: main(a.root)
    except BaseException as exc:
        save(Path(a.root)/'error.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        raise
