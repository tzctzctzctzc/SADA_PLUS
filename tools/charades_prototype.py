"""完整Charades词表/原始视频教师缓存；目标GT从不进入候选、教师和选择器。"""
from collections import defaultdict
from copy import deepcopy
import hashlib,importlib.util,json,math,random
from pathlib import Path
from tools.baseline_protocol import read,save,sha
from tools.prepare_charades_inputs import LABEL_NAMES

SCHEMA='charades-prototype-fixed40of200-v1'
MODEL=Path('/vepfs/vlms/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
POLICY=dict(schema=SCHEMA,teacher='qwen3_vl_8b',labels=LABEL_NAMES,candidate_count=200,selected_count=40,
    proposal_model='Charades-UDA-source-loss-selected-EMA',candidate_seed=6,levels=[0,1,2],
    candidate_threshold='frozen-Charades-baseline-pl_threshold',per_video_cap=30,
    candidate_eligibility='eight-distinct-frames-and-fully-supported-original-feature-grid',
    participation='top20-percent-max-class-probability-no-top1-ties',
    frames=8,input_mode='native_video',readout='ten-digit-single-token-log-likelihood',
    min_pixels=224*224,max_pixels=448*448,temperature=1.,label_order='frozen-official-verb-ids',
    embedding_dim=128,target_alignment_weight=.1,source_weight=1.,
    variants=['adversarial','no_adversarial'],source_selection='original-Charades-source-validation-loss',
    seed=123,batch=2,max_epochs=50,patience_epochs=15,target_GT_used=False)
PHRASES=('taking or picking up an object','putting or placing an object down','sitting or sitting down',
    'walking','holding an object','standing or standing up','opening something','drinking',
    'smiling','laughing')

def mapping(candidate):
    seed=int.from_bytes(hashlib.sha256(str(candidate).encode()).digest()[:8],'big')
    names=list(LABEL_NAMES);random.Random(seed).shuffle(names);return tuple(names)

def prompt(candidate,frame_count=8,prompt_mode='baseline',input_mode='native_video'):
    if frame_count!=8 or prompt_mode!='baseline' or input_mode!='native_video':raise ValueError('冻结Charades教师接口')
    definitions=dict(zip(LABEL_NAMES,PHRASES))
    options='\n'.join('%d: %s (%s)'%(i,name,definitions[name]) for i,name in enumerate(mapping(candidate)))
    return ('The video contains eight chronological frames from one temporal proposal. '
        'Classify the single dominant visible human action in this clip. '
        'Use the observed action, without inventing unseen events. Choose exactly one option.\n'
        +options+'\nReply with exactly one digit from 0 to 9.')

def interface():
    # 独立模块命名空间，既复用已核读单token实现，又不改变厨房默认词表/提示。
    path=Path(__file__).with_name('qwen_source_interface.py')
    spec=importlib.util.spec_from_file_location('charades_specific_qwen_interface',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.LABELS=LABEL_NAMES;module.LABEL_TO_ID={name:i for i,name in enumerate(LABEL_NAMES)}
    module.digit_to_label=mapping;module.build_prompt=prompt
    return module

def choose(rows,count=200):
    key=lambda row:hashlib.sha256(('6:'+row['candidate_id']).encode()).hexdigest()
    by=defaultdict(list)
    for row in sorted(rows,key=key):by[row['video_id']].append(row)
    return sorted([row for vid in sorted(by) for row in by[vid][:30]],key=key)[:count]

def participate(rows):
    good=[r for r in rows if r['class_probabilities'].count(max(r['class_probabilities']))==1]
    return sorted(good,key=lambda r:(-max(r['class_probabilities']),r['video_id'],r['absolute_feature_index']))[:len(rows)//5]

def without_gt(video):
    import torch
    metadata=deepcopy(video['metadata'])
    for key in ['annotations','labels','segments']:metadata.pop(key,None)
    return dict(feats=video['feats'],metadata=metadata,segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long))

def dense(model,video):
    if video['labels'].numel() or video['segments'].numel():raise ValueError('无标签接口禁止GT')
    x,valid=model.preprocessing([video],mode='test')
    f,m,_=model.backbone_target(model.dropout(x),valid);f,m=model.neck_target(f,m)
    logits,_=model.cls_head_target(f,m);offsets,_=model.reg_head_target(f,m)
    return dict(fpn=f,masks=m,logits=logits,offsets=offsets,points=model.point_generator(f))

def prediction(model,video):
    packet=dense(model,video)
    raw=model.inference_single_video(packet['points'],[m[0,0] for m in packet['masks']],
        [v[0].transpose(0,1) for v in packet['logits']],[v[0].transpose(0,1) for v in packet['offsets']],
        [None]*len(packet['points']),[None]*len(packet['points']))
    meta=video['metadata']
    return model.postprocessing([dict(raw,video_id=meta['video_id'],fps=meta['fps'],duration=meta['duration'],
        video_length=meta['video_length'],feat_stride=meta['feat_stride'],feat_num_frames=meta['feat_num_frames'],
        use_temporal_grid_feats=meta['use_temporal_grid_feats'])])[0]

def candidates(root,baseline):
    import torch
    from libs.models.SADA import make_meta_arch
    from tools.baseline_engine import dataset
    cfg=read(baseline/'config.json');saved=torch.load(baseline/'best.pt',map_location='cpu')
    if saved['config']!=cfg:raise ValueError('候选源模型配置改变')
    torch.manual_seed(123);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda().eval()
    model.load_state_dict(saved['ema_state_dict'],strict=True)
    ds=dataset(cfg,cfg['baseline']['target_manifest'],cfg['tgt_train_split'],False,True)
    members=read(cfg['baseline']['target_manifest'])['database']
    if any('annotations' in row for row in members.values()):raise ValueError('目标manifest中出现标注')
    all_rows=[]
    with torch.no_grad():
        for index in range(len(ds)):
            v=without_gt(ds[index]);packet=dense(model,v);meta=v['metadata'];grouped=defaultdict(list)
            for level in POLICY['levels']:
                scores=packet['logits'][level][0].sigmoid().amax(0)
                keep=(packet['masks'][level][0,0] & (scores>=cfg['model']['pl_threshold'])).nonzero().flatten()
                pts=packet['points'][level][keep].cpu();offsets=packet['offsets'][level][0,:,keep].transpose(0,1).cpu();scores=scores[keep].cpu()
                for p,o,score in zip(pts,offsets,scores):
                    coordinate=float(p[0]);absolute=int(round(coordinate))
                    start=max(0.,min(meta['duration'],((coordinate-float(o[0])*float(p[3]))*meta['feat_stride']+.5*meta['feat_num_frames'])/meta['fps']))
                    end=max(0.,min(meta['duration'],((coordinate+float(o[1])*float(p[3]))*meta['feat_stride']+.5*meta['feat_num_frames'])/meta['fps']))
                    if end-start<8/meta['fps']:continue
                    # 先满足原Prototype fully_visible_cache的几何定义再冻结200条，
                    # 不能让200条评分完成后才发现视频边界proposal永远无法参与训练。
                    a=(start*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']
                    b=(end*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']
                    if a<0 or b>v['feats'].shape[-1]:continue
                    row=dict(candidate_id='%s-p%07d'%(meta['video_id'],absolute),video_id=meta['video_id'],
                        absolute_feature_index=absolute,source_fpn_levels=[level],proposal_start_sec=start,proposal_end_sec=end)
                    grouped[absolute].append((float(score),row))
            all_rows.extend(max(values,key=lambda z:z[0])[1] for values in grouped.values())
            if index%200==0:print('UNLABELED_CANDIDATES',index+1,len(ds),len(all_rows),flush=True)
    pool=choose(all_rows)
    if len(pool)!=200:raise ValueError('冻结阈值下有效候选不足200，禁止按目标结果降阈值')
    save(root/'candidate_pool.json',dict(policy=POLICY,candidates=pool,target_members=sorted(members),
        target_manifest=cfg['baseline']['target_manifest'],target_manifest_sha256=sha(cfg['baseline']['target_manifest']),
        checkpoint=str(baseline/'best.pt'),checkpoint_sha256=sha(baseline/'best.pt'),
        eligible=len(all_rows),target_GT_used=False))

def frames(root,raw):
    from concurrent.futures import ThreadPoolExecutor
    from tools.refresh_frames import video_paths_by_id,prepare_video_candidates
    pool=read(root/'candidate_pool.json');grouped=defaultdict(list);paths=video_paths_by_id(raw)
    for row in pool['candidates']:grouped[row['video_id']].append(row)
    destination=root/'frames';destination.mkdir(exist_ok=False)
    rows=[]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures=[executor.submit(prepare_video_candidates,v,r,paths,None,destination) for v,r in grouped.items()]
        for future in futures:
            values,rejected=future.result()
            if rejected:raise ValueError('候选不足八个不同帧，保留现场不补抽')
            rows.extend(values)
    assert len(rows)==200
    rows.sort(key=lambda x:x['candidate_id'])
    save(root/'frames_manifest.json',dict(candidates=rows,frame_root=str(destination),
        candidate_pool_sha256=sha(root/'candidate_pool.json'),raw_video_root=str(raw),target_GT_used=False))

def score(root,shard):
    import torch,numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from tools.refresh_frames import pixel_hash
    reader=interface();manifest=read(root/'frames_manifest.json');teacher=root/'teacher'/str(shard);teacher.mkdir(parents=True,exist_ok=False)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    model,info=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map={'':0},local_files_only=True,output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('教师加载不完整')
    if {str(v).replace('cuda:','') for v in model.hf_device_map.values()}!={'0'}:raise ValueError('教师必须完整在获准单卡，禁止CPU/disk卸载')
    model.eval().requires_grad_(False)
    save(teacher/'device_map.json',dict(map={k:str(v) for k,v in model.hf_device_map.items()},uuid=__import__('os').environ['CUDA_VISIBLE_DEVICES']))
    runtime=dict(torch=torch,model=model,processor=processor,process_vision_info=process_vision_info,
        digit_token_ids=reader.resolve_digit_token_ids(processor.tokenizer),device=model.device)
    args=SimpleNamespace(frame_count=8,prompt_mode='baseline',input_mode='native_video',
        min_pixels=POLICY['min_pixels'],max_pixels=POLICY['max_pixels'],score_temperature=1.,uniform_mixed_frame_grid=True)
    rows=manifest['candidates'][shard::2]
    with (teacher/'scores.jsonl').open('x') as out:
        for index,row in enumerate(rows):
            paths=[Path(manifest['frame_root'])/'frames'/row['candidate_id']/(str(i)+'.png') for i in row['frames']]
            actual=[]
            for p in paths:
                with Image.open(p) as image:actual.append(pixel_hash(np.asarray(image.convert('RGB'))))
            if actual!=row['frame_sha256']:raise ValueError('评分帧像素SHA改变')
            result=reader.infer_one(runtime,paths,row['candidate_id'],args,sample_fps=8/(row['proposal_end_sec']-row['proposal_start_sec']))
            record=dict(candidate_id=row['candidate_id'],digit_logits=[result['class_logits'][reader.LABEL_TO_ID[name]] for name in mapping(row['candidate_id'])],
                class_probabilities=result['class_probabilities'],class_id=result['class_id'],prompt_sha256=result['prompt_sha256'],frame_sha256=actual,
                candidate_mass=result['candidate_mass'])
            out.write(json.dumps(record,allow_nan=False)+'\n');out.flush()
            if index%10==0:print('CHARADES_QWEN',shard,index+1,len(rows),flush=True)
    save(teacher/'complete.json',dict(status='passed',count=len(rows),scores_sha256=sha(teacher/'scores.jsonl'),
        manifest_sha256=sha(root/'frames_manifest.json'),model=str(MODEL),policy=POLICY,target_GT_used=False,
        teacher_model_identity_sha256=sha(root/'teacher_model_identity.json'),device_map_sha256=sha(teacher/'device_map.json')))

def freeze(root):
    reader=interface();manifest=read(root/'frames_manifest.json');by={}
    for shard in [0,1]:
        p=root/'teacher'/str(shard);proof=read(p/'complete.json')
        if proof['count']!=100 or proof['scores_sha256']!=sha(p/'scores.jsonl') or proof['manifest_sha256']!=sha(root/'frames_manifest.json'):raise ValueError('教师评分不完整')
        if proof['teacher_model_identity_sha256']!=sha(root/'teacher_model_identity.json') or proof['device_map_sha256']!=sha(p/'device_map.json'):raise ValueError('教师身份或实际放置改变')
        for line in (p/'scores.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['candidate_id'] in by:raise ValueError('评分重复')
            by[r['candidate_id']]=r
    entries=[]
    for candidate in manifest['candidates']:
        p=by[candidate['candidate_id']];computed=reader.normalize_digit_logits(p['digit_logits'],mapping(candidate['candidate_id']))
        if p['class_probabilities']!=computed['class_probabilities'] or p['class_id']!=computed['class_id']:raise ValueError('教师概率复算不一致')
        if p['prompt_sha256']!=hashlib.sha256(prompt(candidate['candidate_id']).encode()).hexdigest() or p['frame_sha256']!=candidate['frame_sha256']:raise ValueError('教师输入身份不同')
        entries.append(dict(candidate,class_probabilities=p['class_probabilities'],class_id=p['class_id']))
    if len(entries)!=200 or len(by)!=200:raise ValueError('非精确200条教师评分')
    chosen={r['candidate_id'] for r in participate(entries)}
    if len(chosen)!=40:raise ValueError('不能得到40条无top1并列的候选')
    by_video=defaultdict(list)
    for entry in entries:
        row={k:entry[k] for k in ['candidate_id','video_id','absolute_feature_index','source_fpn_levels','proposal_start_sec','proposal_end_sec','class_probabilities']}
        row.update(semantic_label=LABEL_NAMES[entry['class_id']],participation_selected=entry['candidate_id'] in chosen)
        by_video[row['video_id']].append(row)
    save(root/'prepared.json',dict(schema=SCHEMA,teacher='qwen3_vl_8b',label_names=LABEL_NAMES,by_video=dict(by_video),
        selected_count=40,candidate_count=200,policy=POLICY,target_GT_used=False,
        source_artifacts={str(root/'candidate_pool.json'):sha(root/'candidate_pool.json'),str(root/'frames_manifest.json'):sha(root/'frames_manifest.json'),
            **{str(root/'teacher'/str(i)/'complete.json'):sha(root/'teacher'/str(i)/'complete.json') for i in [0,1]}}))
    return root/'prepared.json'
