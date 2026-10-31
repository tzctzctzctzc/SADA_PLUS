"""S1源验证动作的上下文干预；固定GT探针与检测输出同时报告，不训练。"""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha, code_identity

BASE = Path('/vepfs/vlms/prototype_no_adv6_9edee06_1031/formal/S1/prototype')
NEW = Path('/vepfs/vlms/prototype_context_mix18_d99d4a6_1031/formal/S1')
MODELS = dict(control=str(BASE), **{v:str(NEW/v) for v in
              ('source_context','target_context','target_context_pair')})


def iou(a, b):
    return max(0., min(a[1],b[1])-max(a[0],b[0]))/max(1e-8,max(a[1],b[1])-min(a[0],b[0]))


def summarize_views(rows, gt, label):
    """未检出计失败；固定GT探针非检测mAP。坐标均减去插入偏移。"""
    duration = gt[1]-gt[0]
    pairs = list(itertools.combinations(rows, 2))
    return dict(class_flip_rate=statistics.mean(a['class_id']!=b['class_id'] for a,b in pairs),
        score_std=statistics.pstdev(r['gt_score'] for r in rows),
        boundary_pair_l1=statistics.mean(sum(abs(x-y) for x,y in zip(a['box'],b['box']))/(2*duration) for a,b in pairs),
        boundary_error=statistics.mean(sum(abs(x-y) for x,y in zip(r['box'],gt))/(2*duration) for r in rows),
        mean_iou=statistics.mean(iou(r['box'],gt) for r in rows),
        class_accuracy=statistics.mean(r['class_id']==label for r in rows),
        detection_miss_rate=statistics.mean(r['detection'] is None for r in rows),
        detection_success_05=statistics.mean(r['detection'] is not None and r['detection']['class_id']==label
            and iou(r['detection']['box'],gt)>=.5 for r in rows))


def prepare(root):
    import numpy as np
    from tools.baseline_engine import dataset
    from tools.context_mix import feature_length, patch_record
    root.mkdir(parents=True,exist_ok=False)
    cfg=read(BASE/'config.json');options=cfg['dataset'];videos=[]
    native=dataset(cfg,options['json_file'],cfg['src_val_split'],False)
    training=read(options['json_file'])['database']
    source_train={v for v,x in training.items() if x['subset'].lower() in cfg['src_train_split']}
    for item in native.data_list:
        video=native[list(x['id'] for x in native.data_list).index(item['id'])]
        path=Path(options['feat_folder'])/((options.get('file_prefix') or '')+item['id']+options['file_ext'])
        videos.append(dict(video_id=item['id'],path=str(path),length=video['feats'].shape[-1],
            segments=video['segments'].tolist(),labels=video['labels'].tolist(),feature_sha256=sha(path)))
    assert not source_train & {v['video_id'] for v in videos}
    source_count=len(videos)
    target=read(cfg['baseline']['target_manifest'])['database']
    assert all('annotations' not in x for x in target.values())
    for vid in target:
        path=Path(options['feat_folder'])/((options.get('file_prefix') or '')+vid+options['file_ext'])
        videos.append(dict(video_id=vid,path=str(path),length=feature_length(path,options['downsample_rate']),
            segments=[],labels=[],feature_sha256=sha(path)))
    cases=[];eligible={str(c):0 for c in range(10)}
    radius=max(1,math.ceil(.5*options['num_frames']/(options['feat_stride']*options['downsample_rate'])))
    for index,row in enumerate(videos[:source_count]):
        for j,(gt,label) in enumerate(zip(row['segments'],row['labels'])):
            patch=patch_record(row['segments'],row['labels'],j,row['length'],max(radius,math.ceil((gt[1]-gt[0])*.25)))
            if patch is None or len(patch['labels'])!=1:continue
            n=min(512,row['length']);width=patch['end']-patch['start']
            if width>n:continue
            start=max(0,min(row['length']-n,(patch['start']+patch['end']-n)//2))
            u=patch['start']-start
            if not 0<=u<=n-width:continue
            sources=[i for i,x in enumerate(videos[:source_count]) if i!=index and x['length']>=n]
            targets=[i for i in range(source_count,len(videos)) if videos[i]['length']>=n]
            if min(len(sources),len(targets))<3:continue
            ident=row['video_id']+':'+str(j);rng=random.Random(ident+':context-stability-v1')
            contexts=[dict(kind='native',video=index,start=start)]
            for kind,pool in [('source',sources),('target',targets)]:
                for k in rng.sample(pool,3):
                    contexts.append(dict(kind=kind,video=k,start=rng.randint(0,videos[k]['length']-n)))
            eligible[str(label)]+=1
            cases.append(dict(id=ident,video=index,patch=patch,length=n,insert=u,label=label,contexts=contexts))
    # 每类最多10个；仅按固定ID散列，不读取预测或目标标签。
    cases=sorted(cases,key=lambda x:hashlib.sha256(x['id'].encode()).hexdigest())
    selected=[];counts={str(c):0 for c in range(10)}
    for case in cases:
        key=str(case['label'])
        if counts[key]<10:selected.append(case);counts[key]+=1
    if not selected:raise ValueError('源留出集没有合法动作')
    models={}
    for name,path in MODELS.items():
        entry=Path(path);complete=read(entry/'complete.json')
        checkpoint_sha=sha(entry/'best.pt')
        assert complete['artifacts'][str(entry/'best.pt')]==checkpoint_sha
        models[name]=dict(entry=path,complete_sha256=sha(entry/'complete.json'),checkpoint_sha256=checkpoint_sha)
    save(root/'plan.json',dict(schema='source-heldout-context-stability-v1',code=code_identity(),shift='S1',
        data_root='/vepfs/vlms/SADA_PLUS',models=models,videos=videos,cases=selected,
        eligible_by_class=eligible,selected_by_class=counts,source_count=source_count,dataset=options,
        source_manifest_sha256=sha(options['json_file']),target_manifest_sha256=sha(cfg['baseline']['target_manifest']),
        target_gt_read=False,checkpoint_selection='existing_source_loss_best_ema',
        probe='fixed_GT_assigned_FPN_point_not_detection_map',contexts_per_action=7))
    print('PREPARED',len(selected),counts,flush=True)


def forward(model, views):
    """原推理预处理，无输入噪声；GT只用于读出固定探针，不送入网络。"""
    import torch
    x,masks=model.preprocessing([dict(feats=x) for x in views],mode='test')
    features,masks,_=model.backbone_source(model.dropout(x),masks)
    features,masks=model.neck_source(features,masks)
    logits,_=model.cls_head(features,masks);offsets,_=model.reg_head(features,masks)
    return masks,model.point_generator(features),[x.permute(0,2,1) for x in logits],[x.permute(0,2,1) for x in offsets]


def run(root):
    from tools.baseline_gpu import first_cuda_guard
    guard=first_cuda_guard()  # 必须在第一次CUDA调用前，父监督持公共卡锁。
    import numpy as np
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    from tools.context_mix import FeatureBank
    from tools.baseline_engine import tensor_digest
    plan=read(root/'plan.json');bank=FeatureBank(dict(videos=plan['videos'],dataset=plan['dataset']))
    all_rows={}
    for name,record in plan['models'].items():
        entry=Path(record['entry']);assert sha(entry/'complete.json')==record['complete_sha256']
        assert sha(entry/'best.pt')==record['checkpoint_sha256']
        cfg=read(entry/'config.json');fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
        model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').to('cuda:0')
        saved=torch.load(entry/'best.pt',map_location='cpu');model.load_state_dict(saved['ema_state_dict'],strict=True);del saved
        model.eval();before=tensor_digest(model.state_dict());rows=[];unroutable=[]
        with torch.no_grad():
            for case in plan['cases']:
                patch=case['patch'];u=case['insert'];width=patch['end']-patch['start']
                native=bank.load(case['video'])[:,patch['start']:patch['end']]
                views=[]
                for context in case['contexts']:
                    view=bank.load(context['video'])[:,context['start']:context['start']+case['length']].clone()
                    view[:,u:u+width]=native;views.append(view)
                assert torch.equal(views[0],bank.load(case['video'])[:,case['contexts'][0]['start']:case['contexts'][0]['start']+case['length']])
                masks,points,logits,offsets=forward(model,views)
                pts=torch.cat(points);main=torch.tensor([patch['main']],device='cuda:0')+u
                cls,_=model.label_points(points,[main],[torch.tensor([case['label']],device='cuda:0')])
                valid=(cls[0][:,case['label']]>0)&torch.cat([x.squeeze(1) for x in masks],1)[0]
                if not valid.any():unroutable.append(case['id']);continue
                choices=valid.nonzero().flatten();distance=(pts[choices,0]-main.mean()).abs()
                probe=int(choices[distance.argmin()]);prob=torch.cat(logits,1).sigmoid();off=torch.cat(offsets,1)
                outputs=[]
                for k,context in enumerate(case['contexts']):
                    box=[float(pts[probe,0]-off[k,probe,0]*pts[probe,3])-u,
                         float(pts[probe,0]+off[k,probe,1]*pts[probe,3])-u]
                    raw=model.inference_single_video(points,[x[k] for x in masks],[x[k] for x in logits],
                        [x[k] for x in offsets],[None]*len(points),[None]*len(points))
                    raw.update(video_id='diagnostic',fps=1.,duration=case['length'],video_length=case['length'],
                        feat_stride=1.,feat_num_frames=0.,use_temporal_grid_feats=False)
                    pred=model.postprocessing([raw])[0]
                    centers=pred['segments'].mean(-1)
                    candidates=((centers>=main[0,0].cpu())&(centers<=main[0,1].cpu())).nonzero().flatten()
                    detection=None
                    if len(candidates):
                        best=int(candidates[pred['scores'][candidates].argmax()])
                        detection=dict(class_id=int(pred['labels'][best]),box=(pred['segments'][best]-u).tolist())
                    outputs.append(dict(kind=context['kind'],class_id=int(prob[k,probe].argmax()),
                        gt_score=float(prob[k,probe,case['label']]),box=box,detection=detection))
                row=dict(id=case['id'],label=case['label'],gt=patch['main'],probe=probe,views=outputs,
                    native=summarize_views([outputs[0],outputs[0]],patch['main'],case['label']),
                    source=summarize_views(outputs[1:4],patch['main'],case['label']),
                    target=summarize_views(outputs[4:7],patch['main'],case['label']))
                rows.append(row)
                print('CASE',name,len(rows),case['id'],flush=True)
        assert before==tensor_digest(model.state_dict()),'诊断不得更新权重或buffer'
        save(root/(name+'.json'),dict(rows=rows,unroutable=unroutable,weights_unchanged=True,checkpoint=record))
        all_rows[name]=rows;del model;torch.cuda.empty_cache()
    ids=[[x['id'] for x in rows] for rows in all_rows.values()]
    assert all(x==ids[0] for x in ids),'四个模型必须使用相同可路由样本'
    summaries={}
    for name,rows in all_rows.items():
        summaries[name]={}
        for kind in ('native','source','target'):
            # 每类等权汇总，同时保存每个动作的完整读数。
            classes=sorted({r['label'] for r in rows})
            summaries[name][kind]={metric:statistics.mean(statistics.mean(r[kind][metric] for r in rows if r['label']==c)
                for c in classes) for metric in rows[0][kind]}
    save(root/'result.json',dict(status='passed',count=len(ids[0]),summaries=summaries,
        first_cuda_guard=guard,first_cuda_guard_sha256=sha(guard),plan_sha256=sha(root/'plan.json'),
        artifacts={name:sha(root/(name+'.json')) for name in MODELS},
        interpretation='source heldout synthetic context intervention; fixed GT probe is not detection mAP; no target GT'))


def supervise(root):
    from tools.baseline_gpu import slot
    identity=code_identity();assert read(root/'plan.json')['code']==identity
    while True:
        for gpu in range(4):
            context=slot(gpu,Path('/vepfs/vlms/SADA_PLUS'),root,16384,identity['commit'])
            try:lease=context.__enter__()
            except BlockingIOError:continue
            try:
                environment=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=lease['uuid'],
                    OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                with (root/'inference.log').open('x') as stream:
                    child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--run','--output',str(root)],
                        cwd=CODE,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                def stop(signum,frame):
                    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                    raise SystemExit(128+signum)
                signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
                save(root/'state.json',dict(phase='inference',pid=os.getpid(),child=child.pid,gpu=gpu))
                code=child.wait()
                save(root/'state.json',dict(phase='complete' if code==0 else 'failed',pid=os.getpid(),returncode=code))
                if code:raise RuntimeError('机制诊断失败，保留工件')
                return
            finally:context.__exit__(None,None,None)
        save(root/'state.json',dict(phase='waiting_public_gpu_lock',pid=os.getpid()));time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args.output)
    elif args.run:run(args.output)
    else:supervise(args.output)
