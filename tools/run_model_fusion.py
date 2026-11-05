"""固定Proto+Trident推理融合；复用六向best EMA，正式训练预算为零。"""
import argparse,json,os,subprocess,sys,time
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,termination
ORIGIN=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
TRIDENT=Path('/vepfs/vlms/trident_repaired6_820b9e3_1104')
PY='/vepfs/vlms/envs/sada_py39/bin/python'
SHIFTS=tuple('S'+str(i) for i in range(1,7))


def timed(call):
    import torch
    torch.cuda.synchronize();start=time.monotonic();value=call();torch.cuda.synchronize()
    return value,time.monotonic()-start


def assert_same(a,b):
    import torch
    assert a.keys()==b.keys()
    for k in a:
        if a[k] is None:assert b[k] is None;continue
        assert len(a[k])==len(b[k])
        assert all(torch.equal(x,y) for x,y in zip(a[k],b[k])),k


def original_raw(model,video,domain,epoch):
    captured=[];original=model.inference
    def capture(v,points,masks,cls,offsets,left,right):
        captured.append(dict(points=points,masks=masks,cls=cls,offsets=offsets,left=left,right=right));return []
    model.inference=capture
    try:model(video_list_src=video,video_list_tgt=video,mode='test',domain=domain,curr_epoch=epoch)
    finally:model.inference=original
    assert len(captured)==2
    return captured[0 if domain=='source' else 1]


def columns():return {'video-id':[],'t-start':[],'t-end':[],'label':[],'score':[]}


def append(output,item):
    output['video-id'].extend([item['video_id']]*len(item['scores']))
    for k,v in [('t-start',item['segments'][:,0]),('t-end',item['segments'][:,1]),('label',item['labels']),('score',item['scores'])]:
        output[k].extend(v.detach().cpu().tolist())


def worker(root,shift):
    import numpy as np,torch
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    from libs.trainer.da_trainer import move_to_device
    from libs.evaluation import ANETdetection
    from tools import baseline_engine as engine
    from tools.model_fusion import dense_forward,native_result,fused_result
    from tools.native_ap_fast import install
    from tools.audit_head_replacements import metrics as independent_metrics
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    import pandas as pd
    assert read(root/'plan.json')['code']==code_identity()
    install();entry=root/shift;binding=read(entry/'binding.json')
    models=[];configs=[];epochs=[]
    for name in ('proto','trident'):
        source=Path(binding[name]['entry']);cfg=read(source/'config.json')
        assert sha(source/'config.json')==binding[name]['config_sha256']
        assert sha(source/'best.pt')==binding[name]['checkpoint_sha256']
        saved=torch.load(source/'best.pt',map_location='cpu')
        assert saved['epoch']==termination(read(source/'termination.json')['history'])['best_epoch']
        fix_random_seed(cfg['seed'],include_cuda=True);torch.cuda.manual_seed(cfg['seed'])
        model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').to('cuda:0')
        model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval();model.requires_grad_(False)
        models.append(model);configs.append(cfg);epochs.append(saved['epoch']);del saved
    assert configs[0]['dataset']==configs[1]['dataset'] and configs[0]['test_cfg']==configs[1]['test_cfg']
    assert configs[0]['src_val_split']==configs[1]['src_val_split'] and configs[0]['tgt_val_split']==configs[1]['tgt_val_split']
    a,b=models;cfg=configs[0];annotation=cfg['baseline']['annotation_for_offline_evaluation']
    assert sha(annotation)==cfg['baseline']['annotation_sha256']
    report={};acceptance={};timings={};all_artifacts={}
    with torch.no_grad():
        for domain,splits in [('source',cfg['src_val_split']),('target',cfg['tgt_val_split'])]:
            dataset=engine.dataset(cfg,annotation,splits,False)
            loader=engine.loader(dataset,False,torch.Generator().manual_seed(cfg['seed']),1,cfg['loader']['num_workers'])
            outputs={k:columns() for k in ('proto','trident','fusion')};seconds={k:0. for k in ('forward_proto','forward_trident','nms_proto','nms_trident','fusion_decode_nms')}
            video_order=[]
            for index,video in enumerate(loader):
                video=move_to_device(video,'cuda:0');video_order.append(video[0]['metadata']['video_id'])
                # 网络/融合只接收特征和元信息，GT只交给生成预测后的AP评估器。
                for v in video:v['segments']=torch.empty(0,2,device='cuda:0');v['labels']=torch.empty(0,dtype=torch.long,device='cuda:0')
                if index==0:
                    for model,epoch in zip(models,epochs):
                        assert_same(original_raw(model,video,domain,epoch),dense_forward(model,video,domain))
                    acceptance[domain]=dict(original_vs_single_domain_dense_exact=True)
                    torch.cuda.reset_peak_memory_stats()
                ra,ta=timed(lambda:dense_forward(a,video,domain));rb,tb=timed(lambda:dense_forward(b,video,domain))
                pa,tna=timed(lambda:native_result(a,video,ra));pb,tnb=timed(lambda:native_result(b,video,rb))
                pf,tf=timed(lambda:fused_result(a,b,video,ra,rb))
                for name,item in [('proto',pa),('trident',pb),('fusion',pf)]:append(outputs[name],item)
                for name,value in [('forward_proto',ta),('forward_trident',tb),('nms_proto',tna),('nms_trident',tnb),('fusion_decode_nms',tf)]:seconds[name]+=value
                del ra,rb,pa,pb,pf
                if index%5==0 or index+1==len(dataset):
                    save(entry/'progress.json',dict(stage='inference',domain=domain,videos=index+1,total_videos=len(dataset),updated=time.time()))
            timings[domain]=dict(videos=len(dataset),seconds=seconds,proto_service_seconds=seconds['forward_proto']+seconds['nms_proto'],
                trident_service_seconds=seconds['forward_trident']+seconds['nms_trident'],
                fusion_service_seconds=seconds['forward_proto']+seconds['forward_trident']+seconds['fusion_decode_nms'],
                excludes='checkpoint/data loading, first-video equivalence forward, AP evaluation',peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            report[domain]={}
            gt=load_gt_seg_from_json(annotation,split=splits,action_subsampling='10_majority_classes');gt=gt[gt['t-start']!=gt['t-end']].reset_index(drop=True)
            for name,output in outputs.items():
                path=entry/domain/(name+'_predictions.pt');engine.checkpoint(path,output)
                save(entry/'progress.json',dict(stage='evaluation',domain=domain,model=name,updated=time.time()))
                evaluator=ANETdetection(annotation,splits,num_workers=2,action_subsampling='10_majority_classes')
                ap,mean,*_=evaluator.evaluate({k:np.asarray(v) for k,v in output.items()},verbose=True)
                if name=='fusion':
                    independent=independent_metrics(gt,pd.DataFrame(output))
                    assert np.max(np.abs(np.asarray(independent['map_by_tiou'])-ap))<1e-12
                result=dict(map=float(mean),map_by_tiou=ap.tolist(),prediction_count=len(output['score']),videos=len(dataset),independent_AP_matched=name=='fusion')
                if name!='fusion':
                    source=Path(binding[name]['entry']);old=read(source/'evaluation'/(domain+'_val.json'))
                    assert np.array_equal(np.asarray(old['map_by_tiou']),ap),'单模型AP不能逐值复现: '+name+'/'+domain
                    archive=source/'evaluation'/(domain+'_predictions.pt')
                    if archive.exists():assert torch.load(archive,map_location='cpu')==output,'归档单模型预测不能逐值复现'
                    result['original_AP_exact']=True;result['archived_predictions_exact']=archive.exists()
                save(entry/domain/(name+'.json'),result);report[domain][name]=result
                all_artifacts[str(path.relative_to(entry))]=sha(path)
                all_artifacts[str((entry/domain/(name+'.json')).relative_to(entry))]=sha(entry/domain/(name+'.json'))
            save(entry/(domain+'_video_order.json'),video_order)
    result=dict(passed=True,shift=shift,results=report,timings=timings,acceptance=acceptance,training_runs=0,
        baseline_delta_pp=100*(report['target']['fusion']['map']-report['target']['proto']['map']),
        trident_delta_pp=100*(report['target']['fusion']['map']-report['target']['trident']['map']),
        binding_sha256=sha(entry/'binding.json'),artifacts=all_artifacts)
    save(entry/'result.json',result);save(entry/'progress.json',dict(stage='complete',updated=time.time()))


def run(root):
    from tools.run_sada_baseline import acquire
    from tools.baseline_gpu import snapshot
    identity=code_identity();root.mkdir(parents=True,exist_ok=True)
    assert not (root/'plan.json').exists(),'不重复启动已存在的推理队列'
    save(root/'plan.json',dict(code=identity,training_runs=0,shifts=SHIFTS,models=['original Proto','Trident'],
        probability='(p_proto+p_trident)/2',geometry='(p_proto*offset_proto+p_trident*offset_trident)/(p_proto+p_trident)',
        postprocessing='original matched threshold/topk/duration/SoftNMS unchanged',parameter_search=False,
        selection='existing source-selected best EMA for both; no per-direction model picking or target-val tuning',maximum_parallel_jobs=4))
    for shift in SHIFTS:
        binding={}
        for name,e in [('proto',ORIGIN/shift),('trident',TRIDENT/(shift+'_trident'))]:
            binding[name]=dict(entry=str(e),config_sha256=sha(e/'config.json'),checkpoint_sha256=sha(e/'best.pt'))
        save(root/shift/'binding.json',binding)
    save(root/'state.json',dict(stage='evaluating',training_runs=0))
    def task(shift):
        entry=root/shift
        while True:
            cards=sorted([snapshot(i) for i in range(4)],key=lambda x:x['utilization'])
            found=acquire([x['index'] for x in cards],Path('/vepfs/vlms/SADA_PLUS'),root,32768,identity['commit'])
            if found:break
            save(entry/'status.json',dict(stage='waiting_for_gpu'));time.sleep(15)
        context,lease=found
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['uuid'],SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
            PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='123',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(CODE))
        try:
            with (entry/'evaluate.log').open('x') as log:
                p=subprocess.Popen([PY,'-u','-B',str(CODE/'tools/run_model_fusion.py'),'--worker',shift,'--output',str(root)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
                status=dict(stage='evaluating',pid=p.pid,gpu=lease['index'],started=time.time());save(entry/'status.json',status)
                rc=p.wait();status.update(stage='complete' if rc==0 else 'failed',returncode=rc,finished=time.time());save(entry/'status.json',status)
            assert rc==0,shift+' rc='+str(rc)
            return read(entry/'result.json')
        finally:context.__exit__(None,None,None)
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(task,SHIFTS))
    means={k:100*sum(r['results']['target'][k]['map'] for r in results)/6 for k in ('proto','trident','fusion')}
    summary=dict(passed=True,training_runs=0,means_percent=means,delta_vs_proto_pp=means['fusion']-means['proto'],
        delta_vs_trident_pp=means['fusion']-means['trident'],target_met=means['fusion']-means['proto']>=1.,results=results,
        single_seed=True,inference_models=2,no_automatic_extension=True)
    save(root/'summary.json',summary);save(root/'state.json',dict(stage='complete',training_runs=0,target_met=summary['target_met']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',choices=SHIFTS);args=p.parse_args()
    try:
        if args.worker:
            from tools.baseline_gpu import first_cuda_guard
            first_cuda_guard();worker(args.output,args.worker)
        else:run(args.output)
    except Exception as error:
        save(args.output/(args.worker or '')/'error.json',dict(type=type(error).__name__,error=str(error)))
        raise
