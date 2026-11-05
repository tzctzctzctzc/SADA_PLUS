"""六方向已选EMA的只读uncertainty诊断，不训练、不修改模型与NMS。"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
REFERENCE=Path('/vepfs/vlms/trident_repaired6_820b9e3_1104')
PY='/vepfs/vlms/envs/sada_py39/bin/python'
SHIFTS=tuple('S'+str(i) for i in range(1,7))
LEGACY=Path('/vepfs/vlms/trident_uncertainty_6ba8536_1105')


def extract(root,shift):
    from copy import deepcopy
    import numpy as np
    import torch
    from tools.baseline_gpu import first_cuda_guard
    guard=first_cuda_guard()
    from tools import baseline_engine as base
    from tools.trident_uncertainty import DistributionObserver,to_seconds
    from libs.utils import fix_random_seed
    from libs.models.SADA import make_meta_arch
    from libs.trainer.da_trainer import move_to_device
    entry=root/shift;old=REFERENCE/(shift+'_trident');cfg=read(old/'config.json')
    complete=read(old/'complete.json')
    assert read(REFERENCE/'independent_final_audit.json')['passed'] and complete['passed']
    assert cfg['head_replacement']['variant']=='trident' and not cfg['baseline']['target_gt_used']
    saved=torch.load(old/'best.pt',map_location='cpu');assert saved['config']==cfg
    selected=saved['epoch'];checkpoint_sha=sha(old/'best.pt')
    assert checkpoint_sha==complete['artifacts']['best.pt'] and sha(old/'config.json')==complete['artifacts']['config.json']
    assert sha(cfg['baseline']['annotation_for_offline_evaluation'])==cfg['baseline']['annotation_sha256']
    fix_random_seed(cfg['seed'],include_cuda=True);torch.cuda.manual_seed(cfg['seed'])
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').to('cuda:0')
    model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval();del saved
    model_hash=base.tensor_digest(model.state_dict());domains={};total_started=time.monotonic()
    for domain,index,splits in (('source',0,cfg['src_val_split']),('target',1,cfg['tgt_val_split'])):
        reference_path=old/'evaluation'/(domain+'_predictions.pt')
        assert sha(reference_path)==complete['artifacts']['evaluation/'+domain+'_predictions.pt']
        reference=torch.load(reference_path,map_location='cpu')
        report=read(old/'evaluation'/(domain+'_val.json'))
        assert report['checkpoint_sha256']==checkpoint_sha and report['selected_epoch']==selected
        dataset=base.dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],splits,False)
        loader=base.loader(dataset,False,torch.Generator().manual_seed(cfg['seed']),1,cfg['loader']['num_workers'])
        columns={'video-id':[],'t-start':[],'t-end':[],'label':[],'score':[]};video_order=[];benchmark={}
        for number,video in enumerate(loader):
            assert len(video)==1
            # GT只留在后续CPU分析，显式移除推理模型输入中的所有GT。
            for item in video:
                item['segments']=item['segments'].new_empty((0,2));item['labels']=item['labels'].new_empty((0,))
            video=move_to_device(video,'cuda:0');meta=video[0]['metadata'];vid=meta['video_id'];video_order.append(vid)
            def forward():return model(video_list_src=video,video_list_tgt=video,mode='test',domain=domain,curr_epoch=selected)[index][0]
            if number==0:
                forward();torch.cuda.synchronize();start=time.monotonic();plain=forward();torch.cuda.synchronize()
                benchmark['plain_seconds']=time.monotonic()-start
            with DistributionObserver(model,index) as observer:
                torch.cuda.synchronize();start=time.monotonic();prediction=forward();torch.cuda.synchronize();seconds=time.monotonic()-start
                assert observer.call_index==2 and observer.record is not None
                record=to_seconds(observer.record,meta)
                assert torch.equal(record['segments'],prediction['segments_pre_nms'])
                if number==0:
                    for key in ('segments','scores','labels'):assert torch.equal(plain[key],prediction[key]),'观察器改变原输出'
                    benchmark['observed_seconds']=seconds
            base.checkpoint(entry/domain/(vid+'.pt'),dict(video_id=vid,domain=domain,metadata=meta,raw=record))
            n=len(prediction['scores']);columns['video-id'].extend([vid]*n)
            for key,value in (('t-start',prediction['segments'][:,0]),('t-end',prediction['segments'][:,1]),('label',prediction['labels']),('score',prediction['scores'])):
                columns[key].extend(value.cpu().tolist())
            save(entry/'progress.json',dict(stage='extract',domain=domain,videos_complete=number+1,videos_total=len(dataset),video_id=vid,last_seconds=seconds,updated=time.time()))
        equality={k:bool(np.array_equal(np.asarray(v),np.asarray(reference[k]))) for k,v in columns.items()}
        assert all(equality.values()),domain+'没有逐值重现归档检测输出'
        domains[domain]=dict(videos=len(dataset),video_order=video_order,prediction_count=len(columns['score']),all_arrays_exact=equality,
            archived_map=report['map_0.1_to_0.5'],report_sha256=sha(old/'evaluation'/(domain+'_val.json')),benchmark=benchmark)
    assert model_hash==base.tensor_digest(model.state_dict()),'推理修改模型状态'
    record=dict(passed=True,shift=shift,reference=str(old),checkpoint_sha256=checkpoint_sha,config_sha256=sha(old/'config.json'),selected_epoch=selected,
        model_state_unchanged=True,domains=domains,seconds=time.monotonic()-total_started,guard=guard,diagnostic_only=True,training_started=False)
    save(entry/'extraction.json',record);save(entry/'progress.json',dict(stage='extraction_complete',updated=time.time()))


def summarize_six(summaries):
    assert set(summaries)==set(SHIFTS) and all(v['passed'] for v in summaries.values()),'六方向齐全才汇总'
    comparisons={}
    for threshold in ('.3','.5'):
        key=str(float(threshold));comparisons[key]={}
        for domain in ('source_heldout','target'):
            comparisons[key][domain]={}
            for pool in ('all_candidates','score_ge_0.3'):
                rows=[v['evaluation'][key][domain][pool] for v in summaries.values()]
                record={}
                for metric in ('auroc','brier','ece','log_loss'):
                    valid=[r for r in rows if r['nuisance'].get(metric) is not None and r['with_uncertainty'].get(metric) is not None]
                    delta=[r['with_uncertainty'][metric]-r['nuisance'][metric] for r in valid]
                    record[metric]=dict(directions=len(valid),macro_delta=sum(delta)/len(delta) if delta else None,
                        improved_directions=sum(d>0 if metric=='auroc' else d<0 for d in delta))
                comparisons[key][domain][pool]=record
    return dict(all_six_complete=True,count=6,comparisons=comparisons,aggregation='unweighted mean across directions, not pooled candidate metrics',
                interpretation='diagnostic prediction-quality signal; no mAP gain or calibrated target uncertainty guarantee')


def analyze(root):
    import numpy as np
    import torch
    from tools.uncertainty_analysis import features,match_quality,source_partition,fit_logistic,predict,metrics,auc,rank_correlation
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    summaries={}
    for shift in SHIFTS:
        entry=root/shift;extraction=read(entry/'extraction.json');assert extraction['passed']
        cfg=read(REFERENCE/(shift+'_trident')/'config.json');data={}
        for domain,splits in (('source',cfg['src_val_split']),('target',cfg['tgt_val_split'])):
            gt=load_gt_seg_from_json(cfg['baseline']['annotation_for_offline_evaluation'],split=splits,action_subsampling='10_majority_classes')
            gt=gt[gt['t-start']!=gt['t-end']]
            rows=[]
            for vid in extraction['domains'][domain]['video_order']:
                row=torch.load(entry/domain/(vid+'.pt'),map_location='cpu')['raw']
                row={k:(v.numpy() if torch.is_tensor(v) else v) for k,v in row.items()}
                truth=gt[gt['video-id']==vid];base,uncert,signals=features(row)
                quality,error,matched=match_quality(row['segments'],row['labels'],truth[['t-start','t-end']].to_numpy(float),truth.label.to_numpy(int))
                rows.append(dict(base=base,uncert=uncert,quality=quality,error=error,matched=matched,score=row['scores'],video=np.repeat(vid,len(quality)),**signals))
            data[domain]={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
        source=data['source'];fit_videos,held_videos=source_partition(source['video']);fit=np.isin(source['video'],list(fit_videos));held=np.isin(source['video'],list(held_videos))
        models={};evaluation={}
        for threshold in (.3,.5):
            y=source['quality']>=threshold;models[str(threshold)]={}
            for name in ('nuisance','with_uncertainty'):
                x=source['base'] if name=='nuisance' else np.column_stack((source['base'],source['uncert']))
                model=fit_logistic(x[fit],y[fit],source['video'][fit]);models[str(threshold)][name]=model
                for domain,mask in (('source_heldout',held),('target',np.ones(len(data['target']['quality']),dtype=bool))):
                    row=source if domain=='source_heldout' else data['target']
                    x=row['base'] if name=='nuisance' else np.column_stack((row['base'],row['uncert']))
                    probability=predict(model,x);truth=row['quality']>=threshold
                    for pool,use in (('all_candidates',mask),('score_ge_0.3',mask&(row['score']>=.3))):
                        evaluation.setdefault(str(threshold),{}).setdefault(domain,{}).setdefault(pool,{})[name]=metrics(truth[use],probability[use])
        raw={}
        for domain,row in data.items():
            raw[domain]={}
            for pool,use in (('all_candidates',np.ones(len(row['quality']),dtype=bool)),('score_ge_0.3',row['score']>=.3)):
                valid=use&row['matched']&(row['quality']>0)
                raw[domain][pool]=dict(count=int(use.sum()),same_class_overlapping_count=int(valid.sum()),
                    entropy_bad_iou_auc=auc(row['quality'][use]<.5,row['entropy'][use]),std_bad_iou_auc=auc(row['quality'][use]<.5,row['relative_std'][use]),
                    left_std_error_spearman=rank_correlation(row['left_std'][valid],row['error'][valid,0]),right_std_error_spearman=rank_correlation(row['right_std'][valid],row['error'][valid,1]))
        summary=dict(passed=True,technical_extraction_exact=True,calibration_fit_videos=sorted(fit_videos),calibration_heldout_videos=sorted(held_videos),
            calibration_scope='Diagnostic fitting on source-validation fit videos; disjoint source-validation videos and target-validation evaluation. No detector training or target fit.',
            nuisance_features=['class_score_logit','log_duration','log_fpn_stride','class_onehot'],uncertainty_features=['left_entropy','right_entropy','left_relative_std','right_relative_std','left_maxprob','right_maxprob','left_outside_mass','right_outside_mass'],
            target_GT_usage='offline error labels only; no fitting, thresholds or model updates',evaluation=evaluation,raw_signals=raw,
            label_definition='Best same-class GT tIoU quality, NOT one-to-one detection TP; duplicates remain. Endpoint correlations separately require overlapping same-class GT.',
            target_is_development=True,new_training_started=False)
        save(entry/'calibrators_diagnostic_only.json',models);save(entry/'result.json',summary);summaries[shift]=summary
    save(root/'final.json',dict(passed=True,shifts=SHIFTS,results=summaries,six_direction_summary=summarize_six(summaries),training_started=False,raw_distributions_exported=True))


def reuse_legacy(root,shift):
    """仅复用原两向已完成导出；活跃/未知原队列不重复推理。"""
    import shutil
    old=LEGACY/shift;manifest=old/'extraction.json'
    if (LEGACY/'superseded_before_launch.json').exists():return False
    while LEGACY.exists() and not manifest.exists():
        if (LEGACY/'error.json').exists() or (old/'error.json').exists():raise RuntimeError('旧诊断失败，须核实后修复，不能盲目重提')
        receipt=LEGACY.with_name(LEGACY.name+'_launch.json')
        if receipt.exists():
            launch=read(receipt);path=Path('/proc')/str(launch['pid'])
            if not path.exists():raise RuntimeError('旧诊断退出且导出不完整，须保留现场核实')
            command=(path/'cmdline').read_bytes();assert str(Path(launch['code'])/'tools/run_trident_uncertainty.py').encode() in command
        save(root/shift/'progress.json',dict(stage='waiting_for_existing_export',legacy=str(old),updated=time.time()));time.sleep(15)
    if not manifest.exists():return False
    record=read(manifest);assert record['passed'] and record['shift']==shift and record['model_state_unchanged']
    reference=REFERENCE/(shift+'_trident')
    assert record['checkpoint_sha256']==sha(reference/'best.pt') and record['config_sha256']==sha(reference/'config.json')
    entry=root/shift;hashes={}
    for domain,info in record['domains'].items():
        assert all(info['all_arrays_exact'].values())
        (entry/domain).mkdir()
        for vid in info['video_order']:
            source=old/domain/(vid+'.pt');target=entry/domain/(vid+'.pt')
            shutil.copyfile(source,target);hashes[str(target.relative_to(entry))]=sha(target)
    save(entry/'extraction.json',record);save(entry/'reuse.json',dict(legacy=str(old),extraction_sha256=sha(manifest),copied_raw_sha256=hashes,no_inference_repeated=True))
    save(entry/'progress.json',dict(stage='extraction_reused',updated=time.time()));return True


def run(root):
    from tools.baseline_gpu import snapshot,slot
    identity=code_identity();root.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(code=identity,shifts=SHIFTS,reference=str(REFERENCE),diagnostic_only=True,training_runs=0,
        scope='Selected EMA source/target validation inference replay and boundary uncertainty analysis; source-video-disjoint diagnostic calibration; no target fitting'))
    def task(shift):
        entry=root/shift;entry.mkdir()
        if shift in ('S1','S6') and reuse_legacy(root,shift):return
        while True:
            chosen=None
            for gpu in sorted([snapshot(i) for i in range(4)],key=lambda x:x['utilization']):
                context=slot(gpu['index'],'/vepfs/vlms/SADA_PLUS',root,32768,identity['commit'])
                try:lease=context.__enter__();chosen=(context,lease);break
                except BlockingIOError:continue
            if chosen:break
            save(entry/'progress.json',dict(stage='waiting_for_gpu',updated=time.time()));time.sleep(15)
        context,lease=chosen
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['uuid'],SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
                PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='123',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (entry/'extract.log').open('x') as log:
                p=subprocess.Popen([PY,'-u','-B',str(Path(__file__)),'--child',shift,'--output',str(root)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
                save(entry/'launch.json',dict(pid=p.pid,gpu=lease['index'],started=time.time()));rc=p.wait()
            assert rc==0,shift+' extraction failed rc='+str(rc)
        finally:context.__exit__(None,None,None)
    save(root/'state.json',dict(stage='extracting'))
    # 新四向先推进；两向若旧任务已启动则等待复用，不争抢它们的资源。
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(task,('S2','S3','S4','S5','S1','S6')))
    save(root/'state.json',dict(stage='cpu_analysis'));analyze(root)
    save(root/'state.json',dict(stage='diagnostics_complete',training_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--child',choices=SHIFTS);args=p.parse_args()
    try:extract(args.output,args.child) if args.child else run(args.output)
    except Exception as error:
        save(args.output/(args.child or '')/'error.json',dict(type=type(error).__name__,error=str(error)));raise
