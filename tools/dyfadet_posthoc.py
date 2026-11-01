"""用户指定的第15轮目标评估；只读已完成检查点，不改变源选模结果。"""
import argparse,os,sys,subprocess,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.dyfadet_uda_protocol import read,save,sha,identity,guard_audit
from tools.dyfadet_uda_gpu import admit,first_cuda_guard

def gpu(args):
    guard=first_cuda_guard()
    from tools import dyfadet_uda_engine as e
    import torch,numpy as np
    from copy import deepcopy
    os.chdir('/vepfs/vlms/SADA_PLUS')
    entry=args.output/args.shift;proof=read(entry/'input.json');cfg=read(proof['config'])
    assert sha(proof['checkpoint'])==proof['checkpoint_sha256']
    state=torch.load(proof['checkpoint'],map_location='cpu');assert state['epoch']==15 and state['mode']=='uda'
    model=e.base.make_meta_arch(cfg['model_name'],**deepcopy(cfg['model'])).to('cuda:0');model.load_state_dict(state['ema'],strict=True);del state;model.eval()
    manifest=cfg['benchmark']['inference']['target_val'];ds=e.base.BenchmarkDataset(cfg,manifest,unlabeled=True)
    cols={k:[] for k in ('video-id','t-start','t-end','label','score')};seen=[]
    with torch.no_grad():
        for i,v in enumerate(ds):
            pred=e.predict_unlabeled(model,[v])[0];seen.append(v['video_id']);cols['video-id'].extend([v['video_id']]*len(pred['scores']))
            for k,x in [('t-start',pred['segments'][:,0]),('t-end',pred['segments'][:,1]),('label',pred['labels']),('score',pred['scores'])]:cols[k].extend(x.cpu().tolist())
            print('TARGET_INFERENCE',args.shift,i+1,len(ds),flush=True)
    np.savez_compressed(entry/'target_val.npz',**{k:np.asarray(v,dtype=str if k=='video-id' else np.int64 if k=='label' else np.float32) for k,v in cols.items()})
    save(entry/'gpu_complete.json',dict(guard=guard,guard_sha256=sha(guard),videos=seen,manifest_sha256=sha(manifest),
        prediction_sha256=sha(entry/'target_val.npz'),checkpoint_sha256=proof['checkpoint_sha256'],epoch=15,weights='EMA',target_GT_used=False))

def evaluate(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import importlib.util,numpy as np
    entry=args.output/args.shift;p=read(entry/'input.json');cfg=read(p['config']);offline=read(Path(p['config']).parent/'offline_evaluation.json')
    assert sha(offline['annotation'])==offline['annotation_sha256']
    module=CODE/'libs/evaluation/evaluator_anet.py';spec=importlib.util.spec_from_file_location('posthoc_eval',module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with np.load(entry/'target_val.npz') as f:pred={k:f[k] for k in f.files}
    g=read(entry/'gpu_complete.json');assert g['prediction_sha256']==sha(entry/'target_val.npz')
    assert g['videos']==list(read(cfg['benchmark']['inference']['target_val'])['database'])
    ev=m.ANETdetection(offline['annotation'],offline['splits']['target_val'],num_workers=4,action_subsampling='10_majority_classes');ap,mean=ev.evaluate(pred,verbose=True)
    so=read(Path(cfg['benchmark']['reference'])/'evaluation/target_val.json')['map_0.1_to_0.5']
    save(entry/'result.json',dict(shift=args.shift,epoch=15,weights='EMA',map=float(mean),map_by_tiou=[float(x) for x in ap],reference_so_map=so,
        delta_pp=100*(float(mean)-so),scope='user_requested_posthoc_target_diagnostic_not_source_selected_result',checkpoint=p['checkpoint'],
        checkpoint_sha256=p['checkpoint_sha256'],prediction_sha256=sha(entry/'target_val.npz'),annotation_sha256=offline['annotation_sha256'],
        evaluator_sha256=sha(module),videos=len(g['videos'])))

def run(args):
    code=identity();args.output.mkdir(parents=True,exist_ok=False);save(args.output/'plan.json',dict(code=code,epoch=15,shifts=['S2','S4'],weights='EMA',purpose='posthoc_target_diagnostic',original_results_unchanged=True))
    active={};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE)+':'+args.extension,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    def launch(s,stage,card=None):
        local=dict(env)
        if card:local.update(CUDA_VISIBLE_DEVICES=card['uuid'],SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
        with (args.output/s/(stage+'.log')).open('w') as f:
            return subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--output',str(args.output),'--shift',s,'--stage',stage,'--extension',args.extension],cwd=CODE,env=local,stdout=f,stderr=subprocess.STDOUT)
    for s,index in [('S2',2),('S4',3)]:
        old=args.reference/'formal'/s;entry=args.output/s;entry.mkdir();checkpoint=old/'uda/latest.pt'
        assert read(old/'uda/termination.json')['decision']['epoch']==15
        save(entry/'input.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),config=str(old/'config.json'),config_sha256=sha(old/'config.json'),original_complete_sha256=sha(old/'complete.json')))
        active[s]=dict(process=launch(s,'gpu',admit(index,args.output,12288,code['commit'])),stage='gpu')
    while active:
        for s,t in list(active.items()):
            rc=t['process'].poll()
            if rc is None:continue
            assert rc==0,(s,t['stage'],rc)
            if t['stage']=='gpu':
                g=read(args.output/s/'gpu_complete.json');guard_audit(g['guard'],g['guard_sha256'],code['commit']);active[s]=dict(process=launch(s,'evaluate'),stage='evaluate')
            else:del active[s]
        save(args.output/'state.json',dict(pid=os.getpid(),active={s:dict(pid=t['process'].pid,stage=t['stage']) for s,t in active.items()}));time.sleep(2)
    results=[read(args.output/s/'result.json') for s in ['S2','S4']]
    save(args.output/'final.json',dict(status='complete',code=code,results=results,scope='posthoc_diagnostic_only'));print(results,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path,default=Path('/vepfs/vlms/dyfadet_uda6_d5fe4dd_1101'))
    p.add_argument('--shift');p.add_argument('--stage',choices=['gpu','evaluate']);p.add_argument('--extension',required=True);args=p.parse_args();sys.path.insert(0,args.extension)
    if args.stage=='gpu':gpu(args)
    elif args.stage=='evaluate':evaluate(args)
    else:run(args)
