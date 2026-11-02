"""六方向×四种端点容量诊断；独立目标train监督Oracle，原正式训练不受影响。"""
import argparse,os,sys,json,time,math,signal,subprocess,io
from copy import deepcopy
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.phase_endpoint_oracle import SPEC,SCHEMA,MODES
SHIFTS=tuple('S%d'%i for i in range(1,7))
REFERENCE=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')


def prepare_truth(cfg,path):
    import numpy as np
    from tools import baseline_engine as base
    annotation=cfg['baseline']['annotation_for_offline_evaluation'];assert sha(annotation)==cfg['baseline']['annotation_sha256']
    ds=base.dataset(cfg,annotation,cfg['tgt_train_split'],False)
    target=read(cfg['baseline']['target_manifest'])['database'];expected={v for v,x in target.items() if x['subset'] in cfg['tgt_train_split']}
    full=read(annotation)['database'];heldout={v for v,x in full.items() if x['subset'] in cfg['tgt_val_split']}
    assert expected.isdisjoint(heldout)
    records={}
    stride=ds.feat_stride*ds.downsample_rate
    for v in ds.data_list:
        if v['segments'] is None:segments=np.empty((0,2),np.float32);labels=np.empty(0,np.int64)
        else:
            keep=v['labels']<10;segments=(v['segments'][keep]*v['fps']-.5*ds.num_frames)/stride;labels=v['labels'][keep]
        records[v['id']]=dict(segments=segments.tolist(),labels=labels.tolist(),fps=v['fps'],feat_stride=stride,feat_num_frames=ds.num_frames)
    assert set(records)==expected
    save(path,dict(scope='target_train_GT_explicit_oracle_only',records=records,annotation_sha256=sha(annotation),
        target_manifest_sha256=sha(cfg['baseline']['target_manifest']),target_val_excluded=True,heldout_ids=sorted(heldout)))


def build(cfg):
    from tools import baseline_engine as base
    from tools.phase_endpoint_oracle import install,enable_controls
    enable_controls();ctx=base.build(cfg)
    ref=Path(cfg['training_oracle']['reference'])
    assert ctx.initial_state_sha==read(ref/'termination.json')['initial_state_sha']
    truth={}
    if cfg['training_oracle']['mode']!='control':
        p=cfg['training_oracle']['truth'];assert sha(p)==cfg['training_oracle']['truth_sha256'];truth=read(p)['records']
    install(ctx,cfg,truth)
    ctx.mature_initial_sha=base.tensor_digest(ctx.model.state_dict())
    return ctx


def metrics(ctx,epoch,seconds):
    m=ctx.train_metrics.result()
    return dict(epoch=epoch,seconds=seconds,train_loss=float(m['t_final_loss']),source_cls_loss=float(m['t_cls_loss']),source_reg_loss=float(m['t_reg_loss']),
        oracle={k[9:]:float(v) for k,v in m.items() if k.startswith('t_oracle_')},
        target_train_oracle_supervision=ctx.cfg['training_oracle']['mode']!='control' and epoch>=SPEC['onset'])


def probe(cfg,e,guard):
    import torch
    from tools import baseline_engine as base
    ctx=build(cfg);phases={}
    # 新增通道关闭时，实际原forward的所有loss逐Tensor等价，控制量/RNG完整还原。
    src=base.move_to_device(next(iter(ctx.train_loader_src)),ctx.model.device);tgt=base.move_to_device(next(iter(ctx.train_loader_tgt)),ctx.model.device)
    ctx.model.train();before=deepcopy(base.capture(ctx,cfg,0,[]))
    kwargs=dict(video_list_src=src,video_list_tgt=tgt,mode='train',curr_epoch=1,cls_static_anchors=None,mean_class_conf=None)
    expected={k:v.detach().clone() for k,v in ctx.oracle_original_forward(**kwargs)[0].items()};base.restore(ctx,before)
    observed=ctx.model(**kwargs)[0];assert all(torch.equal(v,observed[k].detach()) for k,v in expected.items())
    del observed,expected,src,tgt;base.restore(ctx,before);del before
    for epoch,steps in [(1,2),(5,4)]:
        base.epoch_setup(ctx,epoch);loaders=ctx.train_loader_src,ctx.train_loader_tgt
        ctx.train_loader_src,ctx.train_loader_tgt=[base.StepLoader(x,steps) for x in loaders]
        before=base.tensor_digest(ctx.model.backbone_source.state_dict());reg=base.tensor_digest(ctx.model.reg_head.state_dict())
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();ctx._train_epoch(epoch);torch.cuda.synchronize()
        phases[str(epoch)]=dict(metrics(ctx,epoch,time.monotonic()-start),SGP_updated=before!=base.tensor_digest(ctx.model.backbone_source.state_dict()),regression_updated=reg!=base.tensor_digest(ctx.model.reg_head.state_dict()),peak_mib=torch.cuda.max_memory_reserved()/2**20)
        assert phases[str(epoch)]['SGP_updated'] and phases[str(epoch)]['regression_updated']
        ctx.train_loader_src,ctx.train_loader_tgt=loaders
    assert phases['1']['oracle']['aux_loss']==0
    mode=cfg['training_oracle']['mode']
    if mode=='control':assert phases['5']['oracle']['aux_loss']==0 and phases['5']['oracle']['left']==phases['5']['oracle']['right']==0
    else:
        assert phases['5']['oracle']['left']+phases['5']['oracle']['right']>0
        assert phases['5']['oracle']['target_sgp_gradient']>0 and phases['5']['oracle']['target_regression_gradient']>0
    state=base.capture(ctx,cfg,5,[]);stream=io.BytesIO();torch.save(state,stream);stream.seek(0);state=torch.load(stream,map_location='cpu')
    base.restore(ctx,state);assert base.tensor_digest(base.capture(ctx,cfg,5,[]))==base.tensor_digest(state)
    def draws():
        import random,numpy as np
        return dict(py=random.random(),np=np.random.rand(3),cpu=torch.rand(3),cuda=torch.rand(3,device='cuda:0'),loaders={k:torch.randperm(7,generator=g) for k,g in ctx.generators.items()})
    base.restore(ctx,state);a=base.tensor_digest(draws());base.restore(ctx,state);assert a==base.tensor_digest(draws())
    save(e/'result.json',dict(status='passed',schema=SCHEMA,mode=mode,shift=cfg['baseline']['shift'],phases=phases,mature_initial_sha=ctx.mature_initial_sha,
        full_state_and_RNG_restore='passed',disabled_path_all_losses_exact=True,guard=guard,guard_sha256=sha(guard),code=code_identity(),formal_training=False,
        target_train_GT_used=mode!='control',target_val_GT_used=False,peak_mib=max(p['peak_mib'] for p in phases.values())))


def train(cfg,e):
    import torch
    from tools import baseline_engine as base
    ctx=build(cfg);ctx.oracle_ledger=str(e/'oracle_events.jsonl');history=[];rows=[];best=float('inf')
    save(e/'runtime_config.json',ctx.cfg);save(e/'seed_proof.json',ctx.cuda_seed_proof)
    save(e/'mature_initialization.json',dict(sha256=ctx.mature_initial_sha,cold_identity_sha=ctx.initial_state_sha,new_parameters=0,reference=cfg['training_oracle']['reference']))
    ctx.valid_metrics_src.reset();value=float(ctx._valid_epoch(0,ctx.val_loader_src,domain='src'));history.append([0,value]);best=value
    base.checkpoint(e/'epoch000.pt',base.capture(ctx,cfg,0,history));os.link(e/'epoch000.pt',e/'best.pt')
    for epoch in range(1,SPEC['epochs']+1):
        base.epoch_setup(ctx,epoch);start=time.monotonic();checked=ctx.checked_steps;ctx._train_epoch(epoch);torch.cuda.synchronize()
        assert ctx.checked_steps-checked==cfg['baseline']['epoch_budget']['iterations_per_epoch']
        row=metrics(ctx,epoch,time.monotonic()-start)
        if epoch%5==0:
            ctx.valid_metrics_src.reset();value=float(ctx._valid_epoch(epoch,ctx.val_loader_src,domain='src'));assert math.isfinite(value);history.append([epoch,value]);row['source_validation_loss']=value
            path=e/('epoch%03d.pt'%epoch);base.checkpoint(path,base.capture(ctx,cfg,epoch,history))
            tmp=e/'latest.pt.tmp';os.link(path,tmp);tmp.replace(e/'latest.pt')
            if value<best:
                best=value;tmp=e/'best.pt.tmp';os.link(path,tmp);tmp.replace(e/'best.pt')
        rows.append(row);(e/'metrics.jsonl').write_text(''.join(json.dumps(z,allow_nan=False)+'\n' for z in rows));print('ORACLE_METRICS',json.dumps(row),flush=True)
    chosen=min(history,key=lambda x:(x[1],x[0]))[0]
    save(e/'termination.json',dict(reason='fixed20_extra_budget_complete_no_target_selection',epoch=20,best_epoch=chosen,history=history,
        initial_state_sha=ctx.initial_state_sha,mature_initial_sha=ctx.mature_initial_sha))


def inference(cfg,e):
    import torch,numpy as np
    from tools import baseline_engine as base
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda();model.eval()
    terminal=read(e/'termination.json');selected=terminal['best_epoch'];epochs=sorted(set([0,5,10,15,20,selected]))
    reports=[]
    for epoch in epochs:
        saved=torch.load(e/('epoch%03d.pt'%epoch),map_location='cpu');model.load_state_dict(saved['ema_state_dict'],strict=True)
        domains=['source','target'] if epoch in [20,selected] else ['source']
        for domain in domains:
            manifest=e/(domain+'_val_unlabeled.json');assert all('annotations' not in v for v in read(manifest)['database'].values())
            ds=base.dataset(cfg,manifest,cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'],False,True)
            columns={k:[] for k in ['video-id','t-start','t-end','label','score']}
            with torch.no_grad():
                for i in range(len(ds)):
                    v=base.move_to_device([ds[i]],'cuda:0');assert not v[0]['segments'].numel() and not v[0]['labels'].numel()
                    pred=model(video_list_src=v,video_list_tgt=v,mode='test',domain=domain,curr_epoch=epoch)[0 if domain=='source' else 1][0]
                    columns['video-id'].extend([pred['video_id']]*len(pred['scores']))
                    for k,x in [('t-start',pred['segments'][:,0]),('t-end',pred['segments'][:,1]),('label',pred['labels']),('score',pred['scores'])]:columns[k].extend(x.cpu().tolist())
            path=e/('predictions/%s_e%03d.npz'%(domain,epoch));path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**{k:np.asarray(v) for k,v in columns.items()})
            reports.append(dict(domain=domain,epoch=epoch,prediction=str(path),prediction_sha256=sha(path),checkpoint_sha256=sha(e/('epoch%03d.pt'%epoch))))
    save(e/'inference_manifest.json',dict(reports=reports,GT_loaded=False))


def evaluate_cpu(cfg,e):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    from libs.evaluation import ANETdetection
    annotation=cfg['baseline']['annotation_for_offline_evaluation'];assert sha(annotation)==cfg['baseline']['annotation_sha256'];results=[]
    for row in read(e/'inference_manifest.json')['reports']:
        assert sha(row['prediction'])==row['prediction_sha256']
        with np.load(row['prediction'],allow_pickle=False) as f:pred={k:f[k] for k in f.files}
        evaluator=ANETdetection(annotation,cfg['src_val_split'] if row['domain']=='source' else cfg['tgt_val_split'],num_workers=2,action_subsampling='10_majority_classes')
        ap,mean,*_=evaluator.evaluate(pred,verbose=False);r=dict(row,map=float(mean),map_by_tiou=[float(x) for x in ap]);results.append(r)
        save(e/('evaluation/%s_e%03d.json'%(row['domain'],row['epoch'])),r)
    save(e/'evaluation/final.json',dict(results=results,primary=[r for r in results if r['domain']=='target' and r['epoch']==20][0],source_selection=read(e/'termination.json')['best_epoch'],target_GT_scope='CPU heldout final metrics only'))


def audit(e):
    import torch
    r=read(e/'complete.json');cfg=read(e/'config.json');mode=cfg['training_oracle']['mode']
    assert cfg['training_oracle']['spec']==SPEC
    paired=deepcopy(cfg);paired.pop('training_oracle');old=read(Path(cfg['training_oracle']['reference'])/'config.json')
    for k in ['code','tmpdir','schema']:paired['baseline'][k]=old['baseline'][k]
    assert paired==old
    assert r['schema']==SCHEMA and r['mode']==mode and r['code']==code_identity()
    for p,d in r['artifacts'].items():assert sha(p)==d,p
    terminal=read(e/'termination.json');history=terminal['history'];assert [x[0] for x in history]==[0,5,10,15,20]
    assert terminal['best_epoch']==min(history,key=lambda x:(x[1],x[0]))[0]
    latest=torch.load(e/'epoch020.pt',map_location='cpu');assert latest['config']==cfg and latest['history']==history
    from tools.baseline_protocol import validate_cuda_rng
    validate_cuda_rng(latest['rng']['cuda']);assert len(latest['optimizers'])==len(latest['schedulers'])==3
    rows=[json.loads(x) for x in (e/'metrics.jsonl').read_text().splitlines()];assert [x['epoch'] for x in rows]==list(range(1,21))
    ledger=[json.loads(x) for x in (e/'oracle_events.jsonl').read_text().splitlines()];digest='0'*64
    target=read(cfg['baseline']['target_manifest'])['database'];members={v for v,x in target.items() if x['subset'] in cfg['tgt_train_split']}
    for step,event in enumerate(ledger,1):
        given=event.pop('digest');assert event['step']==step and set(event['target_videos'])<=members
        digest=__import__('hashlib').sha256((digest+json.dumps(event,sort_keys=True)).encode()).hexdigest();assert given==digest
        if event['epoch']<5 or mode=='control':assert not event['stats']['target_train_GT_used'] and event['stats']['aux_loss']==0
    state=latest['model_controls']['oracle_training_state'];assert state['step']==len(ledger)==20*cfg['baseline']['epoch_budget']['iterations_per_epoch'] and state['digest']==digest
    if mode!='control':assert state['events']>0 and any(x['oracle'].get('target_sgp_gradient',0)>0 and x['oracle'].get('target_regression_gradient',0)>0 for x in rows)
    truth=read(cfg['training_oracle']['truth']);assert sha(cfg['training_oracle']['truth'])==cfg['training_oracle']['truth_sha256']
    assert set(truth['records'])==members and not members&set(truth['heldout_ids'])
    for domain in ['source','target']:assert all('annotations' not in x for x in read(e/(domain+'_val_unlabeled.json'))['database'].values())
    for stage in ['train','inference']:
        receipt=read(e/(stage+'_gpu.json'));assert sha(receipt['guard'])==receipt['guard_sha256'];guard=read(receipt['guard']);assert sha(guard['lease'])==guard['lease_sha256']
        lease=read(guard['lease']);assert guard['owner_pid']==lease['owner_pid'] and guard['observed']['uuid']==lease['observed']['uuid'] and guard['observed']['free_mib']>=lease['minimum_free_mib']>=12288
    result=read(e/'evaluation/final.json');return dict(mode=mode,shift=cfg['baseline']['shift'],primary_map=result['primary']['map'],selected_epoch=result['source_selection'],complete_sha256=sha(e/'complete.json'))


def worker(a):
    cfg=read(a.entry/'config.json');assert cfg['baseline']['code']==code_identity()
    if a.stage=='evaluate_cpu':evaluate_cpu(cfg,a.entry);return
    from tools.phase_runner_gpu import first_cuda_guard
    guard=first_cuda_guard();os.chdir(a.data_root)
    if a.stage=='probe':probe(cfg,a.entry,guard)
    elif a.stage=='train':train(cfg,a.entry)
    else:inference(cfg,a.entry)
    save(a.entry/(a.stage+'_gpu.json'),dict(guard=guard,guard_sha256=sha(guard),mode=cfg['training_oracle']['mode'],GT_scope='target_train_oracle_only' if a.stage!='inference' else 'no_GT',code=code_identity()))


def main(a):
    from tools.run_phase_boundary import reuse_reference_snapshot
    from tools.phase_runner_gpu import acquire
    root=a.output;root.mkdir(exist_ok=False);identity=code_identity();refs=reuse_reference_snapshot(root)
    tmp=Path('/vepfs/vlms/tmp')/('endpoint_oracle_'+identity['commit'][:7]);tmp.mkdir(exist_ok=False)
    configs={}
    save(root/'plan.json',dict(spec=SPEC,code=identity,shifts=list(SHIFTS),modes=list(MODES),formal_count=24,not_formal_UDA=True,source_phase_aux_included=False))
    for shift in SHIFTS:
        old=read(REFERENCE/shift/'config.json');truth=root/'oracle_train_truth'/(shift+'.json');prepare_truth(old,truth)
        for mode in MODES:
            cfg=deepcopy(old);cfg['baseline'].update(code=identity,tmpdir=str(tmp),schema=SCHEMA)
            cfg['training_oracle']=dict(spec=SPEC,mode=mode,reference=str(REFERENCE/shift),truth=str(truth),truth_sha256=sha(truth),target_train_GT_used=mode!='control')
            configs[(mode,shift)]=cfg
            for folder in ['probes','formal']:
                e=root/folder/mode/shift;e.mkdir(parents=True);save(e/'config.json',cfg)
                if folder=='formal':
                    for domain in ['source','target']:save(e/(domain+'_val_unlabeled.json'),read(root/'reference_snapshot/formal'/shift/(domain+'_val_unlabeled.json')),sort_keys=False)
    save(root/'inputs.json',dict(references=refs,configs={m+'/'+s:sha(root/'formal'/m/s/'config.json') for m,s in configs}))
    active={};pending=[(m,s,'probe') for s in SHIFTS for m in MODES];done=[];reports={m:{} for m in MODES};formal_started=[];minimum=12288
    def spawn(m,s,stage,card=None):
        e=root/('probes' if stage=='probe' else 'formal')/m/s
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='123',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp))
        if card:env.update(CUDA_VISIBLE_DEVICES=card['uuid'],SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
        with (e/(stage+'.log')).open('x') as f:p=subprocess.Popen([sys.executable,'-u','-B',__file__,'--output',str(root),'--entry',str(e),'--stage',stage],cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        return dict(p=p,mode=m,shift=s,stage=stage,gpu=card['index'] if card else None)
    try:
        while pending or active:
            for key,t in list(active.items()):
                rc=t['p'].poll()
                if rc is None:continue
                if t.get('ctx'):t['ctx'].__exit__(None,None,None)
                del active[key];assert rc==0,(key,rc)
                m,s=t['mode'],t['shift'];e=root/'formal'/m/s
                if t['stage']=='probe':
                    p=root/'probes'/m/s/'result.json';r=read(p);assert r['status']=='passed';reports[m][s]=dict(path=str(p),sha256=sha(p),report=r)
                elif t['stage']=='train':pending.append((m,s,'inference'))
                elif t['stage']=='inference':pending.append((m,s,'evaluate_cpu'))
                else:
                    paths=[p for p in e.rglob('*') if p.is_file() and p.name!='complete.json']
                    save(e/'complete.json',dict(schema=SCHEMA,mode=m,shift=s,code=identity,artifacts={str(p):sha(p) for p in paths},scope='controlled endpoint capacity diagnostic, not formal UDA'))
                    done.append(audit(e))
            for m in MODES:
                if len(reports[m])==6 and m not in formal_started:
                    assert len({r['report']['mature_initial_sha'] for r in reports[m].values()})>=1
                    minimum=max(minimum,math.ceil(max(r['report']['peak_mib'] for r in reports[m].values())+4096))
                    save(root/('acceptance_'+m+'.json'),dict(reports=reports[m],minimum_free_mib=minimum,status='passed'))
                    formal_started.append(m);pending.extend((m,s,'train') for s in SHIFTS)
            for m,s,stage in list(pending):
                if stage=='evaluate_cpu':
                    if sum(t['stage']=='evaluate_cpu' for t in active.values())>=2:continue
                    card=None;context=None
                else:
                    if sum(t.get('gpu') is not None for t in active.values())>=8:continue
                    counts={g:sum(t.get('gpu')==g for t in active.values()) for g in range(4)}
                    cards=[g for g in sorted(counts,key=counts.get) if counts[g]<2 and not any(t['mode']==m and t.get('gpu')==g for t in active.values())]
                    found=acquire(cards,a.data_root,root,minimum,identity['commit'])
                    if not found:continue
                    context,card=found
                t=spawn(m,s,stage,card)
                if context:t['ctx']=context
                active[m+'/'+s+'/'+stage]=t;pending.remove((m,s,stage))
            save(root/'state.json',dict(pid=os.getpid(),phase='probe_and_controlled_training',formal_started=formal_started,pending=pending,completed=done,active={k:dict(pid=t['p'].pid,mode=t['mode'],shift=t['shift'],stage=t['stage'],gpu=t['gpu']) for k,t in active.items()}));time.sleep(3)
        assert len(done)==24
        save(root/'final.json',dict(count=24,results=done,means={m:sum(r['primary_map'] for r in done if r['mode']==m)/6 for m in MODES},scope='target_train-supervised capacity diagnostic only'))
    except BaseException as ex:save(root/'error.json',dict(type=type(ex).__name__,message=str(ex)));raise
    finally:
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        for t in active.values():
            try:t['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['p'].pid,signal.SIGKILL);t['p'].wait()
            if t.get('ctx'):t['ctx'].__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--entry',type=Path);p.add_argument('--stage',choices=['probe','train','inference','evaluate_cpu']);p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));a=p.parse_args()
    def stop(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if a.stage:worker(a)
    else:main(a)
