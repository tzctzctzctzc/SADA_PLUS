"""新背景头在源val的独立只读读出；区分源过拟合与跨域差异，不训练。"""
import argparse,os,subprocess,sys,time,signal
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_foreground_calibration_oracle import SHIFTS
SOURCE=Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
PILOT=Path('/vepfs/vlms/proposal_foreground_source_pilot6_eb30065_1103')


def worker(entry):
    import torch
    from tools import baseline_engine as base
    from tools.proposal_foreground_math import ProposalForeground
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.foreground_oracle_ap import prediction_frame
    from tools.background_payoff_diagnosis import distribution
    from tools.run_foreground_calibration_oracle import load_truth
    guard=first_cuda_guard(entry);s=entry.name;parent=SOURCE/s;old=PILOT/s
    fit=read(parent/'fit_complete.json');head_path=parent/'head_fixed20.pt'
    assert sha(head_path)==fit['artifacts'][str(head_path)]
    index=read(old/'cache_index.json');native=read(parent/'cache_index.json')
    assert index['source_initial_state_digest']==native['detector_digest']
    selection=read(old/'selection.json');records=[r for r in index['records'] if r['role']=='validation']
    assert {r['video'] for r in records}==set(selection['validation'])
    assert not set(selection['validation'])&set(read(parent/'selection.json')['source_fit'])
    head=ProposalForeground(512).cuda();head.load_state_dict(torch.load(head_path,map_location='cpu')['state_dict']);head.eval()
    before=base.tensor_digest(head.state_dict());rows=[]
    for r in records:
        assert sha(r['path'])==r['sha256'];item=torch.load(r['path'],map_location='cpu')
        with torch.no_grad():q=head(item['feature'].cuda(),item['valid'].cuda(),item['boxes'].cuda()).softmax(-1)[:,0].cpu()
        rows.append(dict(video=r['video'],segments=item['segments'],scores=item['scores'],labels=item['labels'],pbg=q))
    assert before==base.tensor_digest(head.state_dict())
    base.checkpoint(entry/'source_val_predictions.pt',rows)
    cfg=read(parent/'config.json');gt=load_truth(cfg['dataset']['json_file'],cfg['src_val_split'])
    dist,_=distribution(prediction_frame(rows),gt)
    save(entry/'complete.json',dict(shift=s,code=code_identity(),distribution=dist,source_val_videos=len(rows),
        source_val_not_used_to_fit_or_select_head=True,original_detector_selected_on_source_val=True,
        head_unchanged=True,optimizer_steps=0,guard=guard,guard_sha256=sha(guard),
        artifacts={str(p):sha(p) for p in [entry/'source_val_predictions.pt',head_path,old/'cache_index.json',parent/'cache_index.json']}))


def run(root):
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(code=code_identity(),source=str(SOURCE),
        purpose='source_val_vs_fit_vs_target_generalization_readout',optimizer_steps=0,target_GT=False))
    active=None;done=[]
    def stop(sig,frame):
        if active and active.poll() is None:os.killpg(active.pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        for s in SHIFTS:
            entry=root/s;entry.mkdir()
            while True:
                choices=[snapshot(g) for g in [2,0,3,1]];choices=[r for r in choices if r['free_mib']>=12288]
                if choices:break
                time.sleep(3)
            observed=max(choices,key=lambda r:r['free_mib']);lease=entry/'lease.json'
            save(lease,dict(owner_pid=os.getpid(),index=observed['index'],uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=observed['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
            with (entry/'readout.log').open('a') as log:
                active=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker','--entry',str(entry)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(root/'state.json',dict(pid=os.getpid(),completed=done,active=dict(shift=s,pid=active.pid,gpu=observed['index'])))
                if active.wait():raise RuntimeError(s+' source val readout failed')
            done.append(s);active=None
        save(root/'state.json',dict(pid=os.getpid(),completed=done,active=None))
        save(root/'final.json',dict(count=6,results=[read(root/s/'complete.json') for s in SHIFTS],optimizer_steps=0))
    except Exception as e:
        save(root/'error.json',dict(error=repr(e)));raise
    finally:
        if active and active.poll() is None:os.killpg(active.pid,signal.SIGTERM)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--worker',action='store_true');p.add_argument('--entry',type=Path);a=p.parse_args()
    if a.run:run(a.run)
    else:
        os.chdir('/vepfs/vlms/SADA_PLUS');worker(a.entry)
