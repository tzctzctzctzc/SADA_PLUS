"""固定项目SSH：默认只读，--deploy仅对不存在的专属ROOT提交一次。"""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import runpy

def call(script):
    connect=runpy.run_path(str(HERE.parent/'.codex_tmp_prototype_alignment_connection_1106.py'))['connection']
    with connect(ipv4=True) as remote:return json.loads(remote(script))
ROOT='/vepfs/vlms/module_b_selective_boundary_formal6_1114_fix5'

QUERY=r'''
import json,pathlib,subprocess,time
root=pathlib.Path(ROOT)
def read(p):return json.loads(p.read_text()) if p.exists() else None
def identity(r):
 if not r:return None
 p=pathlib.Path('/proc')/str(r['pid'])
 try:
  f=(p/'stat').read_text().rsplit(')',1)[1].split();matched=f[19]==str(r['start_ticks'])
  command=(p/'cmdline').read_bytes().decode(errors='replace') if matched else ''
  return dict(pid=r['pid'],matched=matched,alive=matched and f[0] not in ('Z','X'),own_command=str(root) in command)
 except FileNotFoundError:return dict(pid=r['pid'],matched=False,alive=False)
def tail(p):
 if not p.exists():return None
 with p.open('rb') as f:
  f.seek(0,2);f.seek(max(0,f.tell()-32768));return '\n'.join(f.read().decode(errors='replace').splitlines()[-7:])
jobs={}
for i in range(1,7):
 key=f'S{i}';entry=root/'jobs'/key;metrics=[];parse_error=None
 try:
  if (entry/'metrics.jsonl').exists():metrics=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines() if x.strip()]
 except json.JSONDecodeError:parse_error='JSONDecodeError_during_refresh'
 initial=read(entry/'route_initialization.json');probe=read(entry/'gate.json')
 stages={s:dict(launch=read(entry/(s+'_launch.json')),identity=identity(read(entry/(s+'_launch.json'))),complete=read(entry/(s+'_complete.json'))) for s in ('gate','train','evaluate','readout','audit')}
 jobs[key]=dict(claim=read(entry/'train_claim.json'),mature=read(entry/'mature.json'),mature_progress=read(entry/'mature_progress.json'),probe=probe,probe_progress=read(entry/'gate_progress.json'),
  initialization=initial,first_update=read(entry/'first_update.json'),
  cold_initialization_matches_probe=bool(initial and probe and initial['native_sha']==probe['native_initial_sha256'] and initial['full_sha']==probe['full_initial_sha256'] and initial['cold_start']),
  completed_epochs=len(metrics),last_metric=metrics[-1] if metrics else None,recent_epoch_seconds=[x['seconds'] for x in metrics[-3:]],metrics_parse_error=parse_error,
  termination=read(entry/'termination.json'),native_audit=read(entry/'native_ap_audit.json'),result=read(entry/'result.json'),audit=read(entry/'audit.json'),
  checkpoint_sizes={n:(entry/(n+'.pt')).stat().st_size if (entry/(n+'.pt')).exists() else None for n in ('best','latest')},
  stages=stages,logs={s:tail(entry/(s+'.log')) for s in ('gate','train','evaluate','readout','audit')})
print(json.dumps(dict(root=str(root),exists=root.exists(),checked_time=time.time(),
 controller=identity(read(root/'formal/controller_launch.json')),launch=read(root/'formal/controller_launch.json'),
 state=read(root/'formal/state.json'),error=read(root/'formal/controller_error.json'),
 optimization=read(root/'formal/optimization_plan.json'),acceptance=read(root/'formal/startup_acceptance.json'),summary=read(root/'formal/summary.json'),
 deploy_receipt=read(root/'deploy_receipt.json'),jobs=jobs,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
'''

DEPLOY=r'''
import base64,hashlib,io,json,os,pathlib,shutil,subprocess,tarfile,time
root=pathlib.Path(ROOT);payload=base64.b64decode(PAYLOAD)
if root.exists():raise ValueError('专属ROOT已有现场，只观察，不重复提交')
if hashlib.sha256(payload).hexdigest()!=SHA:raise ValueError('传输SHA不符')
if shutil.disk_usage(root.parent).free<100*2**30:raise ValueError('完整状态磁盘余量不足')
root.mkdir()
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tar:
 for item in tar.getmembers():
  dest=(root/item.name).resolve()
  if not item.isfile() or root.resolve() not in dest.parents:raise ValueError('非法归档成员')
  dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb') as f:f.write(tar.extractfile(item).read())
manifest=json.loads((root/'run_manifest.json').read_text())
for name,expected in manifest['files'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('运行文件SHA不符：'+name)
env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='123',
 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',WANDB_MODE='disabled',
 PYTHONPATH=os.pathsep.join([str(root/'code/training'),str(root/'code/inference'),
 '/vepfs/vlms/nas_tad_formal12_1111/nms_build','/vepfs/vlms/nas_tad_formal12_1111/python_deps']),
 LD_PRELOAD='/vepfs/vlms/nas_tad_formal12_1111/runtime_lib/libstdc++.so.6')
python='/vepfs/vlms/envs/rq1-cu121/bin/python'
with (root/'prepare.log').open('x') as log:
 subprocess.run([python,'-B',str(root/'prepare.py')],cwd=root,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
(root/'formal').mkdir()
with (root/'formal/controller.log').open('x') as log:
 child=subprocess.Popen([python,'-Bu',str(root/'formal_controller.py')],cwd=root,env=env,
  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
ticks=pathlib.Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
receipt=dict(pid=child.pid,start_ticks=ticks,payload_sha256=SHA,code_sha256=manifest['code_sha256'],
 time=time.time(),formal_cold_starts=6,seed=123,max_GPU_jobs=6,original_A_fixed=False,multivlm_A=True)
(root/'formal/controller_launch.json').write_text(json.dumps(receipt))
(root/'deploy_receipt.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))
'''


def package():
    from prepare import digest,save
    names=('prepare.py','gate.py','worker.py','common.py','speed.py','formal_controller.py','authorization.json','final_code_review.json','scheduling.py','prototype_adapter.py')
    paths=list((HERE/'code').rglob('*.py'))+list((HERE/'reference').rglob('config.json'))+[HERE/n for n in names]
    manifest=dict(code_sha256=digest(),files={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    save(HERE/'run_manifest.json',manifest);paths.append(HERE/'run_manifest.json')
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w:gz') as tar:
        for p in sorted(paths):
            info=tarfile.TarInfo(p.relative_to(HERE).as_posix());info.size=p.stat().st_size;info.mtime=0;info.mode=0o644
            with p.open('rb') as f:tar.addfile(info,f)
    data=out.getvalue();return data,hashlib.sha256(data).hexdigest()


def main(deploy=False):
    from common import save
    status=call('ROOT='+repr(ROOT)+'\n'+QUERY)
    if deploy and status['exists']:
        from prepare import digest
        receipt=status.get('deploy_receipt') or {}
        if receipt.get('code_sha256')!=digest():
            raise ValueError('远端已存在不同源码绑定，只能观察，禁止覆盖或重复派发')
    if deploy and not status['exists']:
        data,sha=package()
        save(HERE/'formal_campaign.json',dict(root=ROOT,payload_sha256=sha,submission_state='prepared',budget=6,seed=123))
        launch=call('ROOT,PAYLOAD,SHA='+repr((ROOT,base64.b64encode(data).decode(),sha))+'\n'+DEPLOY)
        save(HERE/'formal_campaign.json',dict(root=ROOT,payload_sha256=sha,launch=launch,submission_state='submitted',budget=6,seed=123))
        status=call('ROOT='+repr(ROOT)+'\n'+QUERY)
    elif deploy and status['exists'] and not status['launch']:
        raise ValueError('发现部分提交现场，先核实，不覆盖或重提')
    if status['launch'] and not (HERE/'formal_campaign.json').exists():
        save(HERE/'formal_campaign.json',dict(root=ROOT,launch=status['launch'],submission_state='recovered_receipt',budget=6,seed=123))
    status['checked_utc']=datetime.now(timezone.utc).isoformat()
    save(HERE/'formal_status.json',status);save(HERE/'formal_connectivity.json',dict(status='connected',checked_utc=status['checked_utc']))
    print(json.dumps(dict(root=ROOT,controller=status['controller'],state=status['state'],error=status['error'],
        gpu=status['gpu'],optimization=status.get('optimization'),jobs={key:dict(probe=v['probe']['status'] if v['probe'] else None,claim=bool(v['claim']),
        initialization_paired=v['cold_initialization_matches_probe'],first_update=v['first_update'],epoch=v['completed_epochs'],
        train=v['stages']['train']['identity'],probe_process=v['stages']['gate']['identity'],mature=v['mature'],mature_progress=v['mature_progress'],
        last_metric=v['last_metric'],log=v['logs']['gate'] if not v['probe'] else v['logs']['train'] if not v['last_metric'] else None)
        for key,v in status['jobs'].items()},summary=status['summary']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deploy',action='store_true')
    try:main(p.parse_args().deploy)
    except Exception as error:
        from common import save
        save(HERE/'formal_connectivity.json',dict(status='query_or_submission_error',error_type=type(error).__name__,
            old_snapshot_preserved=True,checked_utc=datetime.now(timezone.utc).isoformat()))
        print(type(error).__name__);raise
