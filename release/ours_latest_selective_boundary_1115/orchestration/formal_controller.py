"""唯一六向控制器：先六向真实GPU门验，再固定原A冷启动，四卡最多六路。"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from common import read,save,sha
from prepare import digest

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'formal'
KEYS=[f'S{i}' for i in range(1,7)]
ORDER=['S5','S6','S4','S2','S1','S3']
ACTIVE={}
LEASES={}
BUDGETS={key:read(ROOT/'reference'/key/'config.json')['baseline']['epoch_budget']['iterations_per_epoch'] for key in KEYS}
NEXT={'gate':'await_train','train':'await_evaluate','evaluate':'await_readout','readout':'await_audit','audit':'complete'}


def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def cards():
    value=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(a),uuid=b.strip(),free_mib=int(c),utilization=int(d))
        for a,b,c,d in (line.split(',') for line in value.splitlines())]


def reserve(stage):
    from scheduling import ranked_cards
    receipts=[j['receipt'] for j in ACTIVE.values()]
    for card in ranked_cards(cards(),receipts,stage,BUDGETS):
        index=card['index']
        if index in LEASES:
            card['admission']='shared_own_queue_lease'
            return card,LEASES[index]
        own=(OUT/f"gpu{index}.lock").open('a')
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:own.close();continue
        shared=(Path('/vepfs/vlms/SADA_PLUS')/f".rq2_gpu{index}.lock").open('a')
        try:
            fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
            card['admission']='project_exclusive_lease';leases=[own,shared]
        except BlockingIOError:
            shared.close();card['admission']='user_authorized_shared_existing_project_lease';leases=[own]
        LEASES[index]=leases
        return card,leases
    return None,None


def launch(key,stage,card=None,leases=()):
    entry=ROOT/'jobs'/key
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=card['uuid'] if card else '',OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED='123',PYTHONUTF8='1',
        PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled')
    cmd=[sys.executable,'-Bu',str(ROOT/'gate.py'),'--entry',str(entry)] if stage=='gate' else [
        sys.executable,'-Bu',str(ROOT/'worker.py'),stage,'--entry',str(entry)]
    with (entry/(stage+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=tuple(f.fileno() for f in leases))
    receipt=dict(pid=child.pid,start_ticks=ticks(child.pid),shift=key,stage=stage,gpu=card,
        time=time.time(),cold_start=stage=='train')
    save(entry/(stage+'_launch.json'),receipt)
    ACTIVE[key]=dict(process=child,receipt=receipt,leases=leases)


def acceptance():
    code=digest();proof={};plan=read(OUT/'optimization_plan.json')
    for key in KEYS:
        entry=ROOT/'jobs'/key;gate=read(entry/'gate.json');cfg=read(entry/'config.json');parent=read(ROOT/'reference'/key/'config.json')
        assert gate['status']=='passed' and gate['code_sha256']==code and gate['config_sha256']==sha(entry/'config.json')
        assert gate['native_initial_sha256']==gate['parent_native_sha256']
        assert not gate['probe_weights_reused'] and gate['selected_fast']==plan['use_fast']
        for name in ('model','prototype_capacity','loader','opt','train_cfg','test_cfg','dataset','src_train_split','src_val_split','tgt_train_split','tgt_val_split'):
            assert cfg[name]==parent[name],(key,name)
        assert cfg['uda_route']['b_evidence_mode']=='shared_joint_path' and cfg['uda_route']['b_semantic_alpha']==.1 and cfg['uda_route']['b_vlm_gate'] and cfg['uda_route']['b_selective_boundary']
        assert cfg['moe_signal']==read(ROOT/'reference/multivlm'/key/'config.json')['moe_signal']
        assert gate['multivlm_A_probe']['target_gradient_norm']>0
        assert sha(cfg['moe_signal']['path'])==cfg['moe_signal']['sha256']
        for expected,path in [(cfg['baseline']['source_manifest_sha256'],cfg['dataset']['json_file']),
                              (cfg['baseline']['target_manifest_sha256'],cfg['baseline']['target_manifest']),
                              (cfg['prototype_capacity']['prepared_cache_sha256'],cfg['prototype_capacity']['prepared_cache'])]:
            assert sha(path)==expected
        assert not (entry/'train_claim.json').exists()
        proof[key]=dict(config_sha256=gate['config_sha256'],native_initial_sha256=gate['native_initial_sha256'],
            full_initial_sha256=gate['full_initial_sha256'],peak_mib=gate['peak_reserved_mib'],native_single_FPN_exact=gate['single_FPN_native_exact'])
    save(OUT/'startup_acceptance.json',dict(status='passed',code_sha256=code,directions=proof,cold_starts=6,seed=123,
        original_A_fixed=False,multivlm_A=True,probe_weights_reused=False,optimization_plan_sha256=sha(OUT/'optimization_plan.json'),
        max_GPU_jobs=6,max_own_jobs_per_card=2,shared_GPU_allowed=True,min_free_MiB=12288))


def summary():
    rows={}
    for key in KEYS:
        audit=read(ROOT/'jobs'/key/'audit.json');assert audit['status']=='passed'
        rows[key]=dict(native=audit['native']['target'],final=audit['final']['final'])
    final=sum(v['final']['mAP_percent'] for v in rows.values())/6
    save(OUT/'summary.json',dict(status='passed',directions=rows,
        native_mean_percent=sum(v['native']['mAP_percent'] for v in rows.values())/6,
          uniform_readout_mean_percent=final,reference_best_percent=55.57286110553307,
          delta_to_best_pp=final-55.57286110553307,reaches_previous_best=final>=55.57286110553307,
        seed=123,original_A_fixed=False,multivlm_A=True,per_direction_tuning=False,target_TRAIN_GT=False,
        limitations=['单seed','统一推理规则曾在同批目标验证视频上筛选']))


def main():
    auth=read(ROOT/'authorization.json')
    assert auth['formal_training_authorized'] and auth['budget']==6 and auth['seed']==123 and auth['entries']==KEYS
    with (OUT/'controller_claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),start_ticks=ticks(os.getpid())),f)
    manifest=read(ROOT/'run_manifest.json')['files']
    for name,expected in manifest.items():assert sha(ROOT/name)==expected,name
    states={key:'await_gate' for key in KEYS};failures=[];validated=False
    while True:
        for key,job in list(ACTIVE.items()):
            rc=job['process'].poll()
            if rc is None:continue
            stage=job['receipt']['stage'];save(ROOT/'jobs'/key/(stage+'_complete.json'),dict(returncode=rc,time=time.time()))
            card=job['receipt']['gpu']
            del ACTIVE[key]
            if card and not any(j['receipt']['gpu'] and j['receipt']['gpu']['index']==card['index'] for j in ACTIVE.values()):
                for lease in LEASES.pop(card['index']):lease.close()
            if rc:states[key]='failed';failures.append(dict(shift=key,stage=stage,returncode=rc))
            else:states[key]=NEXT[stage]
        if not failures and not validated and not any('gate' in v for v in states.values()):
            acceptance();validated=True
        if not failures:
            for key in KEYS:
                if states[key]=='await_audit' and sum(j['receipt']['stage']=='audit' for j in ACTIVE.values())<2:
                    launch(key,'audit');states[key]='running_audit'
            # 长方向先启动，空位立即接续；最多两路评估，CPU审计不占GPU。
            stages=('evaluate','readout','train') if validated else ('gate',)
            for stage in stages:
                for key in ORDER:
                    if states[key]!='await_'+stage:continue
                    if stage=='gate' and key!='S1' and not (OUT/'optimization_plan.json').exists():continue
                    gpu_jobs=[j for j in ACTIVE.values() if j['receipt']['gpu']]
                    if len(gpu_jobs)>=(6 if validated else 4):break
                    if stage in ('evaluate','readout') and sum(j['receipt']['stage'] in ('evaluate','readout') for j in gpu_jobs)>=6:break
                    card,leases=reserve(stage)
                    if card:launch(key,stage,card,leases);states[key]='running_'+stage
        save(OUT/'state.json',dict(states=states,active=[j['receipt'] for j in ACTIVE.values()],failures=failures,
            validated=validated,time=time.time(),max_GPU_jobs=6,max_training_jobs=6,physical_GPUs=4,max_own_jobs_per_card=2,max_GPU_evaluations=6,max_CPU_audits=2,
            CPU_threads_per_process=2,original_A_fixed=False,multivlm_A=True))
        if failures and not ACTIVE:save(OUT/'controller_error.json',dict(failures=failures));break
        if all(s=='complete' for s in states.values()):summary();break
        time.sleep(4)


if __name__=='__main__':
    try:main()
    except BaseException:
        save(OUT/'controller_error.json',dict(traceback=traceback.format_exc(),active=[j['receipt'] for j in ACTIVE.values()],
            no_children_terminated=True));raise
