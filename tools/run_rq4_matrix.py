"""有限24项：先四路真实数据验收，再双卡独立调度；异常停止，不自动覆盖。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.rq4_protocol import file_sha
from tools.run_rq4_entry import save, read, audit_complete
from tools.run_rq2_matched_control import check_code

CODE = Path(__file__).resolve().parents[1]
TEACHERS = ('timelens2_8b','timeple_8b')
STRATEGIES = ('direct','gated')
ACTIVE = {}


def matrix_entries():
    return [(t,s,'S'+str(i)) for i in range(1,7) for t in TEACHERS for s in STRATEGIES]


def stop_children():
    for child, _, _ in ACTIVE.values():
        if child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    for child, stream, _ in ACTIVE.values():
        if child.poll() is None:
            child.wait(timeout=60)
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--caches',type=Path,required=True)
    parser.add_argument('--expected-head',required=True)
    args = parser.parse_args()
    check_code(args.expected_head)
    lock = open('/vepfs/vlms/SADA_PLUS/.rq4_formal24.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.output.mkdir(parents=True,exist_ok=False)
    def interrupted(*_):
        raise RuntimeError('user/system stop; preserve outputs; do not automatically restart')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    pending = [(t,s,'S1') for t in TEACHERS for s in STRATEGIES]
    phase, reports, completed = 'acceptance', [], []
    acceptance = args.output/'acceptance.json'
    state = dict(pid=os.getpid(),code_commit=args.expected_head,completed=completed)
    def status():
        state.update(phase=phase,pending=pending,
            active={gpu:dict(pid=v[0].pid,entry=v[2]) for gpu,v in ACTIVE.items()})
        temporary=args.output/'state.json.tmp'
        temporary.write_text(json.dumps(state,indent=2))
        temporary.replace(args.output/'state.json')
    try:
        while True:
            for gpu,(child,stream,entry) in list(ACTIVE.items()):
                rc=child.poll()
                if rc is None: continue
                stream.close(); del ACTIVE[gpu]
                if rc: raise RuntimeError(f'{phase} failed: {entry}, exit={rc}')
                t,s,shift=entry
                output=args.output/phase/t/shift/s
                if phase=='acceptance':
                    path=output/'smoke.json'
                    if read(path)['status']!='passed': raise ValueError('acceptance did not pass')
                    reports.append(dict(path=str(path),sha256=file_sha(path)))
                else:
                    # 独立再审，不因子进程返回0就记完成。
                    audit_complete(output)
                    completed.append(dict(teacher=t,strategy=s,shift=shift,root=str(output)))
                print('RQ4_MATRIX_ITEM_DONE',phase,entry,flush=True)
            if not pending and not ACTIVE:
                if phase=='acceptance':
                    save(acceptance,dict(status='passed',code_commit=args.expected_head,reports=reports))
                    phase='formal'; pending=matrix_entries()
                    print('RQ4_ALL_REAL_ACCEPTANCE_PASS; FORMAL_24_ENABLED',flush=True)
                else:
                    for row in completed: audit_complete(row['root'])
                    save(args.output/'final_audit.json',dict(status='passed',count=len(completed),entries=completed))
                    phase='complete'; status()
                    print('RQ4_FORMAL24_AUDIT_COMPLETE',flush=True)
                    break
            for gpu in ('0','1'):
                if gpu in ACTIVE: continue
                ready=next((e for e in pending if (args.caches/e[0]/e[2]/'artifacts'/
                           (e[1]+'.json')).exists()),None)
                if ready is None: continue
                pending.remove(ready)
                t,s,shift=ready
                output=args.output/phase/t/shift/s
                output.parent.mkdir(parents=True,exist_ok=True)
                baseline_gpu='2' if int(shift[1:])%2 else '3'
                baseline=f'/vepfs/vlms/sada_unified_f9ac276_1009_gpu{baseline_gpu}/{shift}/uda_preflight.json'
                command=[sys.executable,'-B','-m','tools.run_rq4_entry',
                    '--baseline-config',baseline,'--spec',str(args.caches/t/shift/'artifacts'/(s+'.json')),
                    '--output',str(output),'--expected-head',args.expected_head,'--gpu',gpu]
                if phase=='acceptance': command+=['--smoke-only']
                stream=(output.parent/(s+'.log')).open('x')
                child=subprocess.Popen(command,cwd=CODE,env=dict(os.environ,
                    CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),
                    RQ4_REAL_ACCEPTANCE=str(acceptance)),stdout=stream,stderr=subprocess.STDOUT,
                    start_new_session=True)
                ACTIVE[gpu]=(child,stream,ready)
                print('RQ4_MATRIX_STARTED',phase,ready,'gpu',gpu,'pid',child.pid,flush=True)
            status(); time.sleep(10)
    except BaseException as error:
        state['error']=str(error); status()
        stop_children()
        raise


if __name__=='__main__': main()
