"""有限顺序：旧24全审→软关系12验收/正式全审→原Qwen100六项。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.rq4_artifacts import reference, checked
from tools.rq4_relational import prepare, VARIANT
from tools.rq4_relation_quality import diagnose
from tools.rq4_relation_launch import TEACHERS, SHIFTS, predecessor, validate_acceptance
from tools.run_rq4_entry import read, save, acceptance_ready
from tools.run_rq4_relational_entry import audit_complete
from tools.run_rq2_matched_control import check_code

CODE=Path(__file__).resolve().parents[1]


def entries():
    return [(t,'S'+str(i)) for i in range(1,7) for t in TEACHERS]


def baseline(shift):
    gpu='2' if int(shift[1:])%2 else '3'
    return Path(f'/vepfs/vlms/sada_unified_f9ac276_1009_gpu{gpu}/{shift}/uda_preflight.json')


def stop_children(active):
    for child,stream,entry in active.values():
        if child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    for child,stream,entry in active.values():
        if child.poll() is None: child.wait(timeout=60)
        stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','caches','rq4-root','qwen-code','qwen-references','qwen-output'):
        parser.add_argument('--'+key,type=Path,required=True)
    for key in ('expected-head','rq4-head','qwen-head'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(); check_code(args.expected_head)
    lock=open('/vepfs/vlms/SADA_PLUS/.rq4_soft_relation_matrix.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.output.mkdir(parents=True,exist_ok=False)
    active={}; completed=[]; reports=[]; sources=[]; pending=[]; phase='source_and_cpu_prepare'
    state=dict(pid=os.getpid(),code_commit=args.expected_head,variant=VARIANT,completed=completed)
    acceptance=args.output/'acceptance.json'
    def status():
        state.update(phase=phase,pending=pending,
                     active={g:dict(pid=v[0].pid,entry=v[2]) for g,v in active.items()})
        temporary=args.output/'state.json.tmp'
        temporary.write_text(json.dumps(state,indent=2)); temporary.replace(args.output/'state.json')
    def interrupted(*_): raise RuntimeError('explicit stop; preserve outputs, no automatic restart')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        status()
        for teacher,shift in entries():
            folder=args.output/'prepared'/teacher/shift
            spec=prepare(folder,original_spec=args.caches/teacher/shift/'artifacts/direct.json')
            proof=read(spec['evidence_path'])
            report=diagnose(proof['source'],teacher,read(baseline(shift)))
            save(folder/'source_quality.json',report); sources.append(reference(folder/'source_quality.json'))
            print('SOFT_RELATION_SOURCE_READY',teacher,shift,report['summary'],flush=True)
        if args.prepare_only:
            phase='prepared_only'; status(); return
        phase='waiting_original24_final'; status()
        while not (args.rq4_root/'final_audit.json').exists():
            old=read(args.rq4_root/'state.json')
            if old.get('error'): raise RuntimeError('original RQ4 error: '+old['error'])
            time.sleep(10)
        phase='original24_audit'; status()
        proof=predecessor(args.rq4_root/'final_audit.json',args.rq4_head,full_audit=True)
        save(args.output/'predecessor.json',proof)
        phase='acceptance'; pending=entries()
        while True:
            for gpu,(child,stream,entry) in list(active.items()):
                rc=child.poll()
                if rc is None: continue
                stream.close(); del active[gpu]
                if rc: raise RuntimeError(f'{phase} {entry} failed, exit={rc}')
                teacher,shift=entry; output=args.output/phase/teacher/shift/'relational'
                if phase=='acceptance':
                    report=read(output/'smoke.json')
                    if report['status']!='passed': raise ValueError('real acceptance failed')
                    reports.append(reference(output/'smoke.json'))
                else:
                    audit_complete(output)
                    completed.append(dict(teacher=teacher,strategy='relational',shift=shift,root=str(output)))
                print('SOFT_RELATION_ITEM_DONE',phase,entry,flush=True)
            if not pending and not active:
                if phase=='acceptance':
                    record=dict(status='passed',code_commit=args.expected_head,variant=VARIANT,
                                predecessor=proof,source_diagnostics=sources,reports=reports)
                    validate_acceptance(record); save(acceptance,record)
                    for teacher in TEACHERS: acceptance_ready(acceptance,teacher,args.expected_head,'relational')
                    phase='formal'; pending=entries()
                    print('SOFT_RELATION_TWELVE_REAL_ACCEPTANCE_PASS',flush=True)
                else:
                    if len(completed)!=12 or {(r['teacher'],r['shift']) for r in completed}!=set(entries()):
                        raise ValueError('incomplete twelve-shift matrix')
                    for row in completed: audit_complete(row['root'])
                    save(args.output/'final_audit.json',dict(status='passed',count=12,variant=VARIANT,
                        code_commit=args.expected_head,predecessor=proof,entries=completed))
                    phase='relation_complete'; status(); break
            for gpu in ('0','1'):
                if gpu in active or not pending: continue
                teacher,shift=pending.pop(0)
                output=args.output/phase/teacher/shift/'relational'; output.parent.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,'-B','-m','tools.run_rq4_relational_entry',
                     '--baseline-config',str(baseline(shift)),
                     '--spec',str(args.output/'prepared'/teacher/shift/'spec.json'),
                     '--output',str(output),'--expected-head',args.expected_head,'--gpu',gpu]
                if phase=='acceptance': cmd.append('--smoke-only')
                stream=(output.parent/'relational.log').open('x')
                child=subprocess.Popen(cmd,cwd=CODE,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
                    PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),RQ4_REAL_ACCEPTANCE=str(acceptance)),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(child,stream,(teacher,shift))
                print('SOFT_RELATION_STARTED',phase,teacher,shift,'gpu',gpu,'pid',child.pid,flush=True)
            status(); time.sleep(10)
        # 只有新12项完整独立终审后，才恢复原Qwen六项链路；原等待根不复用。
        qhead=subprocess.check_output(['git','-C',str(args.qwen_code),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(args.qwen_code),'status','--porcelain'],text=True).strip()
        if qhead!=args.qwen_head or dirty or args.qwen_output.exists():
            raise ValueError('Qwen clean code/new output prerequisite failed')
        command=[sys.executable,'-B','-u','-m','tools.run_rq3_full_matrix',
                 '--output',str(args.qwen_output),'--references',str(args.qwen_references),
                 '--rq4-root',str(args.rq4_root),'--expected-head',args.qwen_head,'--rq4-head',args.rq4_head]
        stream=(args.output/'qwen_followup.log').open('x')
        child=subprocess.Popen(command,cwd=args.qwen_code,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
            PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(args.qwen_code)),
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        active['followup']=(child,stream,('qwen100','six'))
        save(args.output/'qwen_handoff.json',dict(pid=child.pid,root=str(args.qwen_output),
            prerequisite=reference(args.output/'final_audit.json'),command=command))
        phase='qwen_followup'; status()
        while child.poll() is None: time.sleep(10)
        stream.close(); del active['followup']
        if child.returncode: raise RuntimeError('Qwen followup failed; preserve outputs')
        final=read(args.qwen_output/'final_audit.json')
        if final.get('status')!='passed' or final.get('count')!=6: raise ValueError('Qwen final audit missing')
        phase='all_complete'; status()
    except BaseException as error:
        state['error']=str(error); stop_children(active); active.clear(); status(); raise


if __name__=='__main__': main()
