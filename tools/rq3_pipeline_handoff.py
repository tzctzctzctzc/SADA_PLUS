"""在完整S1边界接CPU流水线，不中断学生，不重跑完成项；同卡并发默认关闭。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.rq3_colocation import process_start, write_json
from tools.rq2_dense_protocol import sha256_file
from tools.run_rq3_matrix import baseline_root, completed_baseline
from tools.run_rq3_trusted_monitor import audit_complete
from tools.run_rq3_temporal_pair import managed_environment, RECOVERY, RECOVERY_SHA
from tools.score_rq3_vlm_candidates import check_code

CODE=Path(__file__).resolve().parents[1]
DATA=Path('/vepfs/vlms/SADA_PLUS')
TEACHERS=('timelens2_8b','timeple_8b')


def remaining_rows(teacher, completed):
    expected={(teacher,s,'S'+str(i)) for i in range(1,7) for s in ('hard','verify','prototype')}
    if not set(completed).issubset(expected):raise ValueError('unknown completed identity')
    return [dict(teacher=t,strategy=s,shift=h) for t,s,h in sorted(expected-set(completed),
        key=lambda x:(int(x[2][1:]),('hard','verify','prototype').index(x[1])))]


def active_children(pid):
    path=Path(f'/proc/{pid}/task/{pid}/children')
    return [int(x) for x in path.read_text().split()] if path.exists() else []


def alive(pid,start):
    try:return process_start(pid)==start and Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def worker(args):
    check_code(args.expected_head);os.chdir(DATA)
    teacher=args.teacher;slot=TEACHERS.index(teacher)
    old=args.previous/teacher/'formal18'
    root=args.root/teacher;root.mkdir(exist_ok=False)
    state=json.loads((old/'state.json').read_text());pid=state['pid'];start=process_start(pid)
    if str(old) not in Path(f'/proc/{pid}/cmdline').read_text():raise ValueError('old queue identity differs')
    frozen=False;newchild=None
    def stop(*unused):raise InterruptedError('pipeline handoff stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        write_json(root/'state.json',dict(phase='waiting_complete_S1_boundary',old_queue=pid))
        while True:
            state=json.loads((old/'state.json').read_text())
            if state['status']!='running' or not alive(pid,start):
                raise RuntimeError('old queue exited before planned boundary; no automatic recovery')
            if int(state.get('shift','S1')[1:])>1:
                raise RuntimeError('S1 boundary already passed; leave healthy queue untouched')
            if state.get('strategy')=='prototype' and state.get('phase')=='training-evaluation':
                # 只暂停队列父进程；确认仅一个真实matrix子进程在执行完整条目。
                os.kill(pid,signal.SIGSTOP);frozen=True
                children=active_children(pid)
                if len(children)==1:
                    matrix=children[0]
                    command=Path(f'/proc/{matrix}/cmdline').read_text()
                    if 'tools.run_rq3_matrix' in command and str(old) in command and 'prototype' in command:
                        matrix_start=process_start(matrix);break
                os.kill(pid,signal.SIGCONT);frozen=False
            time.sleep(1)
        write_json(root/'state.json',dict(phase='draining_S1_prototype',old_queue=pid,matrix=matrix))
        while alive(matrix,matrix_start):time.sleep(1)
        paths=list(old.glob(f'**/{teacher}/prototype/S1_seed6/complete.json'))
        if len(paths)!=1:raise RuntimeError('S1 Prototype did not complete; preserve old artifacts')
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);frozen=False
        for _ in range(60):
            if not alive(pid,start):break
            time.sleep(1)
        else:raise RuntimeError('old queue did not exit')
        baselines={'S'+str(i):completed_baseline(baseline_root(Path('/vepfs/vlms'),'S'+str(i)),'S'+str(i)) for i in range(1,7)}
        completed={}
        for path in old.rglob('complete.json'):
            key=audit_complete(path,baselines)
            if key in completed:raise ValueError('duplicate complete identity')
            completed[key]=dict(path=str(path),sha256=sha256_file(path))
        if set(completed)!={(teacher,s,'S1') for s in ('hard','verify','prototype')}:
            raise ValueError('handoff requires exact complete S1 triple')
        plan=root/'remaining.json';write_json(plan,dict(schema='rq3-expanded-fresh-matrix-v1',seed=6,fresh=True,
            entries=remaining_rows(teacher,completed)))
        queue=root/('pipeline15_'+teacher+'_'+args.expected_head[:7])
        cmd=[sys.executable,'-u','-m','tools.run_rq3_trusted_queue','--entry-plan',str(plan),
            '--output-root',str(queue),'--expected-head',args.expected_head,'--gpu',str(slot),'--queue-slot',str(slot),
            '--batch-size','2','--early-stop-patience','15','--refresh-teacher-caches','--pipeline-cpu',
            '--prepared-root',str(old),'--recovery-root',str(RECOVERY),'--recovery-manifest-sha256',RECOVERY_SHA]
        with (root/'queue.log').open('x') as f:
            newchild=subprocess.Popen(cmd,cwd=CODE,env=managed_environment(slot),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(root/'state.json',dict(phase='pipeline_running',queue_pid=newchild.pid,queue_root=str(queue),old_completed=list(completed.values())))
        # 新完整项的独立CPU终审与后续训练重叠；不修改正式训练进程或RNG。
        audits={}
        with ThreadPoolExecutor(max_workers=1) as pool:
            while newchild.poll() is None:
                for path in queue.rglob('complete.json'):
                    if path not in audits:audits[path]=pool.submit(audit_complete,path,baselines)
                for future in audits.values():
                    if future.done():future.result()
                time.sleep(2)
            if newchild.returncode:raise RuntimeError('pipeline queue failed; keep partial outputs')
            for path in queue.rglob('complete.json'):
                if path not in audits:audits[path]=pool.submit(audit_complete,path,baselines)
            for path,future in audits.items():
                key=future.result()
                if key in completed:raise ValueError('duplicate pipeline result')
                completed[key]=dict(path=str(path),sha256=sha256_file(path))
        if remaining_rows(teacher,completed):raise ValueError('incomplete final matrix')
        write_json(root/'final_audit.json',dict(status='passed',count=18,results=list(completed.values())))
        write_json(root/'state.json',dict(phase='complete',count=18))
    except BaseException as error:
        write_json(root/'error.json',dict(error=str(error)))
        raise
    finally:
        # 失败或停止不遗留冻结队列；子matrix若失败仍由原队列收尾。
        if frozen and alive(pid,start):os.kill(pid,signal.SIGCONT)
        if newchild is not None and newchild.poll() is None:
            os.killpg(newchild.pid,signal.SIGTERM);newchild.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--expected-head',required=True);p.add_argument('--teacher',choices=TEACHERS,required=True)
    worker(p.parse_args())


if __name__=='__main__':main()
