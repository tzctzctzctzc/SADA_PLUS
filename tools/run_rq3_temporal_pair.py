"""TimeLens2/TimePLE类别辅助36项；GPU0/1各一教师，不改旧RQ3协议。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tools.rq3_extra_teachers import EXTRA, frozen_protocol
from tools.rq3_temporal_teachers import TEACHERS
from tools.run_rq3_trusted_queue import RUNTIME, DATA, OLD_CANDIDATES
from tools.run_rq3_matrix import baseline_root, completed_baseline, load_entry_plan
from tools.run_rq3_trusted_monitor import audit_complete
from tools.score_rq3_vlm_candidates import check_code
from tools.rq2_dense_protocol import sha256_file

CODE = Path(__file__).resolve().parents[1]
BASE = Path('/vepfs/vlms')
PYTHON = str(BASE/'envs/rq1-cu121/bin/python')
RECOVERY = BASE/'data/EPIC_KITCHENS/rq3_frame_recovery_1001'
RECOVERY_SHA = '103d5a67b41715e63c37b41c5910d17e66ff448d63a33fd7607bfda6e7b9a906'


def save(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2))


def write_once(path, obj):
    if path.exists():
        if json.loads(path.read_text()) != obj: raise ValueError('existing configuration differs: '+str(path))
    else:
        with path.open('x') as f: json.dump(obj, f, sort_keys=True)


def worker(args):
    check_code(args.expected_head)
    teacher = args.teacher; slot = TEACHERS.index(teacher); gpu = slot
    root = args.root/teacher; root.mkdir(exist_ok=False)
    spec = EXTRA[teacher]
    model = BASE/'cache/huggingface/hub'/('models--'+spec['repo'].replace('/','--'))/'snapshots'/spec['revision']
    settings = dict(model_path=str(model))
    if teacher == 'timeple_8b': settings['official_repo'] = str(BASE/'repos/TimePLE')
    settings_path = RUNTIME/(teacher+'_settings.json'); anchor = RUNTIME/(teacher+'_anchor.json')
    write_once(settings_path, settings); write_once(anchor, frozen_protocol(teacher))
    plan = dict(schema='rq3-expanded-fresh-matrix-v1', seed=6, fresh=True,
        entries=[dict(teacher=teacher,strategy=s,shift='S'+str(i))
                 for i in range(1,7) for s in ('hard','verify','prototype')])
    save(root/'plan.json', plan); load_entry_plan(root/'plan.json')
    child = None
    def stop(*_): raise InterruptedError('user stop')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def run(command, name, python):
        nonlocal child
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', RQ3_QUEUE_SLOT=str(slot),
            PYTHONPATH=str(CODE)+':/vepfs/vlms/cache/rq3_preparation_deps_0928', PYTHONDONTWRITEBYTECODE='1')
        with (root/(name+'.log')).open('x') as f:
            child = subprocess.Popen([python,'-u',*command], cwd=CODE, env=env,
                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            save(root/'state.json', dict(phase=name, pid=os.getpid(), child=child.pid, gpu=gpu, teacher=teacher))
            print('RQ3_TEMPORAL_CHILD', teacher,name,child.pid,flush=True)
            if child.wait(): raise RuntimeError(name+' failed; keep outputs')
    try:
        run(['-m','tools.score_rq3_vlm_candidates','--model',teacher,'--settings',str(settings_path),
            '--frozen-protocol',str(anchor),'--manifest',str(OLD_CANDIDATES/'S1/manifest.json'),
            '--frame-root',str(OLD_CANDIDATES/'S1/frames'),'--output-root',str(root/'smoke'),
            '--expected-head',args.expected_head,'--gpu',str(gpu),'--wait-gpu'], 'native_class_smoke', PYTHON)
        assert json.loads((root/'smoke/interface_audit.json').read_text())['smoke_status']=='passed'
        run(['-m','tools.run_rq3_trusted_queue','--entry-plan',str(root/'plan.json'),
            '--output-root',str(root/'formal18'),'--expected-head',args.expected_head,'--gpu',str(gpu),
            '--queue-slot',str(slot),'--batch-size','2','--early-stop-patience','15','--refresh-teacher-caches',
            '--prepared-root',str(BASE/'rq3_videochat3_18_5c299e3_1010'),
            '--recovery-root',str(RECOVERY),'--recovery-manifest-sha256',RECOVERY_SHA],
            'queue18',str(BASE/'envs/sada/bin/python'))
        os.chdir(DATA)
        baselines = {'S'+str(i):completed_baseline(baseline_root(BASE,'S'+str(i)),'S'+str(i)) for i in range(1,7)}
        results = {}
        for path in (root/'formal18').rglob('complete.json'):
            key = audit_complete(path,baselines)
            if key in results: raise ValueError('duplicate formal identity')
            results[key] = dict(path=str(path),sha256=sha256_file(path))
        assert set(results) == {(teacher,s,'S'+str(i)) for i in range(1,7) for s in ('hard','verify','prototype')}
        save(root/'final_audit.json',dict(status='passed',count=18,results=list(results.values())))
        save(root/'state.json',dict(phase='complete',count=18))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM); child.wait()
        save(root/'state.json',dict(phase='failed',error=str(error),teacher=teacher))
        raise


def driver(args):
    check_code(args.expected_head); args.root.mkdir(exist_ok=False)
    save(args.root/'protocol.json',dict(code_commit=args.expected_head,teachers=TEACHERS,gpus=[0,1],
        count=36,batch_size=2,seed=6,patience=15,max_epochs=50,
        participation=dict(hard='all-common-candidates',verify=.2,prototype=.2)))
    children=[]
    def stop(*_): raise InterruptedError('user stop')
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    try:
        for teacher in TEACHERS:
            with (args.root/(teacher+'.log')).open('x') as f:
                child=subprocess.Popen([sys.executable,'-u','-m','tools.run_rq3_temporal_pair','worker',
                    '--root',str(args.root),'--expected-head',args.expected_head,'--teacher',teacher],
                    cwd=CODE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,
                    stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child)
                print('RQ3_TEMPORAL_WORKER',teacher,child.pid,flush=True)
        codes=[c.wait() for c in children]
        if any(codes): raise RuntimeError('one or more teacher workers failed: '+str(codes))
        reports={t:dict(path=str(args.root/t/'final_audit.json'),sha256=sha256_file(args.root/t/'final_audit.json')) for t in TEACHERS}
        save(args.root/'final_audit.json',dict(status='passed',count=36,reports=reports))
    finally:
        for child in children:
            if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
        for child in children: child.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['driver','worker'])
    p.add_argument('--root',type=Path,required=True); p.add_argument('--expected-head',required=True)
    p.add_argument('--teacher',choices=TEACHERS); args=p.parse_args(); args.root=args.root.resolve()
    globals()[args.mode](args)


if __name__=='__main__': main()
