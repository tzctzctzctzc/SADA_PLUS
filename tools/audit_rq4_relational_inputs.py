"""两教师六shift及无VLM对照的禁CUDA工件/真实train_da预检；绝不训练。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys

from tools.rq4_relational import prepare, audit_spec
from tools.rq4_protocol import load_cache, file_sha
from tools.run_rq4_entry import read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caches', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/vepfs/vlms/SADA_PLUS'))
    args = parser.parse_args()
    from tools.run_rq2_matched_control import check_code
    check_code(args.expected_head)
    code = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(code),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')

    def one(teacher, shift):
        original = args.caches/'timelens2_8b'/shift/'artifacts/direct.json'
        if teacher=='sada_self':
            evidence = read(read(original)['evidence_path'])
            kwargs = dict(requests=evidence['requests']['path'])
        else:
            kwargs = dict(original_spec=args.caches/teacher/shift/'artifacts/direct.json')
        root = args.output/teacher/shift
        spec = prepare(root/'artifacts',**kwargs)
        cache = load_cache(spec['cache_path'],spec['cache_file_sha256'])
        regions = audit_spec(spec,cache)
        gpu = '2' if int(shift[1:])%2 else '3'
        baseline = args.data_root.parent/f'sada_unified_f9ac276_1009_gpu{gpu}'/shift/'uda_preflight.json'
        command = [sys.executable,'-B','-m','tools.run_rq4_relational_entry',
            '--baseline-config',str(baseline),'--spec',str(root/'artifacts/spec.json'),
            '--output',str(root/'entry'),'--expected-head',args.expected_head,
            '--gpu','0','--audit-only','--data-root',str(args.data_root)]
        with (root/'preflight.log').open('x') as log:
            subprocess.run(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report = dict(teacher=teacher,shift=shift,strategy=spec['strategy'],status='passed',
            candidates=len(cache['records']),core_nonempty=sum(x['core'] is not None for x in regions['rows']),
            preflight=str(root/'entry/preflight.json'),preflight_sha256=file_sha(root/'entry/preflight.json'))
        print('RELATIONAL_CPU_INPUT_PASS',teacher,shift,flush=True)
        return report

    jobs = [(t,'S'+str(i)) for i in range(1,7) for t in ('timelens2_8b','timeple_8b','sada_self')]
    reports = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(one,*item) for item in jobs]):
            reports.append(future.result())
    save(args.output/'audit.json',dict(status='passed',code_commit=args.expected_head,
        cuda_disabled=True,formal_result=False,real_gpu_acceptance=False,
        entries=sorted(reports,key=lambda x:(x['shift'],x['teacher']))))
    print('RELATIONAL_CPU_INPUT_AUDIT_PASS',len(reports),'NOT_A_TRAINING_RESULT',flush=True)


if __name__=='__main__': main()
