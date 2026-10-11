"""仅补评估已有合法结束的共识训练；以文件入口执行，旧代码负责训练终审。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('old-entry','output','training-code-root','evaluation-code-root','data-root'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--training-head',required=True)
    parser.add_argument('--evaluation-head',required=True)
    args=parser.parse_args()
    for key in ('old_entry','output','training_code_root','evaluation_code_root','data_root'):
        setattr(args,key,getattr(args,key).resolve())
    for root,head in ((args.training_code_root,args.training_head),(args.evaluation_code_root,args.evaluation_head)):
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=head
                or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()):
            raise ValueError('recovery requires exact clean code')
    if __package__:
        raise ValueError('execute by absolute file path, not module, to isolate original audit code')
    sys.path.insert(0,str(args.training_code_root))
    from tools.rq4_consensus import read,save
    from tools.rq4_consensus_audit import training_audit,audit_complete
    from tools.rq4_protocol import file_sha
    from tools.run_rq2_matched_control import run_command,checked_evaluation
    from tools.rq3_gpu import gpu_slot
    os.chdir(args.data_root)
    cfg=read(args.old_entry/'preflight.json');spec=cfg['rq4_boundary']
    baseline_path=Path(spec['baseline_config_path']);baseline=read(baseline_path)
    name='rq4_'+read(spec['cache_path'])['teacher']+'_consensus_'+cfg['rq2_formal']['shift']+'_'+__import__('hashlib').sha256(str(args.old_entry).encode()).hexdigest()[:16]
    run_root=args.data_root/'model_results'/name
    config,best,latest,termination,report=training_audit(run_root,cfg,baseline)
    if read(spec['acceptance_path'])['code_commit']!=args.training_head:
        raise ValueError('original training acceptance commit differs')
    args.output.mkdir(parents=True,exist_ok=False)
    for filename in ('preflight.json','spec.json','baseline_pair.json','acceptance_reference.json'):
        shutil.copy2(args.old_entry/filename,args.output/filename)
    recovery=dict(training_code_root=str(args.training_code_root),training_commit=args.training_head,
        evaluation_code_root=str(args.evaluation_code_root),evaluation_commit=args.evaluation_head,
        original_entry=str(args.old_entry),retrained=False,
        original_artifacts={str(p):file_sha(p) for p in (config,best,latest,args.old_entry/'logs/train.log')})
    save(args.output/'recovery.json',recovery)
    import yaml
    shift=cfg['rq2_formal']['shift'];teacher=read(spec['cache_path'])['teacher']
    registry=args.data_root/'configs/rq2/shift_registry.yaml'
    frozen=args.evaluation_code_root/yaml.safe_load(registry.read_text())['shifts'][shift]['config']
    command=[sys.executable,'-B',str(args.evaluation_code_root/'evaluate_da.py'),'-c',str(frozen),
        '-s','6','-i','0','--rq2-shift',shift,'--rq2-registry',str(registry),
        '--rq2-reference-root',str(args.data_root/'data/ek100_slowfast/features'),
        '-n',name+'_recovered_eval','-r',str(best),'--rq2-formal',
        '--rq2-output-dir',str(args.output/'evaluation')]
    with gpu_slot('0',wait=True) as selected:
        run_command(command,args.output/'logs/evaluate.log',args.data_root,selected['uuid'],selected['index'])
    if checked_evaluation(args.output/'evaluation',cfg,best).get('rq4_boundary')!=spec:
        raise ValueError('recovered evaluation lost training identity')
    save(args.output/'supervision_audit.json',report)
    artifacts=[config,best,latest,config.parent/'benchmark_epoch_metrics.jsonl',baseline_path,
        args.output/'recovery.json',args.old_entry/'logs/train.log']
    artifacts += [args.output/x for x in ('preflight.json','spec.json','baseline_pair.json',
        'acceptance_reference.json','supervision_audit.json','evaluation/manifest.json',
        'evaluation/source_val.json','evaluation/target_val.json')]
    artifacts += [Path(spec[k+'_path']) for k in ('cache','evidence','regions','quality','source_policy','acceptance')]
    record=dict(schema='rq4-consensus-complete-v1',variant=spec['variant'],teacher=teacher,
        strategy='consensus',shift=shift,seed=6,batch_size=2,code_commit=args.training_head,
        evaluation_code_commit=args.evaluation_head,termination=termination,
        baseline_config=str(baseline_path),run_root=str(run_root),evaluation_gpu=selected,
        artifacts={str(p):file_sha(p) for p in artifacts})
    save(args.output/'complete.json',record)
    audit_complete(args.output)
    print('RECOVERED_CONSENSUS_COMPLETE',teacher,shift,flush=True)


if __name__=='__main__': main()
