"""固定先角色盲六项，再置换六项；每组全验收后fresh正式，原P/B均不重训。"""
import argparse,json,math,os,signal,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as p
from tools import run_phase_alignment as runner
from tools.phase_alignment_protocol import verify_probe
from tools.phase_control_protocol import prepare,audit
from libs.models.SADA.phase_alignment import CONTROL_SCHEMA,CONTROL_MODES
ACTIVE=runner.ACTIVE


def state(root,phase,mode,completed,pending):
    p.save(root/'state.json',dict(phase=phase,mode=mode,pid=os.getpid(),completed=completed,pending=pending,
        active={s:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage'],mode=mode) for s,t in ACTIVE.items()},target_gt_used=False))


def main(args):
    root=args.output_root;identity=p.code_identity();root.mkdir(exist_ok=False);(root/'tmp').mkdir()
    state(root,'input_audit',None,[],[]);os.chdir(args.data_root)
    configs=prepare(root,args.semantic_root,identity);os.chdir(CODE)
    source_gate=args.semantic_root/'source_gate.json'
    p.save(root/'plan.json',dict(schema=CONTROL_SCHEMA,code=identity,modes=list(CONTROL_MODES),gpus=args.gpus,
        source_gate=str(source_gate),source_gate_sha256=p.sha(source_gate),new_formal_runs=12,
        semantic_reference_retrained=False,original_baseline_retrained=False,
        interpretation='same_weight_rule_not_identical_model_dependent_weight_trajectory'))
    completed=[];pending=[];mode=None
    try:
        for mode in CONTROL_MODES:
            pending=sorted(configs[mode]);accepted=[]
            while pending or ACTIVE:
                for shift,t in list(ACTIVE.items()):
                    if t['process'].poll() is None:continue
                    if t['process'].returncode:raise RuntimeError(mode+'/'+shift+' probe failed; preserved')
                    e=t['entry'];report=p.read(e/'probe_result.json');ref=args.semantic_root/'formal'/shift
                    verify_probe(report,p.read(ref/'detector_initialization.json')['sha256'],mode)
                    p.save(e/'acceptance.json',dict(status='passed',schema=CONTROL_SCHEMA,mode=mode,code=identity,batch=2,seed=123,
                        source_gate_sha256=p.sha(source_gate),report_sha256=p.sha(e/'probe_result.json'),
                        minimum_free_mib=max(args.minimum_free_mib,math.ceil(report['peak_reserved_mib']+2048)),target_gt_used=False))
                    accepted.append(shift);t['context'].__exit__(None,None,None);del ACTIVE[shift]
                    print('CONTROL_ACCEPTED',mode,shift,flush=True)
                for shift in list(pending):
                    if not runner.launch(shift,'probe',root/'formal'/mode/shift,configs[mode][shift],args,identity):break
                    pending.remove(shift)
                state(root,'real_acceptance',mode,completed,pending);time.sleep(5)
            if set(accepted)!=set(configs[mode]):raise ValueError('验收不足六方向')
            group=root/('acceptance_'+mode+'_all6.json')
            p.save(group,dict(status='passed',mode=mode,code=identity,shifts=accepted,
                acceptance={str(root/'formal'/mode/s/'acceptance.json'):p.sha(root/'formal'/mode/s/'acceptance.json') for s in accepted}))
            pending=sorted(configs[mode]);group_completed=[]
            while pending or ACTIVE:
                for shift,t in list(ACTIVE.items()):
                    if t['process'].poll() is None:continue
                    if t['process'].returncode:raise RuntimeError(mode+'/'+shift+' '+t['stage']+' failed; preserved')
                    e=t['entry']
                    if t['stage']=='train':
                        t['stage']='evaluate';t['process']=runner.spawn('evaluate',e,t['lease'],args.data_root,args.control/'formal/control'/shift);continue
                    paths=['config.json','runtime_config.json','seed_proof.json','detector_initialization.json','metrics.jsonl',
                        'termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json','probe_gpu.json','probe_result.json',
                        'acceptance.json','evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json']
                    reference=args.semantic_root/'formal'/shift/'complete.json'
                    p.save(e/'complete.json',dict(schema=CONTROL_SCHEMA,shift=shift,regime='uda',variant=mode,code=identity,target_gt_used=False,
                        semantic_complete=str(reference),semantic_complete_sha256=p.sha(reference),
                        source_gate=str(source_gate),source_gate_sha256=p.sha(source_gate),group_acceptance=str(group),group_acceptance_sha256=p.sha(group),
                        acceptance_sha256=p.sha(e/'acceptance.json'),artifacts={str(e/n):p.sha(e/n) for n in paths}))
                    os.chdir(args.data_root);result=audit(e);os.chdir(CODE)
                    completed.append(result);group_completed.append(result);t['context'].__exit__(None,None,None);del ACTIVE[shift]
                    print('CONTROL_COMPLETE',mode,shift,result['map'],flush=True)
                for shift in list(pending):
                    if not runner.launch(shift,'train',root/'formal'/mode/shift,configs[mode][shift],args,identity):break
                    pending.remove(shift);print('FORMAL_STARTED',mode,shift,ACTIVE[shift]['process'].pid,ACTIVE[shift]['lease']['index'],flush=True)
                state(root,'formal',mode,completed,pending);time.sleep(5)
            if {r['shift'] for r in group_completed}!=set(configs[mode]):raise ValueError('组不是精确六方向')
            p.save(root/(mode+'_final_audit.json'),dict(status='passed',schema=CONTROL_SCHEMA,mode=mode,count=6,results=group_completed))
    except BaseException as error:
        p.save(root/'error.json',dict(error=repr(error),completed=completed,pending=pending,mode=mode))
        state(root,'failed',mode,completed,pending);runner.terminate_owned();raise
    if {(r['variant'],r['shift']) for r in completed}!={(m,s) for m in CONTROL_MODES for s in configs[m]}:raise ValueError('不是精确12项')
    comparison=[]
    for row in completed:
        shift=row['shift'];e=args.semantic_root/'formal'/shift;ref=p.read(e/'complete.json')
        if p.sha(e/'evaluation/target_val.json')!=ref['artifacts'][str(e/'evaluation/target_val.json')]:raise ValueError('P结果漂移')
        old=p.read(e/'evaluation/target_val.json')['map_0.1_to_0.5']
        comparison.append(dict(variant=row['variant'],shift=shift,map=row['map'],semantic_map=old,delta_to_semantic_pp=100*(row['map']-old)))
    p.save(root/'comparison.json',dict(results=comparison,exploratory=True))
    p.save(root/'final_audit.json',dict(status='passed',schema=CONTROL_SCHEMA,count=12,results=completed))
    state(root,'complete',None,completed,[])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--semantic-root',type=Path,default=Path('/vepfs/vlms/phase_alignment6_fullwarmup_f71c0e1_1031'))
    parser.add_argument('--control',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--minimum-free-mib',type=int,default=8192);parser.add_argument('--gpus',default='0,1,2,3')
    args=parser.parse_args();args.gpus=[int(v) for v in args.gpus.split(',')]
    signal.signal(signal.SIGTERM,runner.stop);signal.signal(signal.SIGINT,runner.stop);main(args)
