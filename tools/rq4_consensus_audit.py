"""新共识方法独立终审；不能冒充原Direct/Gated/关系方法完成。"""
import math
from pathlib import Path
from tools.rq4_consensus import VARIANT,read,launch_ready,load_quality
from tools.rq4_protocol import load_cache,file_sha,audit_baseline_pair,audit_termination


def supervision(path,epochs):
    rows=[__import__('json').loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if [r['epoch'] for r in rows]!=list(range(1,epochs+1)):
        raise ValueError('consensus epoch history incomplete')
    keys=('queried','matched','routable','accepted','anchor_points','single_point_anchors',
          'relation_points','relation_pairs','weight_sum','uncertainty_sum','anchor_raw','relation_raw',
          'quality_rejected','loss','raw_loss','gradient_norm','main_head_gradient_norm',
          'auxiliary_head_gradient_norm','gradient_ratio','gradient_cosine','gradient_comparison_valid')
    totals={k:0. for k in keys}
    for row in rows:
        values={k:row['t_rq4_'+k] for k in keys}
        if any(not math.isfinite(v) or (k!='gradient_cosine' and v<0) for k,v in values.items()):
            raise ValueError('nonfinite/invalid consensus diagnostic')
        if not -1.00001<=values['gradient_cosine']<=1.00001:
            raise ValueError('invalid shared-parameter gradient cosine')
        if any(row['t_target_gt_'+k+'_loss']!=0 for k in ('cls','reg','downstream')):
            raise ValueError('consensus target GT leakage')
        if not values['accepted']<=values['routable']<=values['matched']<=values['queried']:
            raise ValueError('consensus routing counts inconsistent')
        if (values['anchor_points']>3*values['accepted']+1e-5
                or values['relation_pairs']>2*values['accepted']+1e-5
                or values['single_point_anchors']>values['accepted']+1e-5
                or values['weight_sum']>values['accepted']+1e-5):
            raise ValueError('consensus point/pair/weight counts invalid')
        if not math.isclose(values['loss'],.1*values['raw_loss'],abs_tol=1e-8,rel_tol=1e-5):
            raise ValueError('consensus auxiliary weight changed')
        if values['loss']>0 and values['gradient_norm']==0:
            raise ValueError('consensus loss has no student gradient')
        for k in keys: totals[k]+=values[k]
    return dict(epoch_mean_sums=totals,intervened=totals['accepted']>0,
        effective_gradient=totals['gradient_norm']>0,zero_coverage_is_reported_not_filtered=True)


def training_audit(run_root,cfg,baseline):
    import torch
    from tools.rq2_evaluate import load_checkpoint,selected_checkpoint_state
    from tools.rq2_formal import validate_formal_resume
    configs=list(Path(run_root).glob('*/config.json'))
    best=list(Path(run_root).glob('*/saved_ckpt/best_ckpt/*.pt'))
    latest=list(Path(run_root).glob('*/saved_ckpt/latest_ckpt/*.pt'))
    if len(configs)!=1 or len(best)!=1 or len(latest)!=1:
        raise ValueError('consensus config/checkpoints missing or ambiguous')
    saved=read(configs[0]); spec=cfg['rq4_boundary']
    audit_baseline_pair(baseline,saved)
    if saved['rq4_boundary']!=spec: raise ValueError('consensus saved/preflight spec differs')
    stop=load_checkpoint(latest[0]); chosen=load_checkpoint(best[0])
    for checkpoint in (stop,chosen): validate_formal_resume(checkpoint,cfg['rq2_formal']['run_signature'])
    audit_termination(stop,spec); selected_checkpoint_state(chosen,saved,True)
    termination=stop['rq4_termination']
    if (chosen['epoch']!=termination['decision']['best_epoch'] or
            [r[0] for r in termination['source_validation_history']]!=list(range(5,stop['epoch']+1,5))):
        raise ValueError('consensus source selection/validation cadence changed')
    quality=load_quality(read(spec['quality_path']))
    for checkpoint in (stop,chosen):
        for name,value in quality.items():
            if not torch.equal(checkpoint['state_dict']['rq4_boundary.quality.'+name].cpu(),value):
                raise ValueError('source quality parameters changed during student training')
    report=supervision(configs[0].parent/'benchmark_epoch_metrics.jsonl',stop['epoch'])
    return configs[0],best[0],latest[0],termination,report


def audit_complete(root):
    from tools.run_rq2_matched_control import checked_evaluation
    root=Path(root); record=read(root/'complete.json'); cfg=read(root/'preflight.json')
    spec=cfg['rq4_boundary']; cache=load_cache(spec['cache_path'],spec['cache_file_sha256'])
    if (record.get('schema')!='rq4-consensus-complete-v1' or record.get('variant')!=VARIANT
            or record['strategy']!='consensus' or record['teacher']!=cache['teacher']
            or record['shift']!=cache['shift'] or record['seed']!=6 or record['batch_size']!=2):
        raise ValueError('consensus completion identity mismatch')
    launch_ready(spec,cache)
    if read(spec['acceptance_path'])['code_commit']!=record['code_commit']:
        raise ValueError('consensus completion code differs from accepted code')
    baseline=read(record['baseline_config'])
    config,best,latest,termination,report=training_audit(record['run_root'],cfg,baseline)
    required=[config,best,latest,config.parent/'benchmark_epoch_metrics.jsonl',root/'preflight.json',
        root/'spec.json',root/'baseline_pair.json',root/'supervision_audit.json',Path(record['baseline_config']),
        root/'evaluation/manifest.json',root/'evaluation/source_val.json',root/'evaluation/target_val.json']
    required += [Path(spec[k+'_path']) for k in ('cache','evidence','regions','quality','source_policy','acceptance')]
    if not {str(x) for x in required}<=set(record['artifacts']):
        raise ValueError('consensus completion lacks required artifact hashes')
    for path,sha in record['artifacts'].items():
        if file_sha(path)!=sha: raise ValueError('consensus completed artifact changed: '+path)
    if (report!=read(root/'supervision_audit.json') or termination!=record['termination'] or
            audit_baseline_pair(baseline,cfg)!=read(root/'baseline_pair.json')):
        raise ValueError('consensus completion audit differs')
    if checked_evaluation(root/'evaluation',cfg,best).get('rq4_boundary')!=spec:
        raise ValueError('consensus offline evaluation provenance differs')
    return record
