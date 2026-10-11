"""完整共识辅助的独立身份、源质量证据与真实验收门禁。"""
import json
import math
from pathlib import Path
from tools.rq4_protocol import digest, file_sha, load_cache
from tools.rq4_artifacts import checked, reference, audit_evidence
from tools.rq4_relational import build_regions, relation_weight

VARIANT='rq4-quality-aware-local-consensus-v1'
RULE=dict(points='nearest-anchor-three-real-original-fpn',edges='shared-context-no-mask-gap',
    reference='detached-ema-component-weighted-median',quality='source-only-frozen-localization-mlp',
    tolerance='source-video-residual-plus-ema-disagreement',target_gt_used=False,
    normalization='crop-visible-original-candidates-no-acceptance-renormalization',
    teacher_endpoints_regressed=False)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,allow_nan=False)


def paired_iou(pair): return relation_weight(pair)


def load_quality(bundle):
    import torch
    return {k:torch.tensor(v,dtype=torch.float32) for k,v in bundle['state_dict'].items()}


def validate_quality(bundle, *, shift, target_ids):
    if (bundle.get('schema')!='rq4-localization-quality-v1' or bundle.get('shift')!=shift
            or bundle.get('seed')!=6 or bundle.get('target_gt_used') is not False
            or bundle.get('fit_split')!='source_train' or bundle.get('calibration_split')!='source_validation'
            or bundle.get('code_sha256')!=file_sha(Path(__file__).parents[1]/'libs/models/SADA/rq4_quality.py')
            or bundle.get('payload_sha256')!=digest({k:v for k,v in bundle.items() if k!='payload_sha256'})):
        raise ValueError('source quality identity/code/fit evidence invalid')
    train=set(bundle['train_video_ids']); val=set(bundle['validation_video_ids'])
    if not train or len(val)<2 or train&val or (train|val)&set(target_ids):
        raise ValueError('source train/calibration/target video leakage')
    margin=bundle['calibration']
    if margin.get('schema') is None:
        records=[margin]
    else:
        if (margin.get('schema')!='source-quality-binned-video-balanced-v1'
                or margin.get('bin_edges')!=[0.,.2,.4,.6,.8,1.]
                or margin.get('unsupported_policy')!='abstain' or len(margin['groups'])!=5):
            raise ValueError('invalid grouped source calibration')
        records=[row for row in margin['groups'] if row['supported']]
        for i,row in enumerate(margin['groups']):
            if (row['lower']!=margin['bin_edges'][i] or row['upper']!=margin['bin_edges'][i+1]
                    or row['supported']!=(row['videos']>=2) or row['rows']<row['videos']):
                raise ValueError('invalid grouped source support')
    for record in records:
        values=[record['quality_margin'],*record['error_margin']]
        if len(values)!=3 or any(not math.isfinite(v) or v<0 for v in values):
            raise ValueError('invalid source uncertainty calibration')
    if margin['target_coverage_guaranteed'] is not False:
        raise ValueError('source calibration is not a target reliability guarantee')
    table=checked(bundle['source_table'])
    if (table['shift']!=shift or table['train_video_ids']!=bundle['train_video_ids']
            or table['validation_video_ids']!=bundle['validation_video_ids']):
        raise ValueError('quality fit membership is not bound to source table')
    audit_source_table(table)
    return bundle


def audit_source_table(table):
    if (table.get('schema')!='rq4-quality-source-table-v1' or table.get('target_gt_used') is not False
            or table.get('seed')!=6): raise ValueError('not a source-only quality table')
    cfg=checked(table['baseline_config']); database=checked(table['annotations'])['database']
    if file_sha(table['checkpoint']['path'])!=table['checkpoint']['sha256']:
        raise ValueError('source detector checkpoint changed')
    groups={}
    for name,key in (('train','src_train_split'),('validation','src_val_split'),('target','tgt_train_split')):
        splits={x.lower() for x in cfg[key]}
        groups[name]={k for k,v in database.items() if v['subset'].lower() in splits}
    if groups['train']&groups['validation'] or (groups['train']|groups['validation'])&groups['target']:
        raise ValueError('source quality video splits overlap')
    for name in ('train','validation'):
        if set(table[name+'_video_ids'])!=groups[name]: raise ValueError('source membership differs from baseline')
        rows=table[name]
        if not rows or any(r['video_id'] not in groups[name] for r in rows):
            raise ValueError('quality row outside source split')
        if any(len(r['x'])!=134 or len(r['y'])!=3 or
               any(not math.isfinite(v) for v in r['x']+r['y']) for r in rows):
            raise ValueError('nonfinite/malformed source quality rows')
    for ref in table['features']:
        if file_sha(ref['path'])!=ref['sha256']: raise ValueError('frozen source feature changed')
    if table['shift']!=cfg['rq2_formal']['shift'] or cfg['seed']!=6:
        raise ValueError('quality table baseline shift/seed mismatch')


def source_policy_summary(bundle,*,minimum_quality,maximum_uncertainty,anchor_fraction):
    """源验证上的阈值诊断，不搜索target收益，也不宣称这是最优分支权重。"""
    import torch
    from libs.models.SADA.rq4_quality import LocalizationQuality,calibrated_bounds,calibrate
    table=checked(bundle['source_table']); audit_source_table(table)
    with torch.random.fork_rng(devices=[]): model=LocalizationQuality()
    model.load_state_dict(load_quality(bundle),strict=True); model.eval().requires_grad_(False)
    x=torch.tensor([r['x'] for r in table['validation']],dtype=torch.float32)
    y=torch.tensor([r['y'] for r in table['validation']],dtype=torch.float32)
    with torch.no_grad(): prediction=model(x)
    calibration=bundle['calibration']
    if calibration.get('schema') and calibrate(prediction,y,
            [r['video_id'] for r in table['validation']],quantile=calibration['quantile'])!=calibration:
        raise ValueError('source residual calibration cannot be reproduced')
    lower,radius,supported=calibrated_bounds(prediction,calibration)
    eligible=supported&(lower>=minimum_quality)&(radius.amax(1)<=maximum_uncertainty)
    return dict(schema='rq4-consensus-source-policy-diagnostic-v1',target_gt_used=False,
        quality_payload_sha256=bundle['payload_sha256'],source_table=bundle['source_table'],
        parameters=dict(minimum_quality=minimum_quality,maximum_uncertainty=maximum_uncertainty,
                        anchor_fraction=anchor_fraction),
        parameter_selection='explicit-settings-not-target-optimized',
        source_rows=len(x),accepted=int(eligible.sum()),coverage=float(eligible.float().mean()),
        mean_iou=float(y[:,0].mean()),accepted_mean_iou=float(y[eligible,0].mean()) if eligible.any() else None,
        calibration_supported=int(supported.sum()),
        endpoint_band_coverage=((y[supported,1:]<=radius[supported]).float().mean(0)).tolist()
            if supported.any() else None,
        scope='individual frozen-detector source boxes; not target crop intervention or student gain')


def validate_spec(spec,cache):
    if (spec.get('variant')!=VARIANT or spec.get('strategy')!='consensus' or spec.get('consensus_rule')!=RULE
            or spec.get('weight')!=.1 or spec.get('early_stop_patience')!=15
            or cache['seed']!=6 or cache['teacher'] not in ('timelens2_8b','timeple_8b')
            or 'budget' not in cache or spec.get('gate_path')):
        raise ValueError('consensus identity or fixed protocol differs')
    for key in ('minimum_quality','maximum_uncertainty','anchor_fraction'):
        value=spec.get(key)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<1:
            raise ValueError('explicit source-frozen consensus parameter required: '+key)
    objects={}
    for key in ('regions','quality'):
        if file_sha(spec[key+'_path'])!=spec[key+'_file_sha256']:
            raise ValueError('consensus artifact SHA changed: '+key)
        objects[key]=read(spec[key+'_path'])
    regions=objects['regions']
    if regions['cache_sha256']!=cache['payload_sha256']:
        raise ValueError('paired region cache binding differs')
    if regions['payload_sha256']!=digest({k:v for k,v in regions.items() if k!='payload_sha256'}):
        raise ValueError('paired region payload SHA changed')
    if len(regions['rows'])!=len(cache['records']) or any(
            region['candidate_sha256']!=digest(row['candidate'])
            for row,region in zip(cache['records'],regions['rows'])):
        raise ValueError('candidate identities differ')
    validate_quality(objects['quality'],shift=cache['shift'],
                     target_ids=[r['candidate']['video_id'] for r in cache['records']])
    return regions,objects['quality']


def audit_spec(spec,cache):
    regions,quality=validate_spec(spec,cache)
    if file_sha(spec['evidence_path'])!=spec['evidence_file_sha256']:
        raise ValueError('paired request/source evidence changed')
    proof=read(spec['evidence_path']); audit_evidence(proof,cache)
    if build_regions(cache,proof,'relational')!=regions:
        raise ValueError('local graph support is not reproducible from native teacher replies')
    if not spec.get('source_policy_path') or file_sha(spec['source_policy_path'])!=spec['source_policy_file_sha256']:
        raise ValueError('source policy diagnostic required')
    if read(spec['source_policy_path'])!=source_policy_summary(quality,**{
            k:spec[k] for k in ('minimum_quality','maximum_uncertainty','anchor_fraction')}):
        raise ValueError('source-only policy evidence differs')
    return regions,quality


def prepare(original_spec,quality,output,*,minimum_quality,maximum_uncertainty,anchor_fraction):
    old=read(original_spec); cache=load_cache(old['cache_path'],old['cache_file_sha256'])
    proof=read(old['evidence_path']); audit_evidence(proof,cache)
    regions=build_regions(cache,proof,'relational'); bundle=read(quality)
    validate_quality(bundle,shift=cache['shift'],target_ids=[r['candidate']['video_id'] for r in cache['records']])
    root=Path(output).resolve(); root.mkdir(parents=True,exist_ok=False)
    spec=dict(schema='rq4-boundary-training-v1',variant=VARIANT,strategy='consensus',consensus_rule=RULE,
        weight=.1,normalizer='common-routable-candidates',activation='sada-adaptation-phase',
        early_stop_patience=15,protocol_frozen=False,minimum_quality=minimum_quality,
        maximum_uncertainty=maximum_uncertainty,anchor_fraction=anchor_fraction)
    policy=source_policy_summary(bundle,minimum_quality=minimum_quality,
        maximum_uncertainty=maximum_uncertainty,anchor_fraction=anchor_fraction)
    for key,obj in (('cache',cache),('regions',regions),('quality',bundle),('evidence',proof),('source_policy',policy)):
        path=root/(key+'.json'); save(path,obj)
        spec[key+'_path']=str(path); spec[key+'_file_sha256']=file_sha(path)
    validate_spec(spec,cache); save(root/'spec.json',spec)
    return spec


def launch_ready(spec,cache):
    audit_spec(spec,cache)
    if not spec.get('acceptance_path'):
        raise ValueError('consensus needs source-frozen parameters and real-data EMA acceptance')
    if file_sha(spec['acceptance_path'])!=spec['acceptance_file_sha256']:
        raise ValueError('consensus acceptance SHA mismatch')
    record=read(spec['acceptance_path'])
    from tools.run_rq2_matched_control import check_code
    check_code(record['code_commit'])
    if record.get('variant')!=VARIANT or record.get('status')!='passed':
        raise ValueError('not consensus real-data acceptance')
    desired=(cache['teacher'],cache['shift']); found=[]
    for ref in record['reports']:
        report=checked(ref)
        if (report.get('variant')!=VARIANT or report.get('formal_result') is not False
                or report.get('code_commit')!=record['code_commit']):
            raise ValueError('invalid consensus diagnostic identity')
        if (report.get('teacher'),report.get('shift'))!=desired: continue
        if (report.get('real_video_features') is not True or report.get('status')!='passed'
                or report.get('seed')!=6 or report.get('batch_size')!=2):
            raise ValueError('real video acceptance is missing')
        expected={k:v for k,v in spec.items() if k not in (
            'acceptance_path','acceptance_file_sha256','protocol_frozen','config_path','config_sha256')}
        if report['specification_sha256']!=digest(expected):
            raise ValueError('diagnostic and formal consensus parameters/artifacts differ')
        m=report['metrics']
        if (any(not math.isfinite(v) for v in m.values()) or m['rq4_gradient_norm']<=0
                or m['rq4_anchor_points']<=0 or m['rq4_relation_pairs']<=0
                or any(m['target_gt_'+k+'_loss']!=0 for k in ('cls','reg','downstream'))):
            raise ValueError('consensus acceptance lacks both mechanisms or leaks target GT')
        found.append(ref)
    if len(found)!=1: raise ValueError('exact matching consensus acceptance required')


def freeze_acceptance(output,reports,expected_head):
    from tools.run_rq2_matched_control import check_code
    check_code(expected_head)
    if not reports: raise ValueError('real-data reports are required')
    refs=[reference(path) for path in reports]; seen=set()
    for ref in refs:
        report=checked(ref); key=(report.get('teacher'),report.get('shift'))
        if (key in seen or report.get('variant')!=VARIANT or report.get('status')!='passed'
                or report.get('diagnostic_epoch')!=50 or report.get('formal_result') is not False
                or report.get('code_commit')!=expected_head):
            raise ValueError('invalid or duplicate consensus acceptance')
        seen.add(key)
        spec=report['spec']; cache=load_cache(spec['cache_path'],spec['cache_file_sha256'])
        if key!=(cache['teacher'],cache['shift']): raise ValueError('acceptance teacher/shift differs')
        audit_spec(spec,cache)
    record=dict(variant=VARIANT,status='passed',code_commit=expected_head,reports=refs)
    # 发布前复用与正式入口相同的所有验收检查；临时文件不冒充最终验收。
    import tempfile
    with tempfile.TemporaryDirectory(prefix='rq4-consensus-accept-') as temp:
        path=Path(temp)/'acceptance.json'; save(path,record)
        for ref in refs:
            spec=dict(checked(ref)['spec'],acceptance_path=str(path),acceptance_file_sha256=file_sha(path))
            launch_ready(spec,load_cache(spec['cache_path'],spec['cache_file_sha256']))
    save(output,record)
    return record


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original-spec','quality','output'): p.add_argument('--'+name,required=True)
    for name in ('minimum-quality','maximum-uncertainty','anchor-fraction'):
        p.add_argument('--'+name,type=float,required=True)
    spec=prepare(**vars(p.parse_args()))
    print('CONSENSUS_PREPARED_NOT_FORMAL',spec['variant'])


if __name__=='__main__': main()
