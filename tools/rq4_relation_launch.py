"""新关系方法的前序和验收身份；原24项、新12项不能互相充数。"""
from pathlib import Path
from tools.rq4_artifacts import checked, reference
from tools.rq4_relational import VARIANT
from tools.run_rq4_entry import read

TEACHERS=('timelens2_8b','timeple_8b')
SHIFTS=tuple('S'+str(i) for i in range(1,7))
KEYS={(t,s) for t in TEACHERS for s in SHIFTS}


def predecessor(final_path, expected_head, full_audit=False):
    from tools.run_rq4_entry import audit_complete
    path=Path(final_path).resolve(); final=read(path)
    keys={(t,s,i) for t in TEACHERS for s in ('direct','gated') for i in SHIFTS}
    entries=final.get('entries',[])
    actual=[(e['teacher'],e['strategy'],e['shift']) for e in entries]
    if final.get('status')!='passed' or final.get('count')!=24 or len(actual)!=24 or set(actual)!=keys:
        raise ValueError('exact original RQ4 24-entry audit required')
    refs=[]
    for e,key in zip(entries,actual):
        root=path.parent/'formal'/key[0]/key[2]/key[1]
        if Path(e['root']).resolve()!=root: raise ValueError('RQ4 predecessor root mismatch')
        record=read(root/'complete.json')
        if (record['schema']!='rq4-complete-v1' or record['code_commit']!=expected_head
                or tuple(record[k] for k in ('teacher','strategy','shift'))!=key
                or record['seed']!=6 or record['batch_size']!=2):
            raise ValueError('RQ4 predecessor identity mismatch')
        if full_audit: audit_complete(root)
        refs.append(reference(root/'complete.json'))
    return dict(final=reference(path),code_commit=expected_head,completed=refs)


def validate_acceptance(record):
    if record.get('variant')!=VARIANT: raise ValueError('soft relation acceptance variant mismatch')
    proof=record['predecessor']; checked(proof['final'])
    if predecessor(proof['final']['path'],proof['code_commit'])!=proof:
        raise ValueError('original RQ4 evidence changed')
    found=[]
    for ref in record['source_diagnostics']:
        report=checked(ref); found.append((report['teacher'],report['shift']))
        if report['target_gt_used'] is not False or report['parameters_fitted'] is not False:
            raise ValueError('relation diagnostic is not frozen source-only')
        for item in report['source_evidence'].values(): checked(item)
    if len(found)!=12 or set(found)!=KEYS: raise ValueError('exact twelve source diagnostics required')
    reports=[checked(ref) for ref in record['reports']]
    from tools.rq4_protocol import load_cache
    actual=[]
    for report in reports:
        spec=report['spec']; cache=load_cache(spec['cache_path'],spec['cache_file_sha256'])
        actual.append((cache['teacher'],cache['shift']))
        if (report['strategy']!='relational' or spec.get('variant')!=VARIANT
                or report['status']!='passed' or report['formal_result'] is not False
                or report['diagnostic_epoch']!=50 or report['seed']!=6 or report['batch_size']!=2):
            raise ValueError('invalid soft relation real-data acceptance')
    if len(actual)!=12 or set(actual)!=KEYS: raise ValueError('exact twelve real-data reports required')
