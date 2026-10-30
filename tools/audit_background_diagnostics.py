"""从冻结预测和点记录独立重算六方向诊断汇总；不运行模型，不修改旧产物。"""
import argparse
from pathlib import Path
import sys
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as p
from tools.background_diagnostics import SPEC,sum_counts,error_counts
from tools.background_routing_replay import replay


def audit(root):
    rows=[]
    for shift in SPEC['shifts']:
        entry=Path(root)/shift;result=p.read(entry/'result.json')
        if result['status']!='passed_read_only_audit' or result['training_performed']:
            raise ValueError('不是只读诊断完成项')
        for name,digest in result['artifacts'].items():
            if p.sha(name)!=digest:raise ValueError('诊断工件改变: '+name)
        for point in p.read(entry/'point_index.json'):
            if p.sha(point['path'])!=point['sha256']:raise ValueError('真实点记录改变')
        for key,value in result['maps']['original'].items():
            if value!=result['maps']['full_backend_reference'][key]:raise ValueError('原完整模型与诊断预测AP不一致')
        routing=p.read(entry/'routing.json')
        actual={domain:{role:sum_counts([x['counts'] for x in routing if x['domain']==domain and x['role']==role and x['active_alignment_level']])
            for role in ['student','ema']} for domain in ['source','target']}
        if actual!=result['routing']:raise ValueError('背景分组统计不能复算')
        if replay(entry,p.read(entry/'plan.json'))!=result['routing_replay']:raise ValueError('四路重放不能复算')
        predictions=p.read(entry/'predictions.json');gt=p.read(entry/'ground_truth_for_diagnosis.json')
        for threshold in SPEC['tiou']:
            if error_counts(predictions['original'],gt,threshold)!=result['errors_by_tiou'][str(threshold)]:raise ValueError('错误分解不能复算')
        base=result['maps']['original']['map_0.1_to_0.5'];route=result['routing']['target']['student']
        rows.append(dict(shift=shift,map=base,map_percent=base*100,
            assigned_anchor_oracle_gain_pp=(result['maps']['assigned_anchor_oracle']['map_0.1_to_0.5']-base)*100,
            occupied_anchor_oracle_gain_pp=(result['maps']['occupied_anchor_oracle']['map_0.1_to_0.5']-base)*100,
            post_nms_background_oracle_gain_pp=(result['maps']['post_nms_background_oracle']['map_0.1_to_0.5']-base)*100,
            target_student_routing=route,routing_replay=result['routing_replay']['counts'],errors=result['errors_by_tiou'],
            historical_gpu_difference_pp=result['historical_gpu_map_difference']*100,
            result_sha256=p.sha(entry/'result.json')))
    import numpy as np
    summary=dict(status='passed_independent_six_shift_audit',count=6,rows=rows,
        mean={key:float(np.mean([x[key] for x in rows])) for key in ['map_percent','assigned_anchor_oracle_gain_pp','occupied_anchor_oracle_gain_pp','post_nms_background_oracle_gain_pp']},
        diagnostic4_scope='offline routing replay only; no retrained performance claim',
        target_gt_scope='exploratory after frozen prediction, never trained or used to tune routing',training_performed=False)
    p.save(Path(root)/'independent_audit.json',summary);print(summary['mean'],flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args();audit(args.root)
