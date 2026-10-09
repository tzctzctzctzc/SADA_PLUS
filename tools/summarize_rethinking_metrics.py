"""Summarize offline metrics; preserve raw logs and explicitly label evidence gaps."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_rethinking_metrics import read, lines, sha, dump, table


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def rank(x):
    x=np.array(x); return np.array([np.sum(x<v)+(np.sum(x==v)-1)/2 for v in x])


def spearman(x,y):
    if len(x)<3 or len(set(x))<2 or len(set(y))<2: return None
    return float(np.corrcoef(rank(x),rank(y))[0,1])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--base',type=Path,default=Path('/vepfs/vlms')); a=p.parse_args(); out=a.output
    r1=csv_rows(out/'rq1_classification.csv'); tal=csv_rows(out/'tal_results.csv'); pairs=csv_rows(out/'paired_deltas.csv'); sup=csv_rows(out/'rq3_supervision.csv')
    rq1_summary=[]
    for model in sorted(set(r['model'] for r in r1)):
        rs=[r for r in r1 if r['model']==model]
        en=[r for r in rs if r['seed']=='ensemble']; tgt=[r for r in en if r['domain']=='target']; src=[r for r in en if r['domain']=='source']
        seed_means=[np.mean([float(r['macro_f1']) for r in rs if r['seed']==str(s) and r['domain']=='target']) for s in range(5)]
        rq1_summary.append({'model':model,'reporting':'five-seed probability ensemble; seeds0-4','source_macro_f1_pct':100*np.mean([float(r['macro_f1']) for r in src]),'target_macro_f1_pct':100*np.mean([float(r['macro_f1']) for r in tgt]),'target_minus_source_pp':100*(np.mean([float(r['macro_f1']) for r in tgt])-np.mean([float(r['macro_f1']) for r in src])),'target_ece':np.mean([float(r['ece_10_equal_width']) for r in tgt]),'target_brier':np.mean([float(r['brier_sum_10_classes']) for r in tgt]),'target_aurc':np.mean([float(r['aurc_discrete_mean']) for r in tgt]),'target_worst_shift_f1_pct':100*min(float(r['macro_f1']) for r in tgt),'individual_seed_mean_f1_pct':100*np.mean(seed_means),'individual_seed_sd_f1_pp':100*np.std(seed_means,ddof=1),'seed_note':'classifier seed variability, frozen backbone unchanged'})
    table(out/'rq1_summary.csv',rq1_summary)
    # Resolve the CLIP-B16 stale index without touching the original index or metrics.
    root=a.base/'SADA_PLUS/artifacts/rq1_semantic_20260915'
    corrected_path=root/'protocol_v2_smoke/rq1_groupcv_one_se_metrics.json'
    corrected=read(corrected_path)['models']['CLIP-B16']
    for r in r1:
        if r['model']=='CLIP ViT-B/16' and r['seed']=='ensemble':
            assert abs(float(r['macro_f1'])-corrected['shifts'][r['shift']][r['domain']]['macro_f1'])<1e-9
    dump(out/'rq1_index_resolution.json',{'issue':'eight-model table points CLIP-B16 at pre-one-SE metrics; prediction file was subsequently updated','old_reference':str(root/'protocol_v2_smoke/clip_groupcv_wide_metrics.json'),'verified_reference':str(corrected_path),'verified_reference_sha256':sha(corrected_path),'all_12_domain_shift_cells_match_current_predictions':True,'original_files_changed':False})
    # Recompute class-conditional output drift; do not substitute a feature-space
    # domain distance or interpret it as a causal measure of invariance.
    manifest=lines(root/'rq1_val_gt_segments.jsonl'); drift=[]
    for model in read(root/'rq1_8vlm_main_table.json')['models']:
        pf=root/model['protocol_file'].replace('.protocol.json','')
        pi={(r['candidate_id'],r['shift']):r for r in lines(pf)}
        for shift in [f'S{i}' for i in range(1,7)]:
            means={}
            for domain in ['source','target']:
                for c in range(10):
                    vals=[pi[(r['candidate_id'],shift)]['class_probabilities'] for r in manifest if r['gt_label_id']==c and {'domain':domain,'shift':shift} in r['memberships']]
                    means[(domain,c)]=np.mean(vals,axis=0)
            for c in range(10):
                s,t=means[('source',c)],means[('target',c)];m=(s+t)/2
                js=.5*(np.sum(s[s>0]*np.log(s[s>0]/m[s>0]))+np.sum(t[t>0]*np.log(t[t>0]/m[t>0])))
                drift.append({'model':model['model'],'shift':shift,'class_id':c,'conditional_prediction_jsd_nats':float(js),'seed':'ensemble0-4'})
    table(out/'rq1_conditional_prediction_drift.csv',drift)
    # Keep source-target gaps distinct from same-target SO/UDA gains.
    lookup={(r['rq'],r['protocol'],r['model'],r['regime'],r['shift'],r['seed'],r['domain']):r for r in tal}
    gaps=[]
    for r in tal:
        if r['domain']!='target': continue
        key=tuple(r[k] for k in ['rq','protocol','model','regime','shift','seed'])
        src=lookup[key+('source',)]
        gaps.append({**dict(zip(['rq','protocol','model','regime','shift','seed'],key)),'source_map_pct':float(src['map_pct']),'target_map_pct':float(r['map_pct']),'target_minus_source_pp':float(r['map_pct'])-float(src['map_pct'])})
    table(out/'tal_domain_gap.csv',gaps)
    # Per-class deltas use exactly the same paired protocol/seed/shift.
    classes=csv_rows(out/'tal_class_ap.csv'); idx={(r['protocol'],r['model'],r['regime'],r['shift'],r['seed'],r['class_id']):r for r in classes if r['domain']=='target'}
    class_pairs=[]
    for r in classes:
        if r['domain']!='target' or not(r['rq']=='RQ3' or (r['rq']=='RQ2' and r['regime']=='uda')):continue
        key=('rq3-matched','slowfast','uda',r['shift'],r['seed'],r['class_id']) if r['rq']=='RQ3' else (r['protocol'],r['model'],'so',r['shift'],r['seed'],r['class_id'])
        b=idx[key]; class_pairs.append({**r,'delta_ap_pp':float(r['ap_pct'])-float(b['ap_pct'])})
    table(out/'paired_class_ap.csv',class_pairs)
    # Per-threshold error changes are fixed-budget diagnostics, never tuned on target.
    err=csv_rows(out/'detection_errors.csv'); idx={(r['protocol'],r['model'],r['regime'],r['shift'],r['seed'],r['tiou'],r['prediction_budget']):r for r in err}; errpairs=[]
    for r in err:
        if not(r['rq']=='RQ3' or (r['rq']=='RQ2' and r['regime']=='uda')):continue
        key=('rq3-matched','slowfast','uda',r['shift'],r['seed'],r['tiou'],r['prediction_budget']) if r['rq']=='RQ3' else (r['protocol'],r['model'],'so',r['shift'],r['seed'],r['tiou'],r['prediction_budget'])
        if key not in idx:continue
        b=idx[key]; d={k:r[k] for k in ['rq','model','regime','seed','shift','tiou','prediction_budget']}
        for k in ['recall','miss_rate','duplicate_fraction','wrong_label_fraction','localization_fraction','confusion_fraction','background_fraction','short_le2s_recall','medium_2to5s_recall','long_gt5s_recall']:
            d[k+'_change_pp']=100*(float(r[k])-float(b[k])) if r[k] and b[k] else None
        errpairs.append(d)
    table(out/'paired_detection_errors.csv',errpairs)
    # Small-n correlations are descriptive and intentionally have no significance claims.
    correlations=[]; joined=[]
    by_pair={(r['model'],r['regime'],r['shift']):r for r in pairs if r['rq']=='RQ3'}
    for r in sup:
        joined.append({**r,'delta_pp':float(by_pair[(r['model'],r['regime'],r['shift'])]['delta_pp'])})
    table(out/'rq3_supervision_and_gain.csv',joined)
    for model in sorted(set(r['model'] for r in sup)):
        for regime,fields in [('verify',['verify_reject_of_observed','verify_reject_of_eligible','verify_uncertain_rate']),('prototype',['prototype_selected_of_visible','alignment_loss_epoch_mean','target_gradient_norm_epoch_mean'])]:
            rs=[r for r in joined if r['model']==model and r['regime']==regime]
            for f in fields:
                valid=[r for r in rs if r[f]!='']
                correlations.append({'scope':'within teacher, across six shifts','model':model,'regime':regime,'x':f,'y':'delta_pp','n':len(valid),'spearman':spearman([float(r[f]) for r in valid],[r['delta_pp'] for r in valid]),'interpretation':'descriptive only; overlapping shifts, n<=6, no causal/significance claim'})
    table(out/'exploratory_correlations.csv',correlations)
    # Exactly three shared VLMs; expose paired values instead of inflating n to 18.
    names={'egovideo_hod':'EgoVideo-HOD','internvideo2_clip_1b':'InternVideo2-CLIP-1B','qwen3_vl_8b':'Qwen3-VL-8B'}
    cross=[]
    for model,name in names.items():
        one=next(r for r in rq1_summary if r['model']==name)
        r={'model':model,'rq1_ensemble_target_macro_f1_pct':one['target_macro_f1_pct'],'rq1_seed_note':'0-4 classifier ensemble'}
        for regime in ['so','uda']:
            rs=[x for x in tal if x['rq']=='RQ2' and x['model']==model and x['regime']==regime and x['seed']=='6' and x['domain']=='target'];assert len(rs)==6
            r['rq2_'+regime+'_seed6_target_map_pct']=np.mean([float(x['map_pct']) for x in rs])
        for regime in ['verify','prototype']:
            rs=[x for x in pairs if x['rq']=='RQ3' and x['model']==model and x['regime']==regime];assert len(rs)==6
            r['rq3_'+regime+'_seed6_gain_pp']=np.mean([float(x['delta_pp']) for x in rs])
        cross.append(r)
    table(out/'cross_rq_common_models.csv',cross)
    # Archive the available raw numerical logs without interpreting their periodic
    # evaluator mAP as the final full-video offline evaluation.
    log_index=[]; prediction_validity=[]
    for run in read(out/'inventory.json'):
        ev=read(Path(run['path'])/'target_val.json'); logdir=Path(ev['checkpoint']).parents[2]
        ps=ev['predictions']; prediction_validity.append({**{k:run[k] for k in ['rq','model','regime','shift','seed']},'predictions':len(ps),'zero_duration_predictions':sum(r['t-end']==r['t-start'] for r in ps),'negative_duration_predictions':sum(r['t-end']<r['t-start'] for r in ps),'policy':'zero length retained as false positive, not dropped'})
        key='_'.join(str(run[k]) for k in ['rq','protocol','model','regime','shift','seed'])
        logs={}; row={'run':key,'run_directory':str(logdir)}
        for name in ['benchmark_epoch_metrics.jsonl','benchmark_eval_metrics.jsonl','vlm_epoch_metrics.jsonl']:
            f=logdir/name; row[name+'_present']=f.exists()
            if f.exists():
                logs[name]=lines(f);row[name+'_rows']=len(logs[name]);row[name+'_sha256']=sha(f)
        dump(out/'training_logs'/f'{key}.json',logs);log_index.append(row)
    table(out/'training_log_inventory.csv',log_index)
    table(out/'prediction_validity.csv',prediction_validity)
    availability=read(out/'availability.json')
    availability['reused_with_crosscheck']=[]
    availability['computed'].append('RQ1 video-bootstrap intervals recomputed (1000 repeats, seed20261009), five classifier-seed variability; per-class/error paired deltas; descriptive correlations; 168-run raw log archive')
    availability['cautions']=['RQ1 CLIP-B16 main-table reference was stale; resolved via one-SE metrics and exact predictions','RQ1 five-seed ensemble is not single seed; RQ2 SlowFast/Ego/Intern three seeds, Qwen one; RQ3 seed6','periodic training evaluation logs may differ from final full-video evaluator; retain raw units, do not mix their mAP','fixed-duration recall is not duration-specific AP; error removal is an oracle diagnostic, not observed gain','classifier ECE/AURC is not teacher reliability; historical RQ3 error-correction quality is unavailable']
    availability['cautions'].append('AURC uses discrete cumulative risk and stable input-order confidence ties; detection diagnostics use stable score ties, so their recomputed AP can differ minutely from legacy quicksort ties; original reported mAP is preserved')
    dump(out/'availability.json',availability)
    dump(out/'metric_catalog.json',{
        'units':'CSV字段以_pct结尾表示百分数，以_pp结尾表示百分点；其余率/概率为0–1。',
        'RQ1':{'主指标':['target Macro-F1：类别均等的分类能力','target−source Macro-F1：跨域下降'], '诊断':['逐类别F1/召回/精确率与混淆矩阵','ECE/Brier：分类器概率是否可信','risk-coverage/AURC：按置信度保留预测时的错误率','按GT类别条件化的预测分布JSD','视频bootstrap区间、五个分类器seed的变动'],'文件':['rq1_summary.csv','rq1_classification.csv','rq1_class_metrics.csv','rq1_confusions_and_bootstrap.json','rq1_reliability.csv','rq1_risk_coverage.csv','rq1_conditional_prediction_drift.csv']},
        'RQ2':{'主指标':['mAP@tIoU0.1:0.5','各tIoU mAP与类别AP','同模型同shift同seed的UDA−SO'],'诊断':['DETAD式重复/错类/定位/混淆/背景误报分解','固定预测预算的漏检率与时长分组召回','逐seed、逐shift稳定性'],'文件':['tal_results.csv','paired_deltas.csv','paired_class_ap.csv','detection_errors.csv','paired_detection_errors.csv']},
        'RQ3':{'主指标':['相对匹配SADA的mAP变化','各策略六shift收益及正向shift数'],'诊断':['Verify可信/不确定/拒绝比例','Prototype可见候选选择比例','辅助loss、alignment loss、目标梯度的epoch轨迹','检测错误变化和类别收益','干预与收益的描述性关联'],'文件':['main_summary.csv','rq3_supervision.csv','rq3_supervision_and_gain.csv','rq3_epoch_trajectories.csv','exploratory_correlations.csv']},
        '不要替代':['非零梯度不等于性能改善','20%候选参与不等于20%的SADA伪标签被拒绝','RQ1分类器ECE不等于RQ3教师的可靠性','视频bootstrap不等于多seed置信区间','固定时长分组召回不等于时长分组mAP','错误删除后的oracle AP不是实际算法收益'],
        '资料':['https://github.com/HumamAlwassel/DETAD','https://proceedings.mlr.press/v70/guo17a.html','https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html']})
    crosschecks=csv_rows(out/'detection_map_crosscheck.csv')
    assert len(crosschecks)==168, 'Full detection diagnostics must finish before final summary.'
    assert max(abs(float(r['map_difference_pp'])) for r in crosschecks)<.01, 'Investigate material AP replay mismatch.'
    dump(out/'analysis_validation.json',{'tal_runs':len(read(out/'inventory.json')),'eval_files_sha_checked':len(tal),'detection_runs':len(crosschecks),'maximum_abs_map_difference_pp':max(abs(float(r['map_difference_pp'])) for r in crosschecks),'rq1_prediction_seed_domain_shift_rows':len(r1),'rq3_epoch_rows':len(csv_rows(out/'rq3_epoch_trajectories.csv')),'supplement_script_sha256':sha(__file__),'training_log_runs':len(log_index)})
    print('SUPPLEMENT_DONE',len(log_index),len(errpairs),flush=True)


if __name__=='__main__':main()
