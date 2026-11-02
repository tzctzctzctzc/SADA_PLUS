"""只读源域反事实：风险排序/覆盖/有效强度与完整分类Oracle的差别。"""
import argparse
from pathlib import Path
import json
import numpy as np


def auc(score, positive):
    n = int(positive.sum()); m = len(positive)-n
    if not n or not m:
        return None
    order = np.argsort(score, kind='stable'); sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0)+1]
    ends = np.r_[starts[1:], len(score)]
    ranks = np.empty(len(score), float)
    for a,b in zip(starts,ends): ranks[order[a:b]] = (a+1+b)/2
    return float((ranks[positive].sum()-n*(n+1)/2)/(n*m))


def focal(p, truth):
    p = np.clip(p.astype(np.float64), 1e-12, 1-1e-12)
    positive = -.25*(1-p)**2*np.log(p)
    negative = -.75*p**2*np.log1p(-p)
    gradient_pos = .25*(1-p)**2*(2*p*np.log(p)-(1-p))
    gradient_neg = .75*p**2*(p-2*(1-p)*np.log1p(-p))
    return np.where(truth,positive,negative), np.where(truth,gradient_pos,gradient_neg), gradient_neg


def measure(probability, raw, valid, truth, calibration):
    p=probability[valid]; y=truth[valid].astype(bool); r=raw[valid]
    score=p.max(-1); candidate=score<.1; unsafe=y.any(-1)
    q=1/(1+np.exp(-np.clip(r*calibration['slope']+calibration['bias'],-80,80)))
    w=q**2; loss,gradient,negative_gradient=focal(p,y)
    np_=max(int(unsafe.sum()),1); nc=max(int(candidate.sum()),1)
    # 同一源EMA静态反事实，normalizer为当前计数proxy，非真实target/student/EMA-normalizer重放。
    oracle=gradient/np_
    learned=.025*w[:,None]*candidate[:,None]*negative_gradient/nc
    return dict(valid=len(p),positives=int(unsafe.sum()),candidates=int(candidate.sum()),
        missed_positive_count=int((candidate&unsafe).sum()),
        missed_positive_fraction=float((candidate&unsafe).sum()/np_),
        unsafe_fraction=float(unsafe[candidate].mean()) if candidate.any() else None,
        predicted_safe_mean=float(q[candidate].mean()) if candidate.any() else None,
        mean_weight=float(w[candidate].mean()) if candidate.any() else None,
        weight_std=float(w[candidate].std()) if candidate.any() else None,
        unsafe_AUC_risk=auc(-r[candidate],unsafe[candidate]),
        unsafe_AUC_confidence=auc(score[candidate],unsafe[candidate]),
        positive_entry_fraction_of_full_focal_loss=float((loss*y).sum()/max(loss.sum(),1e-30)),
        negative_only_proxy_gradient_L2_vs_full_oracle=float(np.linalg.norm(learned)/max(np.linalg.norm(oracle),1e-30)),
        selected_unsafe_mean_weight=float(w[candidate&unsafe].mean()) if (candidate&unsafe).any() else None)


def macro(rows):
    return {k:float(np.mean([r[k] for r in rows if r.get(k) is not None]))
            for k in rows[0] if k!='video' and any(r.get(k) is not None for r in rows)}


def run(root, output):
    import torch
    from types import SimpleNamespace
    from tools.baseline_protocol import read,save,sha
    from tools.dense_assignment_signals import assign_cpu
    from libs.models.SADA.meta_archs import SADA
    output.mkdir(exist_ok=False,parents=True)
    jobs=[]
    for shift in ('S1','S2','S3','S4','S5','S6'):
        entry=root/'formal/negative'/shift
        cfg=read(entry/'config.json')
        db=read(cfg['baseline']['annotation_for_offline_evaluation'])['database']
        assert sha(cfg['baseline']['annotation_for_offline_evaluation'])==cfg['baseline']['annotation_sha256']
        for folder in sorted((entry/'signals').glob('e*')):
            if not (folder/'mechanism_quality.json').exists():continue
            manifest=read(folder/'diagnostics.json'); rows=[]
            for item in manifest['videos']:
                assert db[item['video']]['subset'] in cfg['src_val_split']
                assert sha(item['arrays'])==item['arrays_sha256']
                meta=item['metadata'];gt=[g for g in db[item['video']]['annotations'] if 0<=int(g['label_id'])<10]
                seg=np.asarray([g['segment'] for g in gt],np.float32).reshape(-1,2)
                seg=(seg*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']-meta['crop_start']
                keep=(seg[:,1]>0)&(seg[:,0]<meta['crop_feat_len']);seg=seg[keep].clip(0,meta['crop_feat_len'])
                labels=np.asarray([int(g['label_id']) for g in gt])[keep]
                with np.load(item['arrays'],allow_pickle=False) as a:
                    y=assign_cpu(a['points'],seg,labels,cfg['model']['center_sample_radius'],True)
                    # 原数学模型真实CPU方法，独立验证点/stride/range/最短实例及多热标签。
                    dummy=SimpleNamespace(num_class_orig=10,train_center_sample='radius',
                        train_center_sample_radius=cfg['model']['center_sample_radius'],train_reg_distance=True)
                    real,_=SADA.label_points_single_video(dummy,torch.from_numpy(a['points']),torch.from_numpy(seg),torch.tensor(labels))
                    np.testing.assert_array_equal(real.numpy(),y)
                    rows.append(dict(video=item['video'],**measure(a['probability'],a['raw'],a['valid'],y,item['calibration'])))
            result=dict(shift=shift,epoch=int(folder.name[1:]),video_macro=macro(rows),videos=rows,
                native_label_assignment_exact=True,source_GT_only=True,diagnostics_sha256=sha(folder/'diagnostics.json'),
                gradient_proxy_limit='same source EMA, current-count denominator proxy; not target training gradient or causal AP attribution')
            save(output/(shift+'_'+folder.name+'.json'),result);jobs.append(result)
    training=[]
    for shift in ('S1','S2','S3','S4','S5','S6'):
        entry=root/'formal/negative'/shift;cfg=read(entry/'config.json');rows=read(entry/'metrics.json')
        terminal=read(entry/'termination.json') if (entry/'termination.json').exists() else None
        selected=terminal['decision']['best_epoch'] if terminal else None
        for row in rows:
            r=row['risk'];down=1. if row['epoch']<cfg['train_cfg']['warmup_adv'] else cfg['model']['loss_weights']['weight_downstream']
            contribution=.025*r['ramp']*r['target_cls_loss']*down
            training.append(dict(shift=shift,epoch=row['epoch'],selected=row['epoch']==selected,
                target_loss_added_to_final=contribution,relative_to_total_training_loss=contribution/row['train_loss'],
                mean_target_weight=r['weight_sum']/r['negative'] if r['negative'] else None,
                target_SGP_gradient_after_extra_coeff=r.get('target_SGP_gradient',0)*.025*r['ramp']*down,
                target_reg_gradient=r.get('target_reg_gradient',0)))
    save(output/'final.json',dict(jobs=len(jobs),source_signal_jobs=jobs,training_epochs=training,
        all_source_labels_exact=True,formal_training_modified=False,target_GT_read=False,
        oracle_scope='full positive and negative classification from epoch1; current only low-confidence negative after epoch10',
        full_cold_classification_oracle_gain_pp=6.88335096,negative_only_training_oracle_not_yet_measured=True))
    print(json.dumps(dict(jobs=len(jobs),selected=[r for r in training if r['selected']],
        selected_source_signal=[{k:r[k] for k in ('shift','epoch','video_macro')} for r in jobs
             if any(t['selected'] and t['shift']==r['shift'] and t['epoch']==r['epoch'] for t in training)])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.root,a.output)
