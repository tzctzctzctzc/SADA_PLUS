"""独立审源预检成员、缓存/输出SHA、真值标签、校准分位数及背景排序。"""
from pathlib import Path
import argparse,json
import numpy as np


def rank_auc(score,bg):
    if not bg.any() or bg.all():return None
    order=np.argsort(score,kind='stable');x=score[order]
    starts=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1];ends=np.r_[starts[1:],len(x)]
    ranks=np.empty(len(x))
    for a,b in zip(starts,ends):ranks[order[a:b]]=(a+1+b)/2
    n=int(bg.sum());m=len(bg)-n
    return float((ranks[bg].sum()-n*(n+1)/2)/(n*m))


def audit(root):
    import torch
    from tools.baseline_protocol import read,sha,save
    from tools.proposal_foreground_math import ProposalForeground
    plan=read(root/'plan.json');all_results=[]
    for s in ('S1','S2','S3','S4','S5','S6'):
        entry=root/s
        if not (entry/'complete.json').exists():continue
        complete=read(entry/'complete.json');cfg=read(entry/'config.json');selection=read(entry/'selection.json')
        assert selection==plan['jobs'][s]['selection']
        for path,digest in complete['artifacts'].items():assert sha(path)==digest,path
        for key in ('config','selection','source_truth','source_unlabeled'):
            file=entry/(key+'.json')
            expected='truth_sha256' if key=='source_truth' else key+'_sha256'
            assert sha(file)==plan['jobs'][s][expected]
        assert not set(selection['fit'])&set(selection['calibration'])
        assert not (set(selection['fit'])|set(selection['calibration']))&set(selection['validation'])
        source_db=read(cfg['dataset']['json_file'])['database']
        assert set(selection['validation'])=={v for v,r in source_db.items() if r['subset'] in cfg['src_val_split']}
        gpu=read(entry/'gpu_complete.json');assert gpu['code']==plan['code']
        guard=read(gpu['guard']);assert sha(gpu['guard'])==gpu['guard_sha256']
        lease=read(guard['lease']);assert sha(guard['lease'])==guard['lease_sha256']
        assert guard['owner_pid']==lease['owner_pid'] and guard['observed']['uuid']==lease['uuid']
        assert guard['observed']['free_mib']>=lease['minimum_free_mib']>=12288
        index=read(entry/'cache_index.json');assert index['frozen_detector_unchanged'] and index['direct_original_forward_arrays_exact']
        for r in index['records']:assert r['video'] in selection[r['role']] and sha(r['path'])==r['sha256']
        head=torch.load(entry/'head_fixed20.pt',map_location='cpu')
        assert [r['epoch'] for r in head['history']]==list(range(1,21))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(311103);initial=ProposalForeground(512).state_dict()
        assert any(not torch.equal(v,head['state_dict'][k]) for k,v in initial.items())
        assert all(torch.isfinite(v).all() for v in head['state_dict'].values())
        truth=read(entry/'source_truth.json');rows=torch.load(entry/'source_predictions.pt',map_location='cpu');q={}
        for r in rows:
            b=r['boxes'].numpy();gt=np.asarray(truth[r['video']]['segments'],np.float32).reshape(-1,2)
            inter=np.maximum(0,np.minimum(b[:,None,1],gt[None,:,1])-np.maximum(b[:,None,0],gt[None,:,0]))
            union=(b[:,1]-b[:,0])[:,None]+(gt[:,1]-gt[:,0])[None]-inter
            best=(inter/np.maximum(union,1e-8)).max(-1) if len(gt) else np.zeros(len(b))
            y=np.where(best<.01,0,np.where(best>=.5,2,1));np.testing.assert_array_equal(y,r['states'].numpy())
            bg=y==0;score=r['background_probability'].numpy();tau=complete['calibration_threshold'];remove=score>tau
            group=q.setdefault(r['role'],[])
            group.append(dict(video=r['video'],bg_AUC=rank_auc(score,bg),confidence_bg_AUC=rank_auc(-r['scores'].numpy(),bg),
                bg_removed_fraction=float((remove&bg).sum()/max(bg.sum(),1)),
                original_TP_kept_fraction=float((~remove&r['true_positive'].numpy()).sum()/max(int(r['true_positive'].sum()),1))))
        # 独立NumPy视频等权校准分位数；严格大于阈值才删，同分全部保留。
        positives=[r['background_probability'][r['true_positive']].numpy().astype(float) for r in rows if r['role']=='calibration' and r['true_positive'].any()]
        if positives:
            values=np.concatenate(positives);weights=np.concatenate([np.full(len(x),1/(len(x)*len(positives))) for x in positives])
            order=np.argsort(values);j=min(int(np.searchsorted(np.cumsum(weights[order]),.99)),len(order)-1)
            assert abs(float(values[order[j]])-complete['calibration_threshold'])<1e-7
        else:assert complete['calibration_threshold']==1.01
        macro={role:{k:float(np.mean([r[k] for r in records if r[k] is not None])) for k in ('bg_AUC','confidence_bg_AUC','bg_removed_fraction','original_TP_kept_fraction')} for role,records in q.items()}
        results=complete['results'];base=results['original']['map']
        assert abs(complete['delta_pp']-100*(results['learned']['map']-base))<1e-10
        assert abs(complete['equal_count_confidence_delta_pp']-100*(results['confidence_equal_count']['map']-base))<1e-10
        assert abs(base-read(Path(cfg['proposal_foreground']['reference'])/'evaluation/source_val.json')['map_0.1_to_0.5'])<1e-7 if cfg.get('proposal_foreground') else abs(base-read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/s/'evaluation/source_val.json')['map_0.1_to_0.5'])<1e-7
        report=dict(passed=True,shift=s,delta_pp=complete['delta_pp'],equal_count_confidence_delta_pp=complete['equal_count_confidence_delta_pp'],source_oracle_delta_pp=complete['source_oracle_delta_pp'],
            source_only_not_target_UDA=True,macro_by_role=macro,full_source_val_videos=len(selection['validation']),fit_videos=len(selection['fit']),calibration_videos=len(selection['calibration']),
            results=results,complete_sha256=sha(entry/'complete.json'),all_cache_SHA_labels_memberships_and_calibration_checked=True)
        save(root/'independent_audit'/(s+'.json'),report);all_results.append(report)
    result=dict(count=len(all_results),results=all_results,source_only=True)
    if len(all_results)==6:
        result['mean_source_delta_pp']=float(np.mean([r['delta_pp'] for r in all_results]))
        result['mean_confidence_equal_count_delta_pp']=float(np.mean([r['equal_count_confidence_delta_pp'] for r in all_results]))
    save(root/'independent_final_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();audit(a.root)
