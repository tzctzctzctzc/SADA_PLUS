"""只读诊断：同预算GT零误伤参照、配置特征时间支持与纯BG标注的冲突。"""
import argparse
from pathlib import Path
import sys
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
SOURCE=Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
LAYER=Path('/vepfs/vlms/highscore_layer_signals_bdcc520_1103')
DIAG=Path('/vepfs/vlms/highscore_confusion_diag_36c0916_1103')
UDA=Path('/vepfs/vlms/highscore_uda_ideas_354cf35_1103')
RATES=(.005,.01,.02,.05,.1,.2,1.)


def configured_support(boxes,truth,raw_length,num_frames=32,feat_stride=16):
    """按loader中心时间约定推算窗口；不冒称核过原特征提取器的真实像素支持。"""
    fraction=(np.arange(16,dtype=np.float64)+.5)/16;x=boxes[:,0,None]+(boxes[:,1]-boxes[:,0])[:,None]*fraction
    lo=np.floor(x).astype(int);hi=np.ceil(x).astype(int);valid=(lo>=0)&(hi<raw_length)
    half=num_frames/(2*feat_stride);left=lo-half;right=hi+half
    contaminated=np.zeros_like(valid)
    if len(truth):
        contaminated=((np.minimum(right[:,:,None],truth[None,None,:,1])-np.maximum(left[:,:,None],truth[None,None,:,0]))>0).any(2)&valid
    count=valid.sum(1);hit=contaminated.sum(1);unique=[]
    for a,b,m in zip(lo,hi,valid):unique.append(len(np.unique(np.r_[a[m],b[m]])))
    return dict(valid_samples=count,overlap_samples=hit,any_overlap=hit>0,all_overlap=(count>0)&(hit==count),unique_feature_centers=np.array(unique))


def source_support(root):
    import torch
    from tools.audit_background_ap_cost import overlaps
    results={}
    for s in ('S1','S2'):
        cfg=read(SOURCE/s/'config.json');manifest=read(cfg['dataset']['json_file'])['database'];original=torch.load(DIAG/s/'inputs.pt',map_location='cpu');lookup={}
        for domain in ('source','validation'):
            for rec in read(LAYER/(s+'_'+domain)/'complete.json')['records']:lookup[rec['video']]=rec
        roles={}
        for role,rows in original.items():
            per_video=[];all_bg=[];all_non=[];direct_overlap=0
            for row in rows:
                rec=lookup[row['video']];assert sha(rec['path'])==rec['sha256'];bound=torch.load(rec['path'],map_location='cpu');meta=bound['metadata'];boxes=bound['boxes'][bound['hard']].numpy().astype(float)
                annotations=[a for a in manifest[row['video']]['annotations'] if 0<=int(a['label_id'])<10];truth=np.asarray([a['segment'] for a in annotations],dtype=float).reshape(-1,2);truth=(truth*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']
                matrix=overlaps(boxes,truth);maximum=matrix.max(1) if len(truth) else np.zeros(len(boxes));bg=maximum<.01
                if role!='validation':assert np.array_equal(bg,row['bg'].numpy().astype(bool))
                support=configured_support(boxes,truth,meta['video_length'],meta['feat_num_frames'],meta['feat_stride']);direct_overlap+=int((bg&(maximum>0)).sum())
                def summary(mask):
                    return dict(candidates=int(mask.sum()),any_configured_window_overlap=int(support['any_overlap'][mask].sum()),all_configured_windows_overlap=int(support['all_overlap'][mask].sum()),
                        at_most4_distinct_feature_centers=int((support['unique_feature_centers'][mask]<=4).sum()),zero_valid_samples=int((support['valid_samples'][mask]==0).sum()))
                per_video.append(dict(video=row['video'],BG=summary(bg),nonBG=summary(~bg)))
                for mask,dest in ((bg,all_bg),(~bg,all_non)):
                    for i in np.flatnonzero(mask):dest.append((float((boxes[i,1]-boxes[i,0])*meta['feat_stride']/meta['fps']),int(support['unique_feature_centers'][i]),bool(support['any_overlap'][i]),bool(support['all_overlap'][i])))
            def aggregate(data):
                a=np.asarray(data,float)
                if not len(a):return dict(candidates=0)
                return dict(candidates=len(a),duration_seconds_q10_q50_q90=np.quantile(a[:,0],[.1,.5,.9]).tolist(),distinct_feature_centers_q10_q50_q90=np.quantile(a[:,1],[.1,.5,.9]).tolist(),
                    any_configured_window_overlap_fraction=float(a[:,2].mean()),all_configured_windows_overlap_fraction=float(a[:,3].mean()),at_most4_distinct_feature_centers_fraction=float((a[:,1]<=4).mean()))
            roles[role]=dict(BG=aggregate(all_bg),nonBG=aggregate(all_non),BG_with_nonzero_direct_task_IoU=direct_overlap,per_video=per_video)
        results[s]=roles
    save(root/'configured_input_support.json',dict(source_only=True,results=results,temporal_convention='center=(feature_index*stride+.5*num_frames)/fps, each contributing raw feature assigned num_frames support; interpolate floor/ceil',
        caveats=['configuration-derived timing support, actual extraction pixel windows not independently verified','overlapping a task annotation does not prove representation identity or impossibility of discrimination','SGP receptive fields may be larger; this computes only nominal raw-window support','strict BG IoU<.01 can include small nonzero task overlap']))


def oracle_curves(root):
    import torch
    from tools.foreground_oracle_ap import ClassAP,prediction_frame
    from tools.audit_background_ap_cost import IndependentAP,background
    from tools.run_foreground_calibration_oracle import load_truth
    results={}
    for s in ('S1','S2'):
        e=UDA/(s+'_density');done=read(e/'gpu_complete.json');assert sha(e/'predictions.pt')==done['predictions_sha256'];rows=torch.load(e/'predictions.pt',map_location='cpu')['target_val'];frame=prediction_frame(rows)
        cfg=read(SOURCE/s/'config.json');truth=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split']);bg=background(truth,frame);high=frame.score.to_numpy()>=.3;positions={};engines={};refs={};baseline=[]
        for c in range(10):
            idx=np.flatnonzero(frame.label.to_numpy()==c);positions[c]=idx;engines[c]=ClassAP(truth[truth.label==c],frame.iloc[idx]);refs[c]=IndependentAP(truth[truth.label==c],frame.iloc[idx].reset_index(drop=True))
            ap=engines[c].evaluate(np.ones(len(idx),bool))['ap'];assert np.allclose(ap,refs[c].ap(np.ones(len(idx),bool)),atol=1e-7,rtol=0);baseline.append(ap)
        curve=[]
        for rate in RATES:
            selected=[];requested=0
            for c,idx in positions.items():
                pool=idx[high[idx]];quota=int(len(pool)*rate);requested+=quota;valid=[int(i) for i in pool if bg[i]];valid.sort(key=lambda i:(-float(frame.score.iloc[i]),i));selected.extend(valid[:quota])
            selected.sort();keep=np.ones(len(frame),bool);keep[selected]=False;ap=[engines[c].evaluate(keep[positions[c]])['ap'] for c in range(10)];ref=[refs[c].ap(keep[positions[c]]).tolist() for c in range(10)];assert np.allclose(ap,ref,atol=1e-7,rtol=0)
            assert not np.any(high[~keep]==False) and bg[selected].all();curve.append(dict(rate=rate,requested_deletion_quota=requested,actual_BG_deleted=len(selected),nonBG_deleted=0,delta_pp=float(100*(np.mean(ap)-np.mean(baseline)))))
        results[s]=dict(high_candidates=int(high.sum()),high_BG=int((bg&high).sum()),high_BG_fraction=float(bg[high].mean()),baseline=float(np.mean(baseline)),curve=curve)
        print('ORACLE_CURVE',s,results[s],flush=True)
    means=[dict(rate=rate,mean_delta_pp=float(np.mean([results[s]['curve'][i]['delta_pp'] for s in results]))) for i,rate in enumerate(RATES)]
    save(root/'same_budget_oracle_curves.json',dict(results=results,two_shift_means=means,
        scope='full S1/S2 targetval, fixed original predictions; CPU GT-assisted diagnostic only, never training or threshold selection',
        selection='within each predictedclass quota floor(rate*Nhigh), delete GTpureBG in descending original confidence; if BG insufficient do not delete foreground',
        caveats=['not a proof of globally optimal AP deletion subset; score ties retain original benchmark re-sort behavior','two directions not six-direction mean; fullBG endpoint not assumed equal to known six-direction+4.557834pp','GT-selected deletion curves are diagnostic oracles, not deployable UDA methods']))


def selection_decomposition(root):
    import torch
    from tools.foreground_oracle_ap import ClassAP,prediction_frame
    from tools.audit_background_ap_cost import IndependentAP,background
    from tools.run_foreground_calibration_oracle import load_truth
    original=read(root/'complete.json');assert sha(root/'same_budget_oracle_curves.json')==original['oracle_curve_sha256'];oracle=read(root/'same_budget_oracle_curves.json');audit=read(UDA/'independent_final_audit.json');reports={}
    for s in ('S1','S2'):
        task=s+'_density';entry=UDA/task;record=read(UDA/'independent_results'/(task+'.json'));assert sha(entry/'complete.json')==record['complete_sha256'];done=read(entry/'complete.json');cfg=read(SOURCE/s/'config.json')
        frame=prediction_frame(torch.load(entry/'predictions.pt',map_location='cpu')['target_val']);truth=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split']);bg=background(truth,frame);positions={};engines={};refs={};baseline=[]
        for c in range(10):
            idx=np.flatnonzero(frame.label.to_numpy()==c);positions[c]=idx;engines[c]=ClassAP(truth[truth.label==c],frame.iloc[idx]);refs[c]=IndependentAP(truth[truth.label==c],frame.iloc[idx].reset_index(drop=True));baseline.append(engines[c].evaluate(np.ones(len(idx),bool))['ap'])
        def effect(selected):
            keep=np.ones(len(frame),bool);keep[selected]=False;ap=[engines[c].evaluate(keep[positions[c]])['ap'] for c in range(10)];reference=[refs[c].ap(keep[positions[c]]).tolist() for c in range(10)];assert np.allclose(ap,reference,atol=1e-7,rtol=0)
            return float(100*(np.mean(ap)-np.mean(baseline)))
        report=[]
        for i,budget in enumerate(done['results']['target_val']['density']['budgets']):
            ids=np.asarray(budget['selected'],dtype=int);actual=effect(ids);assert abs(actual-budget['delta_pp'])<1e-7;bgids=ids[bg[ids]];fgids=ids[~bg[ids]];ideal=[]
            for c,idx in positions.items():
                pool=idx[frame.iloc[idx].score.to_numpy()>=.3];quota=int(len(pool)*budget['rate']);choices=[int(j) for j in pool if bg[j]];choices.sort(key=lambda j:(-float(frame.score.iloc[j]),j));ideal.extend(choices[:quota])
            assert len(ideal)==oracle['results'][s]['curve'][i]['actual_BG_deleted']
            report.append(dict(rate=budget['rate'],actual_delta_pp=actual,actual_BG_count=len(bgids),actual_nonBG_count=len(fgids),
                delete_only_model_selected_trueBG_delta_pp=effect(bgids),delete_only_model_selected_nonBG_delta_pp=effect(fgids),
                configured_pureBG_oracle_delta_pp=oracle['results'][s]['curve'][i]['delta_pp'],configured_pureBG_oracle_count=len(ideal),
                model_selected_BG_mean_original_score=float(frame.iloc[bgids].score.mean()) if len(bgids) else None,
                oracle_BG_mean_original_score=float(frame.iloc[ideal].score.mean()) if len(ideal) else None))
        reports[s]=report
    means=[]
    for i,rate in enumerate(RATES[:4]):
        keys=('actual_delta_pp','delete_only_model_selected_trueBG_delta_pp','delete_only_model_selected_nonBG_delta_pp','configured_pureBG_oracle_delta_pp')
        means.append(dict(rate=rate,**{k:float(np.mean([reports[s][i][k] for s in reports])) for k in keys}))
    save(root/'selection_cost_decomposition.json',dict(results=reports,two_shift_means=means,UDA_independent_audit_sha256=sha(UDA/'independent_final_audit.json'),
        original_oracle_curve_sha256=sha(root/'same_budget_oracle_curves.json'),effects_not_additive=True,not_training_or_selection=True,
        caveat='nonBG includes partial/wrongclass task-overlap candidates; not every nonBG is a true positive. AP is recomputed after each full intervention, not summed single-box costs'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--decompose-existing',action='store_true');a=p.parse_args()
    if a.decompose_existing:selection_decomposition(a.root)
    else:
        a.root.mkdir(parents=True,exist_ok=False);save(a.root/'plan.json',dict(code=code_identity(),mode='read_only_CPU_no_training',rates=RATES));source_support(a.root);oracle_curves(a.root)
        save(a.root/'complete.json',dict(passed=True,source_support_sha256=sha(a.root/'configured_input_support.json'),oracle_curve_sha256=sha(a.root/'same_budget_oracle_curves.json')))
