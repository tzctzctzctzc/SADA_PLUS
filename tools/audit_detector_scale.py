"""全视频NumPy独立融合检查、原数组身份及完整官方AP复核。"""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_detector_scale import TASKS,MODES,SOURCE,DENSE


def sigmoid(x):return (1/(1+np.exp(-x.astype(np.float64)))).astype(np.float32)


def independent_fuse(cls,reg,views):
    ps=sigmoid(cls);rs=reg.copy();count=np.ones(cls.shape[1],np.float32)
    for c,r,valid,ratio in views:
        q=np.arange(cls.shape[1],dtype=np.float32)*np.float32(ratio);left=np.floor(q).astype(np.int64);right=left+1;mask=(left>=0)&(right<len(valid));lo=np.clip(left,0,len(valid)-1);hi=np.clip(right,0,len(valid)-1);mask&=valid[lo]&valid[hi];fraction=q-left.astype(np.float32)
        p=sigmoid(c);p=p[:,lo]*(1-fraction)[None]+p[:,hi]*fraction[None];r=r[:,lo]*(1-fraction)[None]+r[:,hi]*fraction[None];ps=ps+p*mask[None];rs=rs+r/np.float32(ratio)*mask[None];count=count+mask
    return ps/count[None],rs/count[None]


def run(root):
    import torch,pandas as pd
    from tools.audit_background_ap_cost import IndependentAP,official
    from tools.run_foreground_calibration_oracle import load_truth
    while not (root/'final.json').exists():
        if (root/'error.json').exists():raise RuntimeError('producer failed; no restart')
        time.sleep(3)
    final=read(root/'final.json');assert set(final['allGPU_sha256'])==set(TASKS);owner=read(root/'state.json')['owner_pid'];plan=read(root/'plan.json');assert plan['spec']['scales']==[.75,1.,1.25] and plan['spec']['primary']=='score_boundary';reports={}
    for task in TASKS:
        s,role=task.split('_',1);e=root/task;g=read(e/'gpu_complete.json');assert sha(e/'gpu_complete.json')==final['allGPU_sha256'][task];assert g['passed'] and g['optimizer_steps']==0 and g['all_baseline_arrays_bitwise_equal'] and g['all_cached_fusion_GPU_bitwise_equal'] and g['CPUCUDA_RNG_unchanged']
        guard=read(g['guard']);lease=read(e/'lease_infer.json');assert sha(g['guard'])==g['guard_sha256'] and sha(e/'lease_infer.json')==guard['lease_sha256'];assert guard['owner_pid']==lease['owner_pid']==owner and guard['observed']['uuid']==lease['uuid'] and guard['observed']['free_mib']>=12288
        assert g['model_digest_unchanged']==read(SOURCE/s/'cache_index.json')['detector_digest'];path=root/'inputs'/(task+'.pt');assert sha(path)==g['input_sha256']==read(root/'prepare.json')['tasks'][task]['sha256'];assert sha(e/'predictions.pt')==g['predictions_sha256'];rows=torch.load(path,map_location='cpu');out=torch.load(e/'predictions.pt',map_location='cpu');assert len(rows)==len(g['cases']);max_p=0.;max_r=0.
        for index,(r,c) in enumerate(zip(rows,g['cases'])):
            assert r['video']==c['video'];assert not {'annotations','bg','quality','dense','delta'}&set(r)
            for key in ('segments','scores','labels'):assert torch.equal(r[key],out['baseline'][index][key])
            assert sha(c['path'])==c['sha256'];case=torch.load(c['path'],map_location='cpu');assert case['raw_sha256']==r['raw_sha256'] and case['video']==r['video'];length=case['base_length'];assert len(case['views'])==2
            for v,scale in zip(case['views'],(.75,1.25)):
                assert v['scale']==scale and v['length']==max(2,round((length-1)*scale)+1) and v['ratio']==(v['length']-1)/(length-1)
            for l in range(6):
                p,z=independent_fuse(case['base_cls'][l].numpy(),case['base_reg'][l].numpy(),[(v['cls'][l].numpy(),v['reg'][l].numpy(),v['valid'][l].numpy(),v['ratio']) for v in case['views']]);actual=sigmoid(case['fused_cls'][l].numpy());dr=float(np.abs(z-case['fused_reg'][l].numpy()).max());dp=float(np.abs(p-actual).max());max_p=max(max_p,dp);max_r=max(max_r,dr);assert np.allclose(p,actual,atol=2e-7,rtol=0);assert np.allclose(z,case['fused_reg'][l].numpy(),atol=3e-6,rtol=1e-6)
        reports[task]=dict(all_videos=len(rows),full_fusion_checked=True,max_probability_error=max_p,max_offset_error=max_r,GPU_sha256=sha(e/'gpu_complete.json'))
        print('SCALE_FUSION_AUDITED',task,len(rows),max_p,max_r,flush=True)
    results={}
    def frame(rows):return pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c),float(v)) for r in rows for b,c,v in zip(r['segments'],r['labels'],r['scores'])],columns=['video-id','t-start','t-end','label','score'])
    for s in ('S1','S2'):
        cfg=read(SOURCE/s/'config.json');report=read(root/(s+'_evaluation.json'));assert sha(root/(s+'_evaluation.json'))==final['evaluation_sha256'][s];results[s]={}
        for role in ('validation','target_val'):
            gt=load_truth(cfg['dataset']['json_file'] if role=='validation' else cfg['baseline']['annotation_for_offline_evaluation'],cfg['src_val_split'] if role=='validation' else cfg['tgt_val_split']);pred=torch.load(root/(s+'_'+role)/'predictions.pt',map_location='cpu');results[s][role]={}
            for mode in MODES:
                f=frame(pred[mode]);curve=[]
                for c in range(10):
                    a=gt[gt.label==c].reset_index(drop=True);b=f[f.label==c].reset_index(drop=True);ref=IndependentAP(a,b).ap(np.ones(len(b),bool));value=official(a,b) if len(a) and len(b) else np.zeros(5);assert np.allclose(ref,value,atol=1e-7,rtol=0);curve.append(ref)
                curve=np.asarray(curve);old=report['results'][role][mode];assert np.allclose(curve,old['per_class_tiou_AP'],atol=1e-7,rtol=0);assert abs(curve.mean()-old['map'])<1e-7;results[s][role][mode]=old
            if role=='target_val':assert abs(results[s][role]['baseline']['map']-read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/s/'evaluation/target_val.json')['map_0.1_to_0.5'])<1e-7
    save(root/'independent_final_audit.json',dict(passed=True,code=code_identity(),production_final_sha256=sha(root/'final.json'),GPU_checks=reports,results=results,exact_tasks=4,exact_methods=3,full_NumPy_fusion_all_videos=True,full_AP_official_and_independent_equal=True,NMS_scope='originalNMS implementation reused, not separatelyrewritten',target_train_GT_read=False,target_val_used_for_selection=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
