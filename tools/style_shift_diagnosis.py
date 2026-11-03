"""无标签风格域判别，只使用sourceFIT与targetTRAIN；不将可分性当作因果证明。"""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
DENSE=Path('/vepfs/vlms/highscore_dense_revisit_5ef4cca_1103')


def auc(y,x):
    a=x[y==1];b=np.sort(x[y==0]);return float((np.searchsorted(b,a,side='left')+np.searchsorted(b,a,side='right')).sum()/(2*len(a)*len(b)))


def run(root):
    import torch
    from tools.run_dense_revisit import raw_load
    result={}
    for s in ('S1','S2'):
        data=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu');rows=data['roles'];features=[];labels=[];videos=[];groups={}
        for role,label in [('fit',0),('target_train',1)]:
            for r in rows[role]:
                x=((raw_load(r)-data['mean'])/data['std']).numpy();mu=x.mean(0,dtype=np.float64);sig=np.log(x.var(0,ddof=1,dtype=np.float64)+1e-6)*.5
                features.append(np.r_[mu,sig]);labels.append(label);videos.append(r['video'])
                if role=='fit':
                    p=r['video'].split('_')[0];groups.setdefault(p,dict(videos=[],annotation_count=0,task_cell_mass=np.zeros(10)))
                    groups[p]['videos'].append(r['video']);groups[p]['annotation_count']+=len(r['annotations']['labels']);groups[p]['task_cell_mass']+=r['dense'][:,:10].sum(0).numpy()
        x=np.array(features);y=np.array(labels);held=[]
        for label in (0,1):
            ids=np.flatnonzero(y==label);ranked=sorted(ids,key=lambda i:hashlib.sha256(('style-probe-v1|'+videos[i]).encode()).hexdigest());held+=ranked[:max(1,int(np.ceil(.2*len(ids))))]
        held=np.array(sorted(held));fit=np.setdiff1d(np.arange(len(y)),held);reports={}
        for name,sl in [('mean',slice(0,2304)),('logstd',slice(2304,None)),('joint',slice(None))]:
            a=x[:,sl];mu=a[fit].mean(0);sd=np.maximum(a[fit].std(0),1e-4);z=(a-mu)/sd;w=np.zeros(z.shape[1]);bias=0.;weights=np.array([.5/(y[fit]==label).sum() for label in y[fit]])
            for _ in range(200):
                pred=1/(1+np.exp(-np.clip(z[fit]@w+bias,-40,40)));error=(pred-y[fit])*weights;w-=.1*(z[fit].T@error+.01*w);bias-=.1*error.sum()
            value=z[held]@w+bias;reports[name]=dict(held_video_AUC=auc(y[held],value),held_source=int((y[held]==0).sum()),held_target=int((y[held]==1).sum()))
        for p,v in groups.items():v['task_cell_mass']=v['task_cell_mass'].tolist()
        result[s]=dict(source_FIT=len(rows['fit']),unlabeled_target_TRAIN=len(rows['target_train']),domain_probe=reports,participants=groups,
            fit_videos=[videos[i] for i in fit],probe_held_videos=[videos[i] for i in held],limitation='stylesummary also contains actiondistribution, motion and duration effects; domainseparability doesnot establish scene shortcut')
        print('STYLE_DIAG',s,reports,flush=True)
    save(root/'style_shift_diagnosis.json',dict(passed=True,results=result,target_train_GT_read=False,target_val_read=False,code=code_identity(),dense_audit_sha256=sha(DENSE/'independent_final_audit.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
