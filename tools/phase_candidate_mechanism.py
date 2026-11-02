"""固定源视频上的fit/泛化及候选约束Oracle；不训练，不访问目标GT。"""
import argparse,json,os,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))


def endpoint_oracles(rows, truth, labels, length):
    errors=[];oracle=[];raw=[];gains=[];eligible=0
    for r in rows:
        ids=[i for i,(g,c) in enumerate(zip(truth,labels)) if c==r['label'] and g[0]<=r['point']<=g[1]]
        if not ids:continue
        gt=truth[min(ids,key=lambda i:truth[i][1]-truth[i][0])];box=np.asarray(r['box']);duration=box[1]-box[0]
        for side in range(2):
            if not .05<gt[side]<length-.05:continue
            coords=box[side]+np.r_[0.,np.linspace(-.25,.25,17)[np.linspace(-.25,.25,17)!=0]]*duration
            coords=coords[(coords>=0)&(coords<length)];best=float(np.min(np.abs(coords-gt[side]))/duration)
            errors.append(abs(r['refined'][side]-gt[side])/duration);oracle.append(best);raw.append(abs(box[side]-gt[side])/duration)
            v=r['view'];point=(r['point']-v['lo']+.5)*v['ratio']-.5
            chosen=coords[np.argmin(np.abs(coords-gt[side]))];mapped=(chosen-v['lo']+.5)*v['ratio']-.5
            mapped_gt=(gt[side]-v['lo']+.5)*v['ratio']-.5
            representable=mapped<=r['student_point'] if side==0 else mapped>=r['student_point']
            if r['student_available'] and 0<=mapped<v['length'] and 0<=mapped_gt<v['length'] and representable:
                eligible+=1;student_error=abs(r['student'][side]-gt[side])/duration
                # 完美门控可以拒绝负监督；保留原候选集，不直接吸附到GT。
                gains.append(max(0.,student_error-best))
    mean=lambda x:float(np.mean(x)) if x else None
    return dict(endpoints=len(errors),raw_error=mean(raw),learned_error=mean(errors),candidate_oracle_error=mean(oracle),
                oracle_eligible=eligible,oracle_gate_positive_gain=mean(gains),oracle_scope='same 17 endpoint coordinates, exact original endpoint retained; source CPU diagnostic, not trained AP')


def main(a):
    from tools.phase_runner_gpu import first_cuda_guard
    guard=first_cuda_guard()
    import torch
    from tools.baseline_protocol import read,save,sha,code_identity
    from tools.phase_boundary import PhaseBoundary,SPEC,select_proposals
    from tools.phase_candidate_components import _device_tree,_source_predictions,_cpu_quality
    cfg=read(a.source/'config.json');idx=read(a.source/'source_candidate_cache_index.json')
    frozen=torch.load(a.snapshot,map_location='cpu');head=PhaseBoundary(torch.load(SPEC['text_cache'],map_location='cpu')['embeddings']).cuda();head.load_state_dict(frozen['state_dict']);head.eval()
    model=SimpleNamespace(fpn_strides=[cfg['model']['scale_factor']**i for i in range(cfg['model']['backbone_arch'][-1]+1)])
    outputs=[];selected={}
    for role in ['train','val']:
        # 只由视频ID选样，不依据GT、误差或当前预测挑视频。
        chosen=sorted(idx['roles'][role],key=lambda x:hashlib.sha256(x['videos'][0].encode()).hexdigest())[:a.videos]
        selected[role]=[x['videos'] for x in chosen]
        for item in chosen:
            assert sha(item['path'])==item['sha256'];data=torch.load(item['path'],map_location='cpu')
            # GPU只接收缓存特征/模型原输出/坐标；GT留在CPU作误差分析。
            gpu=_device_tree({k:data[k] for k in ['weak','strong','views']},'cuda:0')
            with torch.no_grad():
                rows=_source_predictions(model,head,gpu['weak'],gpu['strong'],gpu['views'])
                f,m,c,r=gpu['weak'];props=select_proposals(f,m,c,r,model,0,gpu['views'][0]['original_length'])
            assert len(rows)==len(props)
            for row,prop in zip(rows,props):
                box,label,level,index,score=prop;v=row['view'];stride=model.fpn_strides[level]
                k=int(round(((row['point']-v['lo']+.5)*v['ratio']-.5)/stride))
                row['student_point']=k*stride
                row['student_available']=bool(row['box'][0]>=v['lo'] and row['box'][1]<=v['lo']+v['width']-1 and 0<=k<gpu['strong'][0][level].shape[-1] and gpu['strong'][1][level][0,0,k])
            video=data['videos'][0];truth=video['segments'].numpy();labels=video['labels'].numpy();valid=labels<10;truth=truth[valid];labels=labels[valid]
            actual=_cpu_quality(rows,data['videos']);oracle=endpoint_oracles(rows,truth,labels,video['feats'].shape[-1])
            outputs.append(dict(role=role,videos=item['videos'],cache_sha256=item['sha256'],actual=actual,oracle=oracle,observations=rows))
            del gpu,data
    save(a.output,dict(status='complete',code=code_identity(),guard=guard,guard_sha256=sha(guard),source_only=True,target_GT_used=False,optimizer_steps=0,
        snapshot_sha256=sha(a.snapshot),snapshot_epoch=frozen['epoch'],selected=selected,results=outputs,scope='fixed small source crop/train versus full-video/val probe; neither independent nor full TAL mAP; source Oracle is not a training Oracle'))
    print('COMPLETE',a.output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--videos',type=int,default=3);main(p.parse_args())
