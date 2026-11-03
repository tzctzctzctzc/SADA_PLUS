"""同219视频、同24采样点，抽取早层序列与无标签视频均值。"""
from pathlib import Path
import time
import numpy as np
import torch
from tools.baseline_protocol import read,save,sha
from tools.run_layer_signals import SOURCE,DIAG,seed
from tools.background_linear_temporal import region_samples
LAYER=Path('/vepfs/vlms/highscore_layer_signals_bdcc520_1103')


def extract(entry):
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.run_foreground_calibration_oracle import load_detector,native_features
    guard=first_cuda_guard(entry);seed();shift,domain=entry.name.split('_');begin=time.monotonic()
    assert sha(LAYER/'independent_final_audit.json')=='9f4698d0ed47bdc0a406dfac7e383389df8a27fc0102605b908004e5d69b78fe'
    frozen=read(LAYER/entry.name/'complete.json');cfg=read(SOURCE/shift/'config.json');model=load_detector(SOURCE/shift)
    digest=base.tensor_digest(model.state_dict());assert digest==frozen['detector_digest'];cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    original=torch.load(DIAG/shift/'inputs.pt',map_location='cpu');lookup={r['video']:r for rows in original.values() for r in rows};done=[];(entry/'videos').mkdir()
    for rec in frozen['records']:
        tick=time.monotonic();assert sha(rec['path'])==rec['sha256'];bound=torch.load(rec['path'],map_location='cpu');vid=rec['video'];old=lookup[vid]
        raw=torch.from_numpy(np.load(bound['raw_path'])['feats'].astype(np.float32)[::cfg['dataset']['downsample_rate']].T.copy());assert sha(bound['raw_path'])==bound['raw_sha256']
        video=[dict(feats=raw.cuda(),segments=torch.empty(0,2,device='cuda'),labels=torch.empty(0,dtype=torch.long,device='cuda'),metadata=bound['metadata'])]
        with torch.no_grad():
            x,mask=model.preprocessing(video,mode='test');bf,bm,emb=model.backbone_source(model.dropout(x),mask);fpn,fm=model.neck_source(bf,bm)
            if not done:
                f,v,_=native_features(model,video,True);assert torch.equal(f,fpn[0][0]) and torch.equal(v,fm[0][0,0].bool())
            cached=torch.load(bound['source_cache']['path'],map_location='cpu');assert torch.equal(fpn[0][0].cpu(),cached['feature'])
            valid=fm[0][0,0].bool();assert torch.equal(valid,mask[0,0].bool()) and torch.equal(valid.cpu(),cached['valid'])
            boxes=bound['boxes'][bound['hard']].cuda();ev,em=region_samples(emb[0],valid,boxes);fv,mm=region_samples(fpn[0][0],valid,boxes)
            es=torch.cat(ev,1);fs=torch.cat(fv,1);vm=torch.cat(em,1);assert torch.equal(vm,torch.cat(mm,1))
            assert torch.equal(fs.cpu(),old['seq']) and torch.equal(vm.cpu(),old['valid'])
            inner=(ev[1]*em[1][...,None]).sum(1)/em[1].sum(1,keepdim=True).clamp_min(1)
            assert torch.equal(inner.cpu(),bound['representations']['embedding'])
            a=emb[0][:,valid].mean(1);b=fpn[0][0][:,valid].mean(1)
        path=entry/'videos'/(vid+'.pt');base.checkpoint(path,dict(video=vid,seq_early=es.cpu(),seq_fpn=fs.cpu(),valid=vm.cpu(),global_early=a.cpu(),global_fpn=b.cpu(),
            layer_descriptor=rec,raw_sha256=bound['raw_sha256'],scores=old['scores'],labels=old['labels'],hard=old['hard']))
        done.append(dict(video=vid,path=str(path),sha256=sha(path),seconds=time.monotonic()-tick));save(entry/'progress.json',dict(done=len(done),total=len(frozen['records']),elapsed=time.monotonic()-begin));print('INPUT',entry.name,vid,done[-1]['seconds'],flush=True)
        if len(done)==3:
            projected=(time.monotonic()-begin)*len(frozen['records'])/3;save(entry/'throughput.json',dict(projected_end_to_end_seconds=projected,limit=900))
            if projected>900:save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False,records=done));return
    assert digest==base.tensor_digest(model.state_dict()) and all(p.grad is None for p in model.parameters()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    save(entry/'complete.json',dict(passed=True,guard=guard,guard_sha256=sha(guard),records=done,detector_digest=digest,layer_input_complete_sha256=sha(LAYER/entry.name/'complete.json'),
        all_old_FPN_samples_bitwise_equal=True,all_early_inner_means_bitwise_equal=True,parameters_RNG_unchanged=True,seconds=time.monotonic()-begin))


def load_roles(root,shift,mode):
    from tools.signal_ideas_math import select_sequence
    original=torch.load(DIAG/shift/'inputs.pt',map_location='cpu');lookup={}
    for domain in ('source','validation'):
        complete=read(root/(shift+'_'+domain)/'complete.json');assert complete['passed']
        for rec in complete['records']:lookup[rec['video']]=rec
    roles={}
    for role,rows in original.items():
        roles[role]=[]
        for old in rows:
            rec=lookup[old['video']];assert sha(rec['path'])==rec['sha256'];r=torch.load(rec['path'],map_location='cpu')
            for k in ('scores','labels','hard','valid'):assert torch.equal(r[k],old[k])
            roles[role].append({k:old[k] for k in ('video','segments','scores','labels','hard','valid','metadata') if k in old}|{k:old[k] for k in ('bg','quality') if k in old}|dict(seq=select_sequence(r,mode)))
    return roles
