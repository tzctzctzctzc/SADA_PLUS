"""源细分动作标签仅CPU准备；目标GPU唯一入口是无标注manifest。"""
from pathlib import Path
import time
import numpy as np
import torch
from tools.baseline_protocol import read,save,sha
from tools.run_highscore_pairs import DIAG,OLD,SOURCE
from tools.run_layer_signals import seed
from tools.uda_background_math import semantic_targets


def prepare_source(root):
    from tools import baseline_engine as base
    from tools.highscore_confusion_diagnosis import ORIGIN
    report={}
    for s in ('S1','S2'):
        entry=root/'inputs'/s;entry.mkdir(parents=True);roles=torch.load(DIAG/s/'inputs.pt',map_location='cpu');cfg=read(SOURCE/s/'config.json');manifest=read(cfg['dataset']['json_file'])['database']
        cost=read(ORIGIN/s/'costs.json');lookup={r['record']['video']:r['record'] for r in cost['rows']};semantic_count={};sanitized={}
        for r in roles['fit']:
            vid=r['video'];old=torch.load(lookup[vid]['path'],map_location='cpu');meta=old['metadata'];ann=manifest[vid]['annotations'];labels=torch.tensor([int(a['label_id']) for a in ann]);boxes=torch.tensor([a['segment'] for a in ann],dtype=torch.float32).reshape(-1,2)
            gt=(boxes*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride'];target=semantic_targets(r['segments'][r['hard']],gt,labels)
            assert not ((r['bg']==0)&(target>=10)).any();r['semantic']=target
            for c in target.tolist():semantic_count[str(c)]=semantic_count.get(str(c),0)+1
            sanitized[vid]=dict(segments=gt.tolist(),labels=labels.tolist())
        base.checkpoint(entry/'source.pt',roles);save(entry/'source_fit_97truth.json',sanitized)
        ids={role:[r['video'] for r in rows] for role,rows in roles.items()};assert not set(ids['fit'])&set(ids['heldout']) and not (set(ids['fit'])|set(ids['heldout']))&set(ids['validation'])
        target_sets={}
        for role in ('target_train','target_val'):
            old=SOURCE/s/(role+'_unlabeled.json');data=read(old);assert all('annotations' not in x for x in data['database'].values())
            target_sets[role]=set(data['database']);assert not (set(ids['fit'])|set(ids['heldout'])|set(ids['validation']))&target_sets[role]
            save(entry/(role+'_unlabeled.json'),data,sort_keys=False)
        assert not target_sets['target_train']&target_sets['target_val']
        report[s]=dict(source_sha256=sha(entry/'source.pt'),source_fit_97truth_sha256=sha(entry/'source_fit_97truth.json'),old_diagnostic_inputs_sha256=sha(DIAG/s/'inputs.pt'),source_manifest_sha256=sha(cfg['dataset']['json_file']),source_members=ids,
            target_train_videos=len(target_sets['target_train']),target_val_videos=len(target_sets['target_val']),semantic_counts=semantic_count,
            unlabeled_manifest_sha256={r:sha(entry/(r+'_unlabeled.json')) for r in ('target_train','target_val')})
    save(root/'source_prepare.json',dict(passed=True,target_GT_read=False,results=report))


def extract(entry):
    from tools import baseline_engine as base
    from tools.run_foreground_calibration_oracle import load_detector,native_features
    from tools.proposal_foreground_math import seconds_to_grid
    from tools.background_linear_temporal import region_samples
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);seed();shift,role=entry.name.split('_',1);began=time.monotonic();cfg=read(SOURCE/shift/'config.json')
    manifest=entry.parent/'inputs'/shift/(role+'_unlabeled.json');assert all('annotations' not in r for r in read(manifest)['database'].values())
    dataset=base.dataset(cfg,manifest,cfg['tgt_train_split'] if role=='target_train' else cfg['tgt_val_split'],False,True);model=load_detector(SOURCE/shift)
    digest=base.tensor_digest(model.state_dict());assert digest==read(SOURCE/shift/'cache_index.json')['detector_digest'];cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    previous=read(SOURCE/shift/(role+'_gpu_complete.json'));path=SOURCE/shift/(role+'_predictions.pt');assert sha(path)==previous['predictions_sha256'];original={r['video']:r for r in torch.load(path,map_location='cpu')}
    assert set(original)=={r['id'] for r in dataset.data_list};done=[];(entry/'videos').mkdir()
    for i in range(len(dataset)):
        tick=time.monotonic();video=base.move_to_device([dataset[i]],'cuda:0');assert video[0]['labels'].numel()==video[0]['segments'].numel()==0
        feature,valid,result=native_features(model,video,i==0);vid=video[0]['metadata']['video_id'];old=original[vid]
        for k in ('scores','labels','segments'):assert torch.equal(result[k].cpu(),old[k])
        hard=old['scores']>=.3;boxes=seconds_to_grid(old['segments'],video[0]['metadata']).to('cuda');selected=boxes[hard.to('cuda')]
        with torch.no_grad():
            x,m=region_samples(feature,valid,selected);seq=torch.cat(x,1).cpu();mask=torch.cat(m,1).cpu()
        metadata=torch.cat((torch.nn.functional.one_hot(old['labels'][hard].long(),10).float(),torch.logit(old['scores'][hard].clamp(1e-5,1-1e-5))[:,None]/5,
            torch.log1p((selected[:,1]-selected[:,0]).cpu()/feature.shape[-1])[:,None]),1)
        output=entry/'videos'/(vid+'.pt');base.checkpoint(output,dict(video=vid,segments=old['segments'],scores=old['scores'],labels=old['labels'],hard=hard,seq=seq,valid=mask,metadata=metadata,grid_boxes=boxes.cpu(),video_metadata=video[0]['metadata']))
        done.append(dict(video=vid,path=str(output),sha256=sha(output),seconds=time.monotonic()-tick));save(entry/'progress.json',dict(done=len(done),total=len(dataset),elapsed=time.monotonic()-began));print('UNLABELED',entry.name,vid,len(seq),done[-1]['seconds'],flush=True)
        if len(done)==3:
            projection=(time.monotonic()-began)*len(dataset)/3;save(entry/'throughput.json',dict(projected_seconds=projection,limit=900))
            if projection>900:save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False,records=done));return
    assert digest==base.tensor_digest(model.state_dict()) and all(p.grad is None for p in model.parameters()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    save(entry/'complete.json',dict(passed=True,records=done,guard=guard,guard_sha256=sha(guard),detector_digest=digest,manifest=str(manifest),manifest_sha256=sha(manifest),original_predictions_sha256=previous['predictions_sha256'],
        all_original_arrays_bitwise_equal=True,parameters_RNG_unchanged=True,target_GT_read=False,seconds=time.monotonic()-began))


def target_rows(root,shift,role):
    complete=read(root/(shift+'_'+role)/'complete.json');assert complete['passed'];rows=[]
    for rec in complete['records']:
        assert sha(rec['path'])==rec['sha256'];r=torch.load(rec['path'],map_location='cpu');assert not {'bg','quality','semantic','gt','annotations'}&set(r);rows.append(r)
    return rows


def inner_rows(rows):
    return torch.cat([(r['seq'][:,4:20]*r['valid'][:,4:20,None]).sum(1)/r['valid'][:,4:20].sum(1,keepdim=True).clamp_min(1) for r in rows])


def density(entry):
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);seed();shift=entry.name.split('_')[0];source=torch.load(entry.parent/'inputs'/shift/'source.pt',map_location='cpu')['fit'];target=target_rows(entry.parent,shift,'target_train')
    s=inner_rows(source);t=inner_rows(target);sc=torch.cat([r['metadata'][:,:10].argmax(1) for r in source]);tc=torch.cat([r['metadata'][:,:10].argmax(1) for r in target]);means=torch.stack([s[sc==c].mean(0) if (sc==c).any() else s.mean(0) for c in range(10)])
    scale=(s-means[sc]).std(0,unbiased=False).clamp_min(1e-5);a=((s-means[sc])/scale).cuda();b=((t-means[tc])/scale).cuda()
    def sampling(rows):
        weights=torch.cat([torch.full((len(r['seq']),),1/max(1,len(r['seq']))) for r in rows]);return weights/weights.sum()
    sw=sampling(source);tw=sampling(target);order=torch.Generator().manual_seed(311107);model=torch.nn.Linear(512,1).cuda();torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias);opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.05);history=[]
    for step in range(100):
        si=torch.multinomial(sw,256,replacement=True,generator=order).cuda();ti=torch.multinomial(tw,256,replacement=True,generator=order).cuda();opt.zero_grad(set_to_none=True)
        z=model(torch.cat((a[si],b[ti])))[:,0];labels=torch.cat((torch.zeros(256,device='cuda'),torch.ones(256,device='cuda')));loss=torch.nn.functional.binary_cross_entropy_with_logits(z,labels);loss.backward();opt.step();history.append(float(loss.detach()))
    with torch.no_grad():p=model(a)[:,0].sigmoid().cpu();ratio=(p/(1-p).clamp_min(1e-6)).clamp(.25,4)
    base.checkpoint(entry/'density.pt',dict(weights=ratio,model=model.state_dict(),optimizer=opt.state_dict(),order_rng=order.get_state(),source_class_means=means,source_scale=scale,history=history,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0)))
    save(entry/'complete.json',dict(passed=True,guard=guard,guard_sha256=sha(guard),artifact_sha256=sha(entry/'density.pt'),source_candidates=len(s),target_train_candidates=len(t),target_GT_read=False,
        source_weights_min_mean_max=[float(ratio.min()),float(ratio.mean()),float(ratio.max())],clipped_fraction=float(((ratio==.25)|(ratio==4)).float().mean()),loss_first_last=[history[0],history[-1]],
        limitation='bounded domain-discriminator density ratio is a covariate-shift hypothesis, not a guaranteed true ratio; no targetBG labels'))
