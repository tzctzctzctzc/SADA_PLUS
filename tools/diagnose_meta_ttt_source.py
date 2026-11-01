"""只读源域诊断：拆分准备退化与适应效果，不改训练/选模，不读取目标GT。"""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
from collections import OrderedDict
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))


def relative_change(before,after):
    numerator=denominator=0.;changed=total=0
    for key,a in before.items():
        b=after[key].detach();a=a.detach()
        numerator+=float((b.double()-a.double()).square().sum())
        denominator+=float(a.double().square().sum())
        changed+=int((a!=b).sum());total+=a.numel()
    return dict(relative_l2=math.sqrt(numerator/max(denominator,1e-30)),changed_fraction=changed/max(total,1))


def alignment(a,b):
    dot=aa=bb=0.
    for x,y in zip(a,b):
        if x is not None:aa+=float(x.double().square().sum())
        if y is not None:bb+=float(y.double().square().sum())
        if x is not None and y is not None:dot+=float((x.double()*y.double()).sum())
    return dict(cosine=dot/max(math.sqrt(aa*bb),1e-30),detection_gradient_norm=math.sqrt(aa),recovery_gradient_norm=math.sqrt(bb))


def main(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('此诊断必须禁CUDA')
    import torch
    from tools.run_meta_ttt import load,data,binding,item
    from tools.meta_ttt import (SPEC,Encoder,clean_video,reference_targets,recovery_loss,unroll,
        detection_packet,detection_loss,predict)
    from tools.baseline_protocol import read,save,sha,code_identity
    from tools.baseline_engine import tensor_digest
    from tools.run_tta import evaluate_predictions
    from tools.run_sada_baseline import prepare_file_limit
    prepare_file_limit();torch.set_num_threads(args.threads);os.chdir('/vepfs/vlms/SADA_PLUS')
    job_path=args.root/'jobs'/('train_meta_'+args.shift+'.json');job=read(job_path)
    assert binding(job)==job['input_sha256']
    model,ref,enc,dec,cfg,initial,head_initial=load(job,'cpu')
    ds=data(job,cfg,cfg['src_val_split'],False)
    saved_path=args.root/'formal/meta'/args.shift/'latest.pt';saved=torch.load(saved_path,map_location='cpu')
    assert saved['config']==cfg and saved['specification']==SPEC and saved['arm']=='meta'
    assert saved['parent_checkpoint_sha256']==job['input_sha256']['checkpoint']
    args.output.mkdir(parents=True,exist_ok=False);results={};lengths={}
    def evaluate_case(name,steps):
        rows=[];start=time.perf_counter()
        for index in range(len(ds)):
            video=ds[index];lengths[video['metadata']['video_id']]=video['feats'].shape[-1]
            rows.append(item(predict(model,ref,enc,dec,clean_video(video),steps)))
            if index%10==0:print('SOURCE_PROGRESS',args.shift,name,index+1,len(ds),time.perf_counter()-start,flush=True)
        p=args.output/(name+'_predictions.json')
        with p.open('x') as stream:json.dump(rows,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        metrics=evaluate_predictions(rows,job['source_manifest'],cfg['src_val_split'])
        results[name]=dict(metrics,steps=steps,source_videos=len(ds),elapsed_seconds=time.perf_counter()-start,predictions_sha256=sha(p))
        save(args.output/(name+'.json'),results[name]);print('SOURCE_RESULT',args.shift,name,json.dumps(results[name]),flush=True)
    evaluate_case('original_noadapt',0)
    model.load_state_dict(saved['state_dict'],strict=True);dec.load_state_dict(saved['decoder'],strict=True)
    drift={}
    for key,a,b in [('SGP',ref.backbone_target,model.backbone_target),('FPN',ref.neck_target,model.neck_target),
                    ('classification',ref.cls_head_target,model.cls_head_target),('regression',ref.reg_head_target,model.reg_head_target)]:
        drift[key]=relative_change(dict(a.named_parameters()),dict(b.named_parameters()))
    save(args.output/'parameter_drift.json',drift)
    evaluate_case('prepared_noadapt',0)
    evaluate_case('prepared_adapt5',5)
    fixed=sorted(range(len(ds)),key=lambda i:hashlib.sha256(('123:'+ds.data_list[i]['id']).encode()).hexdigest())[:args.samples]
    diagnostics=[]
    for index in fixed:
        source=ds[index];video=clean_video(source);named=OrderedDict(enc.named_parameters());targets=reference_targets(ref,video)
        packet0=detection_packet(model,enc,video)
        det0=detection_loss(model,packet0,source);gd=torch.autograd.grad(det0,tuple(named.values()),allow_unused=True)
        rec0,coverage=recovery_loss(model,enc,dec,video,targets,named,0)
        ga=torch.autograd.grad(rec0,tuple(named.values()),allow_unused=True)
        grad=alignment(gd,ga);del gd,ga
        fast=unroll(lambda p,s:recovery_loss(model,enc,dec,video,targets,p,s)[0],named,5,SPEC['inner_lr'],False)
        with torch.no_grad():
            packet5=detection_packet(model,enc,video,fast);det5=detection_loss(model,packet5,source)
            rec5,_=recovery_loss(model,enc,dec,video,targets,fast,0)
            held0,_=recovery_loss(model,enc,dec,video,targets,named,1000)
            held5,_=recovery_loss(model,enc,dec,video,targets,fast,1000)
            logit_delta=max(float((a-b).abs().max()) for a,b in zip(packet0['logits'],packet5['logits']))
            offset_delta=max(float((a-b).abs().max()) for a,b in zip(packet0['offsets'],packet5['offsets']))
        row=dict(video_id=video['metadata']['video_id'],tokens=video['feats'].shape[-1],source_annotations=len(source['labels']),
            gradient_alignment=grad,inner_parameter_change=relative_change(named,fast),
            detection_loss_before=float(det0.detach()),detection_loss_after=float(det5),
            same_mask_recovery_before=float(rec0.detach()),same_mask_recovery_after=float(rec5),
            held_mask_recovery_before=float(held0),held_mask_recovery_after=float(held5),
            maximum_logit_change=logit_delta,maximum_offset_change=offset_delta,coverage=coverage)
        diagnostics.append(row);save(args.output/'mechanism_samples.json',diagnostics)
        print('MECHANISM',args.shift,json.dumps(row),flush=True)
        del packet0,packet5,det0,det5,rec0,rec5,fast,source,video,targets
    assert tensor_digest(model.state_dict())==tensor_digest(saved['state_dict'])
    assert tensor_digest(ref.state_dict())==initial
    assert tensor_digest(dec.state_dict())==tensor_digest(saved['decoder'])
    report=dict(status='passed',scope='exploratory-source-only-diagnosis-not-model-selection',shift=args.shift,
        code=code_identity(),training_code=job['code'],training_job_sha256=sha(job_path),
        latest_checkpoint_sha256=sha(saved_path),checkpoint_epoch=saved['epoch'],source_manifest_sha256=sha(job['source_manifest']),
        results=results,parameter_drift=drift,mechanism_samples=diagnostics,
        source_preparation_delta_pp=100*(results['prepared_noadapt']['map']-results['original_noadapt']['map']),
        isolated_adaptation_delta_pp=100*(results['prepared_adapt5']['map']-results['prepared_noadapt']['map']),
        full_video_token_lengths=lengths,source_GT_for_diagnosis=True,target_GT_used=False,training_parameters_changed=False,
        CUDA_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    save(args.output/'complete.json',report);print('COMPLETE',args.shift,report['source_preparation_delta_pp'],report['isolated_adaptation_delta_pp'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shift',choices=['S5','S6'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=2);p.add_argument('--samples',type=int,default=8);main(p.parse_args())
