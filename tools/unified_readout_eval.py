"""对融合版应用冻结的最强版推理设置；不重新选择目标参数。"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from tools.baseline_protocol import read,save,sha
from tools.proto_trident_best import MANIFEST,selected_uncertainty,rescore,native_readout
def evaluate_shift(args,shift):
    from tools import baseline_engine as base
    from libs.evaluation import ANETdetection
    from tools.native_ap_fast import install
    install();manifest=read(MANIFEST);setting=manifest['settings'][shift]
    reference=args.reference_root/shift;cfg=read(reference/'config.json')
    assert cfg['unified_prototype'] and cfg['baseline']['shift']==shift
    setting=dict(setting,checkpoint_sha256=sha(reference/'best.pt'),selected_epoch=read(reference/'termination.json')['decision']['best_epoch'])
    dataset=base.dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],False)
    order=[row['id'] for row in dataset.data_list]
    parameters=manifest['shared_readout_parameters'] if args.shared else setting['readout']
    columns={k:[] for k in ['video-id','t-start','t-end','label','score']};model=None
    if args.cached_root:
        original=read(args.cached_root/'final.json')['results'][shift]
        assert original['passed'] and original['checkpoint_sha256']==setting['checkpoint_sha256']
    else:
        from libs.models.SADA import make_meta_arch
        from libs.utils import fix_random_seed
        from libs.trainer.da_trainer import move_to_device
        from tools.trident_uncertainty import DistributionObserver
        fix_random_seed(123,include_cuda=args.device.startswith('cuda'))
        if args.device.startswith('cuda'):torch.cuda.manual_seed(123)
        saved=torch.load(reference/'best.pt',map_location='cpu')
        assert saved['epoch']==setting['selected_epoch'] and saved['config']==cfg
        model=make_meta_arch('sada',deepcopy(cfg),device=args.device).to(args.device)
        model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval();del saved
        before=base.tensor_digest(model.state_dict());sigma=model.test_nms_sigma
    count=len(order) if args.limit_videos is None else min(args.limit_videos,len(order))
    for index,vid in enumerate(order[:count]):
        if args.cached_root:
            file=args.cached_root/'features'/shift/(vid+'.pt');assert sha(file)==original['feature_hashes'][vid]
            record=torch.load(file,map_location='cpu')
        else:
            with torch.no_grad():
                video=dataset[index];video['segments']=video['segments'][:0];video['labels']=video['labels'][:0]
                video=move_to_device([video],args.device);meta=video[0]['metadata'];assert meta['video_id']==vid
                captures=[];model.test_nms_sigma=sigma
                hook=model.neck_source.register_forward_hook(lambda module,inputs,value:captures.append(value[0]))
                try:
                    with DistributionObserver(model,1) as observer:
                        model(video_list_src=video,video_list_tgt=video,mode='test',domain='target',curr_epoch=setting['selected_epoch'])
                        raw=observer.record
                finally:hook.remove()
                prediction,scores,ids=selected_uncertainty(model,raw,meta,setting['uncertainty'])
                cosine,logits=native_readout(model,captures[1],raw,ids)
                record=dict(segments=prediction['segments'],labels=prediction['labels'],uncertainty_scores=scores,
                    prototype_cosines=cosine,branch_classifier_logits=logits,source_prototype_ready=model.rq3_capacity.branch.source_prototype_ready.cpu().numpy())
        adjusted=rescore(record,parameters);columns['video-id'].extend([vid]*len(adjusted))
        columns['t-start'].extend(record['segments'][:,0].tolist());columns['t-end'].extend(record['segments'][:,1].tolist())
        columns['label'].extend(record['labels'].tolist());columns['score'].extend(adjusted.tolist())
    if model is not None:assert before==base.tensor_digest(model.state_dict())
    output=args.output_root/shift;base.checkpoint(output/'predictions.pt',columns)
    report=dict(shift=shift,videos=count,full_dataset=count==len(order),optimizer_steps=0,parameters=parameters,model_state_unchanged=True,
        checkpoint_sha256=setting['checkpoint_sha256'],predictions_sha256=sha(output/'predictions.pt'))
    if count==len(order):
        ap,mean,*_=ANETdetection(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],num_workers=1,action_subsampling='10_majority_classes').evaluate({k:np.asarray(v) for k,v in columns.items()},verbose=False)
        expected=None
        if expected is not None:assert abs(100*float(mean)-expected)<1e-7
        report.update(map_percent=100*float(mean),by_iou=np.asarray(ap).tolist(),selected_score_reproduced=expected is not None)
    save(output/'result.json',report);return report
