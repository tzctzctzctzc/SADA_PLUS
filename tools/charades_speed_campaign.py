"""原Charades保存点精确接续；全部训练/评估独立审计后原子移交双卡。"""
import json,os
from pathlib import Path
from tools.baseline_protocol import read,save,sha,code_identity,decision,termination

RELEASE=Path('/vepfs/vlms/dual_campaign_1101/charades_gpu_release.json')


def final_audit(root):
    import numpy as np
    import torch
    from libs.evaluation import ANETdetection
    plan=read(root/'plan.json');rows={}
    for arm in ['adversarial','no_adversarial']:
        e=root/'formal'/arm;cfg=read(e/'config.json');term=read(e/'termination.json');expected=termination(term['history'])
        metrics=[json.loads(v) for v in (e/'metrics.jsonl').read_text().splitlines() if v]
        assert [v['epoch'] for v in metrics]==list(range(1,expected['epoch']+1))
        assert all(v['target_gt_cls_loss']==v['target_gt_reg_loss']==v['target_gt_downstream_loss']==0 for v in metrics)
        best=torch.load(e/'best.pt',map_location='cpu');assert best['config']==cfg and best['epoch']==expected['best_epoch']
        assert all(bool(torch.isfinite(v).all()) for v in best['ema_state_dict'].values() if torch.is_tensor(v))
        manifest=read(e/'evaluation/manifest.json');assert manifest['checkpoint_sha256']==sha(e/'best.pt')
        artifacts={}
        for domain in ['source','target']:
            p=e/'evaluation'/(domain+'_predictions.json');result=read(e/'evaluation'/(domain+'_val.json'))
            assert result['GT_read_after_saved_predictions'] and result['predictions_sha256']==sha(p)
            assert result==manifest['results'][domain]
            artifacts[str(p)]=sha(p)
        receipt=read(root/'stage_receipts'/('evaluate_'+arm+'.json'))
        assert receipt['code']==plan['code'] and receipt['target_GT_for_training'] is False
        predictions=read(e/'evaluation/target_predictions.json');members=read(root/'evaluation_unlabeled.json')['database']
        assert {v['video_id'] for v in predictions}==set(members) and len(predictions)==len(members)
        assert all(not v.get('annotations') for v in members.values())
        columns={k:[] for k in ['video-id','t-start','t-end','label','score']}
        for item in predictions:
            count=len(item['scores']);columns['video-id'] += [item['video_id']]*count
            columns['t-start'] += [v[0] for v in item['segments']];columns['t-end'] += [v[1] for v in item['segments']]
            columns['label']+=item['labels'];columns['score']+=item['scores']
        evaluator=ANETdetection(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],num_workers=4,action_subsampling='all')
        _,score,*_=evaluator.evaluate({k:np.asarray(v) for k,v in columns.items()},verbose=False)
        result=read(e/'evaluation/target_val.json');assert abs(float(score)-result['map_0.1_to_0.5'])<1e-12
        rows[arm]=dict(status='passed',best_epoch=best['epoch'],stop_epoch=expected['epoch'],target=result,
            checkpoint_sha256=manifest['checkpoint_sha256'],artifacts=artifacts,evaluation_receipt_sha256=sha(root/'stage_receipts'/('evaluate_'+arm+'.json')))
        del best
    save(root/'final_independent_audit.json',dict(status='passed',count=2,code=plan['code'],results=rows,
        cache_sha256=sha(root/'prepared.json'),target_GT_for_training=False,offline_metric_recomputed=True))
    return rows


def main(args,runner):
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    runner.prepare_file_limit();old=args.resume_root.resolve();root=args.root.resolve();identity=code_identity()
    state=read(old/'state.json')
    for pid in [state['parent_pid']]+[v['pid'] for v in state.get('active',{}).values()]:
        p=Path('/proc')/str(pid)/'cmdline'
        if p.exists() and str(old).encode() in p.read_bytes():raise ValueError('旧自有监督/worker尚未退出')
    review=read(args.speed_review)
    if review['status']!='passed' or review['code']!=identity:raise ValueError('缺少当前代码真实GPU等价门禁')
    previous=read(old/'plan.json');assert previous['policy']==json.loads(json.dumps(runner.POLICY))
    root.mkdir(exist_ok=False)
    names=['prepared.json','evaluation_unlabeled.json','cache_independent_audit.json','participation_coverage.json',
        'teacher_model_identity.json','source_threshold.json','candidate_pool.json','source_val_proposals.json']
    for name in names:os.link(old/name,root/name)
    assert sha(root/'prepared.json')==review['cache_sha256']
    resumes={};bindings={}
    for arm in runner.POLICY['variants']:
        olde=old/'formal'/arm;e=root/'formal'/arm;e.mkdir(parents=True)
        saved=torch.load(olde/'latest.pt',map_location='cpu');cfg=read(olde/'config.json')
        assert saved['config']==cfg and decision(saved['history'])['reason']=='running'
        assert sha(olde/'latest.pt')==review['reports'][arm]['resume_checkpoint_sha256']
        os.link(olde/'config.json',e/'config.json');os.link(olde/'best.pt',e/'best.pt')
        rows=[json.loads(v) for v in (olde/'metrics.jsonl').read_text().splitlines() if v]
        rows=[v for v in rows if v['epoch']<=saved['epoch']]
        assert [v['epoch'] for v in rows]==list(range(1,saved['epoch']+1))
        (e/'metrics.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in rows))
        resumes[arm]=str(olde/'latest.pt');bindings[arm]=dict(epoch=saved['epoch'],checkpoint_sha256=sha(olde/'latest.pt'),config_sha256=sha(olde/'config.json'))
        del saved
    plan=dict(previous,code=identity,resume_from=resumes,original_frozen_config_retained=True,teacher_queries_repeated=0,
        execution_optimization='batched-metric-sync-plus-alias-preserving-EMA',gpu_release=str(RELEASE))
    save(root/'plan.json',plan)
    save(root/'predecessor.json',dict(root=str(old),original_code=previous['code'],resumes=bindings,
        preparation={str(old/n):sha(old/n) for n in names},speed_review=str(args.speed_review),speed_review_sha256=sha(args.speed_review),teacher_queries_repeated=0))
    save(root/'reused_acceptance.json',dict(original_code=previous['code'],original_acceptance=str(old/'acceptance.json'),
        original_acceptance_sha256=sha(old/'acceptance.json'),original_GPU_probes={v:sha(old/'probes'/v/'result.json') for v in runner.POLICY['variants']},
        current_speed_equivalence=str(args.speed_review),training_mathematics_changed=False))
    runner.tasks(root,[dict(variant=v) for v in runner.POLICY['variants']],'train',[2,3])
    runner.tasks(root,[dict(variant=v) for v in runner.POLICY['variants']],'evaluate',[2,3])
    results=final_audit(root)
    save(root/'final_audit.json',dict(status='passed',count=2,results=results))
    save(root/'state.json',dict(phase='complete',active={},parent_pid=os.getpid()))
    save(RELEASE,dict(status='released-after-independent-final-audit',gpus=[2,3],root=str(root),
        audit=str(root/'final_independent_audit.json'),audit_sha256=sha(root/'final_independent_audit.json')))
