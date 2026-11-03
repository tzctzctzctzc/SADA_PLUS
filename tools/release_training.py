"""直接调用作者BaseTrainerDA.train；包装器只记工件，不替换其控制流程。"""
from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
import time

from tools.baseline_protocol import read,save,sha
from tools.release_data_protocol import release_decision,SCHEMA


class RecordingEvaluator:
    def __init__(self,ctx,entry,name,splits,annotation):
        from libs.evaluation import ANETdetection
        self.ctx,self.entry,self.name=ctx,entry,name
        self.splits=splits
        self.inner=ANETdetection(str(annotation),splits,num_workers=1,
            action_subsampling='10_majority_classes')

    def evaluate(self,results,**kwargs):
        import fcntl
        from tools.baseline_engine import checkpoint
        # 只限制独立CPU评测并发，不改变任何候选或AP算法。
        lock=None
        while lock is None:
            for i in range(2):
                candidate=(self.entry.parent/f'private_CPU_evaluation_{i}.lock').open('a')
                try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:candidate.close()
                else:lock=candidate;break
            if lock is None:time.sleep(5)
        started=time.monotonic()
        try:values=self.inner.evaluate(results,**kwargs)
        finally:lock.close()
        prefix=self.entry/'debug_evaluation'/f'epoch{self.ctx.release_epoch:03d}'/self.name
        prefix.parent.mkdir(parents=True,exist_ok=True)
        checkpoint(str(prefix)+'_predictions.pt',results)
        save(str(prefix)+'.json',dict(map_by_tiou=[float(x) for x in values[0]],
            map_0_1_to_0_5=float(values[1]),GT_splits=self.splits,prediction_splits=self.splits,
            prediction_count=len(results['score']),CPU_AP_seconds=time.monotonic()-started))
        return values


def train(cfg,entry):
    import torch
    import wandb
    from libs.trainer.base_trainer_da import BaseTrainerDA
    from tools import baseline_engine as base
    entry=Path(entry);ctx=base.build(cfg)
    save(entry/'runtime_config.json',ctx.cfg);save(entry/'seed_proof.json',ctx.cuda_seed_proof)
    if cfg.get('prototype_capacity'):
        save(entry/'detector_initialization.json',dict(sha256=ctx.initial_detector_sha,
            specification=ctx.model.rq3_capacity.specification,
            auxiliary_parameters=sum(p.numel() for p in ctx.model.rq3_capacity.parameters())))
    # 跟train_da.py一样：非训练loader generator=None，全部persistent_workers=True。
    def evaluation_loader(path,splits):
        return base.loader(base.dataset(cfg,path,splits,False),False,None,1,cfg['loader']['num_workers'],True)
    annotation=cfg['baseline']['annotation_for_offline_evaluation']
    ctx.train_loader_src2=evaluation_loader(cfg['dataset']['json_file'],cfg['src_train_split'])
    ctx.train_loader_tgt2=evaluation_loader(cfg['baseline']['target_manifest'],cfg['tgt_train_split'])
    ctx.val_loader_tgt=evaluation_loader(annotation,cfg['tgt_val_split'])
    for name,splits in [('src_train',cfg['src_train_split']),('src_val',cfg['src_val_split']),
                        ('tgt_train',cfg['tgt_train_split']),('tgt_val',cfg['tgt_val_split'])]:
        setattr(ctx,name+'_evaluator',RecordingEvaluator(ctx,entry,name,splits,annotation))
    ctx.valid_freq=ctx.test_freq=2;ctx.epochs=cfg['opt']['epochs']
    ctx.save_period=cfg['train_cfg']['save_period'];ctx.start_epoch=1;ctx.latest_best_epoch=-1
    ctx.pretraining=cfg['model'].get('pretraining',False)
    ctx.pretraining_epochs=cfg['model'].get('pretraining_epochs',0)
    ctx.discriminator_updates=0;ctx.logger=logging.getLogger('release_trainer')
    ctx.logger.setLevel(logging.INFO);ctx.logger.addHandler(logging.StreamHandler())
    history,rows=[],[]
    original_epoch,original_validation,original_test=ctx._train_epoch,ctx._valid_epoch,ctx._test_epoch
    def phase(name,epoch,seconds,**extra):
        with (entry/'phase_timings.jsonl').open('a') as out:
            out.write(json.dumps(dict(phase=name,epoch=epoch,seconds=seconds,**extra))+'\n')
    def epoch(epoch,*args,**kwargs):
        ctx.release_epoch=epoch;ctx.stream_records=[];ctx.first_gradient_norm=None
        count=ctx.checked_steps;torch.cuda.synchronize();started=time.monotonic()
        value=original_epoch(epoch,*args,**kwargs);torch.cuda.synchronize()
        assert ctx.checked_steps-count==cfg['baseline']['epoch_budget']['iterations_per_epoch']
        stream=entry/'streams'/f'epoch{epoch:03d}.json';save(stream,ctx.stream_records)
        metrics=ctx.train_metrics.result()
        row=dict(epoch=epoch,seconds=time.monotonic()-started,train_loss=float(metrics['t_final_loss']),
            source_gradient_norm=ctx.first_gradient_norm,data_stream_sha256=sha(stream),
            auxiliary={k[2:]:float(v) for k,v in metrics.items() if k.startswith('t_rq3_')})
        rows.append(row)
        (entry/'metrics.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows))
        print('RELEASE_EPOCH',json.dumps(row,allow_nan=False),flush=True)
        phase('train',epoch,row['seconds'])
        return value
    def validation(epoch,loader,domain='src'):
        started=time.monotonic()
        value=original_validation(epoch,loader,domain=domain)
        phase('validation',epoch,time.monotonic()-started,domain=domain)
        if domain=='src':
            recorded=float(value) if math.isfinite(value) else str(float(value))
            history.append([epoch,recorded]);rows[-1]['source_validation_loss']=recorded
            (entry/'metrics.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows))
        return value
    def checkpoint(mode,epoch):
        started=time.monotonic()
        base.checkpoint(entry/('best.pt' if mode=='best' else 'latest.pt'),base.capture(ctx,cfg,epoch,history))
        phase('checkpoint',epoch,time.monotonic()-started,mode=mode)
        if mode=='best':ctx.latest_best_epoch=epoch
    def test(epoch,loader,mode,domain='src'):
        started=time.monotonic();value=original_test(epoch,loader,mode,domain=domain)
        phase('debug_inference_and_AP',epoch,time.monotonic()-started,domain=domain,mode=mode)
        return value
    ctx._train_epoch=epoch;ctx._valid_epoch=validation;ctx._test_epoch=test;ctx.save_ckpt=checkpoint
    original_wandb_log=wandb.log
    def local_log(values,*args,**kwargs):
        with (entry/'release_debug_metrics.jsonl').open('a') as out:
            out.write(json.dumps(dict(epoch=kwargs.get('step'),values={k:float(v) for k,v in values.items()}))+'\n')
    wandb.log=local_log
    save(entry/'release_runtime.json',dict(schema=SCHEMA,raw_train_method='BaseTrainerDA.train',
        raw_train_file_sha256=sha(Path(__file__).resolve().parents[1]/'libs/trainer/base_trainer_da.py'),
        validation_every=2,early_stop_bad_validations=1,test_every=2,
        original_zero_positive_division=True,wandb_sink='local_JSON_only_no_network'))
    try:BaseTrainerDA.train(ctx)
    finally:wandb.log=original_wandb_log
    result=release_decision(history);assert result['reason']!='running'
    save(entry/'termination.json',dict(history=history,decision=result,initial_state_sha=ctx.initial_state_sha))
    assert (entry/'best.pt').exists(),'原版未产生合法best，保留现场，不替换停止规则'
