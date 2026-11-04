"""固定真实batch，对原/固定累加窗口分别完整重放模型、EMA和三优化器。"""
import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity


def cpu_copy(value):
    import torch
    buffer=io.BytesIO();torch.save(value,buffer);buffer.seek(0)
    result=torch.load(buffer,map_location='cpu');buffer.close();return result


def worker(root):
    from tools.baseline_gpu import first_cuda_guard
    guard=first_cuda_guard()
    import torch
    from tools import baseline_engine as base
    from libs.trainer.da_trainer import move_to_device
    original=Path('/vepfs/vlms/trident_complete6_8fdd2ae_1104/S1_trident')
    cfg=read(original/'config.json');cfg['baseline']['code']=code_identity()
    ctx=base.build(cfg);saved=torch.load(original/'probe_state.pt',map_location='cpu');base.restore(ctx,saved)
    source=move_to_device(next(iter(ctx.train_loader_src)),ctx.model.device)
    target=move_to_device(next(iter(ctx.train_loader_tgt)),ctx.model.device)
    epoch=cfg['train_cfg']['warmup_adv'];initial=cpu_copy(base.capture(ctx,cfg,epoch,[]))
    evidence={}
    for fixed in (False,True):
        snapshots=[];steps=[];losses=[];max_diffs=[];reference=None
        for run in range(3):
            # 每次用独立磁盘等价副本，避免诊断器自身产生可变状态别名。
            base.restore(ctx,deepcopy(initial));ctx.model.cfg['head_replacement']['fixed_window_gradient']=fixed
            for opt in base.optimizers(ctx):opt.zero_grad(set_to_none=True)
            ctx.model.train()
            values=ctx.model(video_list_src=source,video_list_tgt=target,mode='train',curr_epoch=epoch)[0]
            loss=values['downstream_loss']*cfg['model']['loss_weights']['weight_downstream']
            if cfg['model'].get('use_adv_discriminator',True):
                for method,key in [('conditional_rev','loss_conditional_adv_rev'),('reverse','loss_adv_rev')]:
                    if method in cfg['model']['adv_method']:loss=loss+cfg['model']['loss_weights']['weight_adv']*values[key]
            loss.backward();grads={n:p.grad for n,p in ctx.model.named_parameters() if p.grad is not None}
            steps.append(base.tensor_digest(dict(loss=loss.detach(),grads=grads)));losses.append(float(loss))
            clip=cfg['model']['weight_clip_value']
            if clip>0:
                for module in list(ctx.model.disc_dom_adv)+list(ctx.model.disc_dom_conditional_adv):
                    for parameter in module.parameters():parameter.data.clamp_(-clip,clip)
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),cfg['train_cfg']['clip_grad_l2norm'])
            ctx.optimizer_disc.step();ctx.optimizer_gen_source_alignment.step();ctx.model_ema.update(ctx.model)
            for scheduler in base.schedulers(ctx):scheduler.step()
            state=base.capture(ctx,cfg,epoch,[]);snapshots.append(base.tensor_digest(state))
            if reference is None:reference=cpu_copy(state)
            else:
                diffs=[]
                for name,tensor in state['state_dict'].items():
                    old=reference['state_dict'][name]
                    current=tensor.detach().cpu()
                    if not torch.equal(old,current):
                        diffs.append(dict(name=name,max_abs=float((old.double()-current.double()).abs().max()),
                            unequal=int((old!=current).sum())))
                max_diffs.append(diffs[:20])
        evidence['fixed' if fixed else 'native']=dict(full_state_unique=len(set(snapshots)),
            loss_all_gradient_unique=len(set(steps)),losses=losses,full_state_SHA=snapshots,
            model_parameter_difference_samples=max_diffs)
    assert evidence['fixed']['full_state_unique']==evidence['fixed']['loss_all_gradient_unique']==1
    assert evidence['fixed']['losses']==evidence['native']['losses'],'修复不得改变相同状态下的前向loss'
    save(root/'model_replay_diagnosis.json',dict(passed=True,evidence=evidence,
        input_digest=base.tensor_digest((source,target)),original_probe_checkpoint_sha256=sha(original/'probe_state.pt'),
        guard=guard,guard_sha256=sha(guard),loaded_diagnostic_weights_not_formal_initialization=True,
        no_formal_training=True,commit=code_identity()['commit']))
    print(json.dumps(evidence),flush=True)


def run(root):
    import fcntl
    from tools.baseline_gpu import snapshot
    root.mkdir(exist_ok=False);identity=code_identity()
    for gpu in range(4):
        view=snapshot(gpu)
        if view['free_mib']<16384:continue
        lockpath=root/f'private_gpu{gpu}.lock';lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lease=root/'lease.json';save(lease,dict(owner_pid=os.getpid(),observed=view,lock=str(lockpath),minimum_free_mib=16384,commit=identity['commit']))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=view['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),
            SADA_BASELINE_LEASE=str(lease),SADA_BASELINE_LEASE_SHA256=sha(lease),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (root/'model.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker','--root',str(root)],
                cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'launch.json',dict(pid=p.pid,owner_pid=os.getpid(),code=identity))
        code=p.wait();lock.close();assert code==0,(root/'model.log').read_text()[-3500:];return
    raise RuntimeError('no free GPU for one real model replay diagnostic')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--worker',action='store_true');a=p.parse_args()
    worker(a.root) if a.worker else run(a.root)
