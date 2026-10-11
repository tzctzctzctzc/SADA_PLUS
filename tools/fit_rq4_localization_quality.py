"""源域CPU质量拟合/校准。冻结既有UDA的EMA；从不构造target dataset。"""
import argparse
import copy
from pathlib import Path
import torch
from tools.rq4_consensus import read, save, audit_source_table
from tools.rq4_artifacts import reference
from tools.rq4_protocol import digest, file_sha


def collect(baseline_config,checkpoint,data_root,output,*,per_video):
    from libs.datasets import make_dataset
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.rq4_quality import descriptor,source_targets
    from tools.rq2_evaluate import load_checkpoint, selected_checkpoint_state
    cfg=read(baseline_config)
    if cfg.get('rq4_boundary') or cfg.get('rq3_auxiliary') or cfg['seed']!=6:
        raise ValueError('quality needs an unmodified matched seed6 baseline')
    saved=load_checkpoint(checkpoint); _,state=selected_checkpoint_state(saved,cfg,True)
    if saved['rq2_run_signature']!=cfg['rq2_formal']['run_signature']:
        raise ValueError('source quality checkpoint/baseline mismatch')
    if per_video<6: raise ValueError('per-video budget must cover the six FPN levels')
    root=Path(data_root).resolve()
    for key in ('json_file','feat_folder'):
        cfg['dataset'][key]=str((root/cfg['dataset'][key]).resolve())
    database=read(cfg['dataset']['json_file'])['database']
    members={name:sorted(k for k,v in database.items() if v['subset'].lower() in
        {s.lower() for s in cfg[split]}) for name,split in
        (('train','src_train_split'),('validation','src_val_split'),('target','tgt_train_split'))}
    if set(members['train'])&set(members['validation']) or (
            set(members['train'])|set(members['validation']))&set(members['target']):
        raise ValueError('source quality membership overlap')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6)
        model=make_meta_arch(cfg['model_name'],cfg,device='cpu')
    model.load_state_dict(state,strict=True); model.eval().requires_grad_(False)
    table=dict(schema='rq4-quality-source-table-v1',shift=cfg['rq2_formal']['shift'],seed=6,
        target_gt_used=False,baseline_config=reference(baseline_config),checkpoint=reference(checkpoint),
        annotations=reference(cfg['dataset']['json_file']),features=[],per_video=per_video,
        extractor_sha256=file_sha(__file__),train_video_ids=members['train'],
        validation_video_ids=members['validation'])
    seen=set()
    for split in ('train','validation'):
        key='src_train_split' if split=='train' else 'src_val_split'
        dataset=make_dataset(cfg['dataset_name'],False,cfg[key],**cfg['dataset'])
        rows=[]
        for video in dataset:
            vid=str(video['metadata']['video_id'])
            if vid not in members[split]: raise ValueError('dataset unexpectedly contains non-source video')
            path=Path(cfg['dataset']['feat_folder'])/(str(cfg['dataset'].get('file_prefix') or '')+
                vid+cfg['dataset']['file_ext'])
            if str(path) not in seen: table['features'].append(reference(path)); seen.add(str(path))
            with torch.no_grad():
                # 与训练EMA相同target分支的定位输出，输入仍严格为source视频。
                raw=model.build_rq4_consensus_teacher([video])
                candidates=[]
                for level,(grid,mask,offset) in enumerate(zip(raw['points'],raw['masks'],raw['offsets'])):
                    ids=torch.where(mask.reshape(-1).bool())[0]
                    # 标签无关、确定性按grid均匀采样；不按source质量挑容易样本。
                    count=min(len(ids),max(1,per_video//len(raw['points'])))
                    ids=ids[torch.linspace(0,len(ids)-1,count).long()] if count else ids
                    for index in ids.tolist(): candidates.append((level,index))
                for level,index in candidates:
                    grid=raw['points'][level]; center=grid[index,0]; stride=grid[index,3]
                    off=raw['offsets'][level][0,index]
                    box=torch.stack((center-off[0]*stride,center+off[1]*stride))[None]
                    if box[0,1]<=box[0,0]: continue
                    x=descriptor(video['feats'],box,center[None],stride[None])
                    y=source_targets(box,video['segments'])
                    rows.append(dict(video_id=vid,level=level,index=index,x=x[0].tolist(),y=y[0].tolist()))
            print('QUALITY_SOURCE_EXTRACT',split,vid,len(rows),flush=True)
        table[split]=rows
    audit_source_table(table); save(output,table)
    return table


def fit(table_path,output,*,epochs,learning_rate,quantile):
    from libs.models.SADA.rq4_quality import fit_quality,calibrate
    if epochs<=0 or not 0<learning_rate<1: raise ValueError('explicit valid source fit budget required')
    table=read(table_path); audit_source_table(table)
    tensor=lambda split,key: torch.tensor([r[key] for r in table[split]],dtype=torch.float32)
    x,y=tensor('train','x'),tensor('train','y')
    model=fit_quality(x,y,epochs=epochs,learning_rate=learning_rate)
    with torch.no_grad(): prediction=model(tensor('validation','x'))
    truth=tensor('validation','y')
    calibration=calibrate(prediction,truth,[r['video_id'] for r in table['validation']],quantile=quantile)
    bundle=dict(schema='rq4-localization-quality-v1',shift=table['shift'],seed=6,target_gt_used=False,
        fit_split='source_train',calibration_split='source_validation',source_table=reference(table_path),
        train_video_ids=table['train_video_ids'],validation_video_ids=table['validation_video_ids'],
        code_sha256=file_sha(Path(__file__).parents[1]/'libs/models/SADA/rq4_quality.py'),
        fit=dict(epochs=epochs,learning_rate=learning_rate,hidden=128,selection='fixed-source-fit-budget'),
        calibration=calibration,state_dict={k:v.tolist() for k,v in model.state_dict().items()},
        source_validation_mae=(prediction-truth).abs().mean(0).tolist(),
        note='Source residual calibration only; no target reliability or positive-gain guarantee.')
    bundle['payload_sha256']=digest(bundle); save(output,bundle)
    return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    extract=sub.add_parser('collect')
    for name in ('baseline-config','checkpoint','data-root','output'): extract.add_argument('--'+name,required=True)
    extract.add_argument('--per-video',type=int,required=True)
    learn=sub.add_parser('fit')
    for name in ('table-path','output'): learn.add_argument('--'+name,required=True)
    learn.add_argument('--epochs',type=int,required=True); learn.add_argument('--learning-rate',type=float,required=True)
    learn.add_argument('--quantile',type=float,required=True)
    args=vars(p.parse_args()); mode=args.pop('mode')
    torch.set_num_threads(4)  # 仅此独立CPU进程，绝不继承进入学生训练。
    (collect if mode=='collect' else fit)(**args)


if __name__=='__main__': main()
