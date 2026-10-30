"""独立入口；预热暂停→源机制审计→GPU真实验收→严格恢复，默认不放行正式训练。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as protocol
from tools.endpoint_risk_protocol import assert_pair,assert_formal_ready,promote_checkpoint


def main(args):
    import torch
    from tools import baseline_engine as engine
    identity=protocol.code_identity()
    if args.mode=='promote':
        print(promote_checkpoint(args.checkpoint,args.calibration,args.output));return
    cfg=protocol.read(args.config)
    if cfg['baseline']['code']!=identity:raise ValueError('必须使用当前已提交的clean代码')
    control=Path(args.control_root)/'formal/control'/cfg['baseline']['shift']
    assert_pair(cfg,protocol.read(control/'config.json'))
    for path,key in [(cfg['dataset']['json_file'],cfg['baseline']['source_manifest_sha256']),
                     (cfg['prototype_capacity']['prepared_cache'],cfg['prototype_capacity']['prepared_cache_sha256'])]:
        if protocol.sha(path)!=key:raise ValueError('配对输入工件改变')
    os.chdir(args.data_root)
    if args.mode not in ('source_audit','preflight'):
        tmp=Path(cfg['baseline']['tmpdir']);tmp.mkdir(parents=True,exist_ok=True)
        for name in ('TMPDIR','TMP','TEMP'):os.environ[name]=str(tmp)
    if args.mode=='source_audit':
        from tools.audit_endpoint_risk_source import collect
        if args.device.startswith('cuda'):
            from tools.baseline_gpu import first_cuda_guard
            first_cuda_guard()
        result=collect(args.checkpoint,args.source_pool,args.output,args.margin,args.device,identity)
        print('SOURCE_GATE',result['status']);return
    if args.mode=='preflight':
        from tools.calibrate_prototype_geometry import cpu_source_preflight
        print(cpu_source_preflight(control));return
    if not args.device.startswith('cuda'):raise ValueError('训练/真实验收必须使用受管GPU')
    # 先检查正式门禁再取首CUDA准入；不会先初始化模型再报缺证据。
    if args.mode=='train':
        saved=torch.load(args.checkpoint,map_location='cpu') if args.checkpoint else {}
        acceptance=protocol.read(args.acceptance) if args.acceptance else None
        assert_formal_ready(cfg,saved,acceptance)
    if args.mode=='source_warmup':
        warm=cfg['endpoint_risk']['warmup_epochs']
        if not 0<warm<50 or warm%5:raise ValueError('预热需在显式冻结的源验证保存点暂停')
        if args.output.exists():raise ValueError('不覆盖已有预热目录')
        args.output.mkdir(parents=True);protocol.save(args.output/'config.json',cfg)
    from tools.baseline_gpu import first_cuda_guard
    guard=first_cuda_guard()
    if args.mode=='source_warmup':engine.train(cfg,args.output,pause_after=warm)
    elif args.mode=='train':
        if protocol.read(args.output/'config.json')!=cfg:raise ValueError('只能精确接续同一预热条目')
        engine.train(cfg,args.output,resume=args.checkpoint)
    elif args.mode=='evaluate':engine.evaluate(cfg,args.output,cfg['baseline']['annotation_for_offline_evaluation'])
    elif args.mode=='probe':
        # 完整专项probe需有真实source保存点/证据，不能用随机头捏造有效门控。
        from tools.endpoint_risk_probe import probe
        probe(cfg,args.checkpoint,args.output,identity,guard)
    protocol.save(args.output/(args.mode+'_gpu.json'),dict(guard=guard,guard_sha256=protocol.sha(guard),
        config_sha256=protocol.sha(args.config)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['preflight','source_warmup','source_audit','promote','probe','train','evaluate'])
    p.add_argument('--config',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--acceptance',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--margin',type=float,default=0.,help='只接受预先冻结的source风险差阈值；默认仅要求教师更好')
    p.add_argument('--device',default='cpu')
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--control-root',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    p.add_argument('--source-pool',type=Path,default=Path('/vepfs/vlms/qwen8b32b_source_0adb413_1030/frozen_candidates.json'))
    a=p.parse_args()
    for key in ('config','checkpoint','calibration','acceptance','output','data_root','control_root','source_pool'):
        if getattr(a,key) is not None:setattr(a,key,getattr(a,key).resolve())
    main(a)
