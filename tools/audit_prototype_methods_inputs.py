"""第三轮CPU审计：六方向真实输入/SHA/去GT/原版协议配对，不用合成数据冒充GPU验收。"""
import argparse
import json
import os
from pathlib import Path
import sys

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as p
from tools.run_prototype_capacity import paired_config,assert_pair
from tools.baseline_engine import dataset
from libs.models.SADA.prototype_methods import VARIANTS


def main(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('本审计必须禁CUDA')
    import torch
    torch.set_num_threads(2)
    os.chdir(args.data_root)
    identity=p.code_identity();prepared=p.read(args.prepared_root/'audit.json')
    inventory=p.read(args.baseline_root/'inputs/feature_inventory.json');changed=[]
    for item in inventory:
        path=Path(item['path']);stat=path.stat()
        if stat.st_size!=item['size'] or stat.st_mtime_ns!=item['mtime_ns']:
            if p.sha(path)!=item['sha256']:raise ValueError('真实特征SHA改变: '+str(path))
            changed.append(str(path))
    rows=[]
    for shift in ['S'+str(i) for i in range(1,7)]:
        baseline=args.baseline_root/'formal'/shift/'uda';cfg=p.read(baseline/'config.json')
        source=dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],True)
        target=dataset(cfg,cfg['baseline']['target_manifest'],cfg['tgt_train_split'],True,True)
        p.assert_no_target_gt([target[0]])
        assert source[0]['feats'].shape[0]==target[0]['feats'].shape[0]==2304
        assert p.sha(cfg['dataset']['json_file'])==cfg['baseline']['source_manifest_sha256']
        assert p.sha(cfg['baseline']['target_manifest'])==cfg['baseline']['target_manifest_sha256']
        item=prepared['entries'][shift];cache=Path(item['path'])
        assert p.sha(cache)==item['sha256'];proof=p.read(cache)
        assert proof['selected_count']==40 and proof['target_gt_used'] is False
        target_members=p.read(cfg['baseline']['target_manifest'])['database']
        assert set(proof['by_video'])<=set(target_members)
        for variant in VARIANTS:
            current=paired_config(cfg,variant,cache,identity,args.output.parent/'tmp')
            assert_pair(current,cfg)
            original=p.read(args.control_root/'formal/control'/shift/'config.json')
            assert original['prototype_capacity']['prepared_cache_sha256']==current['prototype_capacity']['prepared_cache_sha256']
            stripped=json.loads(json.dumps(current));stripped.pop('prototype_capacity')
            stripped['baseline']['code']=original['baseline']['code'];stripped['baseline']['tmpdir']=original['baseline']['tmpdir']
            old=json.loads(json.dumps(original));old.pop('prototype_capacity')
            assert stripped==old
            rows.append(dict(shift=shift,variant=variant,status='passed',source_videos=len(source),target_videos=len(target),
                source_manifest_sha256=p.sha(cfg['dataset']['json_file']),target_manifest_sha256=p.sha(cfg['baseline']['target_manifest']),
                prepared_sha256=p.sha(cache),reference_config_sha256=p.sha(args.control_root/'formal/control'/shift/'config.json'),
                target_gt_empty=True,student_seed=current['seed'],batch=current['loader']['batch_size'],specification=current['prototype_capacity']['specification']))
        print('REAL_INPUT_PAIR_PASS',shift,flush=True)
    p.save(args.output,dict(status='passed',count=12,code=identity,results=rows,
        feature_inventory_sha256=p.sha(args.baseline_root/'inputs/feature_inventory.json'),
        unchanged_feature_metadata=len(inventory)-len(changed),metadata_changed_but_sha_verified=changed,
        target_gt_used=False,gpu_acceptance=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline-root','control-root','prepared-root','output','data-root'):
        ap.add_argument('--'+key,type=Path,required=True)
    main(ap.parse_args())
