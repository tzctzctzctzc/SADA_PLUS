"""只更换B表示/语义条件及运行目录，其余六向协议逐项核对。"""
from copy import deepcopy
import hashlib
from pathlib import Path
from common import ROOT,read,save,sha


def digest(root=ROOT):
    h=hashlib.sha256()
    for p in sorted((root/'code').rglob('*.py'),key=lambda p:p.relative_to(root).as_posix()):
        h.update(p.relative_to(root).as_posix().encode());h.update(p.read_bytes())
    h.update((root/'speed.py').read_bytes());h.update((root/'prototype_adapter.py').read_bytes())
    return h.hexdigest()


def prepare():
    if any((ROOT/'jobs').glob('*/train_claim.json')):raise ValueError('已有正式领取，不重建配置')
    for i in range(1,7):
        ref=ROOT/'reference'/f'S{i}'/'config.json';cfg=read(ref);before=deepcopy(cfg)
        cfg['uda_route'].update(b_variant='path8',b_evidence_mode='shared_joint_path',b_semantic_alpha=.1,
            b_same_view=True,alignment_weight=.1,source_anchor=True,b_state_transfer=False,
            b_vlm_gate=True,b_vlm_gate_threshold=.8,
            b_selective_boundary=True,selective_residual=.1,
            response_source_weight=.1,response_target_weight=.1,response_group_limit=64,
            response_queries_per_video=128,response_queue_capacity=4096)
        multi=read(ROOT/'reference/multivlm'/f'S{i}'/'config.json')
        for name in ('model','prototype_capacity','dataset','loader','opt','train_cfg','test_cfg','src_train_split','src_val_split','tgt_train_split','tgt_val_split'):
            assert cfg[name]==multi[name], (i,name)
        cfg['moe_signal']=deepcopy(multi['moe_signal'])
        payload=read(cfg['moe_signal']['path'])
        assert sha(cfg['moe_signal']['path'])==cfg['moe_signal']['sha256']
        assert not payload['target_GT_used'] and payload['hard_A_interface']
        rows=[r for group in payload['by_video'].values() for r in group]
        chosen=[r for r in rows if r['participation_selected']]
        assert len(chosen)==cfg['moe_signal']['selected_count']
        assert all(r['participation_selected']==(r['routing_score']>=.8) for r in rows)
        assert payload['legacy_B_cache']==cfg['prototype_capacity']['prepared_cache']
        assert payload['legacy_B_cache_sha256']==cfg['prototype_capacity']['prepared_cache_sha256']
        entry=ROOT/'jobs'/f'S{i}';cfg['baseline'].update(route_entry=str(entry),tmpdir=str(entry/'tmp'))
        for key in ('model','prototype_capacity','dataset','loader','opt','train_cfg','test_cfg',
                    'src_train_split','src_val_split','tgt_train_split','tgt_val_split'):
            assert cfg[key]==before[key],key
        if (entry/'config.json').exists():assert read(entry/'config.json')==cfg
        else:save(entry/'config.json',cfg)
        save(entry/'binding.json',dict(parent_config_sha256=sha(ref),seed=123,original_A_fixed=False,multivlm_A=True,moe_signal=cfg['moe_signal'],B_and_readout_fixed=True))


if __name__=='__main__':prepare()
