"""只在原A forward期间替换证据缓存；原始forward、参数及B缓存均保留。"""
import hashlib
import json
from pathlib import Path
import types

def _forward(self,*args,**kwargs):
    target=kwargs.get('target_videos',args[1]if len(args)>1 else ())
    if any(v['segments'].numel() or v['labels'].numel()for v in target):
        raise ValueError('目标训练标注不得进入A')
    legacy=self.cache
    try:
        self.cache=self._moe_target_cache
        return self._moe_original_forward(self,*args,**kwargs)
    finally:
        self.cache=legacy

def attach(branch,cache):
    assert not hasattr(branch,'_moe_original_forward'),'禁止重复安装'
    rows=[r for group in cache.values()for r in group]
    assert all(isinstance(r['participation_selected'],bool)for r in rows)
    assert len({r['candidate_id']for r in rows})==len(rows)
    for r in rows:
        q=r['class_probabilities']
        assert len(q)==10 and q.count(1.)==1 and sum(q)==1.
    before=tuple(branch.state_dict());parameters=tuple(id(p)for p in branch.parameters())
    branch._moe_original_forward=type(branch).forward
    branch._moe_target_cache=cache
    branch.forward=types.MethodType(_forward,branch)
    assert tuple(branch.state_dict())==before and tuple(id(p)for p in branch.parameters())==parameters
    return branch

def install(ctx,cache):
    for model in (ctx.model,ctx.model_ema.module):attach(model.rq3_capacity,cache)
    return ctx

def load_bound(config):
    binding=config['moe_signal'];path=Path(binding['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==binding['sha256']
    payload=json.loads(path.read_text());assert payload['target_GT_used']is False
    assert payload['hard_A_interface']is True
    assert payload['selected_count']==binding['selected_count']
    assert sum(r['participation_selected']for group in payload['by_video'].values()for r in group)==payload['selected_count']
    return payload['by_video']

def patch_builder(base):
    original=base.build
    def build(config):
        ctx=original(config)
        return install(ctx,load_bound(config))if config.get('moe_signal')else ctx
    base.build=build
    return original
