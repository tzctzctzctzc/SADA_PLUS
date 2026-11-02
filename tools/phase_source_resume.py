"""源头完整轮保存点接续；旧保存点必须能恢复历史最优模型。"""
from copy import deepcopy
import random
import numpy as np
import torch


def restore_source_head(head, optimizer, state, spec, index_sha):
    assert state['spec']==spec and state['cache_index_sha256']==index_sha
    history=deepcopy(state['history']);epoch=state['epoch']
    assert [x['epoch'] for x in history]==list(range(epoch+1))
    best=min(history,key=lambda x:(x['source_candidate_score'],x['epoch']))
    # 旧格式没有落盘历史best；仅best正是latest时才可精确接续。
    assert 'best_state_dict' in state or best['epoch']==epoch
    best_state=deepcopy(state.get('best_state_dict',state['state_dict']))
    head.load_state_dict(state['state_dict'],strict=True);optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_cpu'])
    if torch.cuda.is_available():torch.cuda.set_rng_state(state['rng_cuda'],0)
    return history,best['source_candidate_score'],best['epoch'],best_state,deepcopy(best['quality']),epoch+1
