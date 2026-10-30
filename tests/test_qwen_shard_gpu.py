from contextlib import contextmanager
import json
import pytest

from tools.qwen_shard_gpu import acquire_four,verify_cards,MIN_FREE_MIB
from tools.baseline_protocol import sha


def test_four_card_acquisition_releases_every_partial_lock_on_busy_card(tmp_path):
    held=[]
    @contextmanager
    def slot(index,*args):
        if index==2:raise BlockingIOError('busy')
        held.append(index)
        try:yield dict(index=index)
        finally:held.remove(index)
    assert acquire_four([3,2,1,0],tmp_path,tmp_path,'head',slot_fn=slot) is None
    assert held==[]


def test_all_four_locks_remain_held_together_and_release_together(tmp_path):
    held=[]
    @contextmanager
    def slot(index,*args):
        held.append(index)
        try:yield dict(index=index,uuid=str(index))
        finally:held.remove(index)
    ctx,cards=acquire_four([3,2,1,0],tmp_path,tmp_path,'head',slot_fn=slot)
    assert held==[0,1,2,3] and [c['index'] for c in cards]==held
    ctx.close();assert held==[]
    with pytest.raises(ValueError):acquire_four([0,1,2],tmp_path,tmp_path,'head',slot_fn=slot)


def fixture_cards(tmp_path):
    cards=[]
    for i in range(4):
        p=tmp_path/f'{i}.json';p.write_text(json.dumps(dict(owner_pid=123,observed=dict(uuid=f'gpu-{i}'),minimum_free_mib=MIN_FREE_MIB)))
        cards.append(dict(index=i,uuid=f'gpu-{i}',lease=str(p),lease_sha256=sha(p)))
    return cards


def test_first_cuda_checks_each_uuid_parent_and_memory_but_allows_external_pid_changes(tmp_path):
    cards=fixture_cards(tmp_path);visible=','.join(c['uuid'] for c in cards)
    observe=lambda i:dict(uuid=f'gpu-{i}',free_mib=15000,external_pids=[9999])
    _,enough=verify_cards(cards,123,visible,observe);assert enough
    _,enough=verify_cards(cards,123,visible,lambda i:dict(uuid=f'gpu-{i}',free_mib=10000 if i==3 else 15000))
    assert not enough
    with pytest.raises(ValueError):verify_cards(cards,124,visible,observe)
    with pytest.raises(ValueError):verify_cards(cards,123,'wrong-order',observe)
    with pytest.raises(ValueError):verify_cards(cards,123,visible,lambda i:dict(uuid='wrong',free_mib=15000))
    with pytest.raises(ValueError):verify_cards(cards[:3],123,visible,observe)


def test_modified_per_card_lease_rejected(tmp_path):
    cards=fixture_cards(tmp_path);visible=','.join(c['uuid'] for c in cards)
    with open(cards[0]['lease'],'a') as f:f.write(' ')
    with pytest.raises(ValueError):verify_cards(cards,123,visible,lambda i:dict(uuid=f'gpu-{i}',free_mib=15000))
