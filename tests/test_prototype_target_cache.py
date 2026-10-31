"""缓存隔离、原编码器采样等价、检索去重及固定分数的行为核验。"""
from copy import deepcopy
from pathlib import Path
from runpy import run_path
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from tools.prototype_target_cache import (metadata_only, selected_records, duration_levels,
    encode_segments, memory_support, rescore, specification)


def payload():
    rows=[]
    for i in range(200):
        rows.append(dict(candidate_id=str(i), video_id='train', participation_selected=i<40,
                         semantic_label='TAKE', class_probabilities=[.9]+[.1/9]*9,
                         source_fpn_levels=[0], proposal_start_sec=1., proposal_end_sec=2.))
    return dict(schema='prototype-capacity-evidence-v1', teacher='qwen3_vl_8b', target_gt_used=False,
                candidate_count=200, selected_count=40, by_video={'train':rows})


LABELS=('TAKE','PUT','WASH','OPEN','CLOSE','INSERT','TURN_ON','CUT','TURN_OFF','POUR')


def test_metadata_drops_annotations_without_changing_members_or_source():
    db={'b':dict(subset='target_val',duration=3.,annotations=[{'label':5}]),
        'a':dict(subset='target_val',duration=2.),'t':dict(subset='target_train',annotations=[])}
    before=deepcopy(db)
    x=metadata_only(db,['TARGET_VAL'])
    assert list(x['database'])==['b','a'] and all('annotations' not in r for r in x['database'].values())
    assert db==before


def test_only_original_forty_and_known_unique_classes_enter_memory():
    p=payload();p['by_video']['train'][0]['semantic_label']='ABSTAIN'
    p['by_video']['train'][1]['class_probabilities']=[.1]*10
    rows,rejected=selected_records(p,{'train'},LABELS)
    assert len(rows['train'])==38 and len(rejected)==2
    assert all(r['participation_selected'] for r in rows['train'])


def test_memory_uses_vlm_class_not_historical_detector_class():
    p=payload()
    for row in p['by_video']['train']:row['class_id']=8
    rows,_=selected_records(p,{'train'},LABELS)
    assert all(row['memory_label']==0 for row in rows['train'])


@pytest.mark.parametrize('bad',['oracle','GT','validation','budget','ids','label'])
def test_cache_identity_and_split_contamination_rejected(bad):
    p=payload();ids={'train'}
    if bad=='oracle':p['schema']='prototype-gt-oracle-cache-v1'
    if bad=='GT':p['target_gt_used']=True
    if bad=='validation':ids={'val'}
    if bad=='budget':p['by_video']['train'][40]['participation_selected']=True
    if bad=='ids':p['by_video']['train'][1]['candidate_id']='0'
    if bad=='label':p['by_video']['train'][0]['semantic_label']='PUT'
    with pytest.raises(ValueError):selected_records(p,ids,LABELS)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_rescore_absent_classes_and_zero_strength_exact(dtype):
    p=torch.tensor([0.,.0001,.2,.7,1.],dtype=dtype)
    assert torch.equal(rescore(p,torch.zeros_like(p)),p)
    assert torch.equal(rescore(p,torch.ones_like(p),0),p)
    q=rescore(p,torch.ones_like(p))
    assert torch.isfinite(q).all() and (q>=p).all() and (q<=1).all()
    assert q[0]==0 and q[-1]==1


def bank():
    return dict(embedding=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[-1.,0.]]),
                label=torch.tensor([0,0,0,1]),video=torch.tensor([0,0,1,2]),confidence=torch.ones(4))


def test_memory_one_vote_per_video_and_preserves_absent_class():
    query=torch.tensor([[1.,0.],[1.,0.]])
    b=bank();before={k:v.clone() for k,v in b.items()}
    result=memory_support(query,torch.tensor([0,9]),b)
    assert result[0]==pytest.approx((1.+np.exp(-5.))/2.,abs=1e-7)
    assert result[1]==0
    assert all(torch.equal(b[k],v) for k,v in before.items())


def test_duplicate_bank_entry_does_not_multiply_support():
    b=bank();q=torch.tensor([[1.,0.]])
    x=memory_support(q,torch.tensor([0]),b)
    for k in b:b[k]=torch.cat([b[k],b[k][:1].repeat((20,1) if b[k].ndim==2 else 20)])
    assert torch.equal(x,memory_support(q,torch.tensor([0]),b))


def test_similarity_can_improve_ranking_without_changing_labels_or_boundaries():
    q=torch.tensor([[1.,0.],[-1.,0.]])
    b=dict(embedding=torch.tensor([[1.,0.]]),label=torch.tensor([0]),video=torch.tensor([0]),confidence=torch.ones(1))
    original=torch.tensor([.60,.61]);label=torch.tensor([0,0])
    score=rescore(original,memory_support(q,label,b))
    assert score[0]>score[1] and original[0]<original[1]


def test_duration_assignment_matches_original_first_matching_range():
    segments=torch.tensor([[0.,.2],[0.,4.],[0.,8.],[0.,12.],[0.,40.]])
    levels=duration_levels(segments,[[0,4],[2,8],[4,16],[8,10000]])
    assert levels.tolist()==[0,0,0,1,3]


@pytest.mark.parametrize('chunk',[1,3,256])
def test_vectorized_sampler_matches_real_original_encoder_and_rng(chunk):
    module=run_path(str(Path(__file__).resolve().parents[1]/'libs/models/SADA/prototype_capacity.py'))
    branch=module['VLMProposalCoAdaptationBranch'](8,10).eval().requires_grad_(False)
    features=[torch.randn(1,8,51),torch.randn(1,8,27)]
    segments=torch.tensor([[-1.,.5],[1.,20.],[2.,24.],[33.,70.],[40.,41.]])
    levels=torch.tensor([0,1,0,1,0])
    before={k:v.clone() for k,v in branch.state_dict().items()};rng=torch.get_rng_state().clone()
    with torch.no_grad():
        old=branch.encode(features,[dict(segments=segments,levels=levels)],[1,2])['embedding']
        new=encode_segments(branch,features,segments,levels,[1,2],chunk)
    assert torch.allclose(new,old,rtol=1e-5,atol=2e-6)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in branch.state_dict().items())
    assert new.shape==(5,128)


def test_empty_queries_and_memory_have_no_effect():
    q=torch.empty(0,2);b=bank()
    assert memory_support(q,torch.empty(0,dtype=torch.long),b).shape==(0,)
    b={k:v[:0] for k,v in b.items()}
    assert torch.equal(memory_support(torch.ones(2,2),torch.tensor([0,1]),b),torch.zeros(2))


def test_mixed_fpn_levels_use_one_original_order_encoder_batch():
    module=run_path(str(Path(__file__).resolve().parents[1]/'libs/models/SADA/prototype_capacity.py'))
    branch=module['VLMProposalCoAdaptationBranch'](8,10).eval().requires_grad_(False)
    calls=[]
    hook=branch.encoder.register_forward_pre_hook(lambda m,args:calls.append(args[0].shape[0]))
    try:
        encode_segments(branch,[torch.randn(1,8,32),torch.randn(1,8,16)],
                        torch.tensor([[2.,12.],[3.,15.]]),torch.tensor([1,0]),[1,2])
    finally:hook.remove()
    assert calls==[2]  # 不能按层变成[1,1]，否则真实GPU的GEMM舍入路径不同。


def test_specification_has_no_training_or_test_cache_writes():
    s=specification()
    assert s['trainable_parameters']==0 and s['validation_cache_writes'] is False
    assert s['selected_candidates']==40 and s['boundaries_and_labels']=='unchanged'


def test_actual_runner_help_imports_and_parses_without_cuda():
    root=Path(__file__).resolve().parents[1]
    runner=root/'tools/run_prototype_target_cache.py'
    compile(runner.read_text(encoding='utf-8'),str(runner),'exec')
    result=subprocess.run([sys.executable,'-B',str(runner),'--help'],cwd=root,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    assert result.returncode==0,result.stdout
    assert '--stage' in result.stdout and '--output' in result.stdout


def test_repository_device_transfer_receives_video_batch():
    # 加载真实仓库函数，不用另写一个近似实现掩盖接口差异。
    # 函数本身没有外部依赖，独立编译它以避免Windows没有Linux NMS扩展。
    import ast
    path=Path(__file__).resolve().parents[1]/'libs/trainer/da_trainer.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
              if isinstance(n,ast.FunctionDef) and n.name=='move_to_device')
    namespace=dict(torch=torch)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    video=dict(feats=torch.ones(8,16),segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long),metadata={'video_id':'t'})
    result=namespace['move_to_device']([video],'cpu')[0]
    assert result.keys()==video.keys() and torch.equal(result['feats'],video['feats'])
    assert result is not video and result['metadata']['video_id']=='t'
