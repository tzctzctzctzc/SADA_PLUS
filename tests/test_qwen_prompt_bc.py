import json
from copy import deepcopy
import pytest
from tools.qwen_prompt_examples import source_candidates,messages,prompt,DEFINITIONS
from tools.run_qwen_prompt_bc import jobs,shard,task_key
from tools.run_qwen_gt_oracle import LABELS,option_order


def row():return dict(candidate_id='a'*64,video_id='query',start=1.,end=2.,frame_paths=['query']*8,frame_sha256=['h']*8,actual_times=[1.1]*8,unique_frames=8)


def test_definitions_preserve_options_and_explain_broad_insert():
    text=prompt('a'*64)
    for letter,label in zip('ABCDEFGHIJKLMN',option_order('a'*64)):assert '%s: %s ('%(letter,label) in text
    assert 'put in' in DEFINITIONS['INSERT'] and 'single cutting motion' in DEFINITIONS['CUT']
    assert 'brief' in DEFINITIONS['WASH'] and 'A to N' in text


def test_source_selection_never_uses_target_or_validation_and_has_no_overlap():
    db={}
    for label in range(10):
        for k in range(4):db['s%d_%d'%(label,k)]=dict(subset='source_train',annotations=[dict(label_id=label,segment=[1.,3.5])])
    db['target']=dict(subset='target_train',annotations=[dict(label_id=0,segment=[1.,3.5])])
    db['validation']=dict(subset='source_val',annotations=[dict(label_id=0,segment=[1.,3.5])])
    db['ambiguous']=dict(subset='source_train',annotations=[dict(label_id=0,segment=[1.,3.5]),dict(label_id=40,segment=[2.,3.])])
    selected=source_candidates(db,['source_train'])
    assert len(selected)==30 and all(x['video_id'] not in ('target','validation','ambiguous') for x in selected)
    assert selected==source_candidates(dict(reversed(list(db.items()))),['source_train'])


def test_B_and_C_query_identical_but_references_separate_and_labeled(monkeypatch):
    monkeypatch.setattr('tools.qwen_prompt_examples.mixed_frame_grid',lambda *a:{})
    query=row();b=messages(query,[]);bank=[dict(row(),label=name,reason='visible action') for name in LABELS]
    c=messages(query,bank)
    bv=[x for x in b[0]['content'] if x['type']=='video'];cv=[x for x in c[0]['content'] if x['type']=='video']
    assert len(bv)==1 and len(cv)==11 and bv[0]==cv[-1]
    assert b[0]['content'][-1]==c[0]['content'][-1]
    assert all(x['max_pixels']==224*224 for x in cv[:-1]) and cv[-1]['max_pixels']==448*448
    with pytest.raises(ValueError):messages(dict(query,label=5),[])


def test_jobs_reuse_B_across_shifts_but_keep_C_source_banks_distinct():
    r=row();labels=[dict(candidate_id=r['candidate_id'],shift=s,label=5) for s in ('S1','S2')]
    result=jobs(dict(candidates=[r]),labels)
    assert len(result)==3 and sum(x['arm']=='B' for x in result)==1
    assert set(result[0])=={'arm','shift','row','job_id'} and all('label' not in x['row'] for x in result)
    assert len({x['job_id'] for x in result})==3
    assert all(shard(x['job_id']) in range(4) for x in result)
    with pytest.raises(ValueError):jobs(dict(candidates=[r]),labels+labels[:1])
