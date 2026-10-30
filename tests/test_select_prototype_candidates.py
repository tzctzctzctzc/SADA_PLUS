from tools.select_prototype_candidates import limit_per_video,decode_candidates


def test_original_hash_budget_order_is_preserved():
    import hashlib
    from collections import defaultdict
    rows=[dict(video_id='v'+str(i//40),candidate_id='c'+str(i)) for i in range(400)]
    stable=lambda x:sorted(x,key=lambda r:hashlib.sha256(('6:'+r['candidate_id']).encode()).hexdigest())
    grouped=defaultdict(list)
    for r in stable(rows):grouped[r['video_id']].append(r)
    expected=stable([r for v in sorted(grouped) for r in grouped[v][:30]])[:200]
    assert limit_per_video(rows)==expected
    assert limit_per_video(list(reversed(rows)))==expected


def test_decoding_preserves_seconds_level_choice_and_does_not_read_gt():
    v=dict(metadata=dict(video_id='v',fps=30.,feat_stride=16.,feat_num_frames=32,duration=100.),segments='forbidden',labels='forbidden')
    levels=[(0,[[10,0,4,1]],[.6],[2],[[2,3]]),(1,[[10,2,8,2]],[.8],[3],[[1,2]])]
    row=decode_candidates(v,levels)[0]
    assert row['source_fpn_levels']==[1] and row['teacher_top1_class']==3
    assert row['proposal_start_sec']==(8*16+16)/30
    assert row['proposal_end_sec']==(14*16+16)/30
    assert not any('gt' in key for key in row)
