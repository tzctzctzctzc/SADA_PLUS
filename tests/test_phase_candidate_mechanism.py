from tools.phase_candidate_mechanism import endpoint_oracles


def test_oracle_keeps_original_and_cannot_invent_out_of_support_correction():
    r=dict(label=0,point=10.,box=[5.,15.],refined=[4.,16.],student=[3.,17.],student_point=10.,student_available=True,
           view=dict(lo=0,width=30,ratio=1.,length=30))
    q=endpoint_oracles([r],[[5.,15.]],[0],30)
    assert q['raw_error']==q['candidate_oracle_error']==0 and q['learned_error']>0 and q['oracle_gate_positive_gain']>0
    q=endpoint_oracles([r],[[1.,19.]],[0],30)
    assert q['candidate_oracle_error']>0


def test_oracle_does_not_relabel_other_instances_or_use_unavailable_student():
    r=dict(label=0,point=10.,box=[5.,15.],refined=[4.,16.],student=[3.,17.],student_point=10.,student_available=False,
           view=dict(lo=0,width=30,ratio=1.,length=30))
    assert endpoint_oracles([r],[[5.,15.]],[1],30)['endpoints']==0
    q=endpoint_oracles([r],[[5.,15.]],[0],30)
    assert q['endpoints']==2 and q['oracle_eligible']==0
