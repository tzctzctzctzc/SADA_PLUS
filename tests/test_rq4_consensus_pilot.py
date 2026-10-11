import copy
import pytest
from tools.run_rq4_consensus_pilot import pilot_decision,TEACHERS


def rows():
    return [dict(teacher=t,shift='S'+str(i),old=.5,new=.5) for i in range(1,4) for t in TEACHERS]


def test_gate_uses_combined_six_not_teacher_selection():
    data=rows();data[0]['new']=.6;data[1]['new']=.3
    assert not pilot_decision(data)['continue_remaining']
    assert pilot_decision(rows())['continue_remaining']


def test_gate_cannot_run_on_partial_or_duplicate_identities():
    with pytest.raises(ValueError):pilot_decision(rows()[:-1])
    data=rows();data[-1]=copy.deepcopy(data[0])
    with pytest.raises(ValueError):pilot_decision(data)


def test_gate_rejects_nonfinite_or_wrong_metric_scale():
    for value in [float('nan'),float('inf'),55.,-.1]:
        data=rows();data[0]['new']=value
        with pytest.raises(ValueError):pilot_decision(data)
