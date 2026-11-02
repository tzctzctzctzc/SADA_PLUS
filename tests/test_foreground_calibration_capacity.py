from copy import deepcopy
from tools.foreground_calibration_capacity import select_unconstrained


def test_unconstrained_reports_gain_even_when_it_costs_TP_and_ties_keep_more():
    cal=dict(grid=[[.1,.1],[.3,.5],[1.01,.1]],classes=[0],
        per_class={'0':dict(baseline=dict(tp=[100]*5),grid=[dict(map=.6,kept=5),dict(map=.6,kept=10),dict(map=.5,kept=20)])})
    a=select_unconstrained(cal)
    assert a['global_index']==1 and a['per_class']['0']['index']==1
    old=deepcopy(cal);select_unconstrained(cal);assert cal==old


def test_no_TP_per_class_fallback():
    cal=dict(grid=[[.1,.1],[1.01,.1]],classes=[0],per_class={'0':dict(baseline=dict(tp=[0]*5),grid=[dict(map=0,kept=5),dict(map=0,kept=20)])})
    assert select_unconstrained(cal)['per_class']['0']['index']==1
