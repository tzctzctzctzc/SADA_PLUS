import numpy as np
from tools.calibrate_target_teacher import counts,choose


def test_calibration_precision_recall_and_empty_selection():
    c=np.array([.8,.6,.4]);g=np.array([True,False,True])
    x=counts(c,g,.5)
    assert (x['tp'],x['fp'],x['fn'])==(1,1,1) and x['f_half']==.5
    assert counts(c,g,.9)['f_half']==0


def test_macro_videos_not_pooled_and_all_trials_retained():
    a=(np.array([.8,.4]),np.array([True,False]));b=(np.array([.4]),np.array([True]))
    selected,trials=choose([a,b]);assert len(trials)==6
    assert selected['threshold'] in (.3,.4)
    assert np.isclose(trials[0]['macro_f_half'],(1.25/2.25+1)/2)
