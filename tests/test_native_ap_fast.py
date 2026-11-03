from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.native_ap_fast import compute_average_precision_detection as fast


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('case',['ties','empty','mixed'])
def test_exact_native_all_four_outputs(dtype,case):
    from libs.evaluation.evaluator_anet import compute_average_precision_detection as native
    gt=pd.DataFrame({'video-id':['a','a','b'],'t-start':np.array([0.,2.,1.],dtype=dtype),'t-end':np.array([2.,4.,3.],dtype=dtype)})
    pred=pd.DataFrame({'video-id':['a','a','a','b','missing'],'t-start':np.array([0.,0.,2.,1.,0.],dtype=dtype),
        't-end':np.array([2.,2.,4.,2.,1.],dtype=dtype),'score':np.array([.8,.8,.8,.7,.9],dtype=dtype)})
    if case=='empty':pred=pred.iloc[:0]
    if case=='mixed':pred.loc[0,'t-start']=.11111111
    expected=native(gt,pred);actual=fast(gt,pred)
    assert type(actual)==type(expected)
    if isinstance(expected,tuple):assert all(np.array_equal(a,b,equal_nan=True) for a,b in zip(expected,actual))
    else:assert np.array_equal(expected,actual,equal_nan=True)
