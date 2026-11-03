from pathlib import Path
import sys
import ast
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.native_ap_fast import compute_average_precision_detection as fast


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('pred_dtype',[np.float32,np.float64])
@pytest.mark.parametrize('case',['ties','empty','mixed'])
def test_exact_native_all_four_outputs(dtype,pred_dtype,case):
    # 直接执行原文件的三个纯函数，避免Windows已有Numba/NumPy兼容问题影响收集。
    path=Path(__file__).resolve().parents[1]/'libs/evaluation/evaluator_anet.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
        ('compute_average_precision_detection','segment_iou','interpolated_prec_rec')]
    namespace={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    native=namespace['compute_average_precision_detection']
    gt=pd.DataFrame({'video-id':['a','a','b'],'t-start':np.array([0.,2.,1.],dtype=dtype),'t-end':np.array([2.,4.,3.],dtype=dtype)})
    pred=pd.DataFrame({'video-id':['a','a','a','b','missing'],'t-start':np.array([0.,0.,2.,1.,0.],dtype=pred_dtype),
        't-end':np.array([2.,2.,4.,2.,1.],dtype=pred_dtype),'score':np.array([.8,.8,.8,.7,.9],dtype=pred_dtype)})
    if case=='empty':pred=pred.iloc[:0]
    if case=='mixed':pred.loc[0,'t-start']=.11111111
    expected=native(gt,pred);actual=fast(gt,pred)
    assert type(actual)==type(expected)
    if isinstance(expected,tuple):assert all(np.array_equal(a,b,equal_nan=True) for a,b in zip(expected,actual))
    else:assert np.array_equal(expected,actual,equal_nan=True)
