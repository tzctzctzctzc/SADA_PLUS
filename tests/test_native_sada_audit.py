import numpy as np


def test_original_point_assignment_matches_independent_numpy():
    import torch
    from tools.background_four_routes_math import source_symbols
    from tools.audit_native_sada import independent_labels
    import ast
    from pathlib import Path
    source=Path(__file__).parents[1]/'libs/models/SADA/meta_archs.py';tree=ast.parse(source.read_text(encoding='utf-8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SADA');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='label_points_single_video');ns=dict(torch=torch,F=torch.nn.functional);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    from types import SimpleNamespace
    owner=SimpleNamespace(num_class_orig=10,train_center_sample='radius',train_center_sample_radius=1.5,train_reg_distance=True)
    pts=np.column_stack((np.arange(20),np.zeros(20),np.full(20,100),np.ones(20))).astype(np.float32);gt=np.array([[2.,9.],[6.,13.],[6.,13.]],np.float32);labels=np.array([1,2,3])
    actual,_=ns['label_points_single_video'](owner,torch.from_numpy(pts),torch.from_numpy(gt),torch.from_numpy(labels).long())
    expected=independent_labels(pts,gt,labels,'radius',1.5,True);assert np.array_equal(actual.numpy(),expected)
    assert np.array_equal(independent_labels(pts,[],[],'radius',1.5,True),np.zeros((20,10),np.float32))
