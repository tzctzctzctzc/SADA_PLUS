"""Load the three published LAD calibration functions without altering them."""
import ast
from pathlib import Path
import numpy as np

def official_lad_functions():
    path=Path('/vepfs/vlms/1/third_party/LAD/src/agd.py')
    names={'compute_bias_matrix','select_attention_scores','sharpen_distribution'}
    tree=ast.parse(path.read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes}==names
    namespace={'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return {name:namespace[name] for name in names}
