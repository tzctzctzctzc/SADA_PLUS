"""修订只改变适应优化器；不改变候选/原型/融合，不能重跑旧完整对照。"""
import ast
import subprocess
from tools.baseline_protocol import CODE
from tools import tta_core as core


def test_revision_is_only_optimizer_and_identity_change():
    original=subprocess.check_output(['git','show','96cf7625c4f64eb25884cefe9342f667a9c0fa11:tools/tta_core.py'],cwd=CODE,text=True)
    current=(CODE/'tools/tta_core.py').read_text(encoding='utf-8')
    expected=original.replace("schema='prototype-episodic-tta-v1'","schema='prototype-episodic-tta-coverage-revision2-v1'")
    expected=expected.replace("optimizer='SGD-no-momentum'","optimizer='Adam-reset-each-video'")
    expected=expected.replace('torch.optim.SGD(list(named.values()),lr=lr)','torch.optim.Adam(list(named.values()),lr=lr)')
    expected=expected.replace('pseudo_threshold=.5','pseudo_threshold=.1')
    assert current==expected
    assert core.SPEC['learning_rates']==[1e-5,1e-4] and core.SPEC['steps']==[1,3]
    assert core.SPEC['pseudo_threshold']==.1 and core.SPEC['minimum_support']==2


def test_old_controls_are_referenced_and_parent_wait_is_cpu_only():
    tree=ast.parse((CODE/'tools/run_tta.py').read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    text=ast.unparse(main)
    assert 'waiting_route2_revision1_complete' in text and 'reused_banks' in text
    assert 'source_reference_bank_S1' not in text
    assert "target_evaluation_jobs=24" in text and 'reused_multiview=0' in text
