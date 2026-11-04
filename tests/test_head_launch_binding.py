from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.baseline_protocol import save,sha,read
from tools.head_launch_binding import launch_binding


def fixture(root):
    root.mkdir()
    proof=root.parent/'repair.json'
    save(proof,dict(passed=True,code=dict(commit='fixed'),
        new_full_state_replay_bit_exact=True,no_gate_tolerance_relaxed=True))
    save(root/'state.json',dict(owner_pid=123))
    save(str(root)+'_launch.json',dict(root=str(root),pid=123,commit='fixed',
        repair_acceptance=str(proof),repair_acceptance_sha256=sha(proof)))


def test_real_adjacent_launch_receipt_without_fake_legacy_file(tmp_path):
    root=tmp_path/'run';fixture(root)
    assert not (root/'_launch.json').exists()
    assert launch_binding(root)['owner_pid']==123


@pytest.mark.parametrize('field,value',[('root','wrong-root'),('pid',456),('commit','wrong-commit'),
    ('repair_acceptance_sha256','wrong-SHA')])
def test_reject_foreign_launch_owner_commit_or_proof(tmp_path,field,value):
    root=tmp_path/'run';fixture(root)
    receipt=Path(str(root)+'_launch.json');launch=read(receipt);launch[field]=value;save(receipt,launch)
    with pytest.raises(AssertionError):launch_binding(root)
