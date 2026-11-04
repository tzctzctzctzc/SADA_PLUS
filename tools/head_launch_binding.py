"""读取真实启动器凭据，不伪造缺失的旧格式_launch.json。"""
from pathlib import Path
from tools.baseline_protocol import read,sha


def launch_binding(root):
    root=Path(root)
    receipt=Path(str(root)+'_launch.json')
    launch=read(receipt)
    assert launch['root']==str(root)
    owner=int(launch['pid'])
    assert read(root/'state.json')['owner_pid']==owner
    acceptance=Path(launch['repair_acceptance'])
    assert sha(acceptance)==launch['repair_acceptance_sha256']
    proof=read(acceptance)
    assert proof['passed'] and proof['code']['commit']==launch['commit']
    assert proof['new_full_state_replay_bit_exact'] and proof['no_gate_tolerance_relaxed']
    return dict(owner_pid=owner,commit=launch['commit'],receipt=str(receipt),
        receipt_sha256=sha(receipt),repair_acceptance_sha256=sha(acceptance))
