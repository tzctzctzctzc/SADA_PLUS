"""第三人称全部评估与独立审计完成后，才把2/3卡加入现有TTT队列。"""
from pathlib import Path
from tools.baseline_protocol import read,sha


def available_indices(base, release_path):
    if not release_path or not Path(release_path).exists():return list(base)
    receipt=read(release_path)
    if receipt['status']!='released-after-independent-final-audit' or receipt['gpus']!=[2,3]:
        raise ValueError('GPU移交凭证不合法')
    path=Path(receipt['audit'])
    if sha(path)!=receipt['audit_sha256']:raise ValueError('移交审计SHA变化')
    audit=read(path);state=read(Path(receipt['root'])/'state.json')
    if audit['status']!='passed' or audit['count']!=2 or state['phase']!='complete' or state['active']:
        raise ValueError('第三人称任务尚未完成完整评估审计')
    return sorted(set(base)|{2,3})
