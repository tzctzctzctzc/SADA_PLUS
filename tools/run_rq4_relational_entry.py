"""独立关系式辅助入口；不扩展旧formal24矩阵，不自动启动新实验。

用 --audit-only 做真实CPU预检；正式运行仍要求同代码真实数据验收报告。
"""
from tools.run_rq4_entry import main, audit_complete as _audit_complete, read
from pathlib import Path


def audit_complete(root):
    if read(Path(root)/'complete.json').get('schema') != 'rq4-relational-complete-v1':
        raise ValueError('not a relational result')
    return _audit_complete(root)


if __name__ == '__main__': main(relational=True)
