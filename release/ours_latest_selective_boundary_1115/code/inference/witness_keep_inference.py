"""已定稿的单视频推理。输入原生解码候选和对应的目标 FPN，输出秒坐标、类别和最终分数。"""
import json
from pathlib import Path

import numpy as np
from witness_priority_readout import pre_nms_evidence, readout


def open_readout(shift, checkpoint_sha256, bank_root):
    """每个模型初始化一次。bank_root 指向该模型已归档的 banks 目录。"""
    here=Path(__file__).resolve().parent
    spec=json.loads((here/'witness_keep_final.json').read_text(encoding='utf8'))
    row=spec['shifts'][shift]
    if row['checkpoint_sha256']!=checkpoint_sha256:
        raise ValueError('输入检查点与定稿绑定不符')
    cached=Path(bank_root)/shift/row['bank_binding_digest']/'train'
    groups=[]
    for file in cached.glob('*.npz'):
        with np.load(file,allow_pickle=False) as part:
            name=part['ids'][0]
            if not np.all(part['ids']==name):
                raise ValueError('一个参照分片混入多个视频')
            groups.append((str(name),{k:part[k] for k in ('pos','neg','labels','levels')}))
    if len(groups)!=len({name for name,_ in groups}) or len(groups)!=row['bank_videos']:
        raise ValueError('源训练参照视频数不匹配')
    groups.sort(key=lambda pair:pair[0])
    bank={k:np.concatenate([g[k] for _,g in groups]) for k in ('pos','neg','labels','levels')}
    if len(bank['labels'])!=row['bank_pairs']:
        raise ValueError('源训练参照对数不匹配')
    def predict(model,video,maps,decoded):
        factor,_=pre_nms_evidence(model,video,maps,decoded,bank)
        return readout(model,video,maps,decoded,bank,priority=factor**row['priority_exponent'])
    return predict

