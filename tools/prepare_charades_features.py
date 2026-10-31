"""保留官方I3D float32数值，将npy转换为统一加载器的无压缩npz容器。"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def prepare(archive,annotations,output):
    identity=json.loads((annotations/'identity.json').read_text())
    if identity['interval_policy']!='clean' or identity['evaluation_protocol']!='clean_only':
        raise ValueError('仅接受用户批准的清洗后协议')
    metadata_path=annotations/'offline_evaluation.json'
    database=json.loads(metadata_path.read_text())['database']
    if hashlib.sha256(metadata_path.read_bytes()).hexdigest()!=identity['files'].get(str(metadata_path)):
        # identity使用生成时路径；迁移时以文件名解析原冻结记录。
        records={Path(p).name:digest for p,digest in identity['files'].items()}
        if hashlib.sha256(metadata_path.read_bytes()).hexdigest()!=records[metadata_path.name]:
            raise ValueError('清洗后评估标注改变')
    output.mkdir(parents=True,exist_ok=False);rows=[]
    with zipfile.ZipFile(archive) as z:
        names={Path(n).stem:n for n in z.namelist() if '/features/' in n and n.endswith('.npy')}
        missing=set(database)-set(names)
        if missing:raise ValueError('官方特征缺少视频: '+str(sorted(missing)[:10]))
        for index,(vid,meta) in enumerate(database.items()):
            with z.open(names[vid]) as stream:raw=stream.read()
            array=np.load(io.BytesIO(raw),allow_pickle=False)
            if array.ndim!=2 or array.shape[1]!=1024 or array.dtype!=np.float32 or not len(array):
                raise ValueError('不是官方T×1024 float32 I3D特征: '+vid)
            if not np.isfinite(array).all():raise ValueError('特征包含非有限值: '+vid)
            path=output/(vid+'.npz');np.savez(path,feats=array)
            rows.append(dict(video_id=vid,path=str(path.resolve()),shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),native_npy_sha256=hashlib.sha256(raw).hexdigest(),
                source_member=names[vid],subset=meta['subset']))
            if (index+1)%500==0:print('FEATURES',index+1,'/',len(database),flush=True)
    result=dict(status='passed',count=len(rows),feature_dim=1024,feature_dtype='float32',
        numerical_conversion=False,container_only='NPY to ZIP_STORED NPZ',annotation_identity_sha256=hashlib.sha256((annotations/'identity.json').read_bytes()).hexdigest(),files=rows)
    (output/'inventory.json').write_text(json.dumps(result,sort_keys=True));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=prepare(a.archive,a.inputs,a.output);print('FEATURE_INVENTORY_PASS',result['count'])
