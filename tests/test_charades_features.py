from pathlib import Path
from runpy import run_path
import hashlib
import io
import json
import zipfile
import numpy as np
import pytest

M=run_path(str(Path(__file__).resolve().parents[1]/'tools/prepare_charades_features.py'))


def test_container_conversion_exact_and_metadata_relocation(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    annotation=inputs/'offline_evaluation.json'
    annotation.write_text(json.dumps(dict(database={'a':dict(subset='third_training')})))
    identity=dict(interval_policy='clean',evaluation_protocol='clean_only',
        files={'/old/server/offline_evaluation.json':hashlib.sha256(annotation.read_bytes()).hexdigest()})
    (inputs/'identity.json').write_text(json.dumps(identity))
    array=np.arange(3*1024,dtype=np.float32).reshape(3,1024);buffer=io.BytesIO();np.save(buffer,array)
    archive=tmp_path/'features.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('CharadesEgo_v1_processed/features/a.npy',buffer.getvalue())
    output=tmp_path/'out';p=M['prepare'](archive,inputs,output)
    with np.load(output/'a.npz') as data:assert np.array_equal(data['feats'],array)
    assert p['count']==1 and not p['numerical_conversion']
    identity['evaluation_protocol']='raw';(inputs/'identity.json').write_text(json.dumps(identity))
    with pytest.raises(ValueError):M['prepare'](archive,inputs,tmp_path/'bad')
