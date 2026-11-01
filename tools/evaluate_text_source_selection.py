"""CPU源验证AP；只接收源GT文件，不打开全量目标标注库。"""
import argparse, importlib.util, os, sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.dyfadet_protocol import read,save,sha

def evaluate(annotation,prediction,manifest,output):
    import numpy as np
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source=read(annotation)['database'];inputs=read(manifest)['database']
    assert set(source)==set(inputs) and all('annotations' not in x for x in inputs.values())
    with np.load(prediction) as f:columns={k:f[k] for k in f.files}
    assert set(columns['video-id'])<=set(source)
    path=CODE/'libs/evaluation/evaluator_anet.py';s=importlib.util.spec_from_file_location('source_ap',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    result=m.ANETdetection(annotation,None,num_workers=2,action_subsampling='10_majority_classes').evaluate(columns,verbose=False)
    save(output,dict(map_0.1_to_0.5=float(result[1]),map_by_tiou=[float(x) for x in result[0]],
        videos=list(inputs),annotation=str(annotation),annotation_sha256=sha(annotation),
        prediction=str(prediction),prediction_sha256=sha(prediction),manifest_sha256=sha(manifest),
        evaluator_sha256=sha(path),target_GT_used=False,cuda_visible_devices=''))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--annotation',required=True);p.add_argument('--prediction',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    evaluate(a.annotation,a.prediction,a.manifest,a.output)
