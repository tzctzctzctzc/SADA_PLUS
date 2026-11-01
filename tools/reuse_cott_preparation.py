"""零有效生成的技术前序：只复用已完成CPU准备，失败响应保留在旧根。"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.cott_protocol import *

def reuse(root,previous):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    root=Path(root);previous=Path(previous);old=read(previous/'plan.json')
    assert not list((previous/'calls').glob('*/*/result.json')), 'not a zero-valid-generation technical predecessor'
    assert not list((previous/'videos').glob('*/complete.json'))
    assert sha(previous/'manifest.json')==old['manifest_sha256']
    assert sha(previous/'selection.json')==old['selection_sha256']
    assert sha(previous/'evaluation_inputs/source_only.json')==old['evaluation_inputs_sha256']
    manifest=read(previous/'manifest.json');no_annotations(manifest)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
    for row in manifest['videos']:
        for frame in row['frames']:assert sha(frame['path'])==frame['sha256']
    root.mkdir(parents=True,exist_ok=False)
    for name in ['selection.json','evaluation_inputs/source_only.json','model_identity.json']:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(previous/name,target)
    manifest['code']=code;manifest['spec']=SPEC
    save(root/'manifest.json',manifest)
    save(root/'prepared_reuse.json',dict(previous=str(previous),previous_code=old['code'],
        previous_plan_sha256=sha(previous/'plan.json'),previous_manifest_sha256=sha(previous/'manifest.json'),
        selection_sha256=sha(previous/'selection.json'),evaluation_inputs_sha256=sha(previous/'evaluation_inputs/source_only.json'),
        model_identity_sha256=sha(previous/'model_identity.json'),accepted_generations_reused=0,
        failed_attempt_sha256={str(p):sha(p) for p in (previous/'calls').glob('*/*/attempt_*.json')},
        reason='Explicit K-1 transition slots and precise format-retry feedback; no selection, visual evidence, model or GT changes',
        frame_paths_read_only_external=True))
    save(root/'plan.json',dict(old,code=code,spec=SPEC,manifest_sha256=sha(root/'manifest.json'),
        preparation_reuse_sha256=sha(root/'prepared_reuse.json')))
    save(root/'state.json',dict(phase='prepared',complete_videos=0,total_videos=len(manifest['videos'])))
    print('REUSED_PREPARATION',len(manifest['videos']),sha(root/'prepared_reuse.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--previous',required=True)
    a=p.parse_args();reuse(a.root,a.previous)
