"""仅CPU：预先固定完整源留出视频，GT单独保存，VLM清单不含标注。"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.cott_protocol import *
from tools.cott_frames import probe,full_stream

def prepare(root,reference,video_root):
    import numpy as np
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU prepare requires hidden CUDA'
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
    selection={}; videos={}; refs={}; evaluation={}
    # Selection uses only unlabelled duration and identifiers. Freeze before reading any annotation.
    for i in range(1,7):
        s=f'S{i}'; entry=Path(reference)/'formal'/s
        db=read(entry/'source_val_unlabeled.json')['database']
        selection[s]=select_videos(db,s)
        for v in selection[s]:
            path=Path(video_root)/(v+'.MP4')
            assert path.exists() and not Path(str(path)+'.aria2').exists(), str(path)
            videos.setdefault(v,dict(video_id=v,path=str(path),metadata_duration=float(db[v]['duration']),shifts=[]))['shifts'].append(s)
        refs[s]=dict(entry=str(entry),complete_sha256=sha(entry/'complete.json'),
            unlabeled_sha256=sha(entry/'source_val_unlabeled.json'))
    save(root/'selection.json',dict(spec=SPEC,selection=selection,videos=list(videos.values()),code=head))
    for v,row in sorted(videos.items()):
        actual=probe(row['path']); row['probe']=actual
        row['duration']=actual['duration']
        assert abs(actual['duration']-row['metadata_duration'])<.5, ('video mismatch',v)
        assert abs(actual['start_time'])<.1
        row['video_sha256']=sha(row['path'])
        frames=full_stream(row['path'],row['duration'],root/'frames'/v/'full')
        row['frames']=frames
        print('PREPARED_VIDEO',v,len(frames),flush=True)
    manifest=dict(schema=SCHEMA,spec=SPEC,code=head,selection=selection,videos=list(videos.values()))
    no_annotations(manifest); save(root/'manifest.json',manifest)
    for s,ids in selection.items():
        entry=Path(refs[s]['entry']); c=read(entry/'complete.json')
        for name in ['source_val.json','source_val.npz']:
            assert sha(entry/name)==c['artifacts'][str(entry/name)], 'reference SHA changed'
        source=read(entry/'source_val.json')['database']
        train=read(entry/'source_train.json')['database']
        assert not set(ids)&set(train)
        gt=[dict(video_id=v,label=CLASSES[int(a['label_id'])],start=float(a['segment'][0]),end=float(a['segment'][1]))
            for v in ids for a in source[v]['annotations'] if 0<=int(a['label_id'])<10]
        z=np.load(entry/'source_val.npz',allow_pickle=False)
        pred=[]
        for j,v in enumerate(z['video-id']):
            if str(v) in ids:
                pred.append(dict(video_id=str(v),label=CLASSES[int(z['label'][j])],
                    start=float(z['t-start'][j]),end=float(z['t-end'][j]),confidence=float(z['score'][j])))
        evaluation[s]=dict(ground_truth=gt,teacher_predictions=pred,source_val_sha256=sha(entry/'source_val.json'),
            teacher_predictions_sha256=sha(entry/'source_val.npz'),reference=refs[s])
    save(root/'evaluation_inputs/source_only.json',evaluation)
    save(root/'plan.json',dict(schema=SCHEMA,code=head,spec=SPEC,manifest_sha256=sha(root/'manifest.json'),
        selection_sha256=sha(root/'selection.json'),evaluation_inputs_sha256=sha(root/'evaluation_inputs/source_only.json'),
        unique_videos=len(videos),memberships=sum(map(len,selection.values())),source_only=True,
        inference_must_not_read='evaluation_inputs',no_optimizer=True))
    save(root/'state.json',dict(phase='prepared',complete_videos=0,total_videos=len(videos)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--reference',default='/vepfs/vlms/dyfadet_so6_694c64a_1101')
    p.add_argument('--video-root',default='/vepfs/vlms/data/EPIC_KITCHENS/videos')
    a=p.parse_args();prepare(a.root,a.reference,a.video_root)
