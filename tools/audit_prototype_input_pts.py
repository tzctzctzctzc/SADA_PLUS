"""固定哈希抽取每方向4条，仅核验现有RGB输入对应真实显示PTS，不查询VLM。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sys

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.prototype_refresh import SHIFTS
from tools.audit_prototype_chain import REFRESH


def check_video(items,video):
    import av
    import numpy as np
    from PIL import Image
    from tools.refresh_frames import pixel_hash
    with av.open(str(video)) as c:
        stream=c.streams.video[0];fps=float(stream.average_rate);base=float(stream.time_base)
        origin=stream.start_time
        pts=sorted(p.pts for p in c.demux(stream) if p.pts is not None)
        if len(pts)!=stream.frames or len(set(pts))!=len(pts):raise ValueError('显示样本时间索引不完整')
    if origin is None:origin=pts[0]
    result=[]
    for s,row,path in items:
        target=[row['proposal_start_sec']+(row['proposal_end_sec']-row['proposal_start_sec'])*(i+.5)/8 for i in range(8)]
        actual=[(pts[i]-origin)*base for i in row['frames']]
        for i,h in zip(row['frames'],row['frame_sha256']):
            with Image.open(path.parent/'frames'/row['candidate_id']/(str(i)+'.png')) as im:
                if pixel_hash(np.asarray(im.convert('RGB')))!=h:raise ValueError('保存RGB帧像素改变')
        result.append(dict(shift=s,candidate_id=row['candidate_id'],video_id=row['video_id'],fps=fps,
            intended_midpoints=target,actual_display_times=actual,
            max_absolute_error_sec=max(abs(a-b) for a,b in zip(actual,target)),
            chronological=all(a<b for a,b in zip(actual,actual[1:])),
            all_inside_proposal=all(row['proposal_start_sec']<=t<=row['proposal_end_sec'] for t in actual)))
    return result


def run(root):
    from collections import defaultdict
    from tools.refresh_frames import video_paths_by_id
    root.mkdir(parents=True,exist_ok=False);selected=[];grouped=defaultdict(list)
    original=read(REFRESH/'candidate_audit.json');videos=video_paths_by_id(Path('/vepfs/vlms/data/EPIC_KITCHENS/videos'))
    for s in SHIFTS:
        path=Path(original['shifts'][s]['manifest'])
        if sha(path)!=original['shifts'][s]['sha256']:raise ValueError('原评分输入身份改变')
        rows=read(path)['candidates']
        sample=sorted(rows,key=lambda x:hashlib.sha256((s+':'+x['candidate_id']).encode()).hexdigest())[:4]
        for row in sample:grouped[row['video_id']].append((s,row,path))
        selected += [dict(shift=s,candidate_id=x['candidate_id']) for x in sample]
    save(root/'plan.json',dict(code=code_identity(),selection='four per shift deterministic SHA; no GT',selected=selected,
        no_new_VLM_queries=True,no_training=True,input_manifest_audit_sha256=sha(REFRESH/'candidate_audit.json')))
    reports=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(check_video,items,videos[v]) for v,items in grouped.items()]
        for f in futures:reports+=f.result()
    save(root/'audit.json',dict(status='passed',count=len(reports),results=reports,
        max_absolute_error_sec=max(x['max_absolute_error_sec'] for x in reports),
        all_chronological=all(x['chronological'] for x in reports),
        all_inside_proposals=all(x['all_inside_proposal'] for x in reports),
        scope='24 frozen candidate sample; does not prove all 1200 video timelines'))
    print('PTS_AUDIT_COMPLETE',len(reports),max(x['max_absolute_error_sec'] for x in reports),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output.resolve())
