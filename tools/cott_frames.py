"""使用真实视频 PTS 抽帧；全视频 2 fps 与精修局部 12 帧独立留痕。"""
import math
import subprocess
from pathlib import Path
from tools.cott_protocol import read,save,sha,digest

def probe(path):
    import json
    x=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=avg_frame_rate,duration,width,height,start_time:format=duration,start_time',
        '-of','json',str(path)],text=True))
    stream=x['streams'][0]
    return dict(duration=float(stream.get('duration',x['format']['duration'])),
        start_time=float(stream.get('start_time',0)),details=x)

def sample_times(duration,fps=2):
    times=[i/fps for i in range(math.ceil(duration*fps)) if i/fps<duration]
    if duration-times[-1]>.001: times.append(max(0.,duration-.001))
    return times

def extract(path,times,output):
    import av
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    record=output/'frames.json'
    request=dict(video=str(path),requested_times_sec=times)
    if record.exists():
        old=read(record)
        assert old['request']==request
        for f in old['frames']: assert sha(f['path'])==f['sha256']
        return old['frames']
    if times!=sorted(times) or not times: raise ValueError('frame requests must be sorted')
    container=av.open(str(path)); stream=container.streams.video[0]
    origin=float(stream.start_time or 0)*float(stream.time_base)
    if times[0]>1:
        container.seek(max(0,int((times[0]+origin-.5)/float(stream.time_base))),stream=stream,backward=True)
    previous=None; selected=[]; cursor=0
    def append_frame(t,frame,requested):
        file=output/f'pts_{frame.pts}.jpg'
        if not file.exists(): frame.to_image().save(file,quality=95)
        selected.append(dict(path=str(file.resolve()),time_sec=t,requested_time_sec=requested,
            pts=frame.pts,time_base=str(stream.time_base),sha256=sha(file)))
    for frame in container.decode(stream):
        if frame.pts is None: continue
        t=float(frame.pts*stream.time_base)-origin
        if previous is not None and t<=previous[0]: continue
        while cursor<len(times) and t>=times[cursor]:
            choice=(t,frame)
            if previous is not None and abs(previous[0]-times[cursor])<=abs(t-times[cursor]): choice=previous
            append_frame(*choice,times[cursor]); cursor+=1
        previous=(t,frame)
        if cursor==len(times): break
    if previous is None: raise ValueError('video has no decoded frame')
    while cursor<len(times): append_frame(*previous,times[cursor]); cursor+=1
    container.close()
    # Duplicate physical frames remain visible in refinement provenance, never called new observations.
    assert all(abs(f['time_sec']-f['requested_time_sec'])<.25 for f in selected), 'timestamp extraction drift'
    save(record,dict(request=request,frames=selected,unique_pts=len({f['pts'] for f in selected})))
    return selected

def full_stream(path,duration,output):
    frames=extract(path,sample_times(duration),output)
    unique=[]
    for f in frames:
        if not unique or f['pts']!=unique[-1]['pts']: unique.append(f)
    assert len(unique)>=2
    return unique

def refined_stream(path,lo,hi,output,count=12):
    return extract(path,[lo+(hi-lo)*i/(count-1) for i in range(count)],output)
