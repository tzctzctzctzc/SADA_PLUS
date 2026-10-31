"""续用单连接已下载前缀，HTTP Range并行下载剩余原视频，不改变数据。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

URL='https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/CharadesEgo_v1_480.tar'
TOTAL=11375974400


def main(root,workers):
    destination=root/'CharadesEgo_v1_480.tar'
    partial=root/'CharadesEgo_v1_480.tar.part'
    chunks=root/'raw_range_chunks';chunks.mkdir(exist_ok=True)
    state=root/'raw_parallel_download.json'
    if destination.exists():
        if destination.stat().st_size!=TOTAL:raise ValueError('现有视频包大小不符')
        return
    prefix=partial.stat().st_size if partial.exists() else 0
    ledger=chunks/'plan.json'
    if ledger.exists():
        plan=json.loads(ledger.read_text())
        if partial.stat().st_size!=plan['prefix']:raise ValueError('下载前缀被另一个进程修改')
    else:
        if prefix>=TOTAL:raise ValueError('下载前缀大小异常')
        ranges=[(start,min(start+(64<<20),TOTAL)-1) for start in range(prefix,TOTAL,64<<20)]
        plan={'prefix':prefix,'total':TOTAL,'ranges':ranges,'url':URL}
        ledger.write_text(json.dumps(plan))
    started=time.monotonic()
    def fetch(item):
        start,end=item;path=chunks/('%012d.part'%start);length=end-start+1
        for attempt in range(6):
            have=path.stat().st_size if path.exists() else 0
            if have==length:return path
            if have>length:raise ValueError('分块大小异常')
            request=urllib.request.Request(URL,headers={'Range':'bytes=%d-%d'%(start+have,end)})
            try:
                with urllib.request.urlopen(request,timeout=45) as response:
                    expected='bytes %d-%d/%d'%(start+have,end,TOTAL)
                    if response.status!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError('服务器未返回请求的精确分块')
                    with path.open('ab') as out:
                        remaining=length-have
                        while remaining:
                            data=response.read(min(1<<20,remaining))
                            if not data:raise OSError('分块提前结束')
                            out.write(data);remaining-=len(data)
                return path
            except OSError:
                if attempt==5:raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i,_ in enumerate(executor.map(fetch,plan['ranges']),1):
            have=sum(p.stat().st_size for p in chunks.glob('*.part'))
            record={'phase':'downloading','pid':__import__('os').getpid(),'workers':workers,
                    'prefix_bytes':plan['prefix'],'downloaded_bytes':plan['prefix']+have,
                    'total_bytes':TOTAL,'seconds':time.monotonic()-started}
            state.write_text(json.dumps(record));print('RANGE_PROGRESS',i,record,flush=True)
    if partial.stat().st_size!=plan['prefix']:raise ValueError('合并前发现前缀被修改')
    with partial.open('ab') as out:
        for start,end in plan['ranges']:
            path=chunks/('%012d.part'%start)
            if path.stat().st_size!=end-start+1:raise ValueError('不完整分块')
            with path.open('rb') as source:
                for block in iter(lambda:source.read(8<<20),b''):out.write(block)
    if partial.stat().st_size!=TOTAL:raise ValueError('合并后总大小不符')
    digest=hashlib.sha256()
    with partial.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):digest.update(block)
    partial.rename(destination)
    record={'phase':'complete','total_bytes':TOTAL,'sha256':digest.hexdigest(),
            'seconds':time.monotonic()-started,'workers':workers}
    state.write_text(json.dumps(record));print('RAW_DOWNLOAD_COMPLETE',record,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();main(a.root,a.workers)
