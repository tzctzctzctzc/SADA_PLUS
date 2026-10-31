"""从SADA官方预处理包和Ai2公开视频包准备数据；不使用训练环境依赖。"""
import argparse
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__();self.action=None;self.fields={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':self.action=a['action']
        if tag=='input' and a.get('type')=='hidden':self.fields[a['name']]=a['value']


def download(opener,url,path):
    if path.exists():return
    partial=path.with_suffix(path.suffix+'.part')
    offset=partial.stat().st_size if partial.exists() else 0
    request=urllib.request.Request(url,headers={'Range':'bytes=%d-'%offset} if offset else {})
    with opener.open(request,timeout=40) as response:
        if 'text/html' in response.headers.get('Content-Type',''):raise ValueError('下载返回HTML，不能当数据包')
        append=offset>0 and response.status==206
        if not append:offset=0
        started=time.monotonic();count=offset;reported=count
        with partial.open('ab' if append else 'wb') as stream:
            while True:
                chunk=response.read(4<<20)
                if not chunk:break
                stream.write(chunk);count+=len(chunk)
                if count-reported>=128<<20:
                    print('DOWNLOAD',path.name,round(count/2**30,3),'GiB',flush=True);reported=count
    partial.rename(path)
    print('DOWNLOADED',path.name,count,round(time.monotonic()-started,1),'seconds',flush=True)


def main(root,raw):
    root.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    feature=root/'CharadesEgo_v1_processed.zip'
    if not feature.exists():
        with opener.open('https://drive.google.com/uc?export=download&id=1QATjT4H9XDPGL2dWQkn5nEq2wZYblNqO',timeout=30) as r:
            parser=DownloadForm();parser.feed(r.read().decode())
        if parser.action!='https://drive.usercontent.google.com/download':raise ValueError('官方公开下载入口改变')
        download(opener,parser.action+'?'+urllib.parse.urlencode(parser.fields),feature)
    if raw:download(opener,'https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/CharadesEgo_v1_480.tar',root/'CharadesEgo_v1_480.tar')
    records=[]
    for path in root.iterdir():
        if path.suffix not in ('.zip','.tar'):continue
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
        records.append(dict(path=str(path.resolve()),size=path.stat().st_size,sha256=digest.hexdigest()))
    (root/'download_manifest.json').write_text(json.dumps(dict(status='passed',files=records),sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--raw-videos',action='store_true')
    a=p.parse_args();main(a.output,a.raw_videos)
