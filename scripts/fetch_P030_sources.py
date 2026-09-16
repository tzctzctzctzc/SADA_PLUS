import pathlib,json,requests,hashlib,concurrent.futures
root=pathlib.Path('/vepfs/vlms/1')
m=json.loads((root/'literature/PinPoint_source_manifest.json').read_text())
tree=json.loads((root/'literature/PinPoint_tree.json').read_text())['tree']
names=['standardized_results/bge_vl_mllm_s1_retrieval_results_licensed.json','standardized_results/bge_vl_mllm_s1_retrieval_results_licensed_reranked.json']
out=root/'data/PinPoint';out.mkdir(exist_ok=True)
def fetch(name):
 ent=next(e for e in tree if e['path']==name)
 url='https://raw.githubusercontent.com/pinterest/pinpoint-dataset/'+m['revision']+'/'+name
 rr=requests.get(url,timeout=(10,30));rr.raise_for_status();b=rr.content
 assert len(b)==ent['size']
 blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
 assert blob==ent['sha'],name
 path=out/pathlib.PurePosixPath(name).name;path.write_bytes(b)
 return {'upstream_path':name,'local_path':str(path.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'blob_sha1':blob,'url':url}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex: files=list(ex.map(fetch,names))
manifest={'revision':m['revision'],'files':files}
(root/'literature/PinPoint_candidates_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('verified candidate files',[(x['upstream_path'],x['bytes']) for x in files])
