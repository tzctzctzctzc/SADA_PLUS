"""训练旁路只读源评估：不可变checkpoint快照、完整AP、端点误差与可靠性。

GPU只读无annotations的source_val；源GT只在CPU评估器读取。
快照是原子替换latest/best的硬链接，不读取正在写的.tmp、不干预训练。
"""
import argparse,json,os,sys,time,subprocess,hashlib
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha

def pairwise_iou(boxes,gt):
 import numpy as np
 inter=np.maximum(0,np.minimum(boxes[:,None,1],gt[None,:,1])-np.maximum(boxes[:,None,0],gt[None,:,0]))
 return inter/np.maximum(1e-9,(boxes[:,1]-boxes[:,0])[:,None]+(gt[:,1]-gt[:,0])[None]-inter)

def seconds(boxes,metadata):
 if metadata['use_temporal_grid_feats']:
  boxes=(boxes*metadata['feat_stride']+.5*metadata['feat_num_frames'])/metadata['fps']
  return boxes.clamp(0,metadata['duration'])
 return boxes.clamp(0,metadata['video_length'])

def gpu(args):
 import torch,numpy as np
 from copy import deepcopy
 from libs.models.SADA import make_meta_arch
 from tools import baseline_engine as base
 from tools.phase_boundary import PhaseBoundary
 import torch.nn.functional as F
 checkpoint=Path(args.checkpoint);job=checkpoint.parent
 # 确认当前卡仍有余量；只启动一个旁路评估，允许与自有训练共享。
 row=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert row[0].strip()==os.environ['CUDA_VISIBLE_DEVICES'] and int(row[1])>=12288
 saved=torch.load(checkpoint,map_location='cpu');cfg=saved['config'];epoch=saved['epoch']
 assert all('annotations' not in d for d in read(job/'source_val_unlabeled.json')['database'].values())
 os.chdir('/vepfs/vlms/SADA_PLUS');torch.manual_seed(123);torch.cuda.manual_seed(123)
 def model(weights):
  m=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda()
  m.phase_boundary=PhaseBoundary(torch.zeros(10,4,512)).cuda();m.load_state_dict(weights,strict=True);m.eval();return m
 teacher=model(saved['ema_state_dict']);student=model(saved['state_dict']);del saved
 ds=base.dataset(cfg,job/'source_val_unlabeled.json',cfg['src_val_split'],False,True)
 columns={k:[] for k in ['video-id','t-start','t-end','label','score']};raw=[]
 captured={k:[] for k in ['f','c','r']};handles=[]
 for mod,key in [(teacher.neck_source,'f'),(teacher.cls_head,'c'),(teacher.reg_head,'r')]:
  handles.append(mod.register_forward_hook(lambda m,i,o,key=key:captured[key].append(o)))
 with torch.no_grad():
  for i in range(len(ds)):
   video=base.move_to_device([ds[i]],'cuda:0');v=video[0];assert v['segments'].numel()==v['labels'].numel()==0
   for k in captured:captured[k]=[]
   pred=teacher(video_list_src=video,video_list_tgt=video,mode='test',domain='source',curr_epoch=epoch)[0][0]
   for k,z in [('t-start',pred['segments'][:,0]),('t-end',pred['segments'][:,1]),('label',pred['labels']),('score',pred['scores'])]:columns[k].extend(z.cpu().tolist())
   columns['video-id'].extend([pred['video_id']]*len(pred['scores']))
   tf,tm=captured['f'][0];tc=captured['c'][0][0];tr=captured['r'][0][0]
   from tools.phase_boundary import direct
   sf,sm,sc,sr=direct(student,video,training=False)
   for l,(f,mask,c,r,stride) in enumerate(zip(tf,tm,tc,tr,teacher.fpn_strides)):
    score,label=c[0,:10].sigmoid().max(0);ids=mask[0,0].nonzero().flatten()
    ids=ids[score[ids].argsort(descending=True)[:256]]
    p=ids.float()*stride
    box=torch.stack([p-r[0,0,ids]*stride,p+r[0,1,ids]*stride],-1)
    sb=torch.stack([p-sr[l][0,0,ids]*stride,p+sr[l][0,1,ids]*stride],-1)
    _,ts,_=teacher.phase_boundary.dense(f[0,:,ids].T);_,ss,_=student.phase_boundary.dense(sf[l][0,:,ids].T)
    b=seconds(box,v['metadata']).cpu().numpy();sbox=seconds(sb,v['metadata']).cpu().numpy()
    anchor=seconds(torch.stack([p,p],-1),v['metadata'])[:,0].cpu().numpy()
    for j,idx in enumerate(ids.tolist()):
     raw.append(dict(video_id=pred['video_id'],level=l,point_index=idx,anchor=float(anchor[j]),label=int(label[idx]),score=float(score[idx]),
       teacher_box=b[j].tolist(),student_box=sbox[j].tolist(),teacher_scale=ts[j].cpu().tolist(),student_scale=ss[j].cpu().tolist()))
   print('SOURCE_SIGNALS',epoch,i+1,len(ds),flush=True)
 for h in handles:h.remove()
 np.savez(job/'source_val.npz',**{k:np.asarray(v) for k,v in columns.items()})
 save(job/'source_raw.json',raw)
 save(job/'gpu_complete.json',dict(epoch=epoch,checkpoint_sha256=sha(checkpoint),prediction_sha256=sha(job/'source_val.npz'),
  raw_sha256=sha(job/'source_raw.json'),videos=len(ds),source_only=True,target_gt_used=False,training=False))

def cpu(job):
 import numpy as np
 from libs.evaluation import ANETdetection
 job=Path(job);manifest=read(job/'snapshot.json');cfg=read(job/'config.json');receipt=read(job/'gpu_complete.json')
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert sha(job/'source_val.npz')==receipt['prediction_sha256'] and sha(job/'source_raw.json')==receipt['raw_sha256']
 annotation=cfg['baseline']['annotation_for_offline_evaluation'];assert sha(annotation)==cfg['baseline']['annotation_sha256']
 with np.load(job/'source_val.npz',allow_pickle=False) as p:prediction={k:p[k] for k in p.files}
 evaluator=ANETdetection(annotation,cfg['src_val_split'],num_workers=2,action_subsampling='10_majority_classes')
 ap,mean,*_=evaluator.evaluate(prediction,verbose=False)
 database=read(annotation)['database'];gt={v:d.get('annotations',[]) for v,d in database.items() if d['subset'] in cfg['src_val_split']}
 errors=[];wrong_class=0;unmatched=0;rows=read(job/'source_raw.json')
 for r in rows:
  # 按同一物理anchor与预测类别匹配最短源GT，再比较双方到同一GT的端点误差。
  same=[g for g in gt[r['video_id']] if int(g['label_id'])==r['label'] and g['segment'][0]<=r['anchor']<=g['segment'][1]]
  if not same:unmatched+=1;continue
  g=min(same,key=lambda g:g['segment'][1]-g['segment'][0]);truth=np.array(g['segment']);duration=max(1e-9,truth[1]-truth[0])
  te=np.abs(np.array(r['teacher_box'])-truth)/duration;se=np.abs(np.array(r['student_box'])-truth)/duration
  for side in range(2):errors.append(dict(side=side,video_id=r['video_id'],teacher_error=float(te[side]),student_error=float(se[side]),
    teacher_scale=r['teacher_scale'][side],student_scale=r['student_scale'][side],predicted_teacher_better=r['teacher_scale'][side]<r['student_scale'][side],actual_teacher_better=bool(te[side]<se[side])))
 diagnostic={}
 for side in range(2):
  subset=[x for x in errors if x['side']==side];selected=[x for x in subset if x['predicted_teacher_better']]
  def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None
  diagnostic[str(side)]=dict(count=len(subset),selected=len(selected),
    actual_teacher_better_fraction=float(np.mean([x['actual_teacher_better'] for x in subset])) if subset else None,
    selected_teacher_better_fraction=float(np.mean([x['actual_teacher_better'] for x in selected])) if selected else None,
    teacher_scale_error_correlation=corr([x['teacher_scale'] for x in subset],[x['teacher_error'] for x in subset]),
    teacher_mean_error=float(np.mean([x['teacher_error'] for x in subset])) if subset else None,
    student_mean_error=float(np.mean([x['student_error'] for x in subset])) if subset else None)
 ref=Path(cfg['phase_boundary_runtime']['reference']);refmap=read(ref/'evaluation/source_val.json')['map_0.1_to_0.5']
 save(job/'endpoint_rows.json',errors)
 result=dict(epoch=manifest['epoch'],source_map=float(mean),source_map_by_tiou=[float(v) for v in ap],reference_source_map=refmap,
  delta_source_pp=100*(float(mean)-refmap),endpoint_diagnostic=diagnostic,raw_count=len(rows),unmatched_or_wrong_class_count=unmatched,
  scope='complete_source_val_AP_and_stratified_dense_top256_per_level_endpoint_diagnostic; source_val_is_selection_set_not_independent_target',
  target_gt_used=False,checkpoint_sha256=manifest['checkpoint_sha256'],prediction_sha256=receipt['prediction_sha256'])
 save(job/'complete.json',result);print(json.dumps(result),flush=True)

def snapshot_file(source,out,entry):
 import torch
 entry=Path(entry);out=Path(out);source=Path(source)
 candidate=out/('pending_'+source.name);candidate.parent.mkdir(parents=True,exist_ok=True)
 if candidate.exists():candidate.unlink()
 os.link(source,candidate)
 state=torch.load(candidate,map_location='cpu');epoch=state['epoch'];cfg=state['config'];del state
 job=out/('epoch%03d'%epoch);job.mkdir(exist_ok=True)
 if not (job/'snapshot.json').exists():
  candidate.replace(job/'checkpoint.pt')
  save(job/'snapshot.json',dict(epoch=epoch,origin=str(source),checkpoint_sha256=sha(job/'checkpoint.pt'),immutable_atomic_inode=True))
  save(job/'config.json',cfg);save(job/'source_val_unlabeled.json',read(entry/'source_val_unlabeled.json'),sort_keys=False)
 else:candidate.unlink()
 return job

def snapshot(entry,out):
 entry=Path(entry);out=Path(out);found=[]
 for name in ['latest.pt','best.pt']:
  source=entry/name
  if not source.exists():continue
  found.append(snapshot_file(source,out,entry))
 return found

def watch(args):
 root=Path(args.root);output=root/'monitor/source_signals';seen=set();child=None
 for p in (root/'monitor/source_signal_preservation').glob('S*_*.pt_*'):
  shift=p.name.split('_')[0];snapshot_file(p,output/shift,root/'formal'/shift/'phase')
 # CPU supervisor始终保留每个新保存点；单GPU旁路评估串行，训练继续运行。
 while True:
  jobs=[]
  for e in sorted((root/'formal').glob('*/phase')):
   if (e/'latest.pt').exists():
    stamp=(e/'latest.pt').stat().st_ino
    if (str(e),stamp) not in seen:
     snapshot(e,output/e.parent.name);seen.add((str(e),stamp))
  for p in output.glob('S*/epoch*/snapshot.json'):
   if not (p.parent/'complete.json').exists():jobs.append(p.parent)
  jobs.sort(key=lambda p:(int(p.name[5:]),p.parent.name))
  if child and child.poll() is not None:
   if child.returncode:raise RuntimeError('source diagnostic subprocess failed; inspect '+str(current))
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1')
   with (current/'cpu.log').open('w') as log:
    p=subprocess.run([sys.executable,'-u','-B',__file__,'--stage','cpu','--job',str(current)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
   if p.returncode:raise RuntimeError('source CPU evaluation failed '+str(current))
   child=None
  jobs=[j for j in jobs if not (j/'complete.json').exists()]
  if child is None and jobs:
   current=jobs[0];observed=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
   cards=[x.split(',') for x in observed.splitlines()];eligible=[x for x in cards if int(x[2])>=12288]
   if eligible:
    card=min(eligible,key=lambda x:int(x[3]));env=dict(os.environ,CUDA_VISIBLE_DEVICES=card[1].strip(),PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with (current/'gpu.log').open('w') as log:child=subprocess.Popen([sys.executable,'-u','-B',__file__,'--stage','gpu','--checkpoint',str(current/'checkpoint.pt'),'--gpu',card[0].strip()],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  reports=[read(p) | dict(shift=p.parent.parent.name,path=str(p)) for p in output.glob('S*/epoch*/complete.json')]
  save(root/'monitor/source_signals_latest.json',dict(reports=reports,active_job=str(current) if child else None,child_pid=child.pid if child else None,target_gt_used=False))
  if not jobs and child is None and read(root/'state.json')['phase'] in ['complete','failed']:return
  time.sleep(10)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['watch','gpu','cpu'],required=True);p.add_argument('--root');p.add_argument('--checkpoint');p.add_argument('--gpu',type=int);p.add_argument('--job');a=p.parse_args()
 if a.stage=='gpu':gpu(a)
 elif a.stage=='cpu':cpu(a.job)
 else:watch(a)
