"""仅CPU源验证诊断：固定post-NMS预测的分类/端点关联Oracle，不作选模指标。

只把与GT已有tIoU>=.1的检测关联到GT；保留原分数、数量、重复框。
不是候选生成上界/可达性能，不重新NMS，不读取目标GT。
"""
import argparse,os,sys
from pathlib import Path
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha
from tools.phase_signals import pairwise_iou

def oracle_predictions(prediction,gt,mode):
 out={k:v.copy() for k,v in prediction.items()};changed=0
 for video in np.unique(prediction['video-id']):
  ids=np.flatnonzero(prediction['video-id']==video);ann=gt[str(video)]
  if not ann:continue
  boxes=np.stack([prediction['t-start'][ids],prediction['t-end'][ids]],-1)
  truth=np.array([a['segment'] for a in ann]);labels=np.array([a['label_id'] for a in ann])
  iou=pairwise_iou(boxes,truth)
  if mode!='class':iou=np.where(labels[None]==prediction['label'][ids,None],iou,-1.)
  match=iou.argmax(1);valid=iou[np.arange(len(ids)),match]>=.1;chosen=ids[valid];target=truth[match[valid]]
  if mode=='class':out['label'][chosen]=labels[match[valid]]
  if mode in ['start','both']:out['t-start'][chosen]=target[:,0]
  if mode in ['end','both']:out['t-end'][chosen]=target[:,1]
  changed+=len(chosen)
 return out,changed

def evaluate(job):
 from libs.evaluation import ANETdetection
 job=Path(job);cfg=read(job/'config.json');done=read(job/'complete.json')
 assert sha(job/'source_val.npz')==done['prediction_sha256']
 annotation=cfg['baseline']['annotation_for_offline_evaluation'];assert sha(annotation)==cfg['baseline']['annotation_sha256']
 db=read(annotation)['database'];gt={v:[a for a in d.get('annotations',[]) if a['label_id']<10] for v,d in db.items() if d['subset'] in cfg['src_val_split']}
 with np.load(job/'source_val.npz',allow_pickle=False) as p:pred={k:p[k] for k in p.files}
 result={}
 for mode in ['class','start','end','both']:
  proposal,changed=oracle_predictions(pred,gt,mode)
  assert np.array_equal(proposal['score'],pred['score']) and len(proposal['score'])==len(pred['score'])
  evaluator=ANETdetection(annotation,cfg['src_val_split'],num_workers=1,action_subsampling='10_majority_classes')
  ap,mean,*_=evaluator.evaluate(proposal,verbose=False)
  result[mode]=dict(map=float(mean),delta_source_pp=100*(float(mean)-done['source_map']),map_by_tiou=[float(x) for x in ap],associated_predictions=changed)
 report=dict(epoch=done['epoch'],source_map=done['source_map'],oracles=result,
  source_prediction_sha256=done['prediction_sha256'],complete_sha256=sha(job/'complete.json'),target_gt_used=False,
  frozen_rule='post_NMS_same_score_same_count_match_tIoU_ge_.1_class_or_same_class_endpoint_correction',
  scope='source-only mechanism diagnostic; not candidate recall upper bound or attainable model gain; never checkpoint selection')
 save(job/'source_oracle.json',report);print(job.parent.name,job.name,{k:round(v['delta_source_pp'],4) for k,v in result.items()},flush=True)

if __name__=='__main__':
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
 for job in sorted((Path(a.root)/'monitor/source_signals').glob('S*/epoch*/complete.json')):
  if not (job.parent/'source_oracle.json').exists():evaluate(job.parent)
