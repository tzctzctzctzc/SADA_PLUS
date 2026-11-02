"""PDA阶段语义、UTv2端点可靠性启发的SADA联合适配；不是原论文复现。

source GT仅约束动作区间；内部阶段用全部有序变长路径边缘化，禁止三等分伪GT。
target弱教师/裁剪缩放强学生显式坐标映射；两个端点分别比较已源监督的误差尺度。
"""
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from tools.baseline_protocol import assert_no_target_gt

SCHEMA='sada-prototype-candidate-phase-risk-v2'
SPEC=dict(schema=SCHEMA,hidden=128,tokens=24,phases=3,text_phases=4,
 text_cache='/vepfs/vlms/dyfadet_text_cache_e5d83cc_1102/text.pt',
 text_sha256='87ecb25a1331172171ddc30ee8878091d0e116ddbe2fe85b091f35b3065976cc',
 text_provenance='/vepfs/vlms/dyfadet_text_cache_e5d83cc_1102/provenance.json',
 text_provenance_sha256='89f84544cbadefc67a7ac8e81eae9db87b4da7055e89382a9e1e73fc874a9f14',
 text_author='previous_assistant_class_name_only_descriptions_CLIP_B16_not_paper_GPT4o',
 phase_alignment='logsumexp_all_start_middle_end_paths_variable_dwell_plus_reversed_order_negative',
 source_weight=.1,target_weight=.1,target_score=.5,target_nms=.6,target_cap=32,
 onset=5,target_ramp_epochs=5,crop_fraction=.8,view_lengths=[.9,1.,1.1],
 uncertainty='per_endpoint_source_supervised_Laplace_normalized_residual_NLL',
 endpoint_gate='same_candidate_normalized_error_teacher_less_than_student_per_side_and_phase_11class_agreement',
 source_boundary='Gaussian_edge_BCE_all_valid_points',boundary_search_fraction=.25,
 target_classification='teacher_class_and_phase_class_agree_positive_points_only',
 target_regression='phase_edge_refined_teacher_endpoints_separate_masks',
 target_phase='ordered_posterior_KL_on_physical_correspondences',
 precision='float32',initialization='mature_Prototype_EMA_extra_source_head_pretrain_and_joint_adaptation',
 source_head_epochs=15,source_head_lr=.0003,source_head_selection='equal_video_real_raw_candidate_class_NLL_plus_selected_endpoint_error_CPU',
 source_selection='original_complete_source_cls_plus_reg_EMA_every5_patience15_max50',
 adversarial='original_conditional_rev_unchanged',prototype='original128_frozen_VLM40of200_membership',
 inference='original_heads_NMS_voting_no_auxiliary_score_product',target_GT=False)
SPEC.update(scientific_revision=1,foreground='hierarchical_11class_action_or_background_log_probability',
 uncertainty='per_endpoint_candidate_absolute_error_prediction_not_calibrated_probability',
 legacy_dense_scale_auxiliary='source_raw_regression_Laplace_NLL_only_never_the_target_candidate_gate',
 source_candidate_curriculum='all_GT_and_two_nearby_intervals_plus_current_raw_proposals_with_source_GT_assignment',
 source_background='all_major10_GT_gaps_plus_real_unmatched_candidates_not_physical_background',
 candidate_error='log1p_absolute_endpoint_error_per_common_reference_duration_labels_detached',
 candidate_context=dict(points=7,radius_fraction=.125,temporal_convolutions=2,class_and_endpoint_role=True),
 candidate_search=dict(points=17,radius_fraction=.25,first_choice='exact_original_endpoint',tie='original_first'),
 source_boundary='Gaussian_auxiliary_only_never_forced_peak_snap',
 source_head_views='frozen_mature_weak_and_actual_strong_feature_views_cached_on_CPU',
 source_joint_views='additional_actual_strong_source_SGP_path_with_source_GT_and_common_reference_pair_calibration',
 source_pair_error='same_weak_anchor_mapped_to_strong_actual_student_endpoint_source_GT_error_targets_detached',
 source_quality_readiness='CPU_real_source_raw_candidates_nonnegative_refinement_gain_and_positive_accepted_teacher_gain_with_both_sides',
 target_regression='selected_candidate_teacher_endpoint_normalized_before_smoothL1_separate_masks')
SPEC.update(target_phase='ordered_posterior_KL_on_same_refined_physical_interval_with_second_class_verification',
            endpoint_representability='left_le_student_positive_point_le_right_separate_endpoint_masks')

def paired_config(old,identity,tmp):
 c=deepcopy(old);c['baseline'].update(code=identity,tmpdir=str(tmp));c['phase_boundary']=deepcopy(SPEC)
 assert_pair(c,old);return c

def assert_pair(c,old):
 p=deepcopy(c);assert p.pop('phase_boundary')==SPEC
 p.pop('phase_boundary_runtime',None)
 for k in ['code','tmpdir']:p['baseline'][k]=old['baseline'][k]
 assert p==old and c['seed']==123 and c['loader']['batch_size']==2

def path_partition(logp):
 """所有非空三阶段顺序路径的log partition；阶段长度可为任意正整数。"""
 assert logp.shape[-1]==3 and logp.shape[-2]>=3
 a=logp[...,0,0];b=torch.full_like(a,-1e4);c=b.clone()
 for t in range(1,logp.shape[-2]):
  a,b,c=a+logp[...,t,0],torch.logaddexp(a,b)+logp[...,t,1],torch.logaddexp(b,c)+logp[...,t,2]
 return c

def phase_posterior(logits):
 """前向后向精确边缘化，支持端点固定但内部阶段任意变长。"""
 lp=logits.log_softmax(-1);n=lp.shape[-2]
 a=torch.stack([lp[...,0,0],torch.full_like(lp[...,0,0],-1e4),torch.full_like(lp[...,0,0],-1e4)],-1);aa=[a]
 for t in range(1,n):
  prev=torch.cat([torch.full_like(a[...,:1],-1e4),a[...,:-1]],-1)
  a=torch.logaddexp(a,prev)+lp[...,t,:];aa.append(a)
 b=torch.stack([torch.full_like(a[...,0],-1e4),torch.full_like(a[...,0],-1e4),torch.zeros_like(a[...,0])],-1);bb=[b]
 for t in range(n-2,-1,-1):
  q=b+lp[...,t+1,:];nxt=torch.cat([q[...,1:],torch.full_like(q[...,:1],-1e4)],-1)
  b=torch.logaddexp(q,nxt);bb.append(b)
 return (torch.stack(aa,-2)+torch.stack(bb[::-1],-2)-a[...,-1,None,None]).exp()

def sample(x,coordinates,stride):
 """真实FPN线性采样；align_corners_false以格中心精确映射。x=C,T。"""
 t=x.shape[-1];g=2*(coordinates/stride+.5)/t-1
 grid=torch.stack([g,torch.zeros_like(g)],-1)[None,None]
 return F.grid_sample(x[None,:,None],grid,align_corners=False,padding_mode='zeros')[0,:,0].transpose(0,1)

def interval_tokens(x,box,stride,n=24):
 u=torch.linspace(0,1,n,device=x.device);return sample(x,box[0]+u*(box[1]-box[0]),stride)

def strong_view(video,step,index):
 """只变换预提取特征，不冒称像素增强；每个点y=(x-lo+.5)*ratio-.5。"""
 x=video['feats'];length=x.shape[-1];width=max(8,int(length*SPEC['crop_fraction']))
 lo=(step*37+index*13)%max(1,length-width+1);width=min(width,length)
 ratio0=SPEC['view_lengths'][(step+index)%3];n=max(8,int(width*ratio0));ratio=n/width
 y=dict(video);y['feats']=F.interpolate(x[:,lo:lo+width][None],size=n,mode='linear',align_corners=False)[0]
 y['segments']=x.new_empty((0,2));y['labels']=torch.empty(0,dtype=torch.long,device=x.device)
 return y,dict(lo=lo,width=width,ratio=ratio,length=n)

def map_coordinates(x,view,inverse=False):
 return (x+.5)/view['ratio']-.5+view['lo'] if inverse else (x-view['lo']+.5)*view['ratio']-.5

class PhaseBoundary(nn.Module):
 def __init__(self,text):
  super().__init__();self.register_buffer('text',text.float())
  self.visual=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128),nn.GELU())
  self.text_proj=nn.Linear(512,128,bias=False)
  self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,256,dropout=0.,batch_first=True),2)
  self.cross=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
  self.foreground=nn.Linear(128,1);self.edge=nn.Linear(128,2);self.scale=nn.Linear(128,2)
  self.phase_weight=nn.Linear(128,4)
  self.boundary_temporal=nn.Sequential(nn.Conv1d(128,128,3,padding=1),nn.GELU(),nn.Conv1d(128,128,3,padding=1),nn.GELU())
  self.boundary_compress=nn.Linear(7*128,128)
  self.error_model=nn.Sequential(nn.Linear(128*3+6,128),nn.GELU(),nn.Linear(128,1))
  nn.init.zeros_(self.error_model[-1].weight);nn.init.zeros_(self.error_model[-1].bias)
  self.register_buffer('boundary_position',torch.sin(torch.linspace(-1,1,7)[:,None]*torch.arange(1,129)[None]*.1))
 def dense(self,x):
  h=self.visual(x);return self.edge(h),F.softplus(self.scale(h))+.01,self.foreground(h).squeeze(-1)
 def sequence(self,x):
  # 类别×阶段文本全部独立保留，交叉注意力后再按阶段自适应加权。
  h=self.visual(x);pos=torch.linspace(0,1,h.shape[-2],device=h.device)
  h=h+torch.sin(pos[None,:,None]*torch.arange(1,129,device=h.device)[None,None,:]*.1)
  h=self.temporal(h);text=F.normalize(self.text_proj(self.text),dim=-1)
  key=text.flatten(0,1)[None].expand(h.shape[0],-1,-1)
  infused=h+self.cross(h,key,key,need_weights=False)[0]
  similarity=torch.einsum('nth,cph->nctp',F.normalize(infused,dim=-1),text)*10
  logits=similarity[...,:3]
  weight=self.phase_weight(infused).softmax(-1)
  scores=(similarity*weight[:,None]).mean(-2).sum(-1)
  ordered=path_partition(logits.log_softmax(-1))/x.shape[-2]
  # 联合前景/类别概率含显式背景类10，前景项不再在阶段softmax中被抵消。
  fg=self.foreground(infused).mean(-2)
  actions=F.logsigmoid(fg)+(scores+ordered).log_softmax(-1)
  return logits,torch.cat([actions,F.logsigmoid(-fg)],-1)
 def endpoint_error(self,x,reference,coordinates,label,side,stride,length,mask=None):
  from tools.phase_candidate_components import endpoint_error
  return endpoint_error(self,x,reference,coordinates,label,side,stride,length,mask)

def sequence_loss(head,tokens,labels):
 logits,scores=head.sequence(tokens);idx=torch.arange(len(labels),device=labels.device)
 positive=labels<10
 chosen=logits[idx[positive],labels[positive]]
 if not positive.any():return F.cross_entropy(scores,labels),chosen,scores
 normal=path_partition(chosen.log_softmax(-1))
 reverse=path_partition(chosen.flip(-1).log_softmax(-1))
 return F.cross_entropy(scores,labels)+(-normal/tokens.shape[1]).mean()+F.softplus((reverse-normal)/tokens.shape[1]).mean(),chosen,scores

def direct(model,videos,training=True):
 # 不调用带no_grad的缓存helper；学生SGP/FPN/回归塔保留完整梯度。
 x,m=model.preprocessing(videos,mode='train' if training else 'validation');x=model.dropout(x)
 f,m,_=model.backbone_source(x,m);f,m=model.neck_source(f,m)
 cls=model.cls_head(f,m)[0];reg=model.reg_head(f,m)[0]
 return f,m,cls,reg

def select_proposals(f,m,cls,reg,model,video_index,length):
 boxes=[];scores=[];labels=[];levels=[];indices=[]
 for l,(mask,logit,off,stride) in enumerate(zip(m,cls,reg,model.fpn_strides)):
  prob,label=logit[video_index,:10].detach().sigmoid().max(0)
  valid=mask[video_index,0].bool()&(prob>=SPEC['target_score']);ids=valid.nonzero().flatten()
  if not len(ids):continue
  # 先取每层足够多候选再做跨层同类NMS，不按GT挑点。
  ids=ids[prob[ids].argsort(descending=True)[:SPEC['target_cap']*4]]
  p=ids.to(off.dtype)*stride;b=torch.stack([p-off[video_index,0,ids]*stride,p+off[video_index,1,ids]*stride],-1)
  ok=(b[:,0]>=0)&(b[:,1]<length)&((b[:,1]-b[:,0])>.05)
  for j in ok.nonzero().flatten().tolist():
   boxes.append(b[j]);scores.append(prob[ids[j]]);labels.append(label[ids[j]]);levels.append(l);indices.append(int(ids[j]))
 if not boxes:return []
 b=torch.stack(boxes).detach();s=torch.stack(scores);labels=torch.stack(labels);keep=[]
 for j in s.argsort(descending=True).tolist():
  if keep:
   k=torch.tensor(keep,device=b.device);inter=(torch.minimum(b[j,1],b[k,1])-torch.maximum(b[j,0],b[k,0])).clamp_min(0)
   iou=inter/((b[j,1]-b[j,0])+(b[k,1]-b[k,0])-inter).clamp_min(.01)
   if ((labels[k]==labels[j])&(iou>SPEC['target_nms'])).any():continue
  keep.append(j)
  if len(keep)==SPEC['target_cap']:break
 return [(b[j],labels[j],levels[j],indices[j],s[j]) for j in keep]

def _source_base_loss(model,head,f,m,reg,videos):
 losses=[];seq=[];ys=[];stats=dict(source_instances=0,source_scale_error=0.)
 for vi,v in enumerate(videos):
  truth=v['segments'];labels=v['labels'];validgt=(labels<10)&((truth[:,1]-truth[:,0])>.05)
  truth,labels=truth[validgt],labels[validgt]
  for box,c in zip(truth,labels):
   if box[0]<=.05 or box[1]>=v['feats'].shape[-1]-.05:continue
   level=int(torch.clamp(torch.log2((box[1]-box[0]).clamp_min(1)/8).round(),0,len(f)-1))
   seq.append(interval_tokens(f[level][vi],box,model.fpn_strides[level]));ys.append(c)
  stats['source_instances']+=len(truth)
  for l,(x,mask,offset,stride) in enumerate(zip(f,m,reg,model.fpn_strides)):
   valid=mask[vi,0].bool();p=torch.arange(x.shape[-1],device=x.device,dtype=x.dtype)*stride
   edge,scale,fg=head.dense(x[vi].T)
   if not len(truth):losses.append(F.binary_cross_entropy_with_logits(fg[valid],torch.zeros_like(fg[valid])));continue
   d=truth[:,1]-truth[:,0];inside=(p[:,None]>=truth[None,:,0])&(p[:,None]<=truth[None,:,1])
   duration=d[None].expand(len(p),-1).masked_fill(~inside,float('inf'));short,which=duration.min(-1);positive=short.isfinite()&valid
   gaussian=torch.exp(-.5*((p[:,None,None]-truth[None])/torch.maximum(d*.1,d.new_full(d.shape,stride)) [None,:,None])**2).amax(1)
   losses.append(F.binary_cross_entropy_with_logits(edge[valid],gaussian[valid]))
   losses.append(F.binary_cross_entropy_with_logits(fg[valid],inside.any(-1)[valid].float()))
   if positive.any():
    ids=positive.nonzero().flatten();gt=truth[which[ids]];pred=torch.stack([p[ids]-offset[vi,0,ids]*stride,p[ids]+offset[vi,1,ids]*stride],-1)
    err=((pred-gt)/(gt[:,1]-gt[:,0])[:,None]).abs();sc=scale[ids]
    # Laplace NLL同时学习误差尺度和定位；没有按分类分数冒充定位可靠性。
    losses.append(.1*(err/sc+torch.log(2*sc)).mean());stats['source_scale_error']+=float(err.detach().mean())
 if seq:
  tokens=torch.stack(seq);labels=torch.stack(ys).long();batch=[]
  for ids in torch.arange(len(labels),device=tokens.device).split(32):batch.append(sequence_loss(head,tokens[ids],labels[ids])[0])
  losses.append(torch.stack(batch).mean())
 return torch.stack(losses).mean(),stats

def source_loss(model,head,f,m,reg,videos,cls=None):
 from tools.phase_candidate_components import source_candidates_loss
 base,stats=_source_base_loss(model,head,f,m,reg,videos)
 extra,more=source_candidates_loss(model,head,f,m,reg,videos,cls)
 stats.update(more)
 return base+extra,stats

def target_loss(model,teacher,head,weak,strong,views,step):
 from tools.phase_candidate_components import candidate_target_loss
 return candidate_target_loss(model,teacher,head,weak,strong,views,step)

def install(ctx,cfg):
 from tools.baseline_protocol import sha
 assert cfg['phase_boundary']==SPEC and sha(SPEC['text_cache'])==SPEC['text_sha256']
 text=torch.load(SPEC['text_cache'],map_location='cpu')['embeddings'];assert text.shape==(10,4,512)
 model=ctx.model
 # 原成熟检测器作固定共同起点；新增源预训与联合预算单独披露。
 runtime=cfg['phase_boundary_runtime'];ref=Path(runtime['reference'])
 saved=torch.load(ref/'best.pt',map_location='cpu')
 model.load_state_dict(saved['ema_state_dict'],strict=True);ctx.model_ema.module.load_state_dict(saved['ema_state_dict'],strict=True)
 for mm in [model,ctx.model_ema.module]:
  for k,v in saved['ema_controls'].items():
   if torch.is_tensor(v):getattr(mm,k).data.copy_(v)
   elif isinstance(v,list):
    for a,b in zip(getattr(mm,k),v):a.data.copy_(b)
   else:setattr(mm,k,deepcopy(v))
 del saved
 model.phase_boundary=PhaseBoundary(text).to(model.device)
 if Path(runtime['source_head']).exists():
  head=torch.load(runtime['source_head'],map_location='cpu');assert head['spec']==SPEC
  model.phase_boundary.load_state_dict(head['state_dict'],strict=True)
 ctx.model_ema.module.phase_boundary=deepcopy(model.phase_boundary)
 for p in ctx.model_ema.module.phase_boundary.parameters():p.requires_grad_(False)
 for opt in [ctx.optimizer_gen_source,ctx.optimizer_gen_source_alignment]:
  for name,p in model.phase_boundary.named_parameters():opt.param_groups[0 if p.ndim>1 else 1]['params'].append(p)
 model.phase_boundary_state=dict(step=0,events=0,digest='0'*64)
 ctx.model_ema.module.phase_boundary_state=deepcopy(model.phase_boundary_state)
 captured=dict(active=False,reg=[],features=[],cls=[])
 ctx.phase_handles=[]
 for module,key in [(model.reg_head,'reg'),(model.neck_source,'features'),(model.cls_head,'cls')]:
  ctx.phase_handles.append(module.register_forward_hook(lambda m,i,o,key=key:captured[key].append(o) if captured['active'] else None))
 original=model.forward;ctx.phase_gradient_epoch=None
 def forward(*args,**kwargs):
  if kwargs.get('mode')!='train':return original(*args,**kwargs)
  assert_no_target_gt(kwargs['video_list_tgt']);state=model.phase_boundary_state;state['step']+=1
  for key in ['reg','features','cls']:captured[key]=[]
  captured['active']=True
  try:output=original(*args,**kwargs)
  finally:captured['active']=False
  assert len(captured['features'])==len(captured['reg'])==3
  f,m=captured['features'][0];reg=captured['reg'][0][0]
  loss,stats=source_loss(model,model.phase_boundary,f,m,reg,kwargs['video_list_src'],captured['cls'][0][0])
  source_strong=[];source_views=[]
  for i,v in enumerate(kwargs['video_list_src']):
   x,view=strong_view(v,state['step'],i);view['original_length']=v['feats'].shape[-1]
   x['segments']=map_coordinates(v['segments'],view);x['labels']=v['labels']
   source_strong.append(x);source_views.append(view)
  sf,sm,sc,sr=direct(model,source_strong)
  strong_loss,strong_stats=source_loss(model,model.phase_boundary,sf,sm,sr,source_strong,sc)
  from tools.phase_candidate_components import paired_source_error_loss
  paired,pair_stats=paired_source_error_loss(model,model.phase_boundary,(f,m,captured['cls'][0][0],reg),
                                           (sf,sm,sc,sr),source_views,kwargs['video_list_src'])
  loss=.5*(loss+strong_loss)+paired;stats.update(pair_stats)
  epoch=kwargs['curr_epoch'];total=SPEC['source_weight']*loss
  stats['source_loss']=float(loss.detach());stats['target_loss']=0.
  if epoch>=SPEC['onset']:
   strong=[];views=[]
   for i,v in enumerate(kwargs['video_list_tgt']):
    x,view=strong_view(v,state['step'],i);view['original_length']=v['feats'].shape[-1];strong.append(x);views.append(view)
   teacher=ctx.model_ema.module;teacher.eval()
   with torch.no_grad():weak=direct(teacher,kwargs['video_list_tgt'])
   strongdata=direct(model,strong)
   tl,ts=target_loss(model,teacher,model.phase_boundary,weak,strongdata,views,state['step'])
   ramp=min(1.,(epoch-SPEC['onset']+1)/SPEC['target_ramp_epochs'])
   total=total+SPEC['target_weight']*ramp*tl;stats.update(ts,target_loss=float(tl.detach()))
   if ts['target_phase_used'] and ctx.phase_gradient_epoch!=epoch:
    for key,parameters in [('target_sgp_gradient',tuple(model.backbone_source.parameters())),('target_regression_gradient',tuple(model.reg_head.parameters()))]:
     grad=torch.autograd.grad(tl,parameters,retain_graph=True,allow_unused=True)
     stats[key]=sum(float(g.detach().square().sum()) for g in grad if g is not None)**.5
    if stats['target_regression_gradient']>0:ctx.phase_gradient_epoch=epoch
   state['events']+=int(ts['target_phase_used']>0)
  if not torch.isfinite(total):raise ValueError('phase auxiliary loss nonfinite')
  output[0]['downstream_loss']=output[0]['downstream_loss']+total
  for key,value in stats.items():output[0]['phase_'+key]=total.new_tensor(value).detach()
  event=dict(step=state['step'],epoch=epoch,source_videos=[v['metadata']['video_id'] for v in kwargs['video_list_src']],
   target_videos=[v['metadata']['video_id'] for v in kwargs['video_list_tgt']],metrics=stats,target_gt_used=False)
  state['digest']=hashlib.sha256((state['digest']+json.dumps(event,sort_keys=True)).encode()).hexdigest()
  if getattr(ctx,'phase_ledger',None):
   with Path(ctx.phase_ledger).open('a') as log:log.write(json.dumps(dict(event,digest=state['digest']))+'\n')
  return output
 model.forward=forward

def pretrain_source_head(ctx,cfg,output):
 from tools.phase_candidate_components import pretrain_candidate_head
 return pretrain_candidate_head(ctx,cfg,output)

def audit(entry):
 from tools import baseline_protocol as protocol
 e=Path(entry);read,sha=protocol.read,protocol.sha;r=read(e/'complete.json');cfg=read(e/'config.json')
 ref=Path(r['reference']);assert sha(ref/'complete.json')==r['reference_complete_sha256'];assert_pair(cfg,read(ref/'config.json'))
 assert cfg['phase_boundary_runtime']['reference']==str(ref)
 headfile=Path(cfg['phase_boundary_runtime']['source_head']);headreceipt=read(headfile.with_suffix('.json'))
 assert sha(headfile)==headreceipt['head_sha256']
 head=torch.load(headfile,map_location='cpu');history=head['history']
 assert [x['epoch'] for x in history]==list(range(SPEC['source_head_epochs']+1))
 expected=min(history,key=lambda x:(x['source_candidate_score'],x['epoch']))
 assert head['best_epoch']==headreceipt['best_epoch']==expected['epoch'] and headreceipt['source_readiness']['passed']
 latesthead=torch.load(headfile.with_name('source_head_latest.pt'),map_location='cpu')
 assert latesthead['epoch']==SPEC['source_head_epochs'] and latesthead['history']==history and latesthead['optimizer']['state']
 protocol.validate_cuda_rng([latesthead['rng_cuda']])
 index=headfile.parent/'source_candidate_cache_index.json';assert sha(index)==headreceipt['cache_index_sha256']==latesthead['cache_index_sha256']
 cache=read(index)['roles'];database=read(cfg['dataset']['json_file'])['database']
 for role,split in [('train',cfg['src_train_split']),('val',cfg['src_val_split'])]:
  assert {v for a in cache[role] for v in a['videos']}=={v for v,d in database.items() if d['subset'] in split}
 result=protocol.audit_entry(e,expected_schema=SCHEMA)
 assert sha(SPEC['text_cache'])==SPEC['text_sha256'] and sha(SPEC['text_provenance'])==SPEC['text_provenance_sha256']
 rows=[json.loads(z) for z in (e/'metrics.jsonl').read_text().splitlines()]
 assert any(z['phase_boundary'].get('phase_target_sgp_gradient',0)>0 and z['phase_boundary'].get('phase_target_regression_gradient',0)>0 for z in rows)
 assert all(z['phase_boundary'].get('phase_target_loss',0)==0 for z in rows if z['epoch']<SPEC['onset'])
 source=read(cfg['dataset']['json_file'])['database'];train={v for v,d in source.items() if d['subset'] in cfg['src_train_split']}
 target=read(cfg['baseline']['target_manifest'])['database'];targettrain={v for v,d in target.items() if d['subset'] in cfg['tgt_train_split']}
 ledger=[json.loads(z) for z in (e/'phase_events.jsonl').read_text().splitlines()];digest='0'*64
 for step,event in enumerate(ledger,1):
  given=event.pop('digest');assert event['step']==step and not event['target_gt_used']
  assert set(event['source_videos'])<=train and set(event['target_videos'])<=targettrain
  digest=hashlib.sha256((digest+json.dumps(event,sort_keys=True)).encode()).hexdigest();assert digest==given
 latest=torch.load(e/'latest.pt',map_location='cpu');s=latest['model_controls']['phase_boundary_state']
 assert s['step']==len(ledger)==len(rows)*cfg['baseline']['epoch_budget']['iterations_per_epoch'] and s['digest']==digest
 assert any(k.startswith('phase_boundary.') for k in latest['state_dict']) and any(k.startswith('phase_boundary.') for k in latest['ema_state_dict'])
 for domain in ['source','target']:
  assert all('annotations' not in v for v in read(e/(domain+'_val_unlabeled.json'))['database'].values())
  assert read(e/'evaluation'/(domain+'_val.json'))['prediction_sha256']==sha(e/(domain+'_val.npz'))
 return dict(result,original_prototype_map=read(ref/'evaluation/target_val.json')['map_0.1_to_0.5'],
  delta_pp=100*(result['map']-read(ref/'evaluation/target_val.json')['map_0.1_to_0.5']))
