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

SCHEMA='sada-prototype-phase-boundary-v1'
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
 endpoint_gate='teacher_scale_less_than_student_scale_and_teacher_edge_ge_.5',
 source_boundary='Gaussian_edge_BCE_all_valid_points',boundary_search_fraction=.25,
 target_classification='teacher_class_and_phase_class_agree_positive_points_only',
 target_regression='phase_edge_refined_teacher_endpoints_separate_masks',
 target_phase='ordered_posterior_KL_on_physical_correspondences',
 precision='float32',initialization='mature_Prototype_EMA_extra_source_head_pretrain_and_joint_adaptation',
 source_head_epochs=5,source_head_lr=.0003,source_head_selection='complete_source_val_auxiliary_loss',
 source_selection='original_complete_source_cls_plus_reg_EMA_every5_patience15_max50',
 adversarial='original_conditional_rev_unchanged',prototype='original128_frozen_VLM40of200_membership',
 inference='original_heads_NMS_voting_no_auxiliary_score_product',target_GT=False)

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
  gate=self.foreground(infused).sigmoid()[:,None]
  logits=similarity[...,:3]+gate.log().clamp_min(-8)
  weight=self.phase_weight(infused).softmax(-1)
  scores=(similarity*weight[:,None]).mean(-2).sum(-1)
  ordered=path_partition(logits.log_softmax(-1))/x.shape[-2]
  return logits, scores+ordered

def sequence_loss(head,tokens,labels):
 logits,scores=head.sequence(tokens);idx=torch.arange(len(labels),device=labels.device)
 chosen=logits[idx,labels];normal=path_partition(chosen.log_softmax(-1))
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

def source_loss(model,head,f,m,reg,videos):
 losses=[];seq=[];ys=[];stats=dict(source_instances=0,source_scale_error=0.)
 for vi,v in enumerate(videos):
  truth=v['segments'];labels=v['labels'];validgt=(labels<10)&((truth[:,1]-truth[:,0])>.05)
  truth,labels=truth[validgt],labels[validgt]
  for box,c in zip(truth,labels):
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

def target_loss(model,teacher,head,weak,strong,views,step):
 sf,sm,sc,sr=strong;tf,tm,tc,tr=weak;losses=[]
 stats=dict(target_proposals=0,target_class_used=0,target_left_used=0,target_right_used=0,target_phase_used=0,
            target_teacher_scale=0.,target_student_scale=0.)
 for vi,view in enumerate(views):
  proposals=select_proposals(tf,tm,tc,tr,teacher,vi,int(view['original_length']))
  for box,c,l,j,confidence in proposals:
   stats['target_proposals']+=1;stride=model.fpn_strides[l]
   if box[0]<view['lo'] or box[1]>view['lo']+view['width']-1:continue
   mapped=map_coordinates(box,view);point=map_coordinates(box.new_tensor(j*stride),view);k=int((point/stride).round())
   if k<0 or k>=sf[l].shape[-1] or not sm[l][vi,0,k]:continue
   with torch.no_grad():
    tt=interval_tokens(tf[l][vi],box,stride)[None];tl,tclass=teacher.phase_boundary.sequence(tt)
    if int(tclass.argmax(-1))!=int(c):continue
    post=phase_posterior(tl[:,c]).detach()
    te,tunc,_=teacher.phase_boundary.dense(tf[l][vi,:,j][None]);tunc=tunc[0]
    # 每端点在自身邻域找源监督边界证据，保留端点身份；非互斥峰均值。
    refined=box.clone();edgeconf=[];width=box[1]-box[0]
    for side in range(2):
     coords=box[side]+torch.linspace(-.25,.25,17,device=box.device)*width
     valid=(coords>=0)&(coords<view['original_length']);ev=teacher.phase_boundary.dense(sample(tf[l][vi],coords,stride))[0][:,side].sigmoid().masked_fill(~valid,0)
     best=ev.argmax();refined[side]=coords[best];edgeconf.append(ev[best])
    if refined[1]-refined[0]<=.05:continue
    mapped_ref=map_coordinates(refined,view)
   st=interval_tokens(sf[l][vi],mapped,stride)[None];sl,sclass=head.sequence(st)
   losses.append(F.kl_div(sl[:,c].log_softmax(-1),post,reduction='batchmean')/SPEC['tokens'])
   losses.append(F.cross_entropy(sclass,c[None]));stats['target_phase_used']+=1
   label=torch.zeros_like(sc[l][vi,:,k]);label[c]=1
   losses.append(F.binary_cross_entropy_with_logits(sc[l][vi,:,k],label)*confidence);stats['target_class_used']+=1
   _,sunc,_=head.dense(sf[l][vi,:,k][None]);sunc=sunc[0]
   mask=(tunc<sunc.detach())&(torch.stack(edgeconf)>=.5)&(mapped_ref>=0)&(mapped_ref<view['length'])
   stats['target_teacher_scale']+=float(tunc.mean());stats['target_student_scale']+=float(sunc.detach().mean())
   pred=torch.stack([k*stride-sr[l][vi,0,k]*stride,k*stride+sr[l][vi,1,k]*stride])
   if mask.any():
    # 以同一伪实例时长归一化，避免长动作/裁剪缩放造成尺度比较混淆。
    losses.append((F.smooth_l1_loss(pred,mapped_ref.detach(),reduction='none')/(mapped[1]-mapped[0]).clamp_min(1))[mask].mean())
    stats['target_left_used']+=int(mask[0]);stats['target_right_used']+=int(mask[1])
 return (torch.stack(losses).mean() if losses else sf[0].sum()*0),stats

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
  loss,stats=source_loss(model,model.phase_boundary,f,m,reg,kwargs['video_list_src'])
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
 """成熟冻结SGP上缓存原训练crop，完整源split训练新头；不访问target/目标GT。"""
 from tools import baseline_engine as base
 from tools.baseline_protocol import save,sha
 model=ctx.model;model.eval();head=model.phase_boundary
 caches={}
 source_loader=base.loader(ctx.train_loader_src.dataset,True,torch.Generator().manual_seed(123),1,cfg['loader']['num_workers'])
 for role,loader in [('train',source_loader),('val',ctx.val_loader_src)]:
  cache=[]
  for videos in loader:
   videos=base.move_to_device(videos,model.device)
   with torch.no_grad():f,m,c,r=direct(model,videos,training=role=='train')
   cache.append((tuple(x.detach() for x in f),m,tuple(x.detach() for x in r),videos))
  caches[role]=cache
 optimizer=torch.optim.AdamW(head.parameters(),lr=SPEC['source_head_lr'],weight_decay=.05)
 best=float('inf');history=[];beststate=None
 for epoch in range(1,SPEC['source_head_epochs']+1):
  head.train();train=[]
  for f,m,r,videos in caches['train']:
   optimizer.zero_grad();loss,_=source_loss(model,head,f,m,r,videos);assert torch.isfinite(loss)
   loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();train.append(float(loss.detach()))
  head.eval();val=[]
  with torch.no_grad():
   for f,m,r,videos in caches['val']:val.append(float(source_loss(model,head,f,m,r,videos)[0]))
  row=dict(epoch=epoch,train_loss=sum(train)/len(train),validation_loss=sum(val)/len(val));history.append(row)
  if row['validation_loss']<best:
   best=row['validation_loss'];beststate=deepcopy(head.state_dict());bestepoch=epoch
  print('SOURCE_PHASE_HEAD',json.dumps(row),flush=True)
 base.checkpoint(output,dict(spec=SPEC,state_dict=beststate,best_epoch=bestepoch,history=history,
  optimizer=optimizer.state_dict(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(0),
  source_only=True,scope='all_source_training_videos_one_original_GT_crop_cached_for_five_head_epochs',
  source_videos=[[v['metadata']['video_id'] for v in vs] for _,_,_,vs in caches['train']],
  validation_videos=[[v['metadata']['video_id'] for v in vs] for _,_,_,vs in caches['val']]))
 save(Path(output).with_suffix('.json'),dict(spec=SPEC,best_epoch=bestepoch,history=history,head_sha256=sha(output),target_gt_used=False))
 head.load_state_dict(beststate,strict=True);ctx.model_ema.module.phase_boundary.load_state_dict(beststate,strict=True)
 del caches
 torch.cuda.empty_cache()

def audit(entry):
 from tools import baseline_protocol as protocol
 e=Path(entry);read,sha=protocol.read,protocol.sha;r=read(e/'complete.json');cfg=read(e/'config.json')
 ref=Path(r['reference']);assert sha(ref/'complete.json')==r['reference_complete_sha256'];assert_pair(cfg,read(ref/'config.json'))
 assert cfg['phase_boundary_runtime']['reference']==str(ref)
 headfile=Path(cfg['phase_boundary_runtime']['source_head']);headreceipt=read(headfile.with_suffix('.json'))
 assert sha(headfile)==headreceipt['head_sha256']
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
