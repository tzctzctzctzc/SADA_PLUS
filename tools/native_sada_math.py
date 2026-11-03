"""复用原SADA条件判别器/GRL，在当前原生FPN上执行动作与背景分别对齐。"""
import ast
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import einops
from tools.native_joint_math import NativeBackground
from tools.background_linear_temporal import region_samples


def sample_pyramid(features,masks,strides,boxes):
    seq=[];valid=[]
    for f,m,stride in zip(features,masks,strides):
        a,b=region_samples(f,m,boxes/stride);seq.append(torch.cat(a,1));valid.append(torch.cat(b,1))
    return torch.stack(seq),torch.stack(valid)


def conditional_loss(model,source,smasks,target,tmasks,source_labels,target_logits,with_background=True):
    """原SADA类内BCE均值、域/层/背景权重；不以预测no-class冒充真实纯背景。"""
    cfg=model.cfg['model'];total=source[0].sum()*0;counts=[];accuracy=[]
    for l,(sf,sm,tf,tm,sl,tl) in enumerate(zip(source,smasks,target,tmasks,source_labels,target_logits)):
        sx=sf.T[sm];tx=tf.T[tm];sl=sl[sm];prob=tl[tm].detach().sigmoid();v,idx=prob.max(-1)
        pseudo=F.one_hot(idx,num_classes=10).to(prob.dtype)*(v>=cfg['pl_threshold'])[:,None]
        src=[];tgt=[];level_counts=[]
        for c in range(11 if with_background else 10):
            a=(sl==0).all(-1) if c==10 else sl[:,c]==1
            b=(pseudo==0).all(-1) if c==10 else pseudo[:,c]==1
            weight=cfg.get('weight_no_class_adv',1.) if c==10 else 1.
            level_counts.append([int(a.sum()),int(b.sum())])
            for x,mask,terms,label in ((sx,a,src,0.),(tx,b,tgt,1.)):
                if not mask.any():continue
                value=torch.cat((x[mask],model.learnable_embeddings[c].expand(int(mask.sum()),-1)),1)
                z=model.compute_level_dom_preds(value,True,l,0,beta=cfg['beta']).squeeze(-1)
                terms.append(weight*F.binary_cross_entropy_with_logits(z,torch.full_like(z,label)))
                accuracy.append(float(((z.detach()>=0)==bool(label)).float().mean()))
        s=sum(src,sx.new_zeros(()))/max(1,len(src))
        t=sum(tgt,tx.new_zeros(()))/max(1,len(tgt))
        if l in cfg['semantic_alignment_levels']:
            total=total+cfg.get('weight_level_'+str(l),1.)*((cfg['weight_source_cond_adv']*s+cfg['weight_target_cond_adv']*t)/len(cfg['semantic_alignment_levels']))
        counts.append(level_counts)
    return total,dict(class_counts=counts,domain_group_accuracy=sum(accuracy)/max(1,len(accuracy)))


def official_conditional_reference(model,source,smasks,target,tmasks,source_labels,target_logits,with_background=True):
    """直接编译原SADA.losses里的条件分支作数值参照，不另造近似参照。"""
    path=Path(__file__).resolve().parents[1]/'libs/models/SADA/meta_archs.py'
    tree=ast.parse(path.read_text(encoding='utf-8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SADA');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='losses')
    block=next(n for n in ast.walk(method) if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp) and "'conditional_rev'" in ast.unparse(n.test) and 'curr_epoch' in ast.unparse(n.test))
    # 保留原分支正文，函数边界只提供原losses的局部变量。
    names=['self','fpn_masks_src','fpn_masks_tgt','out_cls_logits_src','out_cls_logits_tgt','dom_inv_feats','gts_per_level_src','curr_epoch','semantic_alignment_levels','threshold','bs']
    body=ast.parse('losses={}').body+deepcopy(block.body)+ast.parse("return losses['loss_conditional_adv_rev']").body
    if source[0].device.type=='cpu':
        # 原代码CPU叶子tensor +=非法；CUDA的CPU->GPU拷贝为非叶子，无此限制。
        # CPU参照仅把+=展开为同次序非原位加法，GPU参照保留原分支逐字AST。
        class FunctionalCPU(ast.NodeTransformer):
            def visit_AugAssign(self,node):
                if isinstance(node.target,ast.Name) and node.target.id in ('loss_cond_reverse_level_source','loss_cond_reverse_level_target'):
                    return ast.copy_location(ast.Assign(targets=[node.target],value=ast.BinOp(left=ast.Name(id=node.target.id,ctx=ast.Load()),op=node.op,right=node.value)),node)
                return node
        body=[FunctionalCPU().visit(n) for n in body]
    fun=ast.FunctionDef(name='reference',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in names],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[fun],type_ignores=[]));ns=dict(torch=torch,F=F,einops=einops);exec(compile(module,str(path),'exec'),ns)
    sm=[];tm=[];sc=[];tc=[];both=[];labels=[]
    for sf,a,tf,b,c,z in zip(source,smasks,target,tmasks,source_labels,target_logits):
        length=max(sf.shape[-1],tf.shape[-1]);sm.append(F.pad(a,(0,length-len(a)))[None]);tm.append(F.pad(b,(0,length-len(b)))[None]);both.append(torch.stack((F.pad(sf,(0,length-sf.shape[-1])),F.pad(tf,(0,length-tf.shape[-1])))))
        sc.append(sf.new_zeros(1,length,10));tc.append(F.pad(z.T,(0,length-len(z))).T[None]);labels.append(F.pad(c.T,(0,length-len(c))).T[None])
    previous=model.align_bkg;model.align_bkg=with_background
    try:return ns['reference'](model,sm,tm,sc,tc,both,labels,5,model.cfg['model']['semantic_alignment_levels'],source[0].new_full((10,),model.cfg['model']['pl_threshold']),1)
    finally:model.align_bkg=previous


class NativeSADA(NativeBackground):
    def __init__(self,detector):
        super().__init__(detector)
        self.disc_dom_conditional_adv=detector.disc_dom_conditional_adv
        self.learnable_embeddings=detector.learnable_embeddings
        self.cls_head=detector.cls_head
        self.point_generator=detector.point_generator
        self.cfg=deepcopy(detector.cfg);self.cfg['model']['use_source_pseudo_labels']=False
        self.n_levels=6;self.num_classes=self.num_class_orig=10;self.align_bkg=True
        self.criterion_domain=nn.BCEWithLogitsLoss()
        # 原生point assignment与条件GRL直接复用当前项目的SADA方法。
        from libs.models.SADA.meta_archs import SADA
        self.compute_level_dom_preds=SADA.compute_level_dom_preds.__get__(self)
        self.label_points=SADA.label_points.__get__(self)
        self.label_points_single_video=SADA.label_points_single_video.__get__(self)
        self.train_center_sample=detector.train_center_sample;self.train_center_sample_radius=detector.train_center_sample_radius;self.train_reg_distance=detector.train_reg_distance
        self.cls_head.requires_grad_(False);self.cls_head.eval()

    @property
    def device(self):return next(self.parameters()).device

    def core_state(self):
        return {k:v for k,v in self.state_dict().items() if k.startswith(('backbone.','neck.','head.'))}

    def da_parameters(self):return list(self.disc_dom_conditional_adv.parameters())+[self.learnable_embeddings]

    def source_point_labels(self,features,segments,labels):
        with torch.no_grad():
            points=self.point_generator([f[None] for f in features]);gt,_=self.label_points(points,[segments],[labels]);return list(gt[0].split([f.shape[-1] for f in features]))

    def pseudo_logits(self,features,masks):
        with torch.no_grad():
            logits,_=self.cls_head([f[None] for f in features],[m[None,None] for m in masks]);return [z[0].T for z in logits]
