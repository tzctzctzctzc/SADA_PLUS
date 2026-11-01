"""保持逐步训练数学的日志同步与EMA优化；原Trainer源码通过AST限定单点替换。"""
import ast
import inspect
import textwrap
from collections import defaultdict
from types import MethodType
import torch


def enabled(cfg):
    return (cfg.get('benchmark')=='charades-ego-third-to-first' and bool(cfg.get('prototype_capacity'))
            and cfg['model']['adv_method']==['None'])


def batched_metrics(tracker, losses):
    names=list(losses)
    # float64保证不同dtype的标量提升后仍等于各自原.item()。
    values=torch.stack([losses[k].detach().to(torch.float64).reshape(()) for k in names]).cpu().tolist()
    for name,value in zip(names,values):
        key='t_'+name
        if not tracker.contains(key):tracker.add_new(key)
        tracker.update(key,value)


def fast_ema(self, model):
    # SADA有共享参数别名：必须保留原EMA对同一张量的多次更新，不能简单去重。
    if not hasattr(self,'_speed_ema_groups'):
        occurrences=defaultdict(int);groups=defaultdict(list);fallback=[]
        for e,m in zip(self.module.state_dict().values(),model.state_dict().values()):
            if not e.is_floating_point() or self.device is not None:
                fallback.append((e,m));continue
            key=(e.data_ptr(),tuple(e.shape),e.dtype,e.device)
            turn=occurrences[key];occurrences[key]+=1
            groups[(turn,e.dtype,e.device)].append((e,m))
        self._speed_ema_groups=(list(groups.values()),fallback)
    with torch.no_grad():
        groups,fallback=self._speed_ema_groups
        for pairs in groups:
            es,ms=map(list,zip(*pairs))
            a=torch._foreach_mul(es,self.decay)
            b=torch._foreach_mul(ms,1.-self.decay)
            values=torch._foreach_add(a,b)
            if hasattr(torch,'_foreach_copy_'):torch._foreach_copy_(es,values)
            else:
                for e,v in zip(es,values):e.copy_(v)
        for e,m in fallback:
            if self.device is not None:m=m.to(self.device)
            e.copy_(self.decay*e+(1.-self.decay)*m)


def epoch_function():
    from libs.trainer import da_trainer
    source=textwrap.dedent(inspect.getsource(da_trainer.Trainer._train_epoch))
    tree=ast.parse(source);changed=[]
    class Rewrite(ast.NodeTransformer):
        def visit_For(self,node):
            if (isinstance(node.target,ast.Tuple) and [getattr(v,'id',None) for v in node.target.elts]==['loss_name','loss_tr']):
                changed.append(node)
                return ast.parse('_speed_metrics(self.train_metrics, losses)').body[0]
            return self.generic_visit(node)
    tree=Rewrite().visit(tree);assert len(changed)==1,'只替换原Trainer的标量日志循环'
    ast.fix_missing_locations(tree)
    namespace=dict(vars(da_trainer),_speed_metrics=batched_metrics)
    exec(compile(tree,'<charades-speed-original-epoch>','exec'),namespace)
    return namespace['_train_epoch']


def install(ctx):
    ctx._train_epoch=MethodType(epoch_function(),ctx)
    ctx.model_ema.update=MethodType(fast_ema,ctx.model_ema)
