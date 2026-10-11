"""真实数据单epoch适配验收；无正式checkpoint、不计实验结果。"""
import json
from pathlib import Path
import runpy
import sys


def main():
    report,script = Path(sys.argv[1]),sys.argv[2]
    sys.argv = sys.argv[2:]
    # 数据目录含旧libs；必须像直接执行train_da.py一样优先使用当前验收代码。
    sys.path.insert(0,str(Path(script).resolve().parent))
    import libs.trainer.da_trainer as module
    from tools.run_rq4_entry import save

    def smoke(self):
        import math
        self.train_metrics.reset()
        self._train_epoch(50,ad_lambda=self.adaptation_factor(50/30),
                          cls_static_anchors=None,mean_class_conf=None)
        metrics = {k.removeprefix('t_'): v for k,v in self.train_metrics.result().items()}
        values = {k:float(v) for k,v in metrics.items() if 'rq4_' in k or 'target_gt_' in k}
        if not values or any(not math.isfinite(v) for v in values.values()):
            raise ValueError('missing/nonfinite real RQ4 metrics')
        for key in ('cls','reg','downstream'):
            if metrics['target_gt_'+key+'_loss'] != 0:
                raise ValueError('target GT leakage in acceptance')
        if metrics['rq4_queried'] <= 0:
            raise ValueError('real target candidates were never observed')
        if self.cfg['rq4_boundary']['strategy'] == 'direct' and metrics['rq4_gradient_norm'] <= 0:
            raise ValueError('real Direct boundary gradient did not activate')
        if self.cfg['rq4_boundary']['strategy'] in ('relational','proposal_relation'):
            if metrics['rq4_accepted'] <= 0 or metrics['rq4_gradient_norm'] <= 0:
                raise ValueError('relational interface acceptance needs real multi-point gradient')
        if metrics['rq4_loss'] > 0 and metrics['rq4_gradient_norm'] <= 0:
            raise ValueError('boundary loss detached')
        save(report,dict(status='passed',formal_result=False,real_video_features=True,
            strategy=self.cfg['rq4_boundary']['strategy'],spec=self.cfg['rq4_boundary'],
            run_signature=self.cfg['rq2_formal']['run_signature'],
            metrics=values,diagnostic_epoch=50,seed=6,batch_size=2))
        print('RQ4_REAL_DATA_ACCEPTANCE_PASS',json.dumps(values),flush=True)
    module.Trainer.train = smoke
    runpy.run_path(script,run_name='__main__')


if __name__ == '__main__': main()
