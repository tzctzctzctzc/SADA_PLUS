"""真实数据适配诊断，不保存任何正式checkpoint；不能用环境布尔值放行正式训练。"""
import math
from pathlib import Path
import runpy
import sys
import subprocess


def main():
    report,script=Path(sys.argv[1]),sys.argv[2]
    sys.argv=sys.argv[2:]; sys.path.insert(0,str(Path(script).resolve().parent))
    import libs.trainer.da_trainer as trainer
    from tools import rq4_protocol as protocol
    from tools.rq4_consensus import VARIANT,audit_spec,save,digest

    def diagnostic_ready(cfg):
        spec=cfg['rq4_boundary']
        if spec.get('variant')!=VARIANT: raise ValueError('not the consensus diagnostic wrapper')
        audit_spec(spec,protocol.load_cache(spec['cache_path'],spec['cache_file_sha256']))

    def smoke(self):
        self.train_metrics.reset()
        self._train_epoch(50,ad_lambda=self.adaptation_factor(50/30),
            cls_static_anchors=None,mean_class_conf=None)
        metrics={k.removeprefix('t_'):float(v) for k,v in self.train_metrics.result().items()
                 if 'rq4_' in k or 'target_gt_' in k}
        if (not metrics or any(not math.isfinite(v) for v in metrics.values())
                or metrics['rq4_anchor_points']<=0 or metrics['rq4_relation_pairs']<=0
                or metrics['rq4_gradient_norm']<=0
                or any(metrics['target_gt_'+k+'_loss']!=0 for k in ('cls','reg','downstream'))):
            raise ValueError('consensus real-data mechanisms/gradient/GT isolation failed')
        spec=self.cfg['rq4_boundary']; cache=protocol.load_cache(spec['cache_path'],spec['cache_file_sha256'])
        binding={k:v for k,v in spec.items() if k not in (
            'acceptance_path','acceptance_file_sha256','protocol_frozen','config_path','config_sha256')}
        code_commit=subprocess.check_output(['git','-C',str(Path(script).resolve().parent),
            'rev-parse','HEAD'],text=True).strip()
        save(report,dict(variant=VARIANT,status='passed',code_commit=code_commit,
            formal_result=False,real_video_features=True,
            teacher=cache['teacher'],shift=cache['shift'],seed=6,batch_size=2,diagnostic_epoch=50,
            specification_sha256=digest(binding),spec=spec,metrics=metrics))
        print('RQ4_CONSENSUS_REAL_ACCEPTANCE_PASS',flush=True)

    protocol.assert_launch_ready=diagnostic_ready
    trainer.Trainer.train=smoke
    runpy.run_path(script,run_name='__main__')


if __name__=='__main__': main()
