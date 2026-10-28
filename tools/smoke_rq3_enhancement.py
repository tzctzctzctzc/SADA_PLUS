"""独立真实数据适配epoch验收；不保存正式checkpoint，不计实验完成。"""
import json
from pathlib import Path
import runpy
import sys


def main():
    report, train_script = Path(sys.argv[1]), sys.argv[2]
    sys.argv = sys.argv[2:]
    import torch
    from libs.trainer.da_trainer import Trainer

    def smoke(self):
        self.train_metrics.reset()
        self._train_epoch(50, ad_lambda=self.adaptation_factor(50/30),
                          cls_static_anchors=None, mean_class_conf=None)
        metrics = self.train_metrics.result()
        for key in ('t_target_gt_cls_loss', 't_target_gt_reg_loss', 't_target_gt_downstream_loss'):
            if metrics.get(key) != 0:
                raise ValueError('target GT isolation failed: ' + key)
        for key in ('t_rq3_source_loss', 't_rq3_alignment_loss', 't_rq3_selected',
                    't_rq3_gradient_norm', 't_rq3_target_gradient_norm'):
            if not float(metrics.get(key, 0)) > 0:
                raise ValueError('real supervision missing: ' + key)
        if not all(torch.isfinite(torch.tensor(float(v))) for v in metrics.values()):
            raise ValueError('nonfinite smoke metrics')
        # 严格恢复含新增模块的完整state，避免只在正式离线评估时发现漏接。
        self.model.load_state_dict(self.model.state_dict(), strict=True)
        with report.open('x') as stream:
            json.dump(dict(status='passed', diagnostic_only=True, no_checkpoint=True,
                           specification=self.cfg['rq3_enhancement'], metrics=metrics), stream, sort_keys=True)

    Trainer.train = smoke
    runpy.run_path(train_script, run_name='__main__')


if __name__ == '__main__':
    main()
