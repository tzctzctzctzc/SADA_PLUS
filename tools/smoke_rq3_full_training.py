"""独立真实视频特征适配epoch诊断；不保存checkpoint、不记正式完成。"""
from pathlib import Path
import runpy
import sys


def main():
    report, script = Path(sys.argv[1]), sys.argv[2]
    sys.argv = sys.argv[2:]
    sys.path.insert(0, str(Path(script).resolve().parent))
    import libs.trainer.da_trainer as module
    from tools.rq3_full_launch import save, validate_smoke
    from tools.rq3_full_participation import VARIANT

    def smoke(self):
        self.train_metrics.reset()
        self._train_epoch(50, ad_lambda=self.adaptation_factor(50/30),
                          cls_static_anchors=None, mean_class_conf=None)
        values = {key.removeprefix('t_'): float(value)
                  for key, value in self.train_metrics.result().items()
                  if 'rq3_' in key or 'target_gt_' in key}
        record = dict(status='passed', formal_result=False, real_video_features=True,
                      diagnostic_epoch=50, seed=6, batch_size=2, variant=VARIANT,
                      auxiliary=self.cfg['rq3_auxiliary'], metrics=values,
                      run_signature=self.cfg['rq2_formal']['run_signature'])
        validate_smoke(record, self.cfg)
        save(report, record)
        print('QWEN100_REAL_VIDEO_ACCEPTANCE_PASS', flush=True)
    smoke._rq3_full_diagnostic = True
    module.Trainer.train = smoke
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
