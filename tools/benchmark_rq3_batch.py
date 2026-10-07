"""独立进程实测真实Trainer的前后向吞吐；无checkpoint、无评估、不能作实验结果。"""
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    report = Path(sys.argv[1])
    train_script = sys.argv[2]
    sys.argv = sys.argv[2:]
    import torch
    import libs.trainer.da_trainer as trainer_module

    def benchmark(self):
        measurements = []
        original_progress = trainer_module.tqdm
        try:
            # warmup和适配两阶段都测；每阶段3步预热、12步计时。
            for epoch in (1, 50):
                durations = []

                def bounded_progress(iterable, *args, **kwargs):
                    for step, value in enumerate(iterable):
                        if step == 15:
                            break
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        yield value
                        torch.cuda.synchronize()
                        if step >= 3:
                            durations.append(time.perf_counter() - started)

                trainer_module.tqdm = bounded_progress
                self.train_metrics.reset()
                torch.cuda.reset_peak_memory_stats()
                self._train_epoch(epoch, ad_lambda=self.adaptation_factor(epoch / 30),
                                  cls_static_anchors=None, mean_class_conf=None)
                if len(durations) != 12:
                    raise ValueError('batch benchmark needs 15 real iterations')
                batch = self.cfg['loader']['batch_size']
                measurements.append(dict(epoch=epoch, measured_steps=len(durations),
                    seconds_per_step=sum(durations)/len(durations),
                    source_target_videos_per_second=2*batch*len(durations)/sum(durations),
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
            with report.open('x') as stream:
                json.dump(dict(schema='rq3-throughput-only-v1', batch_size=batch,
                               no_training_result=True, measurements=measurements), stream)
        finally:
            trainer_module.tqdm = original_progress

    trainer_module.Trainer.train = benchmark
    runpy.run_path(train_script, run_name='__main__')


if __name__ == '__main__':
    main()
