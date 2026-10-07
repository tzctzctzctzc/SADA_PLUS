"""单卡双进程吞吐诊断子进程；父进程持GPU锁，同进程组仅用于此次测速。"""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def main():
    output, barrier, slot, script = sys.argv[1:5]
    sys.argv = sys.argv[4:]
    import torch
    import tools.rq3_gpu as gpu
    import libs.trainer.da_trainer as trainer

    def guard():
        uuid = os.environ['RQ3_GPU_UUID']
        if os.environ['CUDA_VISIBLE_DEVICES'] != uuid or os.environ['RQ3_GPU_INDEX'] != '3':
            raise RuntimeError('benchmark must stay on GPU3')
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                       '--format=csv,noheader,nounits'], text=True)
        for line in rows.splitlines():
            card, pid = [x.strip() for x in line.split(',')]
            if card == uuid and os.getpgid(int(pid)) != os.getpgrp():
                raise RuntimeError('external GPU process entered benchmark')
    gpu.guard_before_cuda = guard

    def benchmark(self):
        rows = []
        original = trainer.tqdm
        try:
            for epoch in (1, 50):
                durations = []
                def progress(iterable, *args, **kwargs):
                    for step, value in enumerate(iterable):
                        if step >= 13:
                            break
                        torch.cuda.synchronize()
                        if step == 3 and barrier != '-':
                            ready = Path(barrier)
                            (ready / ('%s_%d' % (slot, epoch))).touch()
                            deadline = time.monotonic() + 600
                            while len(list(ready.glob('*_%d' % epoch))) != 2:
                                if time.monotonic() > deadline:
                                    raise TimeoutError('benchmark peer not ready')
                                time.sleep(.05)
                        start = time.perf_counter()
                        yield value
                        torch.cuda.synchronize()
                        if step >= 3:
                            durations.append(time.perf_counter() - start)
                trainer.tqdm = progress
                self.train_metrics.reset()
                torch.cuda.reset_peak_memory_stats()
                self._train_epoch(epoch, ad_lambda=self.adaptation_factor(epoch/30),
                                  cls_static_anchors=None, mean_class_conf=None)
                if len(durations) < 3:
                    raise ValueError('insufficient measured steps')
                rows.append(dict(epoch=epoch, steps=len(durations), seconds=sum(durations),
                    videos=2*self.cfg['loader']['batch_size']*len(durations),
                    peak_mib=torch.cuda.max_memory_allocated()/2**20))
            with Path(output).open('x') as stream:
                json.dump(dict(diagnostic_only=True, measurements=rows), stream)
        finally:
            trainer.tqdm = original
    trainer.Trainer.train = benchmark
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
