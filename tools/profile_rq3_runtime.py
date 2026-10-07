"""固定batch的独立吞吐诊断：真实训练步比较CPU线程/loader workers/pinned传输。无正式结果。"""
from copy import deepcopy
import json
from pathlib import Path
import random
import runpy
import sys
import time


def main():
    report, script = Path(sys.argv[1]), sys.argv[2]
    sys.argv = sys.argv[2:]
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import libs.trainer.da_trainer as module

    def profile(self):
        initial = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        optimizers = [self.optimizer_gen_source, self.optimizer_gen_source_alignment, self.optimizer_disc]
        schedulers = [self.lr_scheduler_gen_source, self.lr_scheduler_gen_source_alignment, self.lr_scheduler_disc]
        optimizer_states = [deepcopy(o.state_dict()) for o in optimizers]
        scheduler_states = [deepcopy(s.state_dict()) if s else None for s in schedulers]
        original_loaders = [self.train_loader_src, self.train_loader_tgt]
        original_progress, original_move = module.tqdm, module.move_to_device
        rows = []
        # 一次只变一个运行设置；未改变batch、模型精度、目标函数或正式配置。
        cases = [(8,4,False), (4,4,False), (16,4,False), (8,0,False), (8,8,False), (8,4,True)]
        try:
            for threads, workers, pinned in cases:
                torch.set_num_threads(threads)
                random.seed(6); np.random.seed(6); torch.manual_seed(6); torch.cuda.manual_seed_all(6)
                self.model.load_state_dict(initial)
                if self.model_ema is not None:
                    self.model_ema.module.load_state_dict(initial)
                for opt, state in zip(optimizers, optimizer_states): opt.load_state_dict(deepcopy(state))
                for scheduler, state in zip(schedulers, scheduler_states):
                    if scheduler is not None: scheduler.load_state_dict(deepcopy(state))
                timers = dict(data_wait=0., transfer=0.)

                class TimedLoader:
                    def __init__(self, loader): self.loader = loader
                    def __len__(self): return len(self.loader)
                    def __iter__(self):
                        iterator = iter(self.loader)
                        while True:
                            start = time.perf_counter()
                            try: batch = next(iterator)
                            except StopIteration: return
                            timers['data_wait'] += time.perf_counter()-start
                            yield batch

                loaders = [DataLoader(old.dataset, batch_size=old.batch_size, num_workers=workers,
                    collate_fn=old.collate_fn, worker_init_fn=old.worker_init_fn, shuffle=True,
                    drop_last=True, generator=torch.Generator().manual_seed(6+i),
                    pin_memory=pinned, persistent_workers=False) for i,old in enumerate(original_loaders)]
                self.train_loader_src, self.train_loader_tgt = [TimedLoader(x) for x in loaders]
                durations = []

                def progress(iterable, *args, **kwargs):
                    for step, value in enumerate(iterable):
                        if step == 15: break
                        torch.cuda.synchronize(); start = time.perf_counter()
                        yield value
                        torch.cuda.synchronize()
                        if step >= 3: durations.append(time.perf_counter()-start)

                def transfer(videos, device):
                    start = time.perf_counter()
                    for video in videos:
                        for key, value in video.items():
                            if key != 'full_feats' and isinstance(value, torch.Tensor):
                                video[key] = value.to(device, non_blocking=pinned)
                    timers['transfer'] += time.perf_counter()-start
                    return videos

                module.tqdm, module.move_to_device = progress, transfer
                self.train_metrics.reset(); torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                self._train_epoch(50, ad_lambda=self.adaptation_factor(50/30), cls_static_anchors=None, mean_class_conf=None)
                elapsed = time.perf_counter()-started
                if len(durations)<3: raise ValueError('insufficient measured steps')
                row = dict(threads=threads, workers=workers, pinned_nonblocking=pinned,
                    measured_steps=len(durations), seconds_per_step=sum(durations)/len(durations),
                    videos_per_second=2*self.cfg['loader']['batch_size']*len(durations)/sum(durations),
                    phase_wall_seconds=elapsed, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    **timers)
                rows.append(row)
                print('RQ3_RUNTIME_PROFILE',json.dumps(row),flush=True)
                report.write_text(json.dumps(dict(status='running', formal_result=False, measurements=rows)))
            report.write_text(json.dumps(dict(status='complete', formal_result=False,
                batch_size=self.cfg['loader']['batch_size'], measurements=rows,
                best=max(rows,key=lambda x:x['videos_per_second']),
                limitation='worker changes can alter crops/RNG; timings are diagnostic, not automatic formal adoption')))
        finally:
            module.tqdm, module.move_to_device = original_progress, original_move

    module.Trainer.train = profile
    runpy.run_path(script,run_name='__main__')


if __name__ == '__main__': main()
