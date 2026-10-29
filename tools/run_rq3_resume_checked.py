"""在真实GPU Trainer恢复完成、首个续跑batch之前逐项验证完整训练状态。"""
import json
from pathlib import Path
import random
import hashlib
import runpy
import sys


def state_digest(value):
    import numpy as np
    import torch
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            x=v.detach().cpu().contiguous()
            h.update(str((str(x.dtype),tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str((str(v.dtype),v.shape)).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v,dict):
            for key in sorted(v,key=str):
                h.update(repr((type(key).__name__,key)).encode());visit(v[key])
        elif isinstance(v,(list,tuple)):
            for item in v:visit(item)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def main():
    report,checkpoint_path,script = Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
    sys.argv=sys.argv[3:]
    import numpy as np
    import torch
    from libs.trainer.da_trainer import Trainer
    from tools.rq2_evaluate import load_checkpoint
    from tools.rq2_dense_protocol import sha256_file
    original=Trainer.train

    def checked(self):
        checkpoint=load_checkpoint(checkpoint_path)
        if self.start_epoch!=checkpoint['epoch']+1 or not self.resume_rng_restored:
            raise ValueError('resume did not restore the expected epoch and RNG state')
        components=dict(model=(self.model.state_dict(),checkpoint['state_dict']),
            ema=(self.model_ema.module.state_dict(),checkpoint['ema_state_dict']),
            optimizer_source=(self.optimizer_gen_source.state_dict(),checkpoint['optimizer_gen_source']),
            optimizer_alignment=(self.optimizer_gen_source_alignment.state_dict(),checkpoint['optimizer_gen_source_alignment']),
            optimizer_discriminator=(self.optimizer_disc.state_dict(),checkpoint['optimizer_disc']),
            normalizers_and_schedulers=(self.get_formal_training_state(),checkpoint['rq2_training_state']))
        observed=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                      cuda=torch.cuda.get_rng_state_all(),source_loader=self.source_rng_generator.get_state(),
                      target_loader=self.target_rng_generator.get_state())
        components['rng']=(observed,checkpoint['training_rng_state'])
        hashes={}
        for name,(actual,expected) in components.items():
            a,b=state_digest(actual),state_digest(expected)
            if a!=b:
                raise ValueError('restored training state differs: '+name)
            hashes[name]=a
        after=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state_all(),source_loader=self.source_rng_generator.get_state(),
                   target_loader=self.target_rng_generator.get_state())
        if state_digest(after)!=hashes['rng']:
            raise ValueError('resume verification consumed training RNG')
        with report.open('x') as stream:
            json.dump(dict(status='passed',restored_epoch=checkpoint['epoch'],start_epoch=self.start_epoch,
                 checkpoint_sha256=sha256_file(checkpoint_path),component_sha256=hashes,
                 complete_saved_state_bitwise_equal=True,verification_rng_unchanged=True),stream,sort_keys=True)
        del checkpoint,components,observed,after
        return original(self)

    Trainer.train=checked
    runpy.run_path(script,run_name='__main__')


if __name__=='__main__':main()
