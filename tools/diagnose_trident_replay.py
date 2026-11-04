"""独立短GPU核验：重叠窗口反传是否导致逐位重放失败；不训练检测器。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity


def worker(root):
    from tools.baseline_gpu import first_cuda_guard
    guard=first_cuda_guard()
    import torch
    from tools.trident_windows import strided_windows,fixed_windows
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.manual_seed(123);torch.cuda.manual_seed(123)
    x=torch.randn(2,10,2304,device='cuda');upstream=torch.randn(2,10,2304,17,device='cuda')
    results={}
    for left in (True,False):
        grads={};outputs={}
        for name,function in [('native',strided_windows),('fixed',fixed_windows)]:
            values=[]
            for _ in range(20):
                leaf=x.detach().clone().requires_grad_();output=function(leaf,16,left)
                gradient=torch.autograd.grad(output,leaf,upstream)[0]
                values.append(gradient.detach().clone())
            outputs[name]=output.detach();grads[name]=values
        assert torch.equal(outputs['native'],outputs['fixed'])
        assert all(torch.equal(grads['fixed'][0],v) for v in grads['fixed'])
        cpu=x.detach().cpu().requires_grad_();cpu_out=fixed_windows(cpu,16,left)
        cpu_grad=torch.autograd.grad(cpu_out,cpu,upstream.cpu())[0]
        assert torch.equal(cpu_grad,grads['fixed'][0].cpu())
        double=x.detach().cpu().double().requires_grad_()
        exact=torch.autograd.grad(strided_windows(double,16,left),double,upstream.cpu().double())[0]
        new_double=x.detach().cpu().double().requires_grad_()
        actual=torch.autograd.grad(fixed_windows(new_double,16,left),new_double,upstream.cpu().double())[0]
        torch.testing.assert_close(actual,exact,atol=1e-12,rtol=1e-12)
        hashes=[sha_tensor(v) for v in grads['native']]
        results['left' if left else 'right']=dict(native_unique_gradients=len(set(hashes)),
            native_max_run_difference=max(float((v-grads['native'][0]).abs().max()) for v in grads['native']),
            fixed_unique_gradients=1,fixed_CPU_GPU_bit_equal=True,forward_bit_equal=True,
            float64_adjoint_reference_passed=True)
    save(root/'kernel_diagnosis.json',dict(passed=True,results=results,guard=guard,guard_sha256=sha(guard),
        model_training_steps=0,scope='window primitive only; full-model cause still requires state replay',
        torch_version=torch.__version__,commit=code_identity()['commit']))
    print(json.dumps(results),flush=True)


def sha_tensor(x):
    import hashlib
    return hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest()


def run(root):
    import fcntl
    from tools.baseline_gpu import snapshot
    root.mkdir(exist_ok=False);identity=code_identity()
    for gpu in range(4):
        observed=snapshot(gpu)
        if observed['free_mib']<16384:continue
        lockpath=root/f'private_gpu{gpu}.lock';lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lease=root/'lease.json';save(lease,dict(owner_pid=os.getpid(),observed=observed,lock=str(lockpath),minimum_free_mib=16384,commit=identity['commit']))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=observed['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),
            SADA_BASELINE_LEASE=str(lease),SADA_BASELINE_LEASE_SHA256=sha(lease),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (root/'kernel.log').open('x') as out:
            p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker','--root',str(root)],
                cwd=CODE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'launch.json',dict(pid=p.pid,owner_pid=os.getpid(),code=identity))
        code=p.wait();lock.close();assert code==0,(root/'kernel.log').read_text()[-3000:]
        return
    raise RuntimeError('no free GPU for short primitive diagnosis')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--worker',action='store_true');a=p.parse_args()
    worker(a.root) if a.worker else run(a.root)
