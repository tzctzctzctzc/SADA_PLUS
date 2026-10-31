import resource
import subprocess
import sys

def test_large_shared_batch_fd_budget_is_available_to_child():
    # 子进程内降低soft后再恢复，不改变pytest父进程；真实打开2048个句柄。
    script='''
import os,resource
from tools.run_sada_baseline import prepare_file_limit
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE,(1024,hard))
result=prepare_file_limit()
assert result['soft']>=65536
fds=[os.open('/dev/null',os.O_RDONLY) for _ in range(2048)]
for fd in fds:os.close(fd)
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True)
