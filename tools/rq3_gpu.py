"""为RQ3串行任务选择实际空闲的物理卡；跨入口共享单卡锁。"""

from contextlib import contextmanager
from pathlib import Path
import subprocess
import time


def gpu_argument(value):
    if str(value) == "auto":
        return "auto"
    if not str(value).isdigit():
        raise ValueError("GPU must be auto or a nonnegative physical index")
    return str(int(value))


def gpu_indices(request):
    request = gpu_argument(request)
    if request != "auto":
        return [int(request)]
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits",
    ], text=True)
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def idle_gpu_uuid(gpu):
    row = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    uuid, memory, utilization = (part.strip() for part in row)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader",
    ], text=True)
    if int(memory) >= 2048 or int(utilization) > 5 or any(uuid in line for line in processes.splitlines()):
        raise RuntimeError("selected physical GPU is occupied; external jobs are not interrupted")
    return uuid


@contextmanager
def gpu_slot(request="auto", lock_root=Path("/vepfs/vlms/SADA_PLUS"), wait=False):
    """全局锁限制RQ3最多一张卡；每卡锁兼容已有RQ2脚本。"""
    import fcntl
    lock_root = Path(lock_root)
    with (lock_root / ".rq3_single_gpu.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        announced = False
        while True:
            for index in gpu_indices(request):
                card_lock = (lock_root / (".rq2_gpu%d.lock" % index)).open("a")
                try:
                    fcntl.flock(card_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    uuid = idle_gpu_uuid(index)
                except (BlockingIOError, RuntimeError):
                    card_lock.close()
                    continue
                except BaseException:
                    card_lock.close()
                    raise
                try:
                    print("RQ3_GPU_SELECTED index=%d uuid=%s" % (index, uuid), flush=True)
                    yield {"index": index, "uuid": uuid}
                finally:
                    card_lock.close()
                return
            if not wait:
                raise RuntimeError("no idle permitted GPU; no GPU task started")
            if not announced:
                print("RQ3_WAITING_FOR_IDLE_GPU request=%s" % request, flush=True)
                announced = True
            time.sleep(30)
