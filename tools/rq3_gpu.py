"""为RQ3选择实际空闲的物理卡；跨入口最多使用两张卡。"""

from contextlib import contextmanager
from pathlib import Path
import os
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
def gpu_slot(request="auto", lock_root=Path("/vepfs/vlms/SADA_PLUS"), wait=False,
             idle_seconds=0, poll_seconds=30):
    """两个容量锁限制最多两张卡；第一个锁兼容旧单卡队列。"""
    import fcntl
    lock_root = Path(lock_root)
    queue_lock = None
    announced_capacity = False
    while queue_lock is None:
        for name in (".rq3_single_gpu.lock", ".rq3_second_gpu.lock"):
            candidate = (lock_root / name).open("a")
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                queue_lock = candidate
                break
            except BlockingIOError:
                candidate.close()
        if queue_lock is None:
            if not wait:
                raise BlockingIOError("RQ3 already has two active GPU stages")
            if not announced_capacity:
                print("RQ3_WAITING_FOR_GPU_CAPACITY limit=2", flush=True)
                announced_capacity = True
            time.sleep(30)
    with queue_lock:
        announced = False
        idle_since = {}
        while True:
            for index in gpu_indices(request):
                card_lock = (lock_root / (".rq2_gpu%d.lock" % index)).open("a")
                try:
                    fcntl.flock(card_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    uuid = idle_gpu_uuid(index)
                except (BlockingIOError, RuntimeError):
                    idle_since.pop(index, None)
                    card_lock.close()
                    continue
                except BaseException:
                    card_lock.close()
                    raise
                try:
                    now = time.monotonic()
                    previous = idle_since.get(index)
                    if previous is None or previous[0] != uuid:
                        idle_since[index] = (uuid, now)
                        if idle_seconds:
                            print("RQ3_IDLE_TIMER_STARTED index=%d seconds=%d" %
                                  (index, idle_seconds), flush=True)
                    if now - idle_since[index][1] < idle_seconds:
                        continue
                    # 锁内再次复核，确认连续空闲满时长后仍然空闲。
                    try:
                        confirmed_uuid = idle_gpu_uuid(index)
                    except RuntimeError:
                        idle_since.pop(index, None)
                        continue
                    if confirmed_uuid != uuid:
                        idle_since.pop(index, None)
                        continue
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
            time.sleep(poll_seconds)


class GpuLease:
    """连续空闲满时长后取得一张卡，供本分片后续训练/评估复用。"""

    def __init__(self, request, lock_root, idle_seconds):
        self.request = request
        self.lock_root = lock_root
        self.idle_seconds = idle_seconds
        self.context = None
        self.selected = None

    @contextmanager
    def stage(self):
        if self.selected is not None:
            # 子进程刚退出时NVML可能短暂保留利用率，先等待最多30秒释放。
            for attempt in range(7):
                try:
                    if idle_gpu_uuid(self.selected["index"]) != self.selected["uuid"]:
                        raise RuntimeError("GPU identity changed")
                    break
                except RuntimeError:
                    if attempt < 6:
                        time.sleep(5)
            else:
                # 外部任务仍在占用，归还锁，新卡重新观察完整空闲时长。
                self.close()
        if self.selected is None:
            context = gpu_slot(self.request, self.lock_root, wait=True,
                               idle_seconds=self.idle_seconds)
            selected = context.__enter__()
            self.context, self.selected = context, selected
        yield self.selected

    def close(self):
        if self.context is not None:
            self.context.__exit__(None, None, None)
        self.context = self.selected = None


def guard_before_cuda():
    """RQ3子进程CPU加载完成后，在第一次CUDA模型构建前复核物理卡。"""
    index = os.environ.get("RQ3_GPU_INDEX")
    if index is None:
        return
    if os.environ.get('RQ3_EXTERNAL_GPU_LEASE'):
        if os.environ.get('RQ3_COHORT_LEASE'):
            raise RuntimeError('external and cohort sharing cannot be combined')
        from tools.rq3_external_share import shared_guard
        return shared_guard()
    if os.environ.get('RQ3_COHORT_LEASE'):
        from tools.rq3_colocation import shared_guard
        return shared_guard()
    expected = os.environ["RQ3_GPU_UUID"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
        raise RuntimeError("RQ3 CUDA visibility differs from reserved physical GPU")
    if idle_gpu_uuid(int(index)) != expected:
        raise RuntimeError("RQ3 reserved GPU identity changed during CPU preparation")
    print("RQ3_PRE_CUDA_IDLE_CHECK_PASS index=" + index, flush=True)
