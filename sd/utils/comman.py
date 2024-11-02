import time
import torch

from sd.utils import sd_logger
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlShutdown,
)


def calculate_time(show=True):

    def wrapper(func):

        def inner_func(*args, **kwargs):
            torch.cuda.synchronize()
            if show:
                start_time = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            if show:
                end_time = time.time()
                sd_logger.info(
                    f"function {func.__name__} took {round(end_time - start_time, 4)} seconds to execute"
                )
            return result

        return inner_func

    return wrapper


def monitor_gpu_memory(event, interval: float = 0.1, device_index: int = 0):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)

    peak_memory = 0.0
    try:
        while not event.is_set():
            info = nvmlDeviceGetMemoryInfo(handle)
            used_memory = info.used / float(1024**2)

            if used_memory > peak_memory:
                peak_memory = used_memory

            time.sleep(interval)
    finally:
        nvmlShutdown()
        sd_logger.info(f"the peak memory is {peak_memory:.2f} MB")
