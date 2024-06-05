import time
import torch

from sd.utils import logger


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
                logger.info(
                    f"Function {func.__name__} took {round(end_time - start_time, 4)} seconds to execute"
                )
            return result

        return inner_func

    return wrapper
