import torch
import numpy as np

from sd.utils import logger


def get_cosine(tensor1: torch.Tensor | np.ndarray,
               tensor2: torch.Tensor | np.ndarray, prefix: str) -> None:
    if not isinstance(tensor1, torch.Tensor):
        tensor1 = torch.tensor(tensor1)
    if not isinstance(tensor2, torch.Tensor):
        tensor2 = torch.tensor(tensor2)

    tensor1 = tensor1.reshape(1, -1).cpu().to(dtype=torch.float32)
    logger.debug(tensor1[:10])
    tensor2 = tensor2.reshape(1, -1).cpu().to(dtype=torch.float32)
    logger.debug(tensor2[:10])

    logger.debug("{} Cosine = {}".format(
        prefix, torch.nn.functional.cosine_similarity(tensor1, tensor2)))
    logger.debug("{} MaxAbs = {}".format(
        prefix, torch.max(torch.abs(tensor1 - tensor2)).item()))
