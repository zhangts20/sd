import torch
import numpy as np
import onnxruntime as ort

from typing import Dict

__all__ = ["OnnxSession"]


class OnnxSession:

    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path)

    def __call__(
        self,
        input_feed: Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        # Convert from tensor to numpy.
        input_feed_np = input_feed.copy()
        for (k, v) in input_feed_np.items():
            input_feed_np[k] = v.cpu().numpy()

        # Get output names.
        output = self.sess.run(self.output_names, input_feed_np)

        return output

    @property
    def output_names(self) -> Dict[str, np.ndarray]:
        return [out.name for out in self.sess.get_outputs()]
