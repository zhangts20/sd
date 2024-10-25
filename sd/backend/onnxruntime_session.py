import torch
import numpy as np
import onnxruntime as ort

from typing import Dict


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
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            input_feed_np[k] = v

        # Get output names.
        outputs = self.sess.run(self.output_names, input_feed_np)

        return outputs

    @property
    def output_names(self) -> Dict[str, np.ndarray]:
        return [out.name for out in self.sess.get_outputs()]
