import torch
import contextlib
import tensorrt as trt

from dataclasses import dataclass
from typing import Any, Dict, List


@contextlib.contextmanager
def _scoped_stream():
    stream = torch.cuda.current_stream()
    try:
        yield stream.cuda_stream
    finally:
        stream.synchronize()


@dataclass
class TensorInfo:
    name: str
    dtype: trt.DataType
    shape: tuple


_trt_to_torch_dtype_dict = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int64: torch.int64,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
    trt.bfloat16: torch.bfloat16,
    trt.fp8: torch.float8_e4m3fn,
}

_torch_to_trt_dtype_dict = {
    torch.float16: trt.DataType.HALF,
    torch.float32: trt.DataType.FLOAT,
    torch.int64: trt.DataType.INT64,
}


def trt_dtype_to_torch(dtype):
    ret = _trt_to_torch_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.9.0/tensorrt_llm/runtime/session.py#L52
class Session:

    def __init__(self):
        self.logger = trt.Logger(trt.Logger.ERROR)

    def from_serialized_engine(engine: bytes) -> "Session":
        return Session()._init(engine)

    def _init(self, engine_buffer=None):
        trt.init_libnvinfer_plugins(None, "")

        self._runtime = trt.Runtime(self.logger)
        if engine_buffer is not None:
            self._engine = self._runtime.deserialize_cuda_engine(engine_buffer)

        self._context = self._engine.create_execution_context()
        with _scoped_stream() as stream:
            self._context.set_optimization_profile_async(0, stream)

        self.input_names: List[str] = list()
        self.output_names: List[str] = list()
        input_output_names = [engine for engine in self._engine]
        for name in input_output_names:
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        return self

    def infer_shapes(
        self,
        inputs: List[TensorInfo],
    ) -> List[TensorInfo]:
        for i in inputs:
            if self.engine.get_tensor_mode(i.name) != trt.TensorIOMode.INPUT:
                raise ValueError(f"Tensor: {i.name} is not an input tensor")
            if self.engine.get_tensor_dtype(i.name) != i.dtype:
                raise ValueError(f"Tensor: {i.name} has wrong dtype")
            if not self.context.set_input_shape(i.name, i.shape):
                raise RuntimeError(
                    f"Could not set shape {i.shape} for tensor {i.name}. Please check the profile range for which your model was build."
                )

        outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.context.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
                outputs.append(TensorInfo(name, dtype, shape))

        return outputs

    def run(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        stream: torch.cuda.stream,
    ) -> bool:
        for tensor_name in inputs:
            tensor = inputs[tensor_name]
            ptr = tensor.data_ptr() if isinstance(tensor,
                                                  torch.Tensor) else tensor
            self.context.set_tensor_address(tensor_name, ptr)

        for tensor_name in outputs:
            tensor = outputs[tensor_name]
            ptr = tensor.data_ptr() if isinstance(tensor,
                                                  torch.Tensor) else tensor
            self.context.set_tensor_address(tensor_name, ptr)

        ok = self.context.execute_async_v3(stream)

        return ok

    @property
    def runtime(self) -> trt.Runtime:
        return self._runtime

    @property
    def context(self) -> trt.IExecutionContext:
        return self._context

    @property
    def engine(self) -> trt.ICudaEngine:
        return self._engine


class TrtSession:

    def __init__(self,
                 engine_path: str,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype
        self.stream = torch.cuda.current_stream().cuda_stream

        # Load engine.
        with open(engine_path, "rb") as f:
            engine = f.read()
        self.session: Session = Session.from_serialized_engine(engine)

    def __call__(
        self,
        input_feed: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        trt_outputs = self.session.infer_shapes([
            TensorInfo(k, _torch_to_trt_dtype_dict.get(v.dtype), v.shape)
            for (k, v) in input_feed.items()
        ])
        trt_outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device="cuda")
            for t in trt_outputs
        }

        ok = self.session.run(input_feed, trt_outputs, self.stream)
        assert ok
        torch.cuda.synchronize()

        return trt_outputs

    @property
    def output_names(self) -> List[str]:
        return self.session.output_names
