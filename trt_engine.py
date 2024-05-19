import os
import torch
import contextlib
import tensorrt as trt
import onnxruntime as ort

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TensorInfo:
    name: str
    dtype: trt.DataType
    shape: tuple


def export_onnx(model: torch.nn.Module, in_latents: torch.Tensor,
                timesteps: torch.Tensor, prompt_embeds: torch.Tensor,
                onnx_path: str) -> None:
    torch.onnx.export(
        model,
        (in_latents, timesteps, prompt_embeds),
        onnx_path,
        opset_version=17,
        input_names=["in_latents", "timesteps", "prompt_embeds"],
        output_names=["out_latents"],
    )


def export_engine(onnx_path: str, engine_path: str) -> None:
    logger = trt.Logger(trt.Logger.ERROR)

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    # use float16
    config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as model:
        parser.parse(model.read(), onnx_path)

    inputT = network.get_input(0)
    profile.set_shape(inputT.name, [2, 4, 64, 64], [2, 4, 64, 64],
                      [2, 4, 64, 64])
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    assert engine is not None, "build engine failed"
    with open(engine_path, "wb") as f:
        f.write(engine)


@contextlib.contextmanager
def _scoped_stream():
    stream = torch.cuda.current_stream()
    try:
        yield stream.cuda_stream
    finally:
        stream.synchronize()


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

        return self

    def infer_shapes(
        self,
        inputs: List[TensorInfo],
    ) -> List[TensorInfo]:
        for i in inputs:
            if self.engine.get_tensor_mode(i.name) != trt.TensorIOMode.INPUT:
                raise ValueError(f"Tensor:{i.name} is not an input tensor")
            if self.engine.get_tensor_dtype(i.name) != i.dtype:
                raise ValueError(f"Tensor:{i.name} has wrong dtype")
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

    def __init__(self, engine_path: str, dtype: torch.dtype) -> None:
        super().__init__()

        self.stream = torch.cuda.current_stream().cuda_stream

        # load engine
        with open(engine_path, "rb") as f:
            engine = f.read()
        self.session = Session.from_serialized_engine(engine)
        self.dtype = dtype

    def __call__(
        self,
        in_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        trt_inputs = {
            "in_latents": in_latents,
            "timesteps": timesteps,
            "prompt_embeds": prompt_embeds
        }
        trt_outputs = self.session.infer_shapes([
            TensorInfo("in_latents", trt.DataType.HALF, in_latents.shape),
            TensorInfo("timesteps", trt.DataType.HALF, timesteps.shape),
            TensorInfo("prompt_embeds", trt.DataType.HALF, prompt_embeds.shape)
        ])
        trt_outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device="cuda")
            for t in trt_outputs
        }

        ok = self.session.run(trt_inputs, trt_outputs, self.stream)
        assert ok
        torch.cuda.synchronize()

        return trt_outputs["out_latents"]


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


def trt_dtype_to_torch(dtype):
    ret = _trt_to_torch_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


class OnnxSession:

    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path)

    def __call__(
        self,
        in_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        onnx_inputs = self.sess.get_inputs()
        input_feed: Dict[str, Any] = {
            onnx_inputs[0].name: in_latents.cpu().numpy(),
            onnx_inputs[1].name: timesteps.cpu().numpy(),
            onnx_inputs[2].name: prompt_embeds.cpu().numpy(),
        }
        output_name = self.sess.get_outputs()[0].name
        output = self.sess.run([output_name], input_feed)

        return output[0]


if __name__ == "__main__":
    from utils import Pipeline

    # Export Pytorch Model to ONNX Model
    model_dir = "/data/models/stable-diffusion-v1-4"
    config_name = "model_index.json"
    in_latents = torch.randn((2, 4, 64, 64),
                             dtype=torch.float16,
                             device="cuda")
    timesteps = torch.tensor(1, dtype=torch.float16, device="cuda")
    prompt_embeds = torch.randn((2, 77, 768),
                                dtype=torch.float16,
                                device="cuda")
    model = Pipeline.from_pretrained(model_dir, config_name).cuda()
    onnx_path = os.path.join(model_dir, "unet", "onnx", "model.onnx")
    # export_onnx(model.unet.to(dtype=torch.float16), in_latents, timesteps,
    #             prompt_embeds, onnx_path)

    # Export ONNX Model to Engine
    engine_path = os.path.join(model_dir, "unet", "trt", "model.engine")
    # export_engine(onnx_path, engine_path)

    # Inference engine
    trt_sess = TrtSession(engine_path, dtype=torch.float16)
    trt_output = trt_sess(in_latents, timesteps, prompt_embeds)
    # print("shape={}", trt_output.shape)

    # Compare OnnxRuntime and TensorRT
    ort_sess = OnnxSession(onnx_path)
    ort_output = ort_sess(in_latents, timesteps, prompt_embeds)

    # Inference of PyTorch
    pth_output = model.unet.to(dtype=torch.float16)(in_latents, timesteps,
                                                  prompt_embeds).sample
    pth_output = pth_output.reshape(1, -1).cpu().to(dtype=torch.float32)
    print(">>>> pth: \n", pth_output[:10])

    print("The shape of output: ", ort_output.shape)
    trt_output = trt_output.reshape(1, -1).cpu().to(dtype=torch.float32)
    print(">>>> trt: \n", trt_output[:10])
    ort_output = torch.tensor(ort_output.reshape(1,
                                                 -1)).to(dtype=torch.float32)
    print(">>>> ort: \n", ort_output[:10])
    print("Cosine between onnxruntime and tensorrt: ",
          torch.nn.functional.cosine_similarity(trt_output, ort_output).item())
    max_index = torch.argmax(torch.abs(trt_output - ort_output))
    print("The max difference between onnxruntime and tensorrt: ", max_index,
          torch.abs(trt_output - ort_output).reshape(-1)[max_index])
