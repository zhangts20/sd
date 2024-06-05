import os
import onnx
import torch
import tensorrt as trt

from typing import Dict, List
from onnxsim import simplify
from sd.utils.logger import sd_logger

__all__ = ["export_onnx", "export_engine"]


def export_onnx(model: torch.nn.Module, input_feed: Dict[str, torch.Tensor],
                output_names: List[str], onnx_path: str) -> None:
    input_names = list(input_feed.keys())
    inputs = tuple([input_feed[name] for name in input_names])
    # A temp onnx path.
    base_name, extension = onnx_path.rsplit(".", 1)
    onnx_temp_path = f"{base_name}_tmp.{extension}"
    torch.onnx.export(
        model,
        inputs,
        onnx_temp_path,
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "in_latents": {0: "batch_size", 2: "height", 3: "width"},
        }
    )

    # Simplify
    onnx_model = onnx.load(onnx_temp_path)
    model_sim, check = simplify(onnx_model)
    assert check
    onnx.save(model_sim, onnx_path)

    sd_logger.info(f"Export ONNX to {onnx_path} successfully.")

    os.remove(onnx_temp_path)


def export_engine(onnx_path: str, engine_path: str) -> None:
    logger = trt.Logger(trt.Logger.ERROR)

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    # Set flag of data type.
    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html
    config.set_flag(trt.BuilderFlag.FP16)
    # Parse ONNX model.
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as model:
        parser.parse(model.read(), onnx_path)

    # The first input is dynamic shape
    profile = builder.create_optimization_profile()

    nMin = [1, 4, 64, 64]
    nOpt = [2, 4, 64, 64]
    nMax = [2, 4, 64, 128]

    in_latents = network.get_input(0)
    in_latents.shape = [-1, 4, -1, -1]
    profile.set_shape(in_latents.name, nMin, nOpt, nMax)
    config.add_optimization_profile(profile)

    # Set precision for some layers.
    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/FoundationalTypes/DataType.html?
    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html#ilayer
    trt_precision = trt.DataType.FLOAT
    high_type = []
    for layer in network:
        if layer.type in high_type:
            layer.precision = trt_precision

    engine = builder.build_serialized_network(network, config)
    assert engine is not None, "build engine failed"
    with open(engine_path, "wb") as f:
        f.write(engine)

    sd_logger.info(f"Export Engine to {engine_path} successfully.")
