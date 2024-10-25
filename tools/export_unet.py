import os
import onnx
import torch
import argparse
import subprocess
import tensorrt as trt

from onnxsim import simplify
from typing import Dict, List

from sd.utils import sd_logger
from sd.core.pipeline import Pipeline


def export_onnx(
    model: torch.nn.Module,
    input_feed: Dict[str, torch.Tensor],
    output_names: List[str],
    onnx_path: str,
) -> None:
    input_names = list(input_feed.keys())
    inputs = tuple(input_feed[name] for name in input_names)
    # temp onnx path
    base_name, extension = onnx_path.rsplit(".", 1)
    onnx_temp_path = f"{base_name}_temp.{extension}"
    torch.onnx.export(model,
                      args=inputs,
                      f=onnx_temp_path,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=17,
                      dynamic_axes={
                          "in_latents": {
                              0: "num_prompts",
                              2: "height",
                              3: "width",
                          },
                          "prompt_embeds": {
                              0: "num_prompts",
                          }
                      })

    onnx_temp_model = onnx.load(onnx_temp_path)
    model_sim, check = simplify(onnx_temp_model)
    if not check:
        sd_logger.error("failed to simplify onnx model")
    onnx.save(model_sim,
              onnx_path,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location="model.onnx.data")

    sd_logger.info(f"succeed to export onnx to {onnx_path}")
    onnx_dir = os.path.dirname(onnx_path)
    cmd = f"find {onnx_dir}/* -type f | grep -v '\(model.onnx\|model.onnx.data\)' | xargs rm"
    ret = subprocess.run(cmd, shell=True).returncode
    if ret != 0:
        raise RuntimeError("Error when converting onnx weights.")


def export_engine(onnx_path: str, engine_path: str, dtype: str) -> None:
    logger = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html
    # default is float32
    config = builder.create_builder_config()
    if dtype == "float16":
        config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as model:
        parser.parse(model.read(), onnx_path)

    # in_latents
    nMin = [1, 4, 32, 32]
    nOpt = [2, 4, 64, 64]
    nMax = [2, 4, 128, 128]
    in_latents = network.get_input(0)
    profile.set_shape(in_latents.name, nMin, nOpt, nMax)
    # prompt_embeds
    nMin = [1, 77, 768]
    nOpt = [2, 77, 768]
    nMax = [2, 77, 768]
    prompt_embeds = network.get_input(2)
    profile.set_shape(prompt_embeds.name, nMin, nOpt, nMax)
    config.add_optimization_profile(profile)

    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/FoundationalTypes/DataType.html?
    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html#ilayer
    if dtype == "float16":
        trt_precision = trt.DataType.FLOAT
        high_precision_ops = []
        for layer in network:
            if layer.type in high_precision_ops:
                layer.precision = trt_precision

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        sd_logger.error("failed to build engine")
    with open(engine_path, "wb") as f:
        f.write(engine)

    sd_logger.info(f"succeed to export engine to {engine_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",
                        required=True,
                        type=str,
                        help="The directory of stable diffusion models")
    parser.add_argument("--force-onnx",
                        action="store_true",
                        help="Whether to export onnx forcibly")
    parser.add_argument("--force-engine",
                        action="store_true",
                        help="Whether to export engine forcibly")
    parser.add_argument("--dtype",
                        type=str,
                        default="float32",
                        choices=["float16", "float32"],
                        help="The data type of exported unet engine")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        choices=["cpu", "cuda"],
                        help="The dst device to run pipeline")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    in_latents = torch.randn((2, 4, 64, 64),
                             dtype=torch.float32,
                             device=args.device)
    timesteps = torch.tensor(1, dtype=torch.int64, device=args.device)
    prompt_embeds = torch.randn((2, 77, 768),
                                dtype=torch.float32,
                                device=args.device)
    input_feed = {
        "in_latents": in_latents,
        "timesteps": timesteps,
        "prompt_embeds": prompt_embeds,
    }

    # Keep the data type of model is torch.float32.
    model_dir = args.model_dir
    pl = Pipeline.from_pretrained(model_dir=model_dir,
                                  use_trt=False,
                                  pipeline="txt2img",
                                  dtype=args.dtype,
                                  device=args.device)
    pl = pl.cuda()

    onnx_path = os.path.join(model_dir, "unet", "onnx", "model.onnx")
    if not os.path.exists(os.path.dirname(onnx_path)):
        os.makedirs(os.path.dirname(onnx_path))
    if not os.path.exists(onnx_path) or args.force_onnx:
        export_onnx(model=pl.unet,
                    input_feed=input_feed,
                    output_names=["out_latents"],
                    onnx_path=onnx_path)

    engine_path = os.path.join(model_dir, "unet", "trt", "model.engine")
    if not os.path.exists(os.path.dirname(engine_path)):
        os.makedirs(os.path.dirname(engine_path))
    if not os.path.exists(engine_path) or args.force_engine:
        export_engine(onnx_path=onnx_path,
                      engine_path=engine_path,
                      dtype=args.dtype)
