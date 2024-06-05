import os
import torch
import argparse

from sd.core import Pipeline, pth_unet_infer, ort_unet_infer, trt_unet_infer
from sd.utils import export_onnx, export_engine, get_cosine


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd-dir",
                        required=True,
                        type=str,
                        help="The root directory of stable diffusion model.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dtype = torch.float16
    device = "cuda"

    # Export pytorch model to onnx model. Unet has three inputs, whose shape is
    # [2,4,64,64], scaler and [2,77,768], type is float16, int64 and float16.
    in_latents = torch.randn((2, 4, 64, 64),
                             dtype=torch.float16,
                             device=device)
    timesteps = torch.tensor(1, dtype=torch.int64, device=device)
    prompt_embeds = torch.randn((2, 77, 768),
                                dtype=torch.float16,
                                device=device)
    input_feed = {
        "in_latents": in_latents,
        "timesteps": timesteps,
        "prompt_embeds": prompt_embeds,
    }
    # Initialize pytorch pipeline.
    pipeline = Pipeline.from_pretrained(args.sd_dir, "model_index.json").cuda()
    onnx_path = os.path.join(args.sd_dir, "unet", "onnx", "model.onnx")
    if not os.path.exists(onnx_path):
        export_onnx(pipeline.unet.to(dtype=dtype),
                    input_feed,
                    output_names=["out_latents"],
                    onnx_path=onnx_path)

    # Export onnx model to engine model.
    engine_path = os.path.join(args.sd_dir, "unet", "trt", "model.engine")
    if not os.path.exists(engine_path):
        export_engine(onnx_path, engine_path)

    # Compare cosine between pytorch, onnxruntime and tensorrt.
    pth_output = pth_unet_infer(args.sd_dir, input_feed)
    ort_output = ort_unet_infer(onnx_path, input_feed)
    trt_output = trt_unet_infer(engine_path, input_feed)
    get_cosine(pth_output, ort_output, "Pytorch and Onnxruntime")
    get_cosine(pth_output, trt_output, "Pytorch and Tensorrt")
