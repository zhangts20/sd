import os
import torch
import argparse

from core import Pipeline, pth_unet_infer, ort_unet_infer, trt_unet_infer, man_inference, pth_inference
from utils import export_onnx, export_engine, get_cosine


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd-dir",
                        required=True,
                        type=str,
                        help="The root directory of stable diffusion model.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo, many cars, wide rodes, beautiful scenes",
        help="The input prompt.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to debug cosine between pytorch, onnx and tensorrt.")
    parser.add_argument(
        "--negative-prompts",
        type=str,
        help="Negative promptes, the generated images do not contain these.")
    parser.add_argument(
        "--use-trt",
        action="store_true",
        help="Whether to use TensorRT for the inference of UNet.")
    parser.add_argument(
        "--image-path",
        type=str,
        help="If this value is given, it is img2img; otherwise, txt2img.")

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
    if args.debug:
        pth_output = pth_unet_infer(args.sd_dir, input_feed)
        ort_output = ort_unet_infer(onnx_path, input_feed)
        trt_output = trt_unet_infer(engine_path, input_feed)
        get_cosine(pth_output, ort_output, "Pytorch and Onnxruntime")
        get_cosine(pth_output, trt_output, "Pytorch and Tensorrt")

    # # Inference to generate image.
    image = pth_inference(args.sd_dir, args.prompt, args.negative_prompts)
    image.save("pth_out.jpg")
    image = man_inference(args.sd_dir, args.prompt, args.negative_prompts,
                          args.use_trt)
    image.save("trt_out.jpg")
