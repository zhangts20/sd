import os
import argparse

from sd.core import img2img_inference


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
        "--negative-prompts",
        type=str,
        help="Negative promptes, the generated images do not contain these.")
    parser.add_argument(
        "--use-trt",
        action="store_true",
        help="Whether to use TensorRT for the inference of UNet.")
    parser.add_argument("--out-img-path",
                        type=str,
                        default="out/out.jpg",
                        help="The path of generated image path.")
    parser.add_argument(
        "--use-pipeline",
        action="store_true",
        help="Whether to use pipeline to inference, out is `out/pipeline.jpg`.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    dirname = os.path.dirname(os.path.join(os.getcwd(), args.out_img_path))
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    if args.use_pipeline:
        from sd.core import pth_img2img_inference
        image = pth_img2img_inference(args.sd_dir, args.prompt,
                                      args.negative_prompts)
        image.save("pipeline.jpg")

    image = img2img_inference(args.sd_dir, args.prompt, args.negative_prompts,
                              args.use_trt)
    image.save(args.out_img_path)
