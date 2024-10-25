import argparse

from sd.core.inference import infer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",
                        required=True,
                        type=str,
                        help="The directory of stable diffusion models")
    parser.add_argument("--prompt",
                        type=str,
                        default="a photo, cars, rodes, beautiful scenes",
                        help="The positiove prompts")
    parser.add_argument("--negative-prompts",
                        type=str,
                        help="The negative prompts")
    parser.add_argument("--use-trt",
                        action="store_true",
                        help="Whether use TensorRT for UNet")
    parser.add_argument("--image-path",
                        type=str,
                        help="The input path of image")
    parser.add_argument("--output-dir",
                        type=str,
                        default="outputs",
                        help="The output directory of generated image")
    parser.add_argument("--dtype",
                        type=str,
                        default="float32",
                        choices=["float16", "float32"],
                        help="The data type of inference")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        choices=["cpu", "cuda"],
                        help="The dst device to run pipeline")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args_dict = vars(args)

    infer(**args_dict)
