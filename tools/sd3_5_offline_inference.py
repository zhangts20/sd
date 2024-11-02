import argparse
import multiprocessing

from sd.core.inference import sd3_5_infer
from sd.utils import monitor_gpu_memory


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
    parser.add_argument("--output-dir",
                        type=str,
                        default="outputs",
                        help="The output directory of generated image")
    parser.add_argument("--dtype",
                        type=str,
                        default="float32",
                        choices=["float16", "bfloat16", "float32"],
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

    stop_event = multiprocessing.Event()
    monitor_process = multiprocessing.Process(target=monitor_gpu_memory,
                                              args=(stop_event, ))
    monitor_process.start()

    sd3_5_infer(**args_dict)

    stop_event.set()
    monitor_process.join()
