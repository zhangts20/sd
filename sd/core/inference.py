import os

from PIL import Image

from sd.core.pipeline import Pipeline
from sd.core.sd3_5_pipeline import SD35Pipeline
from sd.utils import sd_logger


def infer(
    model_dir: str,
    prompt: str,
    negative_prompts: str,
    use_trt: bool,
    image_path: str = None,
    num_wramups: int = 1,
    output_dir: str = "outputs",
    dtype: str = "float32",
    device: str = "cuda",
) -> None:
    # Init pipeline.
    pipeline = "txt2img" if image_path is None else "img2img"
    sd_logger.info(f"Current inference pipeline is {pipeline}")
    pl: Pipeline = Pipeline.from_pretrained(model_dir=model_dir,
                                            use_trt=use_trt,
                                            pipeline=pipeline,
                                            dtype=dtype,
                                            device=device)
    pl = pl.cuda()

    image = None
    if pipeline == "img2img":
        image = Image.open(image_path).convert("RGB")

    # Warmup.
    for _ in range(num_wramups):
        pl.forward(prompt=prompt,
                   negative_prompts=negative_prompts,
                   image=image)
    sd_logger.info(f"{num_wramups} warmups done")

    # inference
    out_image = pl.forward(prompt=prompt,
                           negative_prompts=negative_prompts,
                           image=image)[0]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, f"{pipeline}_generated.jpg")
    out_image.save(output_path)
    sd_logger.info(f"output image has been save to {output_path}")


def sd3_5_infer(
    model_dir: str,
    prompt: str,
    negative_prompts: str,
    use_trt: bool,
    num_wramups: int = 1,
    output_dir: str = "outputs",
    dtype: str = "float32",
    device: str = "cuda",
) -> None:
    # Init pipeline.
    pl: SD35Pipeline = SD35Pipeline.from_pretrained(model_dir=model_dir,
                                                    use_trt=use_trt,
                                                    dtype=dtype,
                                                    device=device)
    pl = pl.cuda()

    # Warmup.
    for _ in range(num_wramups):
        pl.forward(prompt=prompt, negative_prompts=negative_prompts)
    sd_logger.info(f"{num_wramups} warmups done")

    # Inference.
    out_image = pl.forward(prompt=prompt, negative_prompts=negative_prompts)[0]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, f"sd3_5_generated.jpg")
    out_image.save(output_path)
    sd_logger.info(f"output image has been save to {output_path}")
