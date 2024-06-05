import os
import torch
import numpy as np

from PIL import Image
from typing import Dict
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from sd.core import Txt2ImgPipeline, Img2ImgPipeline
from sd.utils import calculate_time, sd_logger
from sd.backend import OnnxSession, TrtSession


@torch.no_grad()
def pth_img2img_inference(
    model_dir,
    prompt: str,
    image_path: str,
    negative_prompts: str = None,
    num_warmups: int = 1,
) -> Image.Image:

    @calculate_time(show=True)
    def load_model(model_dir: str):
        return StableDiffusionImg2ImgPipeline.from_pretrained(model_dir).to(
            "cuda")

    def warmup(
        model: torch.nn.Module,
        prompt: str,
        image: Image.Image,
        num_warmups: int,
    ) -> None:
        sd_logger.debug("Begin warmup.")
        for _ in range(num_warmups):
            model(prompt, image)
        sd_logger.debug("Finish warmup.")

    @calculate_time(show=True)
    def p_infer(
        model: torch.nn.Module,
        prompt: str,
        image: Image.Image,
        negative_prompts: str,
    ) -> Image.Image:
        return model(prompt, image, negative_prompt=negative_prompts).images[0]

    # load image
    image = Image.open(image_path).convert("RGB")

    model = load_model(model_dir)
    warmup(model, prompt, image, num_warmups)

    return p_infer(model, prompt, image, negative_prompts)


@torch.no_grad()
def img2img_inference(model_dir: str,
                      prompt: str,
                      image_path: str,
                      negative_prompts: str = None,
                      use_trt: bool = False,
                      num_warmups: int = 1) -> Image.Image:

    if use_trt:
        sd_logger.info("Use TensorRT for the inference of UNet.")

    @calculate_time(show=True)
    def load_model(model_dir: str, config_name: str):
        return Img2ImgPipeline.from_pretrained(model_dir,
                                               config_name,
                                               use_trt=use_trt).cuda()

    def warmup(
        model: torch.nn.Module,
        prompt: str,
        image: Image.Image,
        num_warmups: int,
    ) -> None:
        sd_logger.debug("Begin to warmup...")
        for _ in range(num_warmups):
            model(prompt, image)
        sd_logger.debug("Finish to warmup..")

    @calculate_time(show=True)
    def m_infer(
        model: torch.nn.Module,
        prompt: str,
        image: Image.Image,
        negative_prompts: str,
    ) -> Image.Image:
        return model(prompt, image, negative_prompts)[0]

    # load image
    image = Image.open(image_path).convert("RGB")

    model = load_model(model_dir, "model_index.json")
    warmup(model, prompt, image, num_warmups)

    return m_infer(model, prompt, image, negative_prompts)


@torch.no_grad()
def pth_txt2img_inference(model_dir: str,
                          prompt: str,
                          negative_prompts: str = None,
                          num_warmups: int = 1) -> Image.Image:

    @calculate_time(show=True)
    def load_model(model_dir: str):
        return StableDiffusionPipeline.from_pretrained(model_dir).to("cuda")

    def warmup(
        model: torch.nn.Module,
        prompt: str,
        num_warmups: int,
    ) -> None:
        sd_logger.debug("Begin warmup.")
        for _ in range(num_warmups):
            model(prompt)
        sd_logger.debug("Finish warmup.")

    @calculate_time(show=True)
    def p_infer(
        model: torch.nn.Module,
        prompt: str,
        negative_prompts: str,
    ) -> Image.Image:
        return model(prompt, negative_prompt=negative_prompts).images[0]

    model = load_model(model_dir)
    warmup(model, prompt, num_warmups)

    return p_infer(model, prompt, negative_prompts)


@torch.no_grad()
def txt2img_inference(model_dir: str,
                      prompt: str,
                      negative_prompts: str = None,
                      use_trt: bool = False,
                      num_warmups: int = 1) -> Image.Image:

    if use_trt:
        sd_logger.info("Use TensorRT for the inference of UNet.")

    @calculate_time(show=True)
    def load_model(model_dir: str, config_name: str):
        return Txt2ImgPipeline.from_pretrained(model_dir,
                                               config_name,
                                               use_trt=use_trt).cuda()

    def warmup(model: torch.nn.Module, prompt: str, num_warmups: int):
        sd_logger.debug("Begin to warmup...")
        for _ in range(num_warmups):
            model(prompt)
        sd_logger.debug("Finish to warmup..")

    @calculate_time(show=True)
    def m_infer(
        model: torch.nn.Module,
        prompt: str,
        negative_prompts: str,
    ) -> Image.Image:
        return model(prompt, negative_prompts)[0]

    model = load_model(model_dir, "model_index.json")
    warmup(model, prompt, num_warmups)

    return m_infer(model, prompt, negative_prompts)


@torch.no_grad()
def pth_unet_infer(
        model_dir: str,
        input_feed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Use dtype and device to initialize UNet.
    in_latents = input_feed.get("in_latents")

    model = UNet2DConditionModel.from_pretrained(
        os.path.join(model_dir, "unet")).to(dtype=in_latents.dtype,
                                            device=in_latents.device)

    output: UNet2DConditionOutput = model(
        sample=in_latents,
        timestep=input_feed.get("timesteps"),
        encoder_hidden_states=input_feed.get("prompt_embeds"))

    return output.sample


def ort_unet_infer(
        onnx_path: str,
        input_feed: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    session = OnnxSession(onnx_path)

    output = session(input_feed)
    assert isinstance(output, list) and len(output) == 1

    return output[0]


def trt_unet_infer(engine_path: str,
                   input_feed: Dict[str, torch.Tensor]) -> torch.Tensor:
    session = TrtSession(engine_path)
    assert len(session.output_names) == 1

    return session(input_feed).get(session.output_names[0])
