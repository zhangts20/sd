import os
import torch
import numpy as np

from PIL import Image
from utils import calculate_time
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from core import Pipeline
from typing import Dict
from backend import OnnxSession, TrtSession


@torch.no_grad()
def pth_inference(model_dir: str,
                  prompt: str,
                  num_warmups: int = 1) -> Image.Image:

    @calculate_time
    def load_model(model_dir: str):
        return StableDiffusionPipeline.from_pretrained(model_dir).to("cuda")

    def warmup(model: torch.nn.Module, prompt: str, num_warmups: int):
        for _ in range(num_warmups):
            model(prompt)

    @calculate_time
    def p_infer(model: torch.nn.Module, prompt: str) -> Image.Image:
        return model(prompt).images[0]

    model = load_model(model_dir)
    warmup(model, prompt, num_warmups)

    return p_infer(model, prompt)


@torch.no_grad()
def man_inference(model_dir: str,
                  prompt: str,
                  num_warmups: int = 1) -> Image.Image:

    @calculate_time
    def load_model(model_dir: str, config_name: str):
        return Pipeline.from_pretrained(model_dir, config_name,
                                        use_trt=True).cuda()

    def warmup(model: torch.nn.Module, prompt: str, num_warmups: int):
        for _ in range(num_warmups):
            model(prompt)

    @calculate_time
    def m_infer(model: torch.nn.Module, prompt: str) -> Image.Image:
        return model(prompt)[0]

    model = load_model(model_dir, "model_index.json")
    warmup(model, prompt, num_warmups)

    return m_infer(model, prompt)


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
