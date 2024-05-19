import os
import time
import json
import torch
import inspect
import importlib
import numpy as np

from typing import Dict, Any
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

from transformers import CLIPImageProcessor
from diffusers import PNDMScheduler
from transformers import CLIPTextModel
from transformers import CLIPTokenizer
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL

USE_TRT = os.environ.get("USE_TRT", False)
if USE_TRT:
    from trt_engine import TrtSession


def calculate_time(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function {func.__name__} took {round(end_time - start_time, 4)} seconds to execute"
        )
        return result

    return wrapper


def pipeline_infer(model_dir: str, prompt: str, num_warms: int = 1):

    @calculate_time
    def load_model(model_dir: str):
        return StableDiffusionPipeline.from_pretrained(model_dir).to("cuda")

    def warmup(model: torch.nn.Module, prompt: str, num_warms: int):
        for _ in range(num_warms):
            model(prompt)

    @calculate_time
    def p_infer(model: torch.nn.Module, prompt: str) -> Image.Image:
        return model(prompt).images[0]

    model = load_model(model_dir)
    warmup(model, prompt, num_warms)

    image = p_infer(model, prompt)
    image.save("p_out.jpg")


def manual_infer(model_dir: str, prompt: str, num_warms: int = 1):

    @calculate_time
    def load_model(model_dir: str, config_name: str):
        return Pipeline.from_pretrained(model_dir, config_name).cuda()

    def warmup(model: torch.nn.Module, prompt: str, num_warms: int):
        for _ in range(num_warms):
            model(prompt)

    @calculate_time
    def m_infer(model: torch.nn.Module, prompt: str) -> Image.Image:
        return model(prompt)[0]

    model = load_model(model_dir, "model_index.json")
    warmup(model, prompt, num_warms)

    image = m_infer(model, prompt)
    image.save("m_out.jpg")


class Pipeline:

    def __init__(
        self,
        feature_extractor: CLIPImageProcessor,
        scheduler: PNDMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        engine_path: str = None,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet_inchannels = unet.config.in_channels
        self.uset_sample_size = unet.config.sample_size

        if USE_TRT:
            self.unet = TrtSession(engine_path, dtype=torch.float16)
        else:
            self.unet = unet
            del unet

        self.vae = vae
        self.device = "cuda"
        self.dtype = torch.float16
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) -
                                    1)

    @classmethod
    def from_pretrained(cls, model_dir: str, config_name: str) -> "Pipeline":
        # parse config file
        config_path = os.path.join(model_dir, config_name)
        assert os.path.exists(config_path)
        with open(config_path, "r") as f:
            config_dict: Dict = json.load(f)
        # expected modules
        expected_modules = inspect.signature(cls.__init__).parameters
        config_dict = {
            k: v
            for k, v in config_dict.items() if k in expected_modules
        }
        # e.g., name=extractor, library_name=transformers, class_name=CLIPImageProcessor
        init_kwargs: Dict[str, Any] = dict()
        for name, (library_name, class_name) in config_dict.items():
            module = importlib.import_module(library_name)
            cls_ = getattr(module, class_name)
            # call from_pretrained to initialize module
            init_kwargs[name] = cls_.from_pretrained(
                os.path.join(model_dir, name))

        if USE_TRT:
            # Use TensorRT for UNet
            engine_path = os.path.join(model_dir, "unet", "trt",
                                       "model.engine")
            init_kwargs.update({"engine_path": engine_path})

        return cls(**init_kwargs)

    def cuda(self):
        self.text_encoder = self.text_encoder.to(self.device)

        if not USE_TRT:
            self.unet = self.unet.to(self.device)

        self.vae = self.vae.to(self.device)

        return self

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ):
        # Encode input prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self.encode_prompt(prompt, do_classifier_free_guidance)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        latents = self.prepare_latents(
            batch_size=1,
            num_channels_latents=self.unet_inchannels,
            height=self.uset_sample_size * self.vae_scale_factor,
            width=self.uset_sample_size * self.vae_scale_factor,
            dtype=prompt_embeds.dtype,
            device=self.device)

        # Denoising loog
        latents = self.unet_infer(timesteps, latents, prompt_embeds,
                                  do_classifier_free_guidance, guidance_scale)

        # Postprocess
        image = self.decode_latents(latents)

        # Convert from numpy to PIL
        image = self.numpy_to_pil(image)

        return image

    @calculate_time
    def encode_prompt(
        self,
        prompt: str,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        text_input_ids: torch.Tensor = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt").input_ids

        prompt_embeds: torch.Tensor = self.text_encoder(
            text_input_ids.to(device=self.device))[0]
        prompt_embeds = prompt_embeds.to(self.text_encoder.dtype)

        if do_classifier_free_guidance:
            max_length = prompt_embeds.shape[1]
            uncond_input_ids: torch.Tensor = self.tokenizer(
                [""],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt").input_ids

            negative_prompt_embeds: torch.Tensor = self.text_encoder(
                uncond_input_ids.to(device=self.device))[0].to(
                    self.text_encoder.dtype)
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.text_encoder.dtype)

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    @calculate_time
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        shape = (batch_size, num_channels_latents,
                 height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        latents = torch.randn(shape, dtype=dtype, device=device)

        return latents * self.scheduler.init_noise_sigma

    @calculate_time
    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        return image

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> Image.Image:
        if images.ndim == 3:
            images = images[None, ...]

        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            pil_images = [
                Image.fromarray(image.squeeze(), mode="L") for image in images
            ]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @calculate_time
    def unet_infer(self, timesteps: int, latents: torch.Tensor,
                   prompt_embeds: torch.Tensor,
                   do_classifier_free_guidance: bool,
                   guidance_scale: float) -> torch.Tensor:
        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            if USE_TRT:
                noise_pred = self.unet(latent_model_input, t, prompt_embeds)
            else:
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents
