import os
import json
import torch
import inspect
import importlib
import numpy as np

from PIL import Image
from tqdm import tqdm
from typing import Dict, Any
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import PNDMScheduler, UNet2DConditionModel, AutoencoderKL
from sd.utils import calculate_time

__all__ = ["Txt2ImgPipeline"]


class Txt2ImgPipeline:

    def __init__(
        self,
        feature_extractor: CLIPImageProcessor,
        scheduler: PNDMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        engine_path: str = None,
        use_trt: bool = False,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet_inchannels = unet.config.in_channels
        self.uset_sample_size = unet.config.sample_size

        if use_trt:
            from sd.backend import TrtSession
            self.unet = TrtSession(engine_path, dtype=torch.float16)
            del unet
        else:
            self.unet = unet

        self.vae = vae
        self.device = "cuda"
        self.dtype = torch.float16
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) -
                                    1)
        self.use_trt = use_trt

    @classmethod
    def from_pretrained(cls,
                        model_dir: str,
                        config_name: str,
                        use_trt: bool = False) -> "Txt2ImgPipeline":
        # Parse config file.
        config_path = os.path.join(model_dir, config_name)
        assert os.path.exists(config_path)
        with open(config_path, "r") as f:
            config_dict: Dict = json.load(f)
        # Get expected modules.
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
            # Call from_pretrained to initialize module.
            init_kwargs[name] = cls_.from_pretrained(
                os.path.join(model_dir, name))

        if use_trt:
            # Use TensorRT for UNet.
            engine_path = os.path.join(model_dir, "unet", "trt",
                                       "model.engine")
            assert os.path.exists(engine_path), f"{engine_path} doesn't exist"
            init_kwargs.update({
                "engine_path": engine_path,
                "use_trt": use_trt
            })

        return cls(**init_kwargs)

    def cuda(self):
        self.text_encoder = self.text_encoder.to(self.device)

        if not self.use_trt:
            self.unet = self.unet.to(self.device)

        self.vae = self.vae.to(self.device)

        return self

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompts: str = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        # Encode input prompt, and negative prompt when guidance_scale larger than 1.0.
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self.encode_prompt(prompt,
                                           do_classifier_free_guidance,
                                           negative_prompts=negative_prompts)

        # TODO: Prepare timesteps.
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Prepare latent used as input of UNet.
        latents = self.prepare_latents(
            batch_size=1,
            num_channels_latents=self.unet_inchannels,
            height=self.uset_sample_size * self.vae_scale_factor,
            width=self.uset_sample_size * self.vae_scale_factor,
            dtype=prompt_embeds.dtype,
            device=self.device)

        # Denoising.
        latents = self.unet_infer(timesteps, latents, prompt_embeds,
                                  do_classifier_free_guidance, guidance_scale)

        # Postprocess.
        image = self.decode_latents(latents)

        # Convert from numpy to PIL.
        image = self.numpy_to_pil(image)

        return image

    @calculate_time(show=True)
    def encode_prompt(
        self,
        prompt: str,
        do_classifier_free_guidance: bool,
        negative_prompts: str = None,
    ) -> torch.Tensor:
        # Pad input to model_max_length and do truncation if necessary.
        text_input_ids: torch.Tensor = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt").input_ids

        # (bs, model_max_length, hidden_size)
        prompt_embeds: torch.Tensor = self.text_encoder(
            text_input_ids.to(device=self.device))[0]
        prompt_embeds = prompt_embeds.to(self.text_encoder.dtype)

        # Add negative prompt.
        if do_classifier_free_guidance:
            if negative_prompts is None:
                negative_prompts = [""]
            max_length = prompt_embeds.shape[1]
            uncond_input_ids: torch.Tensor = self.tokenizer(
                negative_prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt").input_ids

            negative_prompt_embeds: torch.Tensor = self.text_encoder(
                uncond_input_ids.to(device=self.device))[0].to(
                    self.text_encoder.dtype)

            # (2 * bs, model_max_length, hidden_size)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    @calculate_time(show=True)
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

    @calculate_time(show=True)
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

    @calculate_time(show=True)
    def unet_infer(self, timesteps: int, latents: torch.Tensor,
                   prompt_embeds: torch.Tensor,
                   do_classifier_free_guidance: bool,
                   guidance_scale: float) -> torch.Tensor:
        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            if self.use_trt:
                input_feed = {
                    "in_latents": latent_model_input.to(self.unet.dtype),
                    "timesteps": t,
                    "prompt_embeds": prompt_embeds.to(self.unet.dtype),
                }
                # The dtype is determined by the dtype when initializing TrtSession.
                output = self.unet(input_feed)
                output_names = self.unet.output_names
                assert len(output_names) == 1
                noise_pred = output[output_names[0]].to(
                    latent_model_input.dtype)
            else:
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds).sample
            assert (not torch.isnan(noise_pred).any()
                    ), "The output of trt has nan."

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents
