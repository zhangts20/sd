import os
import json
import torch
import importlib
import inspect
import numpy as np

from PIL import Image
from typing import Dict, Any, List
from sd.utils.logger import sd_logger
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.models.autoencoders import AutoencoderKL
from transformers import (
    CLIPTextModel,
    T5EncoderModel,
    CLIPTokenizer,
    T5TokenizerFast,
)


class FLUX1Pipeline(object):

    def __init__(
        self,
        schduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: CLIPTextModel,
        text_encoder_2: T5EncoderModel,
        tokenizer: CLIPTokenizer,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        vae: AutoencoderKL,
        dtype: str,
        device: str,
        engine_path: str = None,
        use_trt: bool = False, 
    ) -> None:
        self.scheduler = schduler
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.vae = vae

        self.vae_scale_factor = (2**(len(self.vae.config.block_out_channels) -
                                     1) if self.vae is not None else 8)
        self.tokenizer_max_length = (self.tokenizer.model_max_length
                                     if self.tokenizer is not None else 77)
        self.default_sample_size = 128

        self.unet_in_channels = transformer.config.in_channels // 4
        self.unet_sample_size = (transformer.config.sample_size 
                                 if transformer is not None else 128)

        self.dtype = getattr(torch, dtype)
        self.device = device

    @classmethod
    def from_pretrained(cls, model_dir: str, use_trt: bool, dtype: str,
                        device: str) -> "FLUX1Pipeline":
        # Parse config file
        config_path = os.path.join(model_dir, "model_index.json")
        if not os.path.exists(config_path):
            sd_logger.error(f"config file {config_path} doesn't exist")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Load modules expected in config file
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
            init_kwargs[name] = cls_.from_pretrained(
                os.path.join(model_dir, name),
                torch_dtype=getattr(torch, dtype))
        init_kwargs.update({
            "dtype": dtype,
            "device": device,
        })

        # Use TensorRT for transformer inference
        if use_trt:
            engine_path = os.path.join(model_dir, "transformer", "trt",
                                       "model.engine")
            if not os.path.exists(engine_path):
                sd_logger.error(
                    f"input {engine_path} doesn't exist, please refer to export_sd3_5_unet.py firstly"
                )
            init_kwargs.update({
                "engine_path": engine_path,
                "use_trt": use_trt,
            })

        return cls(**init_kwargs)

    @torch.no_grad
    def forward(
        self,
        prompt: str,
        prompt_2: str,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
    ) -> Image.Image:
        self.do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
        )

        # Prepare latent variables
        latents, latent_image_ids = self.prepare_latents(
            batch_size=1,
            num_channels_latents=self.unet_in_channels,
            height=self.unet_sample_size * self.vae_scale_factor,
            width=self.unet_sample_size * self.vae_scale_factor,
        ) 

        # TODO: Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu =  self.calculate_shift(
            image_seq_len=image_seq_len,
            base_seq_len=self.scheduler.config.base_image_seq_len,
            base_shift=self.scheduler.config.base_shift,
            max_shift=self.scheduler.config.max_shift,
        )


    @staticmethod
    def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
    
        return mu

    def retrieve_timesteps(
        scheduler,
        num_inference_steps: int,
        sigmas: float,
    ):
        accept_sigmas = "sigmas" in set()

    def _get_clip_prompt_embeds(self,
                                batch_size: int,
                                prompt: List[str],
                                num_images_per_prompt: int = 1):
        prompt_input_ids: torch.Tensor = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )["input_ids"]

        prompt_embeds = self.text_encoder(prompt_input_ids.to(self.device),
                                          output_hidden_states=False)

        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, 
                                         device=self.device)

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def _get_t5_prompt_embeds(self,
                              batch_size: int,
                              prompt: List[str],
                              num_images_per_prompt: int = 1,
                              max_sequence_length: int = 256):
        prompt_input_ids: torch.Tensor = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt"
        )["input_ids"]

        prompt_embeds = self.text_encoder_2(prompt_input_ids.to(self.device),
                                            output_hidden_states=False)
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype,
                                         device=self.device)
        
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str,

    ) -> torch.Tensor:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        pooled_prompt_embeds = self._get_clip_prompt_embeds(
            batch_size=batch_size,
            prompt=prompt,
        )
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt_2,
        )
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            dtype=self.text_encoder.dtype, 
            device=self.device)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def prepare_latents(self, batch_size: int, num_channels_latents: int,
                        height: int, width: int):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2)) 
        shape = (batch_size, num_channels_latents, height, width)

        latents = torch.randn(shape, dtype=self.dtype, device=self.device)
        
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        # Prepare image ids
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        latent_image_ids = latent_image_ids.to(dtype=self.dtype, device=self.device)

        return latents, latent_image_ids
