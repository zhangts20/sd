import os
import torch
import json
import inspect
import importlib

from tqdm import tqdm
from PIL import Image
from typing import Any, Dict, List
from sd.utils import sd_logger, calculate_time
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)


class SD35Pipeline(object):

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        dtype: str,
        device: str,
    ) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2
        self.text_encoder_3 = text_encoder_3
        self.tokenizer_3 = tokenizer_3

        self.vae_scale_factor = (2**(len(self.vae.config.block_out_channels) -
                                     1) if self.vae is not None else 8)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (self.tokenizer.model_max_length
                                     if self.tokenizer is not None else 77)
        self.unet_sample_size = (self.transformer.config.sample_size
                                 if self.transformer is not None else 128)
        self.patch_size = (self.transformer.config.patch_size
                           if self.transformer is not None else 2)

        self.dtype = getattr(torch, dtype)
        self.device = device

    def cuda(self):
        self.transformer = self.transformer.to(self.device)
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.text_encoder_2 = self.text_encoder_2.to(self.device)
        self.text_encoder_3 = self.text_encoder_3.to(self.device)

        return self

    @classmethod
    def from_pretrained(cls, model_dir: str, dtype: str,
                        device: str) -> "SD35Pipeline":
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

        return cls(**init_kwargs)

    @torch.no_grad
    def forward(
        self,
        prompt: str,
        prompt_2: str = None,
        prompt_3: str = None,
        negative_prompts: str = None,
        negative_prompts_2: str = None,
        negative_prompts_3: str = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
    ) -> Image.Image:
        self.do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt, and negative prompt when guidance_scale larger than 1.0.
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt)
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds],
                                      dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # TODO: Prepare timesteps.
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps,
                                     device=self.device)
        timesteps = self.scheduler.timesteps

        # Prepare latent used as input of UNet.
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=self.unet_sample_size * self.vae_scale_factor,
            width=self.unet_sample_size * self.vae_scale_factor)

        # Denoising.
        latents = self.unet_infer(timesteps=timesteps,
                                  latents=latents,
                                  prompt_embeds=prompt_embeds,
                                  pooled_prompt_embeds=pooled_prompt_embeds,
                                  guidance_scale=guidance_scale)

        # Postprocess.
        latents = (latents / self.vae.config.scaling_factor
                   ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")

        return image

    def _get_clip_prompt_embeds(self,
                                batch_size: int,
                                prompt: List[str],
                                num_images_per_prompt: int = 1,
                                model_index: int = 0):
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[model_index]
        text_encoder = clip_text_encoders[model_index]

        prompt_input_ids: torch.Tensor = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt")["input_ids"]

        prompt_embeds = text_encoder(prompt_input_ids.to(self.device),
                                     output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt,
                                           seq_len, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def _get_t5_prompt_embeds(self,
                              batch_size: int,
                              prompt: List[str],
                              num_images_per_prompt: int = 1,
                              max_sequence_length: int = 256):
        prompt_input_ids: torch.Tensor = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")["input_ids"]

        prompt_embeds = self.text_encoder_3(prompt_input_ids.to(
            self.device))[0]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt,
                                           seq_len, -1)

        return prompt_embeds

    def encode_prompt(self,
                      prompt: str,
                      prompt_2: str = None,
                      prompt_3: str = None,
                      negative_prompt: str = None,
                      negative_prompt_2: str = None,
                      negative_prompt_3: str = None):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        prompt_3 = prompt_3 or prompt
        prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

        prompt_embeds, pooled_prompt_embeds = self._get_clip_prompt_embeds(
            batch_size=batch_size, prompt=prompt, model_index=0)
        prompt_embeds_2, pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
            batch_size=batch_size, prompt=prompt_2, model_index=1)
        clip_prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2],
                                       dim=-1)

        t5_prompt_embeds = self._get_t5_prompt_embeds(batch_size=batch_size,
                                                      prompt=prompt_3)

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1]))

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds],
                                  dim=-2)
        pooled_prompt_embeds = torch.cat(
            [pooled_prompt_embeds, pooled_prompt_embeds_2], dim=-1)

        if self.do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt
            negative_prompt_2 = (batch_size * [negative_prompt_2]
                                 if isinstance(negative_prompt_2, str) else
                                 negative_prompt_2)
            negative_prompt_3 = (batch_size * [negative_prompt_3]
                                 if isinstance(negative_prompt_3, str) else
                                 negative_prompt_3)

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                batch_size=batch_size,
                prompt=negative_prompt,
                model_index=0,
            )
            negative_prompt_embed_2, negative_pooled_prompt_embed_2 = self._get_clip_prompt_embeds(
                batch_size=batch_size,
                prompt=negative_prompt_2,
                model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat(
                [negative_prompt_embed, negative_prompt_embed_2], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                batch_size=batch_size, prompt=negative_prompt_3)

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] -
                 negative_clip_prompt_embeds.shape[-1]))

            negative_prompt_embeds = torch.cat(
                [negative_clip_prompt_embeds, t5_negative_prompt_embed],
                dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_embed_2],
                dim=-1)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def prepare_latents(self, batch_size: int, num_channels_latents: int,
                        height: int, width: int):
        shape = (batch_size, num_channels_latents,
                 height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        latents = torch.randn(shape, dtype=self.dtype, device=self.device)

        return latents

    @calculate_time(show=True)
    def unet_infer(
        self,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = torch.cat(
                [latents] * 2) if self.do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents
