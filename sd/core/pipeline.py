import os
import json
import torch
import inspect
import importlib
import numpy as np

from sd.utils import sd_logger, calculate_time

from tqdm import tqdm
from PIL import Image
from typing import Dict, Any
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import PNDMScheduler, UNet2DConditionModel, AutoencoderKL


class Pipeline(object):

    def __init__(
        self,
        feature_extractor: CLIPImageProcessor,
        scheduler: PNDMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        dtype: str,
        device: str,
        pipeline: str,
        engine_path: str = None,
        use_trt: bool = False,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) -
                                    1)
        self.use_trt = use_trt
        self.dtype = getattr(torch, dtype)
        self.device = device

        self.unet_in_channels = unet.config.in_channels
        self.unet_sample_size = unet.config.sample_size
        if use_trt:
            del unet
            from sd.backend import TrtSession
            sd_logger.info("Use TensorRT for the inference of UNet")

            self.unet = TrtSession(engine_path, dtype=self.dtype)
        else:
            self.unet = unet

        assert pipeline in ["txt2img", "img2img"]
        if pipeline == "img2img":
            from diffusers.image_processor import VaeImageProcessor
            self.image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor)
        else:
            self.image_processor = None
        self.pipeline = pipeline

    @classmethod
    def from_pretrained(cls, model_dir: str, use_trt: bool, pipeline: str,
                        dtype: str, device: str) -> "Pipeline":
        # Parse config file.
        config_path = os.path.join(model_dir, "model_index.json")
        if not os.path.exists(config_path):
            sd_logger.error(f"config file {config_path} doesn't exist")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Load modules expected in config file.
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
            "pipeline": pipeline
        })

        # Use TensorRT for UNet inference.
        if use_trt:
            engine_path = os.path.join(model_dir, "unet", "trt",
                                       "model.engine")
            if not os.path.exists(engine_path):
                sd_logger.error(
                    f"input {engine_path} doesn't exist, please refer to export_unet.py firstly"
                )
            init_kwargs.update({
                "engine_path": engine_path,
                "use_trt": use_trt,
            })

        return cls(**init_kwargs)

    def cuda(self):
        self.text_encoder = self.text_encoder.to(self.device)
        if not self.use_trt:
            self.unet = self.unet.to(self.device)
        self.vae = self.vae.to(self.device)

        return self

    @torch.no_grad()
    def forward(
        self,
        prompt: str,
        negative_prompts: str = None,
        image: Image.Image = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        # Encode input prompt, and negative prompt when guidance_scale larger than 1.0.
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self.encode_prompt(prompt,
                                           do_classifier_free_guidance,
                                           negative_prompts=negative_prompts)

        # TODO: Prepare timesteps.
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps,
                                     device=self.device)
        timesteps = self.scheduler.timesteps

        # Prepare latent used as input of UNet.
        if self.pipeline == "img2img":
            if image is None:
                sd_logger.error(
                    f"input image is None when pipeline is img2img")
            image = self.image_processor.preprocess(image)

            latents = self.prepare_img2img_latents(image=image,
                                                   timesteps=timesteps[:1])
        else:
            latents = self.prepare_txt2img_latents(
                batch_size=1,
                num_channels_latents=self.unet_in_channels,
                height=self.unet_sample_size * self.vae_scale_factor,
                width=self.unet_sample_size * self.vae_scale_factor)

        # Denoising.
        latents = self.unet_infer(
            timesteps=timesteps,
            latents=latents,
            prompt_embeds=prompt_embeds,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=guidance_scale)

        # Postprocess.
        image = self.decode_latents(latents)
        image = self.numpy_to_pil(image)

        return image

    def encode_prompt(self, prompt: str, do_classifier_free_guidance: bool,
                      negative_prompts: str) -> torch.Tensor:
        # Pad input to model_max_length and do truncation if necessary.
        prompt = prompt * 100
        text_input_ids: torch.Tensor = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt").input_ids

        # (bs, model_max_length, hidden_size)
        prompt_embeds: torch.Tensor = self.text_encoder(
            text_input_ids.to(device=self.device))[0]
        prompt_embeds = prompt_embeds.to(device=self.device)

        # Add negative prompt.
        if do_classifier_free_guidance:
            if negative_prompts is None:
                negative_prompts = [""]
            neg_prompt_ids: torch.Tensor = self.tokenizer(
                negative_prompts,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_tensors="pt").input_ids
            neg_prompt_embeds = self.text_encoder(
                neg_prompt_ids.to(device=self.device))[0]
            neg_prompt_embeds = neg_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=self.device)

            # (2 * bs, model_max_length, hidden_size)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def prepare_txt2img_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        shape = (batch_size, num_channels_latents,
                 height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        latents = torch.randn(shape, dtype=self.dtype, device=self.device)

        return latents * self.scheduler.init_noise_sigma

    def prepare_img2img_latents(
        self,
        image: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        image = image.to(device=self.device, dtype=self.dtype)

        init_latents = self.vae.encode(image).latent_dist.sample(
            generator=None) * self.vae.config.scaling_factor
        init_latents = torch.cat([init_latents], dim=0)

        # Add noise to get latents.
        noise = torch.randn(init_latents.shape,
                            dtype=self.dtype,
                            device=self.device)
        latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        return latents

    @calculate_time(show=True)
    def unet_infer(
        self,
        timesteps: int,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
    ) -> torch.Tensor:
        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            if self.use_trt:
                input_feed = {
                    "in_latents": latent_model_input.to(self.dtype),
                    "timesteps": t,
                    "prompt_embeds": prompt_embeds.to(self.dtype),
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
                    ), "The output of TensorRT has nan."

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

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
