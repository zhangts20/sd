import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("/data/models/sd/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    # "A capybara holding a sign that reads Hello World",
    "A girl running on a wide road",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("outputs/demo_sd3_5_generated.png")
