
import os
import torch
from diffusers import DiffusionPipeline

os.makedirs("./DENOISDING_STEP", exist_ok=True)
model_id = "cagliostrolab/animagine-xl-3.1"
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.to("cuda")


prompt = "A boy, wearing a black suit, jumping into a swimming pool"
CFG_strength = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
for i in CFG_strength:
    image = pipe(
        prompt,
        # negative_prompt=negative_prompt,
        # width=832,
        # height=1216,
        guidance_scale=7,
        num_inference_steps=i,
    ).images[0]
    image.save(f"./DENOISDING_STEP/output_{i}.png")
