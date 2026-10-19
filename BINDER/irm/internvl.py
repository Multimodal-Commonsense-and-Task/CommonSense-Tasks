
import os

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image


model_name = "OpenGVLab/InternVL3_5-2B"
llm = LLM(
    model = model_name,
    limit_mm_per_prompt={"image": 10, "video": 10},
    enable_lora=False,
    # max_model_len=8192,
    dtype="float16",
    trust_remote_code=True,
    kv_cache_dtype="fp8",
)
sampling = SamplingParams(max_tokens=256, temperature=0.2, top_p=0.9)

sample_image_path = "./test1.png"
if os.path.exists(sample_image_path):
    img = Image.open(sample_image_path).convert("RGB")
    prompt = "<image>\nPlease describe the image in detail."

    outputs = llm.generate(
        {
          "prompt": prompt,
          "multi_modal_data": {"image": img}
        },
        sampling_params=sampling
    )

    print(outputs[0].outputs[0].text)
else:
    print(f"Skipping sample inference: image not found at {sample_image_path}")