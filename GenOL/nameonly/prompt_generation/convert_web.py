import json

output_path = 'static_totalprompts_wo_cot_wo_hierarchy.json'
prompt_list = [
    "A serene photorealistic portrait of [concept] in a lush green forest.",
    "Capture a bustling city street at night with [concept] in vivid neon lights.",
    "A peaceful sunrise over a mountain landscape with [concept] in the foreground.",
    "A photorealistic image of [concept] during a vibrant, colorful festival.",
    "The interior of an antique library with [concept] at the center, bathed in soft light.",
    "A realistic depiction of [concept] under the deep blue tones of twilight.",
    "An early morning view of [concept] with dew-covered surroundings in soft light.",
    "A snowy winter scene with [concept], using a monochrome color palette.",
    "A photo of [concept] with the golden hues of an autumn forest in the background.",
    "A detailed realistic image of [concept] on a rainy day with reflective surfaces.",
    "A high-resolution image of [concept] in a modern office space.",
    "A vibrant sunset at the beach with [concept] silhouetted against the sky.",
    "A photo of [concept] framed by the bright and dark contrast of chiaroscuro lighting.",
    "Capture [concept] during a serene foggy morning in a photorealistic style.",
    "A photorealistic snapshot of [concept] in a bustling market environment.",
    "A realistic depiction of [concept] in a tranquil zen garden.",
    "The stark beauty of [concept] in a desert at midday.",
    "A nighttime cityscape with [concept] under streetlights.",
    "A detailed photorealistic image of [concept] during the cherry blossom season.",
    "Capture [concept] on a busy pedestrian street in broad daylight.",
    "A photorealistic image of [concept] with the drama of stormy weather in the background.",
    "A crisp winter morning with [concept] showcased in icy blue tones.",
    "An idyllic countryside setting with [concept] during the golden hour.",
    "A realistic image of [concept] in a classic diner at dusk.",
    "A high-resolution photo of [concept] against a backdrop of dramatic mountain peaks.",
    "Capture [concept] with a backdrop of a colorful street art mural.",
    "A photo of [concept] on a quiet lake at dawn with mist rising from the water.",
    "A photorealistic portrait of [concept] with a soft pastel color palette.",
    "A dynamic sports scene featuring [concept] in action.",
    "A photo of [concept] in a vibrant, crowded festival setting.",
    "A nighttime shot of [concept] with fireworks in the background.",
    "A realistic image of [concept] on a historic cobblestone street.",
    "A photorealistic capture of [concept] under the lush canopy of a rainforest.",
    "A serene beach scene at sunset with [concept] in the foreground.",
    "A detailed image of [concept] in a snowy setting with soft winter light.",
    "A photorealistic view of [concept] in an underground subway station.",
    "Capture [concept] in the rustic charm of a mountain cabin.",
    "A bright, sunny day at the park with [concept] in full detail.",
    "A photo of [concept] in a modern, minimalist kitchen.",
    "A realistic image of [concept] at a lively sports event.",
    "A dramatic thunderstorm setting with [concept] showcased in dark, moody colors.",
    "A photo of [concept] in a vintage car on a historic route.",
    "A detailed, realistic nighttime photo of [concept] under a starry sky.",
    "A sunny coastal scene with [concept] against the backdrop of crashing waves.",
    "A photo of [concept] in a bustling café during the morning rush.",
    "A high-resolution image of [concept] in a serene monastery.",
    "A realistic depiction of [concept] on a vibrant spring day with blossoms.",
    "Capture [concept] in a photorealistic manner against the backdrop of an ancient castle.",
    "A detailed image of [concept] in a modern urban loft.",
    "A photorealistic snapshot of [concept] during a tranquil, foggy sunrise."
]

result_dict = {"metaprompts": []}
for i, prompt in enumerate(prompt_list):
    result_dict['metaprompts'].append({
        'index': i,
        "metaprompt": "dummy",
        "prompts": [
            {
                "index": 0,
                "content": prompt
            }
        ]
    })

with open(output_path, 'w') as f:
    json.dump(result_dict, f)