# Nested prompt structure -> list of prompts
import random
import json

prompt_path = './prompts/gpt4_wo_cot_wo_hierarchy_50.json'
NUM_PROMPTS = 50
prompt_list = [
    "A photo of a [concept] in the forest.",
    "A detailed close-up of a [concept] underwater.",
    "A vivid shot of a [concept] under a sunset sky.",
    "A photorealistic depiction of a [concept] in a field.",
    "A vibrant image of a [concept] surrounded by mountains.",
    "A [concept] in a futuristic cityscape.",
    "A close-up of a [concept] in the rain.",
    "A panoramic view of a [concept] in the desert.",
    "A photorealistic painting of a [concept] in the clouds.",
    "A colorful image of a [concept] in a garden.",
    "A natural scene with a [concept] in a dense forest.",
    "A [concept] in a snow-covered village at dawn.",
    "A bright scene of a [concept] near a crystal-clear lake.",
    "A photorealistic view of a [concept] in outer space.",
    "A high-definition photo of a [concept] on a tropical beach.",
    "A serene shot of a [concept] in a calm river.",
    "A highly detailed image of a [concept] in a city park.",
    "A photorealistic sunset featuring a [concept] by the ocean.",
    "A striking view of a [concept] in a colorful market.",
    "A detailed scene with a [concept] in a winter wonderland.",
    "A [concept] under the northern lights in the Arctic.",
    "A surreal image of a [concept] in a dreamlike landscape.",
    "A panoramic view of a [concept] in a valley at dusk.",
    "A photorealistic image of a [concept] in a modern kitchen.",
    "A vivid close-up of a [concept] in a rainforest.",
    "A peaceful scene of a [concept] by a countryside cottage.",
    "A stunning photo of a [concept] in a flower field.",
    "A dynamic image of a [concept] on a stormy day.",
    "A detailed photograph of a [concept] in an urban setting.",
    "A [concept] in a medieval castle during a sunset.",
    "A photo-realistic view of a [concept] on an icy tundra.",
    "A close-up shot of a [concept] in a bustling city.",
    "A vibrant portrayal of a [concept] in an underwater cave.",
    "A clear image of a [concept] in a modern art gallery.",
    "A striking view of a [concept] in a glowing sunset.",
    "A serene depiction of a [concept] on a calm river.",
    "A vibrant and colorful scene of a [concept] at a carnival.",
    "A [concept] in an open field under a dramatic sky.",
    "A photo of a [concept] on a rocky mountain top.",
    "A high-resolution shot of a [concept] in a bamboo forest.",
    "A detailed view of a [concept] in the heart of a metropolis.",
    "A cinematic shot of a [concept] in an open-air market.",
    "A photorealistic capture of a [concept] in the snow.",
    "A vivid depiction of a [concept] in a rural village.",
    "A photo of a [concept] in an underwater reef.",
    "A high-definition image of a [concept] in a serene meadow.",
    "A close-up shot of a [concept] in a desert landscape.",
    "A wide-angle view of a [concept] in a historic monument.",
    "A colorful scene of a [concept] in a tropical forest.",
    "A beautiful shot of a [concept] at a waterfall.",
    "A photorealistic image of a [concept] in an art studio.",
    "A vibrant photo of a [concept] in a busy marketplace.",
    "A detailed shot of a [concept] in an alpine landscape.",
    "A serene scene of a [concept] in an ancient temple.",
    "A highly detailed photo of a [concept] in a futuristic landscape.",
    "A [concept] under a clear starry night sky.",
    "A photorealistic close-up of a [concept] in springtime.",
    "A dramatic capture of a [concept] during a thunderstorm.",
    "A realistic shot of a [concept] in a foggy forest.",
    "A colorful view of a [concept] by the lakeside.",
    "A photorealistic image of a [concept] near a bonfire.",
    "A stunning photo of a [concept] in the wilderness.",
    "A vibrant depiction of a [concept] in a sunflower field.",
    "A detailed image of a [concept] in a cozy cabin.",
    "A surreal capture of a [concept] floating above a canyon.",
    "A [concept] in a misty landscape at dawn.",
    "A highly detailed image of a [concept] in an urban plaza.",
    "A bright and colorful shot of a [concept] in a carnival.",
    "A serene photo of a [concept] near a mountain lake.",
    "A clear shot of a [concept] in a quaint village.",
    "A photorealistic view of a [concept] in the mountains at dusk.",
    "A dynamic scene with a [concept] near the coastline.",
    "A detailed close-up of a [concept] in a dense jungle.",
    "A vibrant photo of a [concept] in a sunlit meadow.",
    "A serene capture of a [concept] in an alpine village.",
    "A clear shot of a [concept] by a river at sunrise.",
    "A photorealistic depiction of a [concept] in a quiet street.",
    "A bright and lively image of a [concept] in a city market.",
    "A highly detailed capture of a [concept] on a rocky beach.",
    "A colorful scene of a [concept] in a garden of roses.",
    "A photorealistic image of a [concept] in a futuristic mall.",
    "A striking shot of a [concept] in an autumn forest.",
    "A peaceful view of a [concept] near a calm lake.",
    "A detailed photo of a [concept] in a modern house.",
    "A vivid scene with a [concept] in a foggy forest.",
    "A photorealistic close-up of a [concept] on a rainy street.",
    "A cinematic shot of a [concept] in a busy town square.",
    "A colorful depiction of a [concept] at a beach sunset.",
    "A high-definition image of a [concept] in a serene orchard.",
    "A vibrant photo of a [concept] in a snowy pine forest.",
    "A stunning scene of a [concept] on a bridge over a river.",
    "A photorealistic image of a [concept] in a lush jungle.",
    "A colorful shot of a [concept] at an amusement park.",
    "A dramatic photo of a [concept] under a stormy sky.",
    "A highly detailed depiction of a [concept] in a vineyard.",
    "A close-up of a [concept] in a field of lavender.",
    "A photorealistic view of a [concept] in a bustling harbor.",
    "A serene photo of a [concept] in a quiet village.",
    "A detailed shot of a [concept] in a vibrant sunflower field.",
    "A photorealistic capture of a [concept] on a cliffside path."
]

prompt_list = random.sample(prompt_list, NUM_PROMPTS)

prompt_result_dict = {"metaprompts": [
    {
        "index": 0,
        "metaprompt": "dummy",
        "prompts": []
    }
]}

for i, prompt in enumerate(prompt_list):
    prompt_result_dict["metaprompts"][0]['prompts'].append(
        {
            "index": i,
            "content": prompt
        }
    )

with open(prompt_path, 'w') as f:
    json.dump(prompt_result_dict, f)

