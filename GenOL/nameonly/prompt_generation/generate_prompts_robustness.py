import random
import json

prompt_path = './prompts/robustness.json'

prompts = [
    'a bad photo of a [concept].',
    'a photo of many [concept].',
    'a sculpture of a [concept].',
    'a photo of the hard to see [concept].',
    'a low resolution photo of the [concept].',
    'a rendering of a [concept].',
    'graffiti of a [concept].',
    'a bad photo of the [concept].',
    'a cropped photo of the [concept].',
    'a tattoo of a [concept].',
    'the embroidered [concept].',
    'a photo of a hard to see [concept].',
    'a bright photo of a [concept].',
    'a photo of a clean [concept].',
    'a photo of a dirty [concept].',
    'a dark photo of the [concept].',
    'a drawing of a [concept].',
    'a photo of my [concept].',
    'the plastic [concept].',
    'a photo of the cool [concept].',
    'a close-up photo of a [concept].',
    'a black and white photo of the [concept].',
    'a painting of the [concept].',
    'a painting of a [concept].',
    'a pixelated photo of the [concept].',
    'a sculpture of the [concept].',
    'a bright photo of the [concept].',
    'a cropped photo of a [concept].',
    'a plastic [concept].',
    'a photo of the dirty [concept].',
    'a jpeg corrupted photo of a [concept].',
    'a blurry photo of the [concept].',
    'a photo of the [concept].',
    'a good photo of the [concept].',
    'a rendering of the [concept].',
    'a [concept] in a video game.',
    'a photo of one [concept].',
    'a doodle of a [concept].',
    'a close-up photo of the [concept].',
    'a photo of a [concept].',
    'the origami [concept].',
    'the [concept] in a video game.',
    'a sketch of a [concept].',
    'a doodle of the [concept].',
    'a origami [concept].',
    'a low resolution photo of a [concept].',
    'the toy [concept].',
    'a rendition of the [concept].',
    'a photo of the clean [concept].',
    'a photo of a large [concept].',
    'a rendition of a [concept].',
    'a photo of a nice [concept].',
    'a photo of a weird [concept].',
    'a blurry photo of a [concept].',
    'a cartoon [concept].',
    'art of a [concept].',
    'a sketch of the [concept].',
    'a embroidered [concept].',
    'a pixelated photo of a [concept].',
    'itap of the [concept].',
    'a jpeg corrupted photo of the [concept].',
    'a good photo of a [concept].',
    'a plushie [concept].',
    'a photo of the nice [concept].',
    'a photo of the small [concept].',
    'a photo of the weird [concept].',
    'the cartoon [concept].',
    'art of the [concept].',
    'a drawing of the [concept].',
    'a photo of the large [concept].',
    'a black and white photo of a [concept].',
    'the plushie [concept].',
    'a dark photo of a [concept].',
    'itap of a [concept].',
    'graffiti of the [concept].',
    'a toy [concept].',
    'itap of my [concept].',
    'a photo of a cool [concept].',
    'a photo of a small [concept].',
    'a tattoo of the [concept].'
]

# Randomly select 50 prompts
selected_prompts = random.sample(prompts, 50)

prompt_result_dict = {"metaprompts": [
    {
        "index": 0,
        "metaprompt": "dummy",
        "prompts": []
    }
]}

for i, prompt in enumerate(selected_prompts):
    prompt_result_dict["metaprompts"][0]['prompts'].append(
        {
            "index": i,
            "content": prompt
        }
    )

with open(prompt_path, 'w') as f:
    json.dump(prompt_result_dict, f)