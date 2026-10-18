import os
import json
import requests
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from classes import get_count_dict

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, required=True)
args = parser.parse_args()
count_dict = get_count_dict(args.image_dir)

def main():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    
    # Generated jsonl file
    jsonl_data = []
    output_path = os.path.join(args.image_dir, "metadata.jsonl")
    classes = list(count_dict.keys())
    for cls in tqdm(classes):
        cls_path = os.path.join(args.image_dir, cls)
        for image_path in tqdm(os.listdir(cls_path)):
            img = Image.open(os.path.join(cls_path, image_path)).convert('RGB')
            inputs = processor(img, return_tensors="pt").to("cuda")
            out = model.generate(**inputs)
            response = processor.decode(out[0], skip_special_tokens=True)

            jsonl_data.append({
                "file_name": os.path.join(cls, image_path),
                "text": response
            })

    with open(output_path, 'w') as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry) + '\n')
            

if __name__ == "__main__":
    main()
