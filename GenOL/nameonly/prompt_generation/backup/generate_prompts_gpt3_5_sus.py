import re
import pickle
import json
from openai import OpenAI
from tqdm import tqdm
from classes import *

def generate_prompt_sus(client, concept_name):
    message1 = f"""I need to create prompts for generating images using a text-to-image model. The subject of the image is '{concept_name}'. Please generate 10 answers starting with "A {concept_name}" to the following question: "Describe what a {concept_name} looks like" A good example for 'dog' could be "A dog typically has four legs, a tail, and pointy ears.". Please ensure the response format is strictly 'prompt: answer' (without number).\n"""
    message2 = f"""I need to create prompts for generating images using a text-to-image model. The subject of the image is '{concept_name}'. Please generate 10 answers starting with "A {concept_name}" to the following question: "How can you identify {concept_name}?" A good example for 'dog' could be "A dog typically has four legs, a tail, and pointy ears.". Please ensure the response format is strictly 'prompt: answer' (without number).\n"""
    message3 = f"""I need to create prompts for generating images using a text-to-image model. The subject of the image is '{concept_name}'. Please generate 10 answers starting with "A {concept_name}" to the following question: "What does {concept_name} look like?" A good example for 'dog' could be "prompt: A dog typically has four legs, a tail, and pointy ears.". Please ensure the response format is strictly 'prompt: answer' (without number).\n"""
    message4 = f"""I need to create prompts for generating images using a text-to-image model. The subject of the image is '{concept_name}'. Please generate 10 answers starting with "A {concept_name}" to the following question: "Describe an image from the internet of a {concept_name}?" A good example for 'dog' could be "A dog typically has four legs, a tail, and pointy ears.". Please ensure the response format is strictly 'prompt: answer' (without number).\n"""
    message5 = f"""I need to create prompts for generating images using a text-to-image model. The subject of the image is '{concept_name}'. Please generate 10 answers starting with "A {concept_name}" to the following question: "A caption of an image of {concept_name}?" A good example for 'dog' could be "A dog typically has four legs, a tail, and pointy ears.". Please ensure the response format is strictly 'prompt: answer' (without number).\n"""

    messages = [message1, message2, message3, message4, message5]

    responses = []
    # Generate one prompt using GPT-4 API
    while True:
        total_response_results = []
        for message in tqdm(messages):
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message},
                ]
            )

            response_content = response.choices[0].message.content
            pattern1 = rf"A {cls}.*?(?:\n|$)"; pattern2 = rf"An {cls}.*?(?:\n|$)"
            matches1 = re.findall(pattern1, response_content); matches2 = re.findall(pattern2, response_content)
            matches = matches1 if len(matches1) > len(matches2) else matches2
            matches = [match.strip() for match in matches]

            print(f"Length: {len(matches)}")
            if len(matches) > 10:
                matches = matches[:10]
            elif len(matches) < 10:
                print(f"Less than 10...")
            total_response_results.extend(matches)
        
        if len(total_response_results) >= 50:
            total_response_results = total_response_results[:50]
            break
    
    return total_response_results

cls_count_dict = DomainNet_count
result_json_path = './prompts/sus_PACS_before_process.json'

client = OpenAI()
classes = list(cls_count_dict.keys())

totalprompt_dict = {}
for cls in tqdm(classes):
    result_50_prompts = generate_prompt_sus(client, cls)
    totalprompt_dict[cls] = result_50_prompts

with open(result_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)