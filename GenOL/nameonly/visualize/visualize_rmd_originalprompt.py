import pickle
import os
import shutil

pickle_path = './RMD_scores_PACS_prompt.pkl'
with open(pickle_path, 'rb') as f:
    RMD_scores = pickle.load(f)

prompt_dict = {prompt: [] for prompt in range(50)}
for (prompt, cls), images in RMD_scores.items():
    for image in images:
        path = image[0]
        score = image[1]
        prompt_dict[prompt].append((cls, path, score))

avg_rmd_dict = {prompt: 0 for prompt in range(50)}
for prompt in range(50):
    images = prompt_dict[prompt]
    RMD_accumulator = 0
    for cls, path, score in images:
        RMD_accumulator += score
    
    avg_rmd_dict[prompt] = RMD_accumulator / len(images)

# Get the top-3 and bottom-3 prompts
avg_rmd_dict = {k: v for k, v in sorted(avg_rmd_dict.items(), key=lambda item: item[1])}
top3_prompts = {k: avg_rmd_dict[k] for k in list(avg_rmd_dict.keys())[-5:][::-1]}
bottom3_prompts = {k: avg_rmd_dict[k] for k in list(avg_rmd_dict.keys())[:5]}

prompt_counter = 0
for prompt_idx, avg_rmd in top3_prompts.items():
    prompt_counter += 1
    images = prompt_dict[prompt_idx] # [(cls, path, score), ...]
    cls_rmd_dict = {} # class -> [(path, score), ...]
    for cls, path, score in images:
        if cls not in cls_rmd_dict:
            cls_rmd_dict[cls] = []
        cls_rmd_dict[cls].append((path, score))
    
    # Sort the images by RMD score
    for cls in cls_rmd_dict:
        cls_rmd_dict[cls] = sorted(cls_rmd_dict[cls], key=lambda x: x[1])
    
    # Save top 5 images for each class
    for cls, images in cls_rmd_dict.items():
        save_path = os.path.join('./rmd_prompt_results_50prompts', f"prompt_{prompt_idx}_top_{prompt_counter}", cls)
        os.makedirs(save_path, exist_ok=True)
        for k, (path, score) in enumerate(images[-10:][::-1]):
            image_name = f"{k}.png"
            shutil.copy(path, os.path.join(save_path, image_name))

        # Save the top 5 scores in a text file
        with open(os.path.join(save_path, "top5_scores.txt"), 'w') as f:
            for k, (path, score) in enumerate(images[-10:][::-1]):
                f.write(f"{k} ({score})\n")

# Bottom3_prompts
prompt_counter = 0
for prompt_idx, avg_rmd in bottom3_prompts.items():
    prompt_counter += 1
    images = prompt_dict[prompt_idx] # [(cls, path, score), ...]
    cls_rmd_dict = {} # class -> [(path, score), ...]
    for cls, path, score in images:
        if cls not in cls_rmd_dict:
            cls_rmd_dict[cls] = []
        cls_rmd_dict[cls].append((path, score))
    
    # Sort the images by RMD score
    for cls in cls_rmd_dict:
        cls_rmd_dict[cls] = sorted(cls_rmd_dict[cls], key=lambda x: x[1])
    
    # Save bottom 5 images for each class
    for cls, images in cls_rmd_dict.items():
        save_path = os.path.join('./rmd_prompt_results_50prompts', f"prompt_{prompt_idx}_bottom_{prompt_counter}", cls)
        os.makedirs(save_path, exist_ok=True)
        for k, (path, score) in enumerate(images[:10]):
            image_name = f"{k}.png"
            shutil.copy(path, os.path.join(save_path, image_name))

        # Save the bottom 5 scores in a text file
        with open(os.path.join(save_path, "bottom5_scores.txt"), 'w') as f:
            for k, (path, score) in enumerate(images[:10]):
                f.write(f"{k} ({score})\n")

breakpoint()
# Get the top-3 and bottom-3 metaprompts
avg_rmd_dict = {k: v for k, v in sorted(avg_rmd_dict.items(), key=lambda item: item[1])}
top3_metaprompts = {k: avg_rmd_dict[k] for k in list(avg_rmd_dict.keys())[-3:][::-1]}
bottom3_metaprompts = {k: avg_rmd_dict[k] for k in list(avg_rmd_dict.keys())[:3]}

prompt_counter = 0
for prompt_idx, avg_rmd in top3_metaprompts.items():
    prompt_counter += 1
    images = metaprompt_dict[prompt_idx] # [(cls, path, score), ...]
    cls_rmd_dict = {} # class -> [(path, score), ...]
    for cls, path, score in images:
        if cls not in cls_rmd_dict:
            cls_rmd_dict[cls] = []
        cls_rmd_dict[cls].append((path, score))
    
    # Sort the images by RMD score
    for cls in cls_rmd_dict:
        cls_rmd_dict[cls] = sorted(cls_rmd_dict[cls], key=lambda x: x[1])
    
    # Save top 5 images for each class
    for cls, images in cls_rmd_dict.items():
        save_path = os.path.join('./rmd_prompt_results', f"metaprompt_{prompt_idx}_top_{prompt_counter}", cls)
        os.makedirs(save_path, exist_ok=True)
        for k, (path, score) in enumerate(images[:5]):
            image_name = f"{k}.png"
            shutil.copy(path, os.path.join(save_path, image_name))

        # Save the top 5 scores in a text file
        with open(os.path.join(save_path, "top5_scores.txt"), 'w') as f:
            for k, (path, score) in enumerate(images[:5]):
                f.write(f"{k} ({score})\n")

prompt_counter = 0
for prompt_idx, avg_rmd in bottom3_metaprompts.items():
    prompt_counter += 1
    images = metaprompt_dict[prompt_idx] # [(cls, path, score), ...]
    cls_rmd_dict = {} # class -> [(path, score), ...]
    for cls, path, score in images:
        if cls not in cls_rmd_dict:
            cls_rmd_dict[cls] = []
        cls_rmd_dict[cls].append((path, score))
    
    # Sort the images by RMD score
    for cls in cls_rmd_dict:
        cls_rmd_dict[cls] = sorted(cls_rmd_dict[cls], key=lambda x: x[1])
    
    # Save bottom 5 images for each class
    for cls, images in cls_rmd_dict.items():
        save_path = os.path.join('./rmd_prompt_results', f"metaprompt_{prompt_idx}_bottom_{prompt_counter}", cls)
        os.makedirs(save_path, exist_ok=True)
        for k, (path, score) in enumerate(images[:5]):
            image_name = f"{k}.png"
            shutil.copy(path, os.path.join(save_path, image_name))

        # Save the bottom 5 scores in a text file
        with open(os.path.join(save_path, "bottom5_scores.txt"), 'w') as f:
            for k, (path, score) in enumerate(images[:5]):
                f.write(f"{k} ({score})\n")

breakpoint()

breakpoint()
print(f"Metaprompt {prompt_idx} (avg RMD: {avg_rmd})")
for cls, images in cls_rmd_dict.items():
    print(f"\tClass {cls} (n={len(images)})")
    for path, score in images[:3]:
        print(f"\t\t{path} ({score})")
print()

# Print the top 5 and bottom 5 prompts
prompt_pickle = './PACS_prompts.pkl'
with open(prompt_pickle, 'rb') as f:
    prompt_str_dict = pickle.load(f)

# Get the top-3 and bottom-3 prompts
avg_rmd_dict = {k: v for k, v in sorted(avg_rmd_dict.items(), key=lambda item: item[1])}

top3_prompts = {k: avg_rmd_dict[k] for k in list(avg_rmd_dict.keys())[-3:][::-1]}
bottom3_prompts = {k: avg_rmd_dict[k] for k in list(avg_rmd_dict.keys())[:3]}

# Top3_prompts
for prompt_idx, avg_rmd in top3_prompts.items():
    temp_cls = next(iter(prompt_str_dict[prompt_idx])); temp_prompt = prompt_str_dict[prompt_idx][temp_cls]
    prompt_str = temp_prompt.replace(temp_cls, 'CLASS_NAME')
    
    images = prompt_dict[prompt_idx] # [(cls, path, score), ...]
    cls_rmd_dict = {}
    
