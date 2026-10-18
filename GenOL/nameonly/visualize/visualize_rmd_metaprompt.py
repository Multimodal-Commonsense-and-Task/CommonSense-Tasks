import pickle
import os
import shutil

# Metaprompt
metaprompt_mapping = {(0,4): 1, (5,9): 2, (10,14): 3, (15,19): 4, (20,24): 5, (25,29): 6, (30,34): 7, (35,39): 8, (40,44): 9, (45,49): 10}
metaprompt_mapping = {0:1, 1:1, 2:1, 3:1, 4:1, 5:2, 6:2, 7:2, 8:2, 9:2, 10:3, 11:3, 12:3, 13:3, 14:3, 15:4, 16:4, 17:4, 18:4, 19:4, 20:5, 21:5, 22:5, 23:5, 24:5, 25:6, 26:6, 27:6, 28:6, 29:6, 30:7, 31:7, 32:7, 33:7, 34:7, 35:8, 36:8, 37:8, 38:8, 39:8, 40:9, 41:9, 42:9, 43:9, 44:9, 45:10, 46:10, 47:10, 48:10, 49:10}

pickle_path = './RMD_scores_PACS_prompt.pkl'
with open(pickle_path, 'rb') as f:
    RMD_scores = pickle.load(f)

prompt_dict = {prompt: [] for prompt in range(50)}
for (prompt, cls), images in RMD_scores.items():
    for image in images:
        path = image[0]
        score = image[1]
        prompt_dict[prompt].append((cls, path, score))

metaprompt_dict = {metaprompt: [] for metaprompt in range(1, 11)}
for prompt in range(50):
    metaprompt = metaprompt_mapping[prompt]
    metaprompt_dict[metaprompt].extend(prompt_dict[prompt])

# Calculate the average RMD score for each metaprompt
avg_rmd_dict = {metaprompt: 0 for metaprompt in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
for metaprompt in avg_rmd_dict.keys():
    images = metaprompt_dict[metaprompt]
    RMD_accumulator = 0
    for cls, path, score in images:
        RMD_accumulator += score
    
    if len(images) != 0:
        avg_rmd_dict[metaprompt] = RMD_accumulator / len(images)

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
        save_path = os.path.join('./rmd_metaprompt_results', f"metaprompt_{prompt_idx}_top_{prompt_counter}", cls)
        os.makedirs(save_path, exist_ok=True)
        for k, (path, score) in enumerate(images[-200:][::-1]):
            image_name = f"{k}.png"
            shutil.copy(path, os.path.join(save_path, image_name))

        # Save the top 5 scores in a text file
        with open(os.path.join(save_path, "top5_scores.txt"), 'w') as f:
            for k, (path, score) in enumerate(images[-200:][::-1]):
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
        save_path = os.path.join('./rmd_metaprompt_results', f"metaprompt_{prompt_idx}_bottom_{prompt_counter}", cls)
        os.makedirs(save_path, exist_ok=True)
        for k, (path, score) in enumerate(images[:200]):
            image_name = f"{k}.png"
            shutil.copy(path, os.path.join(save_path, image_name))

        # Save the bottom 5 scores in a text file
        with open(os.path.join(save_path, "bottom5_scores.txt"), 'w') as f:
            for k, (path, score) in enumerate(images[:200]):
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
    
