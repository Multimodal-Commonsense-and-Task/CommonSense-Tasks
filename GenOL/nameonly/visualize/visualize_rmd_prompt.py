import pickle

# Metaprompt
metaprompt_mapping = {(0,4): 1, (5,9): 2, (10,14): 3, (15,19): 4, (20,24): 5, (25,29): 6, (30,34): 7, (35,39): 8, (40,44): 9, (45,49): 10}

pickle_path = './RMD_scores_PACS_prompt.pkl'
with open(pickle_path, 'rb') as f:
    RMD_scores = pickle.load(f)

prompt_dict = {prompt: [] for prompt in range(50)}
for (prompt, cls), images in RMD_scores.items():
    breakpoint()
    for image in images:
        path = image[0]
        score = image[1]
        prompt_dict[prompt].append((cls, path, score))

# Calculate the average RMD score for each prompt
avg_rmd_dict = {prompt: 0 for prompt in range(50)}
for prompt in range(50):
    images = prompt_dict[prompt]
    RMD_accumulator = 0
    for cls, path, score in images:
        RMD_accumulator += score
    
    avg_rmd_dict[prompt] = RMD_accumulator / len(images)

breakpoint()

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
    
