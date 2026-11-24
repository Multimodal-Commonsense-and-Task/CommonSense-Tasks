#### 
import json
import random

vizwiz_prompt = """First carefully understand the given examples. 
Then use the given image and answer the question in the same way as the examples. 
If the question can not be answered, respond unanswerable. """
####

def open_data(dataset_name, path):

    jsonl_format_dataset = ["vizwiz", "okvqa"]
    list_format_dataset = ["flower", "cub", "dtd", "pororo", "pororov2", "pororov3", "pororov4", "pororov4_new", "flintstonesv3", "flintstonesv3_new", "flintstones", "storystream", "storystreamv2", "storystreamv2_new", "story_visual_ICL", "story_visual_ICLv2", "stanford10000", "stanford10000_5", "movie101", "moviegraphs", "memor", "isekai", "isekai_4choice", "truemicl", "mic", "vlicl", "vliclv2", "SMMILE", "illusionvqa-comp", "idk"]

    with open(path, 'r') as json_file:
        if dataset_name in jsonl_format_dataset:
            dataset = list(json_file)
        elif dataset_name in list_format_dataset:
            dataset = json.load(json_file)
    return dataset


### Each format function should return (full_text, image_list, answer, question_id)
def get_format_func(cur_dataset):

    if cur_dataset == "vizwiz":
        return format_vizwiz
    if cur_dataset == "okvqa":
        return format_okvqa
    if cur_dataset == "flower":
        return format_flower
    if cur_dataset == "cub":
        return format_cub
    if cur_dataset == "dtd":
        return format_dtd
    else:
        return format_custom


####All return format will be in the form (Text, list of images, Answer, Question_id)
def format_vizwiz(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    prompt = '<image>{} Answer:'

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question, answer, question_id = data['image'], data['question'], data['answer'], data['question_id']

    few_shot_prompt = ''
    if num_shot > 0:

        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(sample['question']) + f" {sample['answer']}"
            image_list.append("../" + sample["image"])


    full_text = vizwiz_prompt + few_shot_prompt + prompt.format(question)
    image_list.append("../" + image)

    return full_text, image_list, answer, question_id


def format_okvqa(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    prompt = '<image>{} Answer:'

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question, answer, question_id = data['image'], data['question'], data['answer'], data['question_id']

    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(sample['question']) + f"{sample['answer']}"
            image_list.append(sample["image"])

    
    full_text = few_shot_prompt + prompt.format(question)
    image_list.append(image)

    return full_text, image_list, answer, question_id


def format_flower(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1


def format_cub(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1
    

def format_dtd(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1


def format_custom(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        # [Train 모드] Task Vector 추출 단계
        sample = random.choice(all_data)
        data = sample if isinstance(sample, dict) else json.loads(sample)
        
        if split == "test":
            mode = "extract_query"
        else:
            mode = "extract_examples"
    else:
        # [Test 모드] 평가 단계
        data = cur_item if isinstance(cur_item, dict) else json.loads(cur_item)
        mode = "extract_query"

    image_list = data.get("image", [])
    question_id = data.get("id", 0)
    
    conversations = data.get("conversations", [])
    if not conversations:
        return "", [], "", question_id

    full_text = conversations[0]["value"]
    answer = conversations[1]["value"] if len(conversations) > 1 else ""

    is_many_shot = len(image_list) > 1 or full_text.count("Answer:") > 1

    if is_many_shot:
        last_image_token_idx = full_text.rfind("<image>")
        
        if last_image_token_idx != -1:
            if mode == "extract_examples":
                full_text = full_text[:last_image_token_idx].strip()
                
                image_token_count = full_text.count("<image>")
                image_list = image_list[:image_token_count]
                
                answer = "" 
                
            elif mode == "extract_query":
                full_text = full_text[last_image_token_idx:].strip()
                
                image_list = image_list[-1:]

    return full_text, image_list, answer, question_id
