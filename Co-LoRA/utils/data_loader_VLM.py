
import torch
import copy
import os
from torch.utils.data import Dataset
import transformers
from configuration.VLM_config_new import DataArguments
from typing import Dict, Sequence, List
from PIL import Image
from dataclasses import dataclass
from models.llava.constants import IGNORE_INDEX
from models.llava import conversation as conversation_lib_llava
from packaging import version
# from utils.augment import DataAugmentation
# IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
IS_TOKENIZER_GREATER_THAN_0_14 = True
Image.MAX_IMAGE_PIXELS = None

class GenerationDataset(Dataset):
    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 processor=None):
        super(GenerationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.processor = processor
        self.conv = conversation_lib_llava.default_conversation.copy()
        
        if self.conv.version == 'qwen':
            self.processor.chat_template = """{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""
        elif self.conv.version == 'llama3':
            self.processor.chat_template = """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{%- if system_message %}\n    {{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n    {%- if tools is not none %}\n        {{- "Environment: ipython\\n" }}\n    {%- endif %}\n    {{- "Cutting Knowledge Date: December 2023\\n" }}\n    {{- "Today Date: " + date_string + "\\n\\n" }}\n    {%- if tools is not none and not tools_in_user_message %}\n        {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n        {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n        {{- "Do not use variables.\\n\\n" }}\n        {%- for t in tools %}\n            {{- t | tojson(indent=4) }}\n            {{- "\\n\\n" }}\n        {%- endfor %}\n    {%- endif %}\n    {{- system_message }}\n    {{- "<|eot_id|>" }}\n{%- endif %}\n\n{%- if tools_in_user_message and not tools is none %}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}\n    {%- endif %}\n    {{- "<|start_header_id|>user<|end_header_id|>\\n\\n" }}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>" }}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- "<|start_header_id|>" + message['role'] + "<|end_header_id|>\\n\\n" + message['content'] | trim + "<|eot_id|>" }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}\n        {{- '{"name": "' + tool_call.name + '", ' }}\n        {{- '"parameters": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n\n{%- if add_generation_prompt %}\n    {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}\n{%- endif %}\n"""

    def __getitem__(self, index):
        source = self.datalist[index]
        qs = source["conversations"][0]['value']
        gold = source["conversations"][1]['value']

        conversation = [
            {
            "role": "user",
            "content": qs
            },
        ]
        
        # conversation = [
        #     {
        #     "role": "user",
        #     "content": [{"type":"text","text":qs}
        #                 ]
        #     },
        # ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. " + prompt
        
        if 'image' in source.keys():
            image_file = source['image']
            if isinstance(image_file, list):
                image = [Image.open(image_path).convert('RGB') for image_path in image_file] #.split(' |sep| ')
            else:
                image = [Image.open(image_file).convert('RGB')]
        
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                images = [expand2square(img, tuple(int(x*255) for x in self.processor.image_processor.image_mean)) for img in image]
            inputs = self.processor(images=images, text=prompt, return_tensors='pt')
        else:
            inputs = self.processor(text=prompt, return_tensors='pt')
        # return input_ids, pixel_values, gold, prompt, image_file
        return_dict = {
            'input_ids':inputs['input_ids'][0],
            'gold':str(gold),
            'prompt':prompt,
        }
        
        if 'pixel_values' in inputs.keys():
            return_dict['pixel_values'] = inputs['pixel_values']
            return_dict['image_file'] = image_file
        return return_dict

    def __len__(self):
        return len(self.datalist)

@dataclass
class DataCollatorForGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, gold, prompt = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", 'gold', 'prompt'))
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # Find the length of the longest sequence
        max_length = max(seq.size(0) for seq in input_ids)
        input_ids = torch.stack([
            torch.cat([torch.full((max_length - seq.size(0),), self.tokenizer.pad_token_id), seq])
            for seq in input_ids
        ])
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = dict(
            input_ids=input_ids,#.to(self.device),
            attention_mask=attention_mask,#.to(self.device),
            gold=gold,
            prompt=prompt
        )
        if 'pixel_values' in instances[0].keys():
            image_file = [instance['image_file'] for instance in instances]
            batch['image_file'] = image_file
            images = [instance['pixel_values'] for instance in instances]
            # if all(x is not None and x.shape == images[0].shape for x in images):
            if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
                images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
                # b, n, c, h, w = images.shape
                # images = images.reshape(b*n,c,h,w)
                # images = self.transform(images).to(dtype=torch.bfloat16)
                batch['pixel_values'] = images#.reshape(b,n,c,h,w)
            else:
                # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
                batch['pixel_values'] = [x.to(dtype=torch.bfloat16) for x in images]

        return batch


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 processor,
                 model_id=None):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.processor = processor
        
        if model_id and 'llama' in model_id.lower():
            self.conv = conversation_lib_llava.conv_templates['llama3'].copy()
        elif model_id and 'qwen' in model_id.lower():
            self.conv = conversation_lib_llava.conv_templates['qwen'].copy()
        else:
            self.conv = conversation_lib_llava.default_conversation.copy()
        
        assistant_start = self.conv.roles[1]
        eot_token = self.conv.sep
        
        print(self.conv.version)
        if self.conv.version == 'qwen':
            self.label_start_id = self.tokenizer(assistant_start, return_tensors="pt")['input_ids'][0].tolist()
            self.label_end_id = self.tokenizer(eot_token, return_tensors="pt")['input_ids'][0].tolist()
            
            self.processor.chat_template = """{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""
        elif self.conv.version == 'llama3':
            self.label_start_id = self.tokenizer(assistant_start, return_tensors="pt")['input_ids'][0][1:].tolist()
            self.label_end_id = self.tokenizer(eot_token, return_tensors="pt")['input_ids'][0][1:].tolist()
        
            self.processor.chat_template = """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{%- if system_message %}\n    {{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n    {%- if tools is not none %}\n        {{- "Environment: ipython\\n" }}\n    {%- endif %}\n    {{- "Cutting Knowledge Date: December 2023\\n" }}\n    {{- "Today Date: " + date_string + "\\n\\n" }}\n    {%- if tools is not none and not tools_in_user_message %}\n        {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n        {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n        {{- "Do not use variables.\\n\\n" }}\n        {%- for t in tools %}\n            {{- t | tojson(indent=4) }}\n            {{- "\\n\\n" }}\n        {%- endfor %}\n    {%- endif %}\n    {{- system_message }}\n    {{- "<|eot_id|>" }}\n{%- endif %}\n\n{%- if tools_in_user_message and not tools is none %}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}\n    {%- endif %}\n    {{- "<|start_header_id|>user<|end_header_id|>\\n\\n" }}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>" }}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- "<|start_header_id|>" + message['role'] + "<|end_header_id|>\\n\\n" + message['content'] | trim + "<|eot_id|>" }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}\n        {{- '{"name": "' + tool_call.name + '", ' }}\n        {{- '"parameters": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n\n{%- if add_generation_prompt %}\n    {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}\n{%- endif %}\n"""
        
    @property
    def lengths(self):
        length_list = []
        for sample in self.datalist:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.datalist:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        source = self.datalist[i]
        
        if self.data_args.get_prompt:
            qs = copy.deepcopy(source)
        
        conversation = []
        for j in range(len(source["conversations"])):
            if j % 2 == 0:
                conversation.append({
                    "role": "user",
                    "content": str(source["conversations"][j]['value'])
                })
            else:
                conversation.append({
                    "role": "assistant",
                    "content": str(source["conversations"][j]['value'])
                })
        prompt = self.processor.apply_chat_template(conversation, tokenize=False)
        
        if self.data_args.is_multimodal:
            assert 'image' in source.keys()
            image_file = source['image']
            # image_folder = "/home/mila/s/sparsha.mishra/scratch/LLaVA/llava_ft"
            # image_folder = "dataset/llava_finetune"
            image_folder = "."
            if isinstance(image_file, list):
                image = [Image.open(os.path.join(image_folder, image_path)).convert('RGB') for image_path in image_file] #.split(' |sep| ')
            else:
                image = [Image.open(os.path.join(image_folder, image_file)).convert('RGB')]
            # if isinstance(image_file, list):
            #     image = [Image.open(image_path).convert('RGB') for image_path in image_file] #.split(' |sep| ')
            # else:
            #     image = [Image.open(image_file).convert('RGB')]
        
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                images = [expand2square(img, tuple(int(x*255) for x in self.processor.image_processor.image_mean)) for img in image]
            inputs = self.processor(images=images, text=prompt, return_tensors='pt')
        else:
            inputs = self.processor(text=prompt, return_tensors='pt')
        # Create labels with ignore_index (-100)
        input_ids = inputs['input_ids'][0]
        label_ids = input_ids.clone().fill_(IGNORE_INDEX)  # Initialize all labels as -100 (ignore_index)

        # Find all occurrences of the assistant start and end tokens
        assistant_start_tokens = self.label_start_id
        assistant_end_tokens = self.label_end_id
        start_indices = []
        end_indices = []
        
        for idx in range(len(input_ids) - len(assistant_start_tokens) + 1):
            if input_ids[idx:idx + len(assistant_start_tokens)].tolist() == assistant_start_tokens:
                start_indices.append(idx + len(assistant_start_tokens))  # Start of assistant response

        for idx in range(len(input_ids) - len(assistant_end_tokens) + 1):
            if input_ids[idx:idx + len(assistant_end_tokens)].tolist() == assistant_end_tokens:
                end_indices.append(idx + len(assistant_end_tokens))  # End of assistant response

        # Match start and end indices to create valid ranges
        for start_idx in start_indices:
            end_idx = next((end for end in end_indices if end > start_idx), None)
            if end_idx is not None:
                label_ids[start_idx:end_idx] = input_ids[start_idx:end_idx]
        
        data_dict = {
            'input_ids':input_ids,
            'labels': label_ids
        }
        
        if 'pixel_values' in inputs.keys():
            data_dict['pixel_values'] = inputs['pixel_values']
        
        if self.data_args.get_prompt:
            data_dict['prompt'] = qs
        
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # device:torch.device
    # transform:DataAugmentation

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = dict(
            input_ids=input_ids,#.to(self.device),
            labels=labels,#.to(self.device),
            attention_mask=attention_mask,#.to(self.device),
            use_cache=False,
        )

        if 'pixel_values' in instances[0]:
            images = [instance['pixel_values'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
                # b, n, c, h, w = images.shape
                # images = images.reshape(b*n,c,h,w)
                # images = self.transform(images).to(dtype=torch.bfloat16)
                # batch['images'] = images#.reshape(b,n,c,h,w)
                batch['pixel_values'] = images
            else:
                # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
                # batch['images'] = [x.to(dtype=torch.bfloat16) for x in images]
                batch['pixel_values'] = [x.to(dtype=torch.bfloat16) for x in images]

        if 'prompt' in instances[0]:
            batch['prompt'] = [instance['prompt'] for instance in instances]

        return batch