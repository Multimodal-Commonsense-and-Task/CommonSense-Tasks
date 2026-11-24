
import torch
import copy
import os
from torch.utils.data import Dataset
import transformers
from configuration.VLM_config_new import DataArguments
from typing import Dict, Sequence, Union
from PIL import Image, ImageFile
from dataclasses import dataclass
from packaging import version
import shutil
from transformers.trainer_pt_utils import LabelSmoother
import json
from qwen_vl_utils import process_vision_info
import re
import itertools
from typing import Dict, Optional, Sequence, List, Tuple, Any
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from collections import Counter
import numpy as np

from models.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava import conversation as conversation_lib_llava
from models.llava.mm_utils import process_images
from models.internvl.train.dataset import dynamic_preprocess

ImageFile.LOAD_TRUNCATED_IMAGES = True
# from utils.augment import DataAugmentation
# IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
IS_TOKENIZER_GREATER_THAN_0_14 = True
Image.MAX_IMAGE_PIXELS = None
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
SYSTEM_MESSAGE = "You are a helpful assistant."
### Intern
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


GRPO_SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

GRPO_XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def _map_role(src: str) -> str:
    src = (src or "").lower().strip()
    if src in ("human", "user"):
        return "user"
    if src in ("gpt", "assistant"):
        return "assistant"
    if src == "system":
        return "system"
    return src  # fallback


def _load_pil_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

class IdeficsICLJsonDataset(Dataset):
    """
    GRPO / generation-time dataset for Idefics

    Output sample:
      {
        "prompt": [{"role": "...", "content": "..."} , ...],
        "image": List[PIL.Image] or List[str],
        "id": ...,
        "task": ...,
      }
    """

    def __init__(
        self,
        datalist: Optional[List[Dict[str, Any]]] = None,
        image_mode: str = "query_last",   # kept for compatibility
        return_image: str = "pil",        # "pil" | "path"
        add_image_token: bool = True,
    ):
        self.records = datalist or []
        self.image_mode = image_mode
        self.return_image = return_image
        self.add_image_token = add_image_token

    def __len__(self) -> int:
        return len(self.records)

    def _select_image_paths(self, rec: Dict[str, Any]) -> Optional[List[str]]:
        paths = rec.get("image", None)
        if not paths or self.image_mode == "none":
            return None
        if not isinstance(paths, list):
            paths = [paths]
        return paths

    def _make_idefics_image_prefix(self, num_images: int) -> str:
        if num_images <= 0:
            return ""
        return "<image>" * num_images + "\n"

    def _build_prompt(self, rec: Dict[str, Any], num_images: int = 0) -> List[Dict[str, Any]]:
        conv = rec.get("conversations", [])
        prompt: List[Dict[str, Any]] = []

        image_prefix = self._make_idefics_image_prefix(num_images) if self.add_image_token else ""
        image_prefix_added = False

        # for m in conv:
        #     role = _map_role(m.get("from"))
        #     content = m.get("value", "")

        #     # 기존 <image> / DEFAULT_IMAGE_TOKEN 제거 후 다시 통일해서 넣기
        #     content = content.replace(DEFAULT_IMAGE_TOKEN, "").replace("<image>", "").strip()

        #     if role == "user" and image_prefix and not image_prefix_added:
        #         content = image_prefix + content
        #         image_prefix_added = True

        #     prompt.append({
        #         "role": role,
        #         "content": content,
        #     })
        image_files = rec["image"]
        gpt_response = conv[1]["value"]
        prompts = ""
        for conversation in conv:
            if conversation["from"] == "human":
                content = []
                qs = conversation['value']
                qs = qs.replace("\n", " ").strip()
                if "<image> <image>" in qs:
                    double_image = True
                    chunks = [c.strip() for c in qs.split("<image> <image") if c.strip()]
                    if not qs.startswith("<image> <image"):
                        chunks = [" ".join(chunks[:2])] + chunks[2:]
                else:
                    double_image = False
                    chunks = [c.strip() for c in qs.split("<image>") if c.strip()]
                    if not qs.startswith("<image>"):
                        chunks = [" ".join(chunks[:2])] + chunks[2:]

                pairs = []

                for chunk in chunks:
                    # Try to extract question and answer with flexible patterns
                    # 1) With "Question:" and "Answer:"
                    match = re.search(r"Question\s*[:：]\s*(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
                    # 2) Fallback: look for last "Answer" keyword
                    if not match:
                        match = re.search(r"(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
                    
                    if match:
                        q, a = match.groups()
                    else:
                        q, a = chunk, ""
                    pairs.append((q.strip(), a.strip()))
                
                if double_image:
                    for pair_i, pair in enumerate(pairs):
                        image1, image2 = image_files[pair_i*2], image_files[pair_i*2+1]
                        if pair_i == len(pairs) - 1:
                            prompts += f"User:<image><image>{pair[0]}<end_of_utterance>Assistant:{gpt_response}<end_of_utterance>\n"
                        else:
                            prompts += f"User:<image><image>{pair[0]}<end_of_utterance>Assistant:{pair[1]}<end_of_utterance>\n"
                else:
                    for image_file_i, image_path in enumerate(image_files):
                        pair = pairs[image_file_i]
                        if image_file_i == len(image_files) - 1:
                            prompts += f"User:<image>{pair[0]}<end_of_utterance>Assistant:{gpt_response}<end_of_utterance>\n"
                        else:
                            prompts += f"User:<image>{pair[0]}<end_of_utterance>Assistant:{pair[1]}<end_of_utterance>\n"

        return prompts

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        img_paths = self._select_image_paths(rec)
        num_images = len(img_paths) if img_paths is not None else 0
        sample: Dict[str, Any] = {
            "id": rec.get("id"),
            "task": rec.get("task"),
            "prompt": self._build_prompt(rec, num_images=num_images),
        }
        if img_paths is not None:
            if self.return_image == "pil":
                sample["image"] = [_load_pil_rgb(p) for p in img_paths]
            elif self.return_image == "path":
                if len(img_paths) == 1:
                    img_paths = img_paths[0]
                sample["image"] = img_paths
                if "neg_image" in rec:
                    neg = rec["neg_image"]
                    sample["neg_image"] = neg if isinstance(neg, list) else [neg]
            else:
                raise ValueError(f"Unknown return_image={self.return_image}")

        return sample


class InternVL35ICLJsonDataset(Dataset):
    """
    GRPO / generation-time dataset for InternVL3.5

    Output sample:
      {
        "prompt": [{"role": "...", "content": "..."} , ...],
        "image": List[PIL.Image] or List[str] or str,
        "id": ...,
        "task": ...,
      }
    """

    def __init__(
        self,
        datalist: Optional[List[Dict[str, Any]]] = None,
        image_mode: str = "query_last",   # kept for compatibility
        return_image: str = "pil",        # "pil" | "path"
        add_image_token: bool = True,
    ):
        self.records = datalist or []
        self.image_mode = image_mode
        self.return_image = return_image
        self.add_image_token = add_image_token

    def __len__(self) -> int:
        return len(self.records)

    def _select_image_paths(self, rec: Dict[str, Any]) -> Optional[List[str]]:
        paths = rec.get("image", None)
        if not paths or self.image_mode == "none":
            return None

        if not isinstance(paths, list):
            paths = [paths]
        return paths

    def _make_internvl_image_prefix(self, num_images: int) -> str:
        if num_images <= 0:
            return ""
        if num_images == 1:
            return "<image>\n"
        return "".join([f"Image-{i+1}: <image>\n" for i in range(num_images)])

    def _build_prompt(self, rec: Dict[str, Any], num_images: int = 0) -> List[Dict[str, Any]]:
        conv = rec.get("conversations", [])
        prompt: List[Dict[str, Any]] = []

        image_prefix = self._make_internvl_image_prefix(num_images) if self.add_image_token else ""
        image_prefix_added = False

        # for m in conv:
        #     role = _map_role(m.get("from"))
        #     content = m.get("value", "")

        #     # 첫 user message에만 image token prefix 삽입
        #     if role == "user" and image_prefix and not image_prefix_added:
        #         if "<image>" not in content:
        #             content = image_prefix + content
        #         image_prefix_added = True

        #     prompt.append({
        #         "role": role,
        #         "content": content,
        #     })

        return conv

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        img_paths = self._select_image_paths(rec)
        num_images = len(img_paths) if img_paths is not None else 0

        sample: Dict[str, Any] = {
            "id": rec.get("id"),
            "task": rec.get("task"),
            "prompt": self._build_prompt(rec, num_images=num_images),
        }

        if img_paths is not None:
            if self.return_image == "pil":
                sample["image"] = [_load_pil_rgb(p) for p in img_paths]
            elif self.return_image == "path":
                sample["image"] = img_paths
                if "neg_image" in rec:
                    neg = rec["neg_image"]
                    sample["neg_image"] = neg if isinstance(neg, list) else [neg]
            else:
                raise ValueError(f"Unknown return_image={self.return_image}")

        return sample


class ICLJsonDataset(Dataset):
    """
    Supports:
      - jsonl_path: each line is one dict
      - or in_memory_records: list[dict]

    Output sample (what _generate_and_score_completions expects per element):
      {
        "prompt": [{"role": "...", "content": "..."} , ...],
        "image": PIL.Image or str path (optional),
        "id": ... (optional passthrough),
        "task": ... (optional passthrough),
      }
    """

    def __init__(
        self,
        datalist: Optional[list] = None,
        in_memory_records: Optional[List[Dict[str, Any]]] = None,
        image_mode: str = "query_last",     # "query_last" | "first" | "none"
        return_image: str = "pil",          # "pil" | "path"
    ):
        self.image_mode = image_mode
        self.return_image = return_image
        # if jsonl_path is not None:
        #     self.records: List[Dict[str, Any]] = []
        #     with open(jsonl_path, "r", encoding="utf-8") as f:
        #         for line in f:
        #             line = line.strip()
        #             if line:
        #                 self.records.append(json.loads(line))
        # else:
        #     self.records = in_memory_records or []
        self.records = datalist

    def __len__(self) -> int:
        return len(self.records)

    def _select_image_path(self, rec: Dict[str, Any]) -> Optional[str]:
        paths = rec.get("image", None)
        if not paths or self.image_mode == "none":
            return None
        else:
            return paths
        raise ValueError(f"Unknown image_mode={self.image_mode}")

    def _build_prompt(self, rec: Dict[str, Any]) -> Union[str, List[Dict[str, Any]]]:
        """
        Your trainer supports both:
          - prompt as list[{"role","content"}] (conversational)
          - prompt as string
        Since your sample is conversational, we output chat format.
        """
        conv = rec.get("conversations", [])
        prompt: List[Dict[str, Any]] = []
        for m in conv:
            prompt.append(
                {"role": _map_role(m.get("from")), "content": m.get("value", "")}
            )
        return prompt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        sample: Dict[str, Any] = {
            "id": rec.get("id"),
            "task": rec.get("task"),
        }

        img_paths = self._select_image_path(rec)
        if img_paths is not None:
            if self.return_image == "pil":
                sample["image"] = [_load_pil_rgb(img_path) for img_path in img_paths]
            elif self.return_image == "path":
                sample["image"] = img_paths[0]
                sample["neg_image"] = rec["neg_image"]
            else:
                raise ValueError(f"Unknown return_image={self.return_image}")
            # rec["image"] = [rec["image"][0]]
            sample["prompt"] = self._build_prompt(rec)
        return sample


def make_inputs_dataloader(
    datalist: list,
    return_image: str = "pil",
    model: str = "qwenvl"
):
    print("init grpo dataloader")

    if "onevision" in model or "qwen" in model:
        ds = ICLJsonDataset(
            datalist=datalist,
            return_image=return_image,
        )
    elif "internvl" in model:
        ds = InternVL35ICLJsonDataset(
            datalist=datalist,
            return_image=return_image,
        )
    elif "idefics" in model:
        ds = IdeficsICLJsonDataset(
            datalist=datalist,
            return_image=return_image,
        )
    return ds

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

# class Intern_LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""
#     def __init__(self, datalist,
#             tokenizer,
#             data_args,
#             preprocess=False,
#             config=None):
#         super(Intern_LazySupervisedDataset, self).__init__()

#         self.tokenizer = tokenizer
#         self.datalist = datalist
#         self.data_args = data_args
#         self.preprocess = preprocess
#         self.dynamic_image_size = False
#         self.is_train = True
#         self.group_by_length=False
#         self.use_packed_ds=True

#         self.config = config
#         image_size = self.config.force_image_size or self.config.vision_config.image_size
#         patch_size = self.config.vision_config.patch_size
#         self.num_image_token = int((image_size // patch_size) ** 2 * (self.config.downsample_ratio ** 2))

#     @property
#     def lengths(self):
#         length_list = []
#         for sample in self.datalist:
#             img_tokens = 128 if 'image' in sample else 0
#             length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
#         return length_list

#     def __len__(self):
#         return len(self.datalist)

#     def get_preprocess_function(self):
#         from models.internvl.train.dataset import preprocess_internvl2_5
#         preprocess_function = preprocess_internvl2_5
#         return preprocess_function

#     def build_transform(self, input_size):
#         MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
#         transform = T.Compose([
#             T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#             T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#             T.ToTensor(),
#             T.Normalize(mean=MEAN, std=STD)
#         ])
#         return transform

#     def load_image(self, image_file):
#         image = Image.open(image_file).convert('RGB')
#         return image


#     def multi_modal_get_item(self, data_item, input_size=448, max_num=12):

#         # Ensure the first conversation contains an image placeholder
#         if '<image>' not in data_item['conversations'][0]['value']:
#             data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

#         image_path = data_item['image'][0]
#         # Load the image using tcs_loader if available, otherwise use PIL
#         image = self.load_image(image_path)
#         transform = self.build_transform(input_size=input_size)
#         if self.dynamic_image_size:
#             images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#         else:
#             images = [image]
#         pixel_values = [transform(image) for image in images]
#         pixel_values = torch.stack(pixel_values)
#         # Ensure that there is only one patch if dynamic image size is not enabled
#         num_patches = pixel_values.size(0)
#         if not self.dynamic_image_size:
#             assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

#         # Select the appropriate preprocessing function based on the template name
#         preprocess_function = self.get_preprocess_function()

#         # Preprocess the conversations and generate the return dictionary
#         ret = preprocess_function("internvl2_5", [copy.deepcopy(data_item['conversations'])],
#                                   self.tokenizer, [self.num_image_token * num_patches],
#                                   group_by_length=self.group_by_length,
#                                   use_packed_ds=self.use_packed_ds)

#         # Calculate position_ids for packed dataset
#         position_ids = ret['attention_mask'].long().cumsum(-1) - 1
#         position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
#         image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
#         # Create the final return dictionary
#         ret = dict(
#             input_ids=ret['input_ids'][0],
#             labels=ret['labels'][0],
#             attention_mask=ret['attention_mask'][0],
#             position_ids=position_ids[0],
#             pixel_values=pixel_values,
#             image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
#         )
#         return ret

#     def multi_modal_multi_image_get_item(self, data_item, input_size=448, max_num=4):
#         transform = self.build_transform(input_size=input_size)
        
#         images, num_tiles = [], []
#         num_image = len(data_item['image'])
#         for image_path in data_item['image']:
#             # Load the image using tcs_loader if available, otherwise use PIL
#             image = self.load_image(image_path)
            
#             if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
#                 image = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#                 images+=image
#                 num_tiles.append(len(image))
#             else:  # Otherwise, use the original image as a single patch
#                 images.append(image)
#                 num_tiles.append(1)
#         pixel_values = [transform(image) for image in images]
#         pixel_values = torch.stack(pixel_values)
#         num_patches = pixel_values.size(0)

#         # Select the appropriate preprocessing function based on the template name
#         preprocess_function = self.get_preprocess_function()

#         # Preprocess the conversations and generate the return dictionary
#         num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
#         ret = preprocess_function("internvl2_5", [copy.deepcopy(data_item['conversations'])],
#                                   self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
#                                   use_packed_ds=self.use_packed_ds, num_image=num_image)

#         # Calculate position_ids for packed dataset
#         position_ids = ret['attention_mask'].long().cumsum(-1) - 1
#         position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
#         image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
#         # Create the final return dictionary
#         ret = dict(
#             input_ids=ret['input_ids'][0],
#             labels=ret['labels'][0],
#             attention_mask=ret['attention_mask'][0],
#             position_ids=position_ids[0],
#             pixel_values=pixel_values,
#             image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
#         )
#         return ret

#     def __getitem__(self, i):
#         from models.internvl.conversation import get_conv_template
#         from models.internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig

#         sources = self.datalist[i]
#         conv = get_conv_template("internvl2_5")

#         if 'image' in sources:
#             all_images = sources["image"]
#             if not isinstance(all_images, list):
#                 all_images = [all_images]
#                 sources["image"] = all_images
#             if len(all_images) > 1:
#                 ret = self.multi_modal_multi_image_get_item(sources)
#             else:
#                 ret = self.multi_modal_get_item(sources)

#         return ret

@dataclass
class Intern_DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # device:torch.device
    # transform:DataAugmentation

    def __call__(self, features, max_item_length=None, pad_id=0):

        first = features[0]
        batch = {}

        batch_lens = [feat['input_ids'].shape for feat in features]
        max_item_length = max_item_length or max(batch_lens)[0]
        for idx in range(len(features)):
            feat = features[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
            feat['input_ids'] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[:feat['labels'].shape[0]] = feat['labels']
            feat['labels'] = temp_labels
            feat['attention_mask'] = feat['input_ids'].ne(pad_id)

            if 'position_ids' in feat:
                temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
                temp_position_ids[:feat['position_ids'].shape[0]] = feat['position_ids']
                feat['position_ids'] = temp_position_ids

            if 'loss_weight' in feat:
                temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
                temp_loss_weight[:feat['loss_weight'].shape[0]] = feat['loss_weight']
                feat['loss_weight'] = temp_loss_weight

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if 'label' in first and first['label'] is not None:
            label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
            dtype = torch.long if isinstance(label, int) else torch.float
            batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
        elif 'label_ids' in first and first['label_ids'] is not None:
            if isinstance(first['label_ids'], torch.Tensor):
                batch['labels'] = torch.stack([f['label_ids'] for f in features])
            else:
                dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
                batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                    v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
            if k in ('pixel_values', 'image_flags'):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.concat([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.concat(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.concat([f[k] for f in features])
        return batch
    
class Intern_GenerationDataset(Dataset):
    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(Intern_GenerationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess

    def __getitem__(self, index):
        source = self.datalist[index]
        qs = source["conversations"][0]['value']
        gold = source["conversations"][1]['value']

        qs.replace("<image>", "<IMG_CONTEXT>")
        conv = conversation_lib_llava.default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if "image" in source.keys():
            image_file = source["image"]
            
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
                image = torch.stack([self.data_args.image_processor.preprocess(expand2square(img, tuple(int(x*255) for x in self.data_args.image_processor.image_mean)), return_tensors='pt')['pixel_values'][0] for img in image])
            else:
                image = torch.stack([self.data_args.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in image])
        else: 
            image = torch.zeros(0)
            image_file = []
            
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        indices_regex = (input_ids == -200).nonzero(as_tuple=True)
        # return input_ids, image, gold, prompt, image_file
        return {
            'input_ids':input_ids,
            'image':image,
            'gold':gold,
            'prompt':prompt,
            'image_file':image_file,
            'image_indices': indices_regex
        }

    def __len__(self):
        return len(self.datalist)

# class Intern3_5_LazySupervisedDataset(Intern_LazySupervisedDataset):
#     """Dataset for InternVL 3.5 supervised fine-tuning."""
#     def __init__(self, datalist, tokenizer, data_args, preprocess=False, config=None):
#         super().__init__(datalist, tokenizer, data_args, preprocess, config)
        
#         self.dynamic_image_size = True

#     def get_preprocess_function(self):
#         from models.internvl.train.dataset import preprocess_internvl2_5
#         return preprocess_internvl2_5

#     def __getitem__(self, i):
#         sources = self.datalist[i]
#         if 'image' in sources:
#             all_images = sources["image"]
#             if not isinstance(all_images, list):
#                 all_images = [all_images]
#                 sources["image"] = all_images
            
#             if len(all_images) > 1:
#                 ret = self.multi_modal_multi_image_get_item(sources)
#             else:
#                 ret = self.multi_modal_get_item(sources)
#             return ret
        
#         return super().__getitem__(i)



class Intern_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        datalist,
        tokenizer,
        data_args,
        preprocess=False,
        config=None,
        onlyquery=False,
    ):
        super(Intern_LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        self.dynamic_image_size = False
        self.is_train = True
        self.group_by_length = False
        self.use_packed_ds = True
        self.onlyquery = onlyquery

        self.config = config
        image_size = self.config.force_image_size or self.config.vision_config.image_size
        patch_size = self.config.vision_config.patch_size
        self.num_image_token = int((image_size // patch_size) ** 2 * (self.config.downsample_ratio ** 2))

    @property
    def lengths(self):
        length_list = []
        for sample in self.datalist:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    def __len__(self):
        return len(self.datalist)

    def get_preprocess_function(self):
        from models.internvl.train.dataset import preprocess_internvl2_5
        preprocess_function = preprocess_internvl2_5
        return preprocess_function

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def load_image(self, image_file):
        image = Image.open(image_file).convert("RGB")
        return image

    def multi_modal_get_item(self, data_item, input_size=448, max_num=12):
        data_item = copy.deepcopy(data_item)

        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["conversations"][0]["value"]:
            data_item["conversations"][0]["value"] = "<image>\n" + data_item["conversations"][0]["value"]

        image_path = data_item["image"][0]
        image = self.load_image(image_path)
        transform = self.build_transform(input_size=input_size)

        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        else:
            images = [image]

        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f"The number of patches should be 1, but got {num_patches}."

        preprocess_function = self.get_preprocess_function()

        ret = preprocess_function(
            "internvl2_5",
            [copy.deepcopy(data_item["conversations"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
        )

        position_ids = ret["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item, input_size=448, max_num=4):
        data_item = copy.deepcopy(data_item)
        transform = self.build_transform(input_size=input_size)

        images, num_tiles = [], []
        num_image = len(data_item["image"])

        for image_path in data_item["image"]:
            image = self.load_image(image_path)

            if self.dynamic_image_size:
                image = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
                images += image
                num_tiles.append(len(image))
            else:
                images.append(image)
                num_tiles.append(1)

        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        preprocess_function = self.get_preprocess_function()

        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(
            "internvl2_5",
            [copy.deepcopy(data_item["conversations"])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            num_image=num_image,
        )

        position_ids = ret["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def __getitem__(self, i):
        from models.internvl.conversation import get_conv_template
        from models.internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig

        sources = copy.deepcopy(self.datalist[i])
        conv = get_conv_template("internvl2_5")

        if "image" in sources:
            all_images = sources["image"]
            if not isinstance(all_images, list):
                all_images = [all_images]
                sources["image"] = all_images

            if self.onlyquery:
                query_sources = copy.deepcopy(sources)
                if len(query_sources["conversations"]) > 0:
                    first_value = query_sources["conversations"][0]["value"]
                    if "<image>" in first_value:
                        query_question = first_value.split("<image>")[-1]
                        query_sources["conversations"][0]["value"] = "<image>\n" + query_question.lstrip("\n")
                    else:
                        query_sources["conversations"][0]["value"] = "<image>\n" + first_value

                if len(all_images) > 1:
                    ret = self.multi_modal_multi_image_get_item(sources)
                    query_ret = self.multi_modal_multi_image_get_item(query_sources)
                else:
                    ret = self.multi_modal_get_item(sources)
                    query_ret = self.multi_modal_get_item(query_sources)

                return ret, query_ret

            else:
                if len(all_images) > 1:
                    ret = self.multi_modal_multi_image_get_item(sources)
                else:
                    ret = self.multi_modal_get_item(sources)

                return ret


class Intern3_5_LazySupervisedDataset(Intern_LazySupervisedDataset):
    """Dataset for InternVL 3.5 supervised fine-tuning."""
    def __init__(
        self,
        datalist,
        tokenizer,
        data_args,
        preprocess=False,
        config=None,
        onlyquery=False,
    ):
        super().__init__(
            datalist=datalist,
            tokenizer=tokenizer,
            data_args=data_args,
            preprocess=preprocess,
            config=config,
            onlyquery=onlyquery,
        )
        self.dynamic_image_size = True

    def get_preprocess_function(self):
        from models.internvl.train.dataset import preprocess_internvl2_5
        return preprocess_internvl2_5

    def __getitem__(self, i):
        sources = copy.deepcopy(self.datalist[i])

        if "image" in sources:
            all_images = sources["image"]
            if not isinstance(all_images, list):
                all_images = [all_images]
                sources["image"] = all_images

            if self.onlyquery:
                query_sources = copy.deepcopy(sources)
                if len(query_sources["conversations"]) > 0:
                    first_value = query_sources["conversations"][0]["value"]
                    if "<image>" in first_value:
                        query_question = first_value.split("<image>")[-1]
                        query_sources["conversations"][0]["value"] = "<image>\n" + query_question.lstrip("\n")
                    else:
                        query_sources["conversations"][0]["value"] = "<image>\n" + first_value

                if len(all_images) > 1:
                    ret = self.multi_modal_multi_image_get_item(sources)
                    query_ret = self.multi_modal_multi_image_get_item(query_sources)
                else:
                    ret = self.multi_modal_get_item(sources)
                    query_ret = self.multi_modal_get_item(query_sources)

                return ret, query_ret

            else:
                if len(all_images) > 1:
                    ret = self.multi_modal_multi_image_get_item(sources)
                else:
                    ret = self.multi_modal_get_item(sources)
                return ret

        return super().__getitem__(i)


@dataclass
class Intern3_5_DataCollatorForSupervisedDataset(Intern_DataCollatorForSupervisedDataset):
    """Collate examples for InternVL 3.5."""
    pass

class Onevision_LazySupervisedDataset(Dataset):
    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(Onevision_LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        self.data_args.is_multimodal = True

    def __len__(self):
        return len(self.datalist)

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

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(image_file).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

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

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.datalist[i]
        if self.data_args.get_prompt:
            prompt = sources["conversations"][0]['value']
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = self.datalist[i]["image"]
            image = [self.process_image(f) for f in image_file]
            # Handling multi images
            # overwrite to process with simple pad
            
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        has_image = ("image" in self.datalist[i]) or ("video" in self.datalist[i])
        data_dict = preprocess_qwen(sources, self.tokenizer, has_image=has_image)
        # data_dict = preprocess_text_llava(sources, self.tokenizer, has_image=has_image)
        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        # image exist in the data
        if "image" in self.datalist[i]:
            data_dict["image"] = image
        # elif "video" in self.list_data_dict[i]:
        #     data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.datalist[i].get("id", i)

        return data_dict


@dataclass
class Onevision_DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch



@dataclass
class OneVision1_5_DataCollatorForGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, gold, prompt, image_grid_thw, image_indices = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", 'gold', 'prompt', 'image_grid_thw', 'image_indices'))
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
            image_grid_thw=image_grid_thw,
            image_indices=image_indices
        )
        if 'pixel_values' in instances[0].keys():
            image_file = [instance['image_file'] for instance in instances]
            batch['image_file'] = image_file
            images = [instance['pixel_values'] for instance in instances]
            if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
                images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
                batch['pixel_values'] = images#.reshape(b,n,c,h,w)
            else:
                batch['pixel_values'] = [x.to(dtype=torch.bfloat16) for x in images]
        batch['prompt'] = prompt
        return batch
    

class OneVision1_5_GenerationDataset(Dataset):
    def __init__(self, datalist, tokenizer,
                 processor,
                 data_args,
                 preprocess=False):
        super(OneVision1_5_GenerationDataset, self).__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        
        self.image_min_pixel = (256 * 28 * 28)
        self.image_max_pixel = (1280 * 28 * 28)
        self.image_resized_w = None
        self.image_resized_h = None

    def __getitem__(self, index):
        
        sources = self.datalist[index]
        gold = sources["conversations"][1]['value']
        qs = sources["conversations"][0]['value']

        prompt = qs
        is_video = False
        is_image = False
        processor = self.processor
        if "image" in sources:
            is_image = True
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
        messages = copy.deepcopy(llava_to_openai_generate(sources['conversations'],image_files, is_video=is_video))
        # print("messages", messages)
        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if is_image:
            images, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, padding=False, return_tensors='pt')
            prompt_input_ids = inputs['input_ids']
            pixel_values = inputs[pixel_key]
            image_thw = inputs[grid_key]

            t_flat = prompt_input_ids.flatten()
            
            idx = (t_flat == 151655).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                indices_regex = torch.tensor([], dtype=torch.long, device=t.device)  # no matches
            elif len(idx) == 1:
                indices_regex = idx
            else:
                gaps = torch.diff(idx) > 1  # length N-1
                keep = torch.cat([torch.tensor([True], device=idx.device), gaps])
                first_indices = idx[keep]
                indices_regex = idx[keep]

        else:
            prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

        attention_mask = (prompt_input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=prompt_input_ids,
            prompt=prompt,
            image_indices=indices_regex
        )

        if pixel_key and grid_key:

            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            data_dict['image_file'] = image_files
            data_dict['gold'] = str(gold)
            data_dict['prompt'] = prompt

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict
        
    def __len__(self):
        return len(self.datalist)

# class OneVision1_5_LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, datalist, processor, data_args, preprocess=False, padding=True):
#         super().__init__()

#         self.processor = processor
#         self.datalist = datalist
#         self.data_args = data_args
#         self.padding = padding

#         self.image_min_pixel = 224 * 224
#         self.image_max_pixel = 1024 * 1024
#         self.image_resized_w = 384
#         self.image_resized_h = 384

#     def __len__(self):
#         return len(self.datalist)

#     def load_image(self, image_file):
#         return Image.open(image_file).convert("RGB")

#     def _get_item(self, i):
#         sources = self.datalist[i]
#         is_video = False

#         processor = self.processor
#         tokenizer = processor.tokenizer

#         if "image" in sources:
#             grid_key = "image_grid_thw"
#             pixel_key = "pixel_values"

#             image_files = sources["image"]
#             image_folder = self.data_args.image_folder

#             if isinstance(image_files, str):
#                 image_files = [image_files]

#             images = []
#             for image_file in image_files:
#                 if not os.path.exists(image_file) and not image_file.startswith("http"):
#                     image_file = os.path.join(image_folder, image_file)
#                 image_input = get_image_info(
#                     image_file,
#                     self.image_min_pixel,
#                     self.image_max_pixel,
#                     self.image_resized_w,
#                     self.image_resized_h,
#                 )
#                 images.append(image_input)
#         else:
#             grid_key = None
#             pixel_key = None
#             images = None

#         sources = copy.deepcopy(llava_to_openai(sources["conversations"], is_video=is_video))

#         all_input_ids = []
#         all_labels = []
#         all_pixel_values = []
#         all_image_grid_thw = []
#         all_second_gird = []

#         if len(SYSTEM_MESSAGE) > 0:
#             system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
#             system_ids = tokenizer(system_message, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
#             system_labels = torch.full_like(system_ids, IGNORE_INDEX)
#             all_input_ids.append(system_ids)
#             all_labels.append(system_labels)

#         image_curr_count = 0
#         for j in range(0, len(sources), 2):
#             user, gpt = sources[j], sources[j + 1]

#             user_text = (
#                 f"{DEFAULT_IM_START_TOKEN}{user['role']}\n"
#                 f"{user['content']}"
#                 f"{DEFAULT_IM_END_TOKEN}\n"
#                 f"{DEFAULT_IM_START_TOKEN}{gpt['role']}\n"
#             )
#             gpt_text = f"{gpt['content']}{DEFAULT_IM_END_TOKEN}\n"

#             if "<|image_pad|>" in user_text:
#                 num_images = user_text.count("<|image_pad|>")
#                 images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
#                 inputs = processor(
#                     text=[user_text],
#                     images=images_for_this_turn,
#                     padding=False,
#                     return_tensors="pt",
#                 )
#                 prompt_ids = inputs["input_ids"]
#                 all_pixel_values.append(inputs[pixel_key])
#                 all_image_grid_thw.append(inputs[grid_key])
#                 image_curr_count += num_images
#             else:
#                 prompt_ids = tokenizer(user_text, add_special_tokens=False, return_tensors="pt")["input_ids"]

#             response_ids = tokenizer(gpt_text, add_special_tokens=False, return_tensors="pt")["input_ids"]

#             input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
#             labels = torch.cat(
#                 [
#                     torch.full((prompt_ids.size(1),), IGNORE_INDEX, dtype=torch.long),
#                     response_ids.squeeze(0),
#                 ],
#                 dim=0,
#             )

#             all_input_ids.append(input_ids)
#             all_labels.append(labels)

#         input_ids = torch.cat(all_input_ids, dim=0).long()
#         labels = torch.cat(all_labels, dim=0).long()

#         attention_mask = (input_ids > -1000000).to(torch.long)

#         data_dict = dict(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#         )

#         if pixel_key and grid_key and len(all_pixel_values) > 0:
#             data_dict["pixel_values"] = torch.cat(all_pixel_values, dim=0)
#             data_dict["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

#         if len(all_second_gird) > 0:
#             data_dict["second_per_grid_ts"] = all_second_gird

#         return data_dict

#     def __getitem__(self, i) -> Dict[str, Any]:
#         try:
#             return self._get_item(i)
#         except Exception as e:
#             print(e)
#             print("error in :")
#             print(self.datalist[i])
#             return self._get_item(0)
class OneVision1_5_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        datalist,
        processor,
        data_args,
        preprocess=False,
        padding=True,
        onlyquery=False,
    ):
        super().__init__()

        self.processor = processor
        self.datalist = datalist
        self.data_args = data_args
        self.padding = padding
        self.preprocess = preprocess
        self.onlyquery = onlyquery

        self.image_min_pixel = 224 * 224
        self.image_max_pixel = 1024 * 1024
        self.image_resized_w = 384
        self.image_resized_h = 384

    def __len__(self):
        return len(self.datalist)

    def load_image(self, image_file):
        return Image.open(image_file).convert("RGB")

    def _get_item(self, i):
        sources = self.datalist[i]
        is_video = False

        processor = self.processor
        tokenizer = processor.tokenizer

        if "image" in sources:
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                if not os.path.exists(image_file) and not image_file.startswith("http"):
                    image_file = os.path.join(image_folder, image_file)
                image_input = get_image_info(
                    image_file,
                    self.image_min_pixel,
                    self.image_max_pixel,
                    self.image_resized_w,
                    self.image_resized_h,
                )
                images.append(image_input)
        else:
            grid_key = None
            pixel_key = None
            images = None

        sources = copy.deepcopy(llava_to_openai(sources["conversations"], is_video=is_video))

        query_sources = None
        if self.onlyquery:
            query_sources = copy.deepcopy(sources)
            query_question = query_sources[0]["content"].split("<|vision_start|><|image_pad|><|vision_end|>")[-1]
            query_sources[0]["content"] = "<|vision_start|><|image_pad|><|vision_end|>" + query_question

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        all_query_input_ids = []
        all_query_labels = []
        all_query_pixel_values = []
        all_query_image_grid_thw = []
        all_query_second_gird = []

        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_ids = tokenizer(
                system_message,
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"].squeeze(0)
            system_labels = torch.full_like(system_ids, IGNORE_INDEX)
            all_input_ids.append(system_ids)
            all_labels.append(system_labels)

            if self.onlyquery:
                all_query_input_ids.append(system_ids.clone())
                all_query_labels.append(system_labels.clone())

        image_curr_count = 0
        query_img_curr_count = 0

        for j in range(0, len(sources), 2):
            user, gpt = sources[j], sources[j + 1]

            user_text = (
                f"{DEFAULT_IM_START_TOKEN}{user['role']}\n"
                f"{user['content']}"
                f"{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{gpt['role']}\n"
            )
            gpt_text = f"{gpt['content']}{DEFAULT_IM_END_TOKEN}\n"

            if "<|image_pad|>" in user_text:
                num_images = user_text.count("<|image_pad|>")
                images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(
                    text=[user_text],
                    images=images_for_this_turn,
                    padding=False,
                    return_tensors="pt",
                )
                prompt_ids = inputs["input_ids"]
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images
            else:
                prompt_ids = tokenizer(
                    user_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"]

            response_ids = tokenizer(
                gpt_text,
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"]

            input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.full((prompt_ids.size(1),), IGNORE_INDEX, dtype=torch.long),
                    response_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

            if self.onlyquery:
                query_user, query_gpt = query_sources[j], query_sources[j + 1]

                query_user_text = (
                    f"{DEFAULT_IM_START_TOKEN}{query_user['role']}\n"
                    f"{query_user['content']}"
                    f"{DEFAULT_IM_END_TOKEN}\n"
                    f"{DEFAULT_IM_START_TOKEN}{query_gpt['role']}\n"
                )
                query_gpt_text = f"{query_gpt['content']}{DEFAULT_IM_END_TOKEN}\n"

                if "<|image_pad|>" in query_user_text:
                    num_images = query_user_text.count("<|image_pad|>")
                    images_for_this_turn = images[query_img_curr_count : query_img_curr_count + num_images]
                    query_inputs = processor(
                        text=[query_user_text],
                        images=images_for_this_turn,
                        padding=False,
                        return_tensors="pt",
                    )
                    query_prompt_ids = query_inputs["input_ids"]
                    all_query_pixel_values.append(query_inputs[pixel_key])
                    all_query_image_grid_thw.append(query_inputs[grid_key])
                    query_img_curr_count += num_images
                else:
                    query_prompt_ids = tokenizer(
                        query_user_text,
                        add_special_tokens=False,
                        return_tensors="pt"
                    )["input_ids"]

                query_response_ids = tokenizer(
                    query_gpt_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"]

                query_input_ids = torch.cat([query_prompt_ids, query_response_ids], dim=1).squeeze(0)
                query_labels = torch.cat(
                    [
                        torch.full((query_prompt_ids.size(1),), IGNORE_INDEX, dtype=torch.long),
                        query_response_ids.squeeze(0),
                    ],
                    dim=0,
                )

                all_query_input_ids.append(query_input_ids)
                all_query_labels.append(query_labels)

        input_ids = torch.cat(all_input_ids, dim=0).long()
        labels = torch.cat(all_labels, dim=0).long()

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key and len(all_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        if len(all_second_gird) > 0:
            data_dict["second_per_grid_ts"] = all_second_gird

        if self.onlyquery:
            query_input_ids = torch.cat(all_query_input_ids, dim=0).long()
            query_labels = torch.cat(all_query_labels, dim=0).long()
            query_attention_mask = (query_input_ids > -1000000).to(torch.long)

            query_data_dict = dict(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                labels=query_labels,
            )

            if pixel_key and grid_key and len(all_query_pixel_values) > 0:
                query_data_dict["pixel_values"] = torch.cat(all_query_pixel_values, dim=0)
                query_data_dict["image_grid_thw"] = torch.cat(all_query_image_grid_thw, dim=0)

            if len(all_query_second_gird) > 0:
                query_data_dict["second_per_grid_ts"] = all_query_second_gird

            return data_dict, query_data_dict

        return data_dict

    def __getitem__(self, i) -> Dict[str, Any]:
        try:
            return self._get_item(i)
        except Exception as e:
            print(e)
            print("error in :")
            print(self.datalist[i])
            return self._get_item(0)
            
class OneVision1_5_DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_image_thw = []

        for example in examples:
            keys = example.keys()
            if "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            batch_label_ids,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        attention_mask = input_ids != self.tokenizer.pad_token_id

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if len(batch_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)

        return data_dict

class GenerationDataset(Dataset):
    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(GenerationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess

    def __getitem__(self, index):
        source = self.datalist[index]
        qs = source["conversations"][0]['value']
        gold = source["conversations"][1]['value']

        conv = conversation_lib_llava.default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if "image" in source.keys():
            image_file = source["image"]
            
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
                image = torch.stack([self.data_args.image_processor.preprocess(expand2square(img, tuple(int(x*255) for x in self.data_args.image_processor.image_mean)), return_tensors='pt')['pixel_values'][0] for img in image])
            else:
                image = torch.stack([self.data_args.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in image])
        else: 
            image = torch.zeros(0)
            image_file = []
            
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        indices_regex = (input_ids == -200).nonzero(as_tuple=True)
        return {
            'input_ids':input_ids,
            'image':image,
            'gold':gold,
            'prompt':prompt,
            'image_file':image_file,
            'image_indices': indices_regex
        }

    def __len__(self):
        return len(self.datalist)

class Qwen_GenerationDataset(Dataset):
    def __init__(self, datalist, processor,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(Qwen_GenerationDataset, self).__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        
        self.image_min_pixel = (256 * 28 * 28)
        self.image_max_pixel = (1280 * 28 * 28)
        self.image_resized_w = None
        self.image_resized_h = None

    def __getitem__(self, index):
        
        sources = self.datalist[index]
        gold = sources["conversations"][1]['value']
        qs = sources["conversations"][0]['value']
        prompt = qs
        is_video = False
        is_image = False
        processor = self.processor
        if "image" in sources:
            is_image = True
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
            image_files=None
        messages = copy.deepcopy(llava_to_openai_generate(sources['conversations'],image_files, is_video=is_video))
        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if is_image:
            images, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, padding=False, return_tensors='pt')
            prompt_input_ids = inputs['input_ids']
            pixel_values = inputs[pixel_key]
            image_thw = inputs[grid_key]
        else:
            prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

        attention_mask = (prompt_input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=prompt_input_ids,
            prompt=prompt,
        )

        if pixel_key and grid_key:
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            data_dict['image_file'] = image_files
            data_dict['gold'] = str(gold)
            data_dict['prompt'] = prompt

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict
        
    def __len__(self):
        return len(self.datalist)

class QwenVL3_GenerationDataset(Dataset):
    def __init__(self, datalist, processor,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(QwenVL3_GenerationDataset, self).__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        
        self.image_min_pixel = (256 * 28 * 28)
        self.image_max_pixel = (1280 * 28 * 28)
        self.image_resized_w = None
        self.image_resized_h = None
        
    def __getitem__(self, index):
        
        sources = self.datalist[index]
        gold = sources["conversations"][1]['value']
        qs = sources["conversations"][0]['value']

        prompt = qs
        is_video = False
        is_image = False
        processor = self.processor
        if "image" in sources:
            is_image = True
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
        messages = copy.deepcopy(llava_to_openai_generate(sources['conversations'],image_files, is_video=is_video))
        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if is_image:
            images, _ = process_vision_info(messages, image_patch_size=16)
            inputs = processor(text=text, images=images, videos=videos, do_resize=False, return_tensors='pt')
            prompt_input_ids = inputs['input_ids']
            pixel_values = inputs[pixel_key]
            image_thw = inputs[grid_key]
            
        else:
            prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

        attention_mask = (prompt_input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=prompt_input_ids,
            prompt=prompt,
        )

        if pixel_key and grid_key:
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            data_dict['image_file'] = image_files
            data_dict['gold'] = str(gold)
            data_dict['prompt'] = prompt

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict
        
    def __len__(self):
        return len(self.datalist)

class QwenVL3_GenerationDataset_CoT(Dataset):
    def __init__(self, datalist, processor,
                 tokenizer,
                 data_args,
                 prompt_ver=1,
                 preprocess=False):
        super(QwenVL3_GenerationDataset_CoT, self).__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        
        self.image_min_pixel = (256 * 28 * 28)
        self.image_max_pixel = (1280 * 28 * 28)
        self.image_resized_w = None
        self.image_resized_h = None
        
        self.cot_prompt_version = prompt_ver
    def __getitem__(self, index):
        
        sources = self.datalist[index]
        gold = sources["conversations"][1]['value']
        qs = sources["conversations"][0]['value']

        prompt = qs
        is_video = False
        is_image = False
        processor = self.processor
        if "image" in sources:
            is_image = True
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
            
        if self.cot_prompt_version == 1:
            messages = copy.deepcopy(llava_to_openai_generate_cot(sources['conversations'],image_files, is_video=is_video))
        elif self.cot_prompt_version == 2:
            messages = copy.deepcopy(llava_to_openai_generate_cot2(sources['conversations'],image_files, is_video=is_video))    
        
        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if is_image:
            images, _ = process_vision_info(messages, image_patch_size=16)
            inputs = processor(text=text, images=images, videos=videos, do_resize=False, return_tensors='pt')
            # print("processed", inputs)
            prompt_input_ids = inputs['input_ids']
            pixel_values = inputs[pixel_key]
            image_thw = inputs[grid_key]
            
        else:
            prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

        attention_mask = (prompt_input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=prompt_input_ids,
            prompt=prompt,
        )

        if pixel_key and grid_key:
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            data_dict['image_file'] = image_files
            data_dict['gold'] = str(gold)
            data_dict['prompt'] = prompt

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict
        
    def __len__(self):
        return len(self.datalist)

class QwenVL3_LazySupervisedDataset_GRPO(Dataset):
    def __init__(
        self,
        datalist,
        processor,
        data_args,
        preprocess=False,
        padding=True,
    ):
        super().__init__()

        self.datalist = datalist
        self.processor = processor
        self.data_args = data_args
        self.padding = padding
        self.preprocess = preprocess
        self.ignore_index = -100
        
        self.image_min_pixel = (256 * 28 * 28)
        self.image_max_pixel = (1280 * 28 * 28)
        self.image_resized_w = None
        self.image_resized_h = None

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        sample = self.datalist[idx]
        processor = self.processor
        tokenizer = processor.tokenizer
        is_video = False
        # print("tokenizer.eos_token", tokenizer.pad_token,tokenizer.eos_token)
        # print("tokenizer.eos_token_id", tokenizer.pad_token_id,tokenizer.eos_token_id)
        # tokenizer.pad_token = tokenizer.eos_token
        if "image" in sample:
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sample["image"]
            image_folder = self.data_args.image_folder
            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                if not os.path.exists(image_file) and not image_file.startswith("http"):
                    image_file = os.path.join(image_folder, image_file)
                image_input = get_image_info(
                    image_file,
                    self.image_min_pixel,
                    self.image_max_pixel,
                    self.image_resized_w,
                    self.image_resized_h,
                )
                images.append(image_input)
        else:
            grid_key = None
            pixel_key = None
            images = None

        sources = copy.deepcopy(llava_to_openai(sample["conversations"], is_video=is_video))

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []
    
        image_curr_count = 0

        if len(GRPO_SYSTEM_PROMPT) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{GRPO_SYSTEM_PROMPT}{DEFAULT_IM_END_TOKEN}\n"
            system_ids = tokenizer(system_message, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
            system_labels = torch.full_like(system_ids, IGNORE_INDEX)
            all_input_ids.append(system_ids)
            all_labels.append(system_labels)
        
        for j in range(0, len(sources), 2):
            user, gpt = sources[j], sources[j + 1]

            user_text = (
                f"{DEFAULT_IM_START_TOKEN}{user['role']}\n"
                f"{user['content']}"
                f"{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{gpt['role']}\n"
            )
            gpt_text = f"{gpt['content']}{DEFAULT_IM_END_TOKEN}\n"
            if "<|image_pad|>" in user_text:
                num_images = user_text.count("<|image_pad|>")
                images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(
                    text=[user_text],
                    images=images_for_this_turn,
                    padding=False,
                    do_resize=True,
                    return_tensors="pt",
                )
                prompt_ids = inputs["input_ids"]
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images
            else:
                prompt_ids = tokenizer(user_text, add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"]

            response_ids = tokenizer(gpt_text, add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"]
            input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.full((prompt_ids.size(1),), IGNORE_INDEX, dtype=torch.long),
                    response_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids = torch.cat(all_input_ids, dim=0).long()
        labels = torch.cat(all_labels, dim=0).long()
        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if pixel_key and grid_key and len(all_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        if len(all_second_gird) > 0:
            data_dict["second_per_grid_ts"] = all_second_gird

        return data_dict



class QwenVL3_LazySupervisedDataset(Dataset):
    def __init__(
        self,
        datalist,
        processor,
        data_args,
        preprocess=False,
        padding=True,
        onlyquery=False
    ):
        super().__init__()

        self.datalist = datalist
        self.processor = processor
        self.data_args = data_args
        self.padding = padding
        self.preprocess = preprocess
        self.ignore_index = -100
        
        self.onlyquery = onlyquery
        self.image_min_pixel = (256 * 28 * 28)
        self.image_max_pixel = (1280 * 28 * 28)
        self.image_resized_w = None
        self.image_resized_h = None

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        sample = self.datalist[idx]
        processor = self.processor
        tokenizer = processor.tokenizer
        is_video = False
 
        if "image" in sample:
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sample["image"]
            image_folder = self.data_args.image_folder
            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                image_input = get_image_info(
                    image_file,
                    self.image_min_pixel,
                    self.image_max_pixel,
                    self.image_resized_w,
                    self.image_resized_h,
                )
                images.append(image_input)
        else:
            grid_key = None
            pixel_key = None
            images = None

        sources = copy.deepcopy(llava_to_openai(sample["conversations"], is_video=is_video))
        query_sources = copy.deepcopy(sources)
        query_question = query_sources[0]["content"].split("<|vision_start|><|image_pad|><|vision_end|>")[-1]
        query_sources[0]["content"] = "<|vision_start|><|image_pad|><|vision_end|>"+query_question
        
        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []
        
        all_query_input_ids = []
        all_query_labels = []
        all_query_pixel_values = []
        all_query_image_grid_thw = []
        all_query_second_gird = []

        image_curr_count = 0
        query_img_curr_count = 0
        for j in range(0, len(sources), 2):
            user, gpt = sources[j], sources[j + 1]

            user_text = (
                
                f"{DEFAULT_IM_START_TOKEN}{user['role']}\n"
                f"{user['content']}"
                f"{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{gpt['role']}\n"
            )
            gpt_text = f"{gpt['content']}{DEFAULT_IM_END_TOKEN}\n"
            if "<|image_pad|>" in user_text:
                num_images = user_text.count("<|image_pad|>")
                images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(
                    text=[user_text],
                    images=images_for_this_turn,
                    padding=False,
                    do_resize=True,
                    return_tensors="pt",
                )

                prompt_ids = inputs["input_ids"]
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images
            else:
                prompt_ids = tokenizer(user_text, add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"]
            response_ids = tokenizer(gpt_text, add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"]
            input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.full((prompt_ids.size(1),), IGNORE_INDEX, dtype=torch.long),
                    response_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            
            if self.onlyquery:
                query_user, query_gpt = query_sources[j], query_sources[j + 1]

                query_user_text = (
                    f"{DEFAULT_IM_START_TOKEN}{query_user['role']}\n"
                    f"{query_user['content']}"
                    f"{DEFAULT_IM_END_TOKEN}\n"
                    f"{DEFAULT_IM_START_TOKEN}{query_gpt['role']}\n"
                )
                query_gpt_text = f"{query_gpt['content']}{DEFAULT_IM_END_TOKEN}\n"
                if "<|image_pad|>" in query_user_text:
                    num_images = query_user_text.count("<|image_pad|>")
                    images_for_this_turn = images[query_img_curr_count : query_img_curr_count + num_images]
                    query_inputs = processor(
                        text=[query_user_text],
                        images=images_for_this_turn,
                        padding=False,
                        do_resize=True,
                        return_tensors="pt",
                    )
                    query_prompt_ids = query_inputs["input_ids"]
                    all_query_pixel_values.append(query_inputs[pixel_key])
                    all_query_image_grid_thw.append(query_inputs[grid_key])
                    query_img_curr_count += num_images
                else:
                    query_prompt_ids = tokenizer(query_user_text, add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"]

                query_response_ids = tokenizer(query_gpt_text, add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"]
                query_input_ids = torch.cat([query_prompt_ids, query_response_ids], dim=1).squeeze(0)
                query_labels = torch.cat(
                    [
                        torch.full((query_prompt_ids.size(1),), IGNORE_INDEX, dtype=torch.long),
                        query_response_ids.squeeze(0),
                    ],
                    dim=0,
                )

                all_query_input_ids.append(query_input_ids)
                all_query_labels.append(query_labels)

        input_ids = torch.cat(all_input_ids, dim=0).long()
        labels = torch.cat(all_labels, dim=0).long()
        attention_mask = (input_ids > -1000000).to(torch.long)
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if pixel_key and grid_key and len(all_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        if len(all_second_gird) > 0:
            data_dict["second_per_grid_ts"] = all_second_gird

        if self.onlyquery:
            query_input_ids = torch.cat(all_query_input_ids, dim=0).long()
            query_labels = torch.cat(all_query_labels, dim=0).long()
            query_attention_mask = (query_input_ids > -1000000).to(torch.long)
            query_data_dict = {
                "input_ids": query_input_ids,
                "attention_mask": query_attention_mask,
                "labels": query_labels,
            }

            if pixel_key and grid_key and len(all_query_pixel_values) > 0:
                query_data_dict["pixel_values"] = torch.cat(all_query_pixel_values, dim=0)
                query_data_dict["image_grid_thw"] = torch.cat(all_query_image_grid_thw, dim=0)

            if len(all_query_second_gird) > 0:
                query_data_dict["second_per_grid_ts"] = all_query_second_gird

            return data_dict, query_data_dict
        else:

            return data_dict


@dataclass
class QwenVL3_DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ignore_index = -100

    def __call__(self, examples):
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_grid_thw = []
        
        query_batch_input_ids = []
        query_batch_labels = []
        query_batch_pixel_values = []
        query_batch_image_grid_thw = []
        if len(examples[0]) == 2:
            query_examples = [examples[0][-1]]
            examples = [examples[0][0]]
            onlyquery = True
        else:
            onlyquery = False
        for example in examples:
            keys = example.keys()

            if "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_grid_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_labels.append(example["labels"])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels,
            batch_first=True,
            padding_value=self.ignore_index,
        )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if len(batch_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)
        
        if onlyquery:
            for query_example in query_examples:
                keys = query_example.keys()

                if "pixel_values" in keys:
                    query_batch_pixel_values.append(query_example["pixel_values"])
                    query_batch_image_grid_thw.append(query_example["image_grid_thw"])

                query_batch_input_ids.append(query_example["input_ids"])
                query_batch_labels.append(query_example["labels"])

            query_input_ids = torch.nn.utils.rnn.pad_sequence(
                query_batch_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )

            query_labels = torch.nn.utils.rnn.pad_sequence(
                query_batch_labels,
                batch_first=True,
                padding_value=self.ignore_index,
            )

            query_attention_mask = query_input_ids.ne(self.tokenizer.pad_token_id)

            query_data_dict = {
                "input_ids": query_input_ids,
                "labels": query_labels,
                "attention_mask": query_attention_mask,
            }

            if len(query_batch_pixel_values) > 0:
                query_data_dict["pixel_values"] = torch.cat(query_batch_pixel_values, dim=0)
                query_data_dict["image_grid_thw"] = torch.cat(query_batch_image_grid_thw, dim=0)
            return data_dict, query_data_dict
            

        return data_dict


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * 2

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


class Llama_GenerationDataset(Dataset):
    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 processor=None, qwen=False):
        super(Llama_GenerationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.processor = processor
        self.conv = conversation_lib_llava.default_conversation.copy()
        
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

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        
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
        t_flat = inputs['input_ids'][0].flatten()
        idx = (t_flat == 128257).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            print("no match")
        elif len(idx) == 1:
            indices_regex = idx
        else:
            gaps = torch.diff(idx) > 1  # length N-1
            keep = torch.cat([torch.tensor([True], device=idx.device), gaps])
            first_indices = idx[keep]
            indices_regex = idx[keep]
        return_dict = {
            'input_ids':inputs['input_ids'][0],
            'gold':str(gold),
            'prompt':prompt,
            'image_indices': indices_regex
        }
        if 'pixel_values' in inputs.keys():
            return_dict['pixel_values'] = inputs['pixel_values']
            return_dict['image_file'] = image_file
        return return_dict

    def __len__(self):
        return len(self.datalist)


class Otter_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning of Otter."""

    def __init__(
        self,
        datalist,
        tokenizer,
        data_args,
        preprocess=False,
        is_mimic=False,
    ):
        super(Otter_LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        self.image_processor = self.data_args.image_processor
        self.is_mimic = is_mimic

    @property
    def lengths(self):
        length_list = []
        for sample in self.datalist:
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]))
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.datalist:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.datalist)

    def _build_source(self, source, mimic=False):
        source = copy.deepcopy(source)

        if mimic and len(source["conversations"]) > 0:
            first_val = source["conversations"][0]["value"].strip()

            if "<image> <image>" in first_val:
                query_question = first_val.split("<image> <image>")[-1].strip()
                source["conversations"][0]["value"] = "<image>\n" + query_question
            elif "<image>" in first_val:
                query_question = first_val.split("<image>")[-1].strip()
                source["conversations"][0]["value"] = "<image>\n" + query_question

        for conv in source["conversations"]:
            val = conv["value"].strip()
            val = val.replace("\nAnswer:", " GPT: <answer>")

            if "<image>" in val:
                parts = [p.strip() for p in val.split("<image>")]
                new_val = parts[0]
                for idx in range(1, len(parts)):
                    if idx == 1:
                        new_val += f"<image>User:{parts[idx]}"
                    else:
                        new_val += f"<|endofchunk|><image>User:{parts[idx]}"
                val = new_val

            val = val.replace("\nGPT: <answer>", " GPT: <answer>")
            conv["value"] = val

        return source

    def _load_pixel_values(self, source):
        if "image" in source:
            image_files = source["image"]
            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                try:
                    if not os.path.exists(image_file) and not image_file.startswith("http"):
                        image_file = os.path.join(self.data_args.image_folder, image_file)
                    image = Image.open(image_file).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"Error reading image {image_file}: {e}")

            try:
                if len(images) > 0:
                    pixel_values = torch.stack(
                        [
                            self.image_processor(img, return_tensors="pt")["pixel_values"][0]
                            for img in images
                        ]
                    )
                else:
                    pixel_values = torch.zeros(0)
            except Exception as e:
                print(f"Image processing error: {e}")
                pixel_values = torch.zeros(0)
        else:
            pixel_values = torch.zeros(0)

        return pixel_values

    def _process_single(self, source, mimic=False):
        built_source = self._build_source(source, mimic=mimic)
        pixel_values = self._load_pixel_values(built_source)

        wrapped_sources = [built_source]
        source_convs = [s["conversations"] for s in wrapped_sources]

        data_dict = preprocess_otter(
            source_convs,
            self.tokenizer,
            has_image=("image" in built_source),
        )

        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
        )
        data_dict["image"] = pixel_values

        return data_dict

    def __getitem__(self, i):
        try:
            source = self.datalist[i]

            data_dict = self._process_single(source, mimic=False)

            if self.is_mimic:
                query_data_dict = self._process_single(source, mimic=True)
                return data_dict, query_data_dict

            return data_dict

        except Exception as e:
            print(f"Error in sample {i}: {e}")
            return self.__getitem__((i + 1) % len(self))

class Otter_GenerationDataset(Dataset):
    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(Otter_GenerationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        self.image_processor = self.data_args.image_processor

    def __getitem__(self, index):
        source = self.datalist[index]
        qs = source["conversations"][0]['value'].strip()
        gold = source["conversations"][1]['value'].strip()

        print(f"\n[GenerationDataset] --- Index {index} ---")
        print(f"[DEBUG] Original QS: {qs[:50]}...")

        qs = qs.replace("Learn from the demos and give only the answer.", "")
        qs = qs.replace("\nAnswer:", " Answer:").replace("Answer:", "GPT:<answer>")
        qs = re.sub(r'\s+', ' ', qs).strip()

        if "<image>" in qs:
            parts = [p.strip() for p in qs.split("<image>")]
            
            global_instruction = parts[0] if parts[0] != "" else ""
            
            new_qs = ""
            image_related_parts = parts[1:]
            
            for i, part in enumerate(image_related_parts):
                processed_part = part.replace("GPT:<answer>", "GPT:<answer>").strip()
                
                if i == 0:
                    prefix = f"{global_instruction} " if global_instruction else ""
                    new_qs += f"<image>User:{prefix.strip()}{processed_part}"
                else:
                    new_qs += f"<|endofchunk|><image>User:{processed_part}"

            qs = new_qs

        prompt = qs.strip()
        if not prompt.endswith("GPT:<answer>"):
            if prompt.endswith("Answer:"):
                prompt = prompt[:-len("Answer:")].strip()
            prompt += "GPT:<answer>"

        print(f"[DEBUG] Formatted Prompt: {repr(prompt)}")
        print(f"[DEBUG] Gold: {gold}")

        safe_max_length = min(self.tokenizer.model_max_length, 32767)

        if "image" in source.keys():
            image_file = source["image"]
            
            if isinstance(image_file, list):
                image = [Image.open(image_path).convert('RGB') for image_path in image_file]
            else:
                image = [Image.open(image_file).convert('RGB')]

            try:
                image = torch.stack([self.image_processor(img, return_tensors='pt')['pixel_values'][0] for img in image])
                image = image.unsqueeze(1)
            except Exception as e:
                print(f"Error processing image: {e}, file: {image_file}")
        else:
            image = torch.zeros(0)
            image_file = []

        inputs = self.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True, max_length=safe_max_length)
        
        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]

        return {
            'input_ids': input_ids,
            'pixel_values': image,
            'gold': gold,
            'prompt': prompt,
            'image_file': image_file,
            'attention_mask': attention_mask
        }

    def __len__(self):
        return len(self.datalist)

@dataclass
class Llama_DataCollatorForGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, gold, prompt, image_indices = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", 'gold', 'prompt', 'image_indices'))
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
            prompt=prompt,
            image_indices=image_indices
        )
        
        if 'pixel_values' in instances[0].keys():
            image_file = [instance['image_file'] for instance in instances]
            batch['image_file'] = image_file
            images = [instance['pixel_values'] for instance in instances]
            if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
                images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
                batch['pixel_values'] = images#.reshape(b,n,c,h,w)
            else:
                batch['pixel_values'] = [x.to(dtype=torch.bfloat16) for x in images]

        return batch

class OneVision_GenerationDataset(Dataset):
    def __init__(self, datalist,
                 tokenizer,
                 processor,
                 data_args,
                 preprocess=False):
        super(OneVision_GenerationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess

    def __getitem__(self, index):
        source = self.datalist[index]
        qs = source["conversations"][0]['value']
        gold = source["conversations"][1]['value']

        conv = conversation_lib_llava.conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if "image" in source.keys():
            image_file = source["image"]
            
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
                image = torch.stack([self.data_args.image_processor.preprocess(expand2square(img, tuple(int(x*255) for x in self.data_args.image_processor.image_mean)), return_tensors='pt')['pixel_values'][0] for img in image])
            else:
                image = torch.stack([self.data_args.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in image])
        else: 
            image = torch.zeros(0)
            image_file = []
            
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        indices_regex = (input_ids == -200).nonzero(as_tuple=True)
        return {
            'input_ids':input_ids,
            'image':image,
            'gold':gold,
            'prompt':prompt,
            'image_file':image_file,
            'image_indices': indices_regex
        }

    def __len__(self):
        return len(self.datalist)


@dataclass
class OneVision_DataCollatorForGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, gold, prompt, image_file = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", 'gold', 'prompt', 'image_file'))

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
            prompt=prompt,
            image_file=image_file
        )
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
            images = torch.stack(images)#.to(device=self.device)
            batch['images'] = images#.reshape(b,n,c,h,w)
        else:
            batch['images'] = [x.to(dtype=torch.bfloat16) for x in images]
        return batch

@dataclass
class Qwen_DataCollatorForGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, gold, prompt, image_grid_thw = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", 'gold', 'prompt', 'image_grid_thw'))

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
            image_grid_thw=image_grid_thw,
        )
        if 'pixel_values' in instances[0].keys():
            image_file = [instance['image_file'] for instance in instances]
            batch['image_file'] = image_file
            images = [instance['pixel_values'] for instance in instances]
            if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
                images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
                batch['pixel_values'] = images#.reshape(b,n,c,h,w)
            else:
                batch['pixel_values'] = [x.to(dtype=torch.bfloat16) for x in images]
        batch['prompt'] = prompt
        return batch
  
@dataclass
class Otter_DataCollatorForGenerationDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        keys = ("input_ids", 'gold', 'prompt', 'image_file')
        input_ids = [instance['input_ids'] for instance in instances]
        gold = [instance['gold'] for instance in instances]
        prompt = [instance['prompt'] for instance in instances]
        image_file = [instance['image_file'] for instance in instances]

        max_length = max(seq.size(0) for seq in input_ids)
        input_ids_padded = torch.stack([
            torch.cat([torch.full((max_length - seq.size(0),), self.tokenizer.pad_token_id), seq])
            for seq in input_ids
        ])
        input_ids_padded = input_ids_padded[:, :self.tokenizer.model_max_length]
        
        attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id)
        
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids_padded:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask,
            'gold': gold,
            'prompt': prompt,
            'image_file': image_file,
            'pixel_values': None
        }

        if any('pixel_values' in instance for instance in instances):
            images = [instance['pixel_values'] for instance in instances if 'pixel_values' in instance]
            
            if len(images) > 0:
                max_t_img = max(img.shape[0] for img in images) 
                padded_images = []
                for img in images:
                    pad_size = max_t_img - img.shape[0]
                    if pad_size > 0:
                        c, h, w = img.shape[1:]
                        padding = torch.zeros(pad_size, c, h, w, dtype=img.dtype)
                        padded_img = torch.cat([img, padding], dim=0)
                    else:
                        padded_img = img
                    padded_images.append(padded_img)
                
                batch['pixel_values'] = torch.stack(padded_images).to(torch.bfloat16)
        
        return batch

@dataclass
class DataCollatorForGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, gold, prompt, image_file, image_indices = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", 'gold', 'prompt', 'image_file', 'image_indices'))

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
            prompt=prompt,
            image_file=image_file,
            image_indices=image_indices
        )
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
            images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
            batch['images'] = images#.reshape(b,n,c,h,w)
        else:
            batch['images'] = [x.to(dtype=torch.bfloat16) for x in images]

        return batch

def get_image_info(image_path, min_pixel, max_pixel, width, height):
    # Using this because of process_vision_info function
    # Need to fix this in the future


    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]
    image_input, _ = process_vision_info(messages)

    return image_input[0]

def replace_image_tokens(input_string):

    pattern = r'\n?' + re.escape("<image>") + r'\n?'
    replacement = "<|vision_start|>"+ "<|image_pad|>" + "<|vision_end|>"

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"])
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def llava_to_openai_generate_cot(conversations, image_files, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}
    cot_prompt = "Let's think step by step."
    prompts = []
    for conversation in conversations:
        if conversation["from"] == "human":
            content = []
            qs = conversation['value']
            qs = qs.replace("\n", " ").strip()
            if "<image> <image>" in qs:
                double_image = True
                chunks = [c.strip() for c in qs.split("<image> <image") if c.strip()]
                if not qs.startswith("<image> <image"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]
            else:
                double_image = False
                chunks = [c.strip() for c in qs.split("<image>") if c.strip()]
                if not qs.startswith("<image>"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]

            pairs = []

            for chunk in chunks:
                match = re.search(r"Question\s*[:：]\s*(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
                if not match:
                    match = re.search(r"(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
                
                if match:
                    q, a = match.groups()
                else:
                    q, a = chunk, ""
                pairs.append((q.strip(), a.strip()))
            
            if double_image:
                for pair_i, pair in enumerate(pairs):
                    image1, image2 = image_files[pair_i*2], image_files[pair_i*2+1]
                    if pair_i == len(pairs) - 1:
                        user_dict = {"role": "user", "content": [{"type": "image", "image": image1}, {"type": "image", "image": image2}, {"type": "text", "text":f"User: {pair[0]} \nAssistant: "}]}
                    else:
                        user_dict = {"role": "user", "content": [{"type": "image", "image": image1}, {"type": "image", "image": image2}, {"type": "text", "text":f"User: {pair[0]} \nAssistant: {pair[1]}"}]}
                    prompts.append(user_dict)
            else:
                for image_file_i, image_path in enumerate(image_files):
                    pair = pairs[image_file_i]
                    if image_file_i == len(image_files) - 1:
                        user_dict = {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text":f"User: {pair[0]} {cot_prompt} \nAssistant: "}]}
                    else:
                        user_dict = {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text":f"User: {pair[0]} {cot_prompt} \nAssistant: {pair[1]}"}]}
                    prompts.append(user_dict)

    return prompts

def llava_to_openai_generate_cot2(conversations, image_files, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}
    cot_prompt = "Let's think step by step. Write each reasoning step on a new line."
    prompts = []
    for conversation in conversations:
        if conversation["from"] == "human":
            content = []
            qs = conversation['value']
            qs = qs.replace("\n", " ").strip()
            if "<image> <image>" in qs:
                double_image = True
                chunks = [c.strip() for c in qs.split("<image> <image") if c.strip()]
                if not qs.startswith("<image> <image"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]
            else:
                double_image = False
                chunks = [c.strip() for c in qs.split("<image>") if c.strip()]
                if not qs.startswith("<image>"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]

            pairs = []

            for chunk in chunks:
                match = re.search(r"Question\s*[:：]\s*(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
                if not match:
                    match = re.search(r"(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
                
                if match:
                    q, a = match.groups()
                else:
                    q, a = chunk, ""
                pairs.append((q.strip(), a.strip()))
            
            if double_image:
                for pair_i, pair in enumerate(pairs):
                    image1, image2 = image_files[pair_i*2], image_files[pair_i*2+1]
                    if pair_i == len(pairs) - 1:
                        user_dict = {"role": "user", "content": [{"type": "image", "image": image1}, {"type": "image", "image": image2}, {"type": "text", "text":f"User: {pair[0]} {cot_prompt} \nAssistant: "}]}
                    else:
                        user_dict = {"role": "user", "content": [{"type": "image", "image": image1}, {"type": "image", "image": image2}, {"type": "text", "text":f"User: {pair[0]} \nAssistant: {pair[1]}"}]}
                    prompts.append(user_dict)
            else:
                for image_file_i, image_path in enumerate(image_files):
                    pair = pairs[image_file_i]
                    if image_file_i == len(image_files) - 1:
                        user_dict = {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text":f"User: {pair[0]} {cot_prompt} \nAssistant: "}]}
                    else:
                        user_dict = {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text":f"User: {pair[0]} \nAssistant: {pair[1]}"}]}
                    prompts.append(user_dict)

    return prompts
import re

def llava_to_openai_generate(conversations, image_files=None, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    if image_files is None:
        image_files = []

    prompts = []

    for conv_i, conversation in enumerate(conversations):
        if conversation["from"] != "human":
            continue

        qs = conversation["value"].replace("\n", " ").strip()

        has_double_image_token = "<image> <image>" in qs
        has_image_token = "<image>" in qs

        if has_double_image_token:
            double_image = True
            chunks = [c.strip() for c in qs.split("<image> <image>") if c.strip()]
            if not qs.startswith("<image> <image>"):
                chunks = [" ".join(chunks[:2])] + chunks[2:]
        elif has_image_token:
            double_image = False
            chunks = [c.strip() for c in qs.split("<image>") if c.strip()]
            if not qs.startswith("<image>"):
                chunks = [" ".join(chunks[:2])] + chunks[2:]
        else:
            double_image = False
            chunks = [qs] if qs else []

        pairs = []

        for chunk in chunks:
            chunk = chunk.strip()

            chunk_wo_question = re.sub(
                r"^\s*Question\s*[:：]\s*",
                "",
                chunk,
                flags=re.IGNORECASE,
            )

            matches = list(
                re.finditer(r"Answer\s*[:：]\s*", chunk_wo_question, flags=re.IGNORECASE)
            )

            if matches:
                last_match = matches[-1]
                q = chunk_wo_question[:last_match.start()].strip()
                a = chunk_wo_question[last_match.end():].strip()
            else:
                q, a = chunk_wo_question.strip(), ""

            pairs.append((q, a))

        if len(pairs) > 0 and pairs[-1][1] == "":
            if conv_i + 1 < len(conversations) and conversations[conv_i + 1]["from"] == "gpt":
                pairs[-1] = (pairs[-1][0], conversations[conv_i + 1]["value"].strip())

        # Case 1: two images per pair
        if has_double_image_token and len(image_files) >= 2:
            num_pairs = min(len(pairs), len(image_files) // 2)

            for pair_i in range(num_pairs):
                q, a = pairs[pair_i]
                image1 = image_files[pair_i * 2]
                image2 = image_files[pair_i * 2 + 1]

                if pair_i == num_pairs - 1:
                    text = f"User: {q}\nAssistant: "
                else:
                    text = f"User: {q}\nAssistant: {a}"

                prompts.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image1},
                            {"type": "image", "image": image2},
                            {"type": "text", "text": text},
                        ],
                    }
                )

        # Case 2: one image per pair
        elif has_image_token and len(image_files) >= 1:
            num_pairs = min(len(pairs), len(image_files))

            for pair_i in range(num_pairs):
                q, a = pairs[pair_i]
                image_path = image_files[pair_i]

                if pair_i == num_pairs - 1:
                    text = f"User: {q}\nAssistant: "
                else:
                    text = f"User: {q}\nAssistant: {a}"

                prompts.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": text},
                        ],
                    }
                )

        # Case 3: no image
        else:
            for pair_i, (q, a) in enumerate(pairs):
                if pair_i == len(pairs) - 1:
                    text = f"User: {q}\nAssistant: "
                else:
                    text = f"User: {q}\nAssistant: {a}"

                prompts.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    }
                )

    return prompts

class Qwen_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(Qwen_LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path
    
        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = (256 * 28 * 28)
        self.image_max_pixel = (1280 * 28 * 28)
        self.image_resized_w = None
        self.image_resized_h = None
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.get_rope_index = get_rope_index_25

        if "Qwen3" in self.model_id:
            self.image_patch_size = 16
            self.return_video_metadata = True
        else:
            self.image_patch_size = 14
            self.return_video_metadata = False

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            images = []

            for image_file in image_files:
                # if not os.path.exists(image_file):
                #     if not image_file.startswith("http"):
                #         image_file = os.path.join(image_folder, image_file)
                image_input = get_image_info(
                        image_file, 
                        self.image_min_pixel, 
                        self.image_max_pixel, 
                        self.image_resized_w, 
                        self.image_resized_h
                    )
                images.append(image_input)

        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))
        
        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        image_curr_count = 0
        video_curr_count = 0
        
        # Qwen2-VL uses a default system message so I've added this.
        # Qwen3-Vl does not use a system message by default.
        if len(SYSTEM_MESSAGE) > 0 and "Qwen3" not in self.model_id:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            
            if "<|image_pad|>" in user_input:
                num_images = user_input.count("<|image_pad|>")
                # Slice the images list to get the images for the current turn.
                images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(text=[user_input], images=images_for_this_turn, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images
            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird

        return data_dict

class Qwen_DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_position_ids = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            # batch_position_ids.append(example["position_ids"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = torch.nn.utils.rnn.pad_sequence(batch_label_ids, batch_first=True, padding_value=IGNORE_INDEX)
        # position_ids = pad_and_cat(position_ids)
        # position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            # 'position_ids': position_ids
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
    

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
    
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
        sources = self.datalist[i]
        
        if self.data_args.get_prompt:
            prompt = sources["conversations"][0]['value']
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            processor = self.data_args.image_processor
            if self.preprocess:
                image = self.images[i]
            else:
                image_file = self.datalist[i]['image']
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
                image = torch.stack([processor.preprocess(expand2square(img, tuple(int(x*255) for x in processor.image_mean)), return_tensors='pt')['pixel_values'][0] for img in image])
            else:
                image = torch.stack([processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in image])
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            image=torch.zeros(0)
        if 'llava' in self.data_args.model_name_for_dataarg.lower() or 'qwen' in self.data_args.model_name_for_dataarg.lower() :
            data_dict = preprocess_text_llava(
                sources,
                self.tokenizer,
                has_image=('image' in self.datalist[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        data_dict['image'] = image
        
        if self.data_args.get_prompt:
            data_dict['prompt'] = prompt
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
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
                # b, n, c, h, w = images.shape
                # images = images.reshape(b*n,c,h,w)
                # images = self.transform(images).to(dtype=torch.bfloat16)
                batch['images'] = images#.reshape(b,n,c,h,w)
            else:
                # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
                batch['images'] = [x.to(dtype=torch.bfloat16) for x in images]

        if 'prompt' in instances[0]:
            batch['prompt'] = [instance['prompt'] for instance in instances]

        return batch

class Idefics_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning of Idefics."""

    def __init__(
        self,
        datalist,
        tokenizer,
        data_args,
        processor,
        preprocess=False,
        onlyquery=False,
    ):
        super(Idefics_LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        self.processor = processor
        self.onlyquery = onlyquery

        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.do_image_splitting = False

    @property
    def lengths(self):
        length_list = []
        for sample in self.datalist:
            text_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            img_tokens = 0
            if "image" in sample:
                images = sample["image"]
                num_images = len(images) if isinstance(images, list) else 1
                img_tokens = num_images * 64
            length_list.append(text_len + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.datalist:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.datalist)

    def _build_prompts(self, conversations, image_files, final_gpt_response, mimic=False):
        prompts = ""

        for conv_idx, conversation in enumerate(conversations):
            if conversation["from"] != "human":
                continue

            qs = conversation["value"].replace("\n", " ").strip()

            if mimic and conv_idx == 0:
                if "<image> <image>" in qs:
                    qs = "<image> " + qs.split("<image> <image>")[-1].strip()
                elif "<image>" in qs:
                    qs = "<image> " + qs.split("<image>")[-1].strip()

            if "<image> <image>" in qs:
                double_image = True
                chunks = [c.strip() for c in qs.split("<image> <image>") if c.strip()]
                if not qs.startswith("<image> <image>"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]
            else:
                double_image = False
                chunks = [c.strip() for c in qs.split("<image>") if c.strip()]
                if not qs.startswith("<image>"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]

            pairs = []
            for chunk in chunks:
                match = re.search(
                    r"Question\s*[:：]\s*(.*?)\s*Answer\s*[:：]\s*(.*)",
                    chunk,
                    re.IGNORECASE,
                )
                if not match:
                    match = re.search(
                        r"(.*?)\s*Answer\s*[:：]\s*(.*)",
                        chunk,
                        re.IGNORECASE,
                    )

                if match:
                    q, a = match.groups()
                else:
                    q, a = chunk, ""
                pairs.append((q.strip(), a.strip()))

            if double_image:
                for pair_i, pair in enumerate(pairs):
                    if pair_i == len(pairs) - 1:
                        prompts += (
                            f"User:<image><image>{pair[0]}"
                            f"<end_of_utterance>Assistant:{final_gpt_response}<end_of_utterance>\n"
                        )
                    else:
                        prompts += (
                            f"User:<image><image>{pair[0]}"
                            f"<end_of_utterance>Assistant:{pair[1]}<end_of_utterance>\n"
                        )
            else:
                usable_len = min(len(image_files), len(pairs))
                for image_file_i in range(usable_len):
                    pair = pairs[image_file_i]
                    if image_file_i == usable_len - 1:
                        prompts += (
                            f"User:<image>{pair[0]}"
                            f"<end_of_utterance>Assistant:{final_gpt_response}<end_of_utterance>\n"
                        )
                    else:
                        prompts += (
                            f"User:<image>{pair[0]}"
                            f"<end_of_utterance>Assistant:{pair[1]}<end_of_utterance>\n"
                        )

        return prompts

    def _process_single(self, source, mimic=False):
        source = copy.deepcopy(source)

        images = []
        image_files = source.get("image", [])
        if isinstance(image_files, str):
            image_files = [image_files]

        if image_files:
            for image_file in image_files:
                try:
                    if not os.path.exists(image_file) and not image_file.startswith("http"):
                        image_file = os.path.join(self.data_args.image_folder, image_file)
                    image = Image.open(image_file).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"Error reading image {image_file}: {e}")
                    images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        conversations = source["conversations"]
        gpt_response = conversations[1]["value"].strip()

        full_text = self._build_prompts(
            conversations=conversations,
            image_files=image_files,
            final_gpt_response=gpt_response,
            mimic=mimic,
        )

        processor_images = [images] if images else None

        try:
            inputs = self.processor(
                text=[full_text],
                images=processor_images,
                return_tensors="pt",
                padding=False,
            )
        except Exception as e:
            print(f"Error processing prompts: {e}")
            raise e

        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]

        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values[0]

        labels = input_ids.clone()

        eou_token_id = self.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
        eou_indices = (input_ids == eou_token_id).nonzero(as_tuple=True)[0]

        if len(eou_indices) > 0:
            len_prompt = eou_indices[0].item() + 1
            labels[:len_prompt] = IGNORE_INDEX

        data_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if pixel_values is not None:
            data_dict["pixel_values"] = pixel_values

        return data_dict

    def __getitem__(self, i):
        source = self.datalist[i]

        try:
            data_dict = self._process_single(source, mimic=False)

            if self.onlyquery:
                query_data_dict = self._process_single(source, mimic=True)
                return data_dict, query_data_dict

            return data_dict

        except Exception as e:
            print(f"Error in sample {i}: {e}")
            return self.__getitem__((i + 1) % len(self))

# class Idefics_LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning of Idefics."""

#     def __init__(self, datalist,
#                  tokenizer,
#                  data_args,
#                  processor,
#                  preprocess=False):
#         super(Idefics_LazySupervisedDataset, self).__init__()

#         self.tokenizer = tokenizer
#         self.datalist = datalist
#         self.data_args = data_args
#         self.preprocess = preprocess
#         self.processor = processor

#         if hasattr(self.processor, "image_processor"):
#             self.processor.image_processor.do_image_splitting = False
    
#     @property
#     def lengths(self):
#         length_list = []
#         for sample in self.datalist:
#             text_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
#             img_tokens = 0
#             if 'image' in sample:
#                 images = sample['image']
#                 num_images = len(images) if isinstance(images, list) else 1
#                 img_tokens = num_images * 64 
#             length_list.append(text_len + img_tokens)
#         return length_list

#     @property
#     def modality_lengths(self):
#         length_list = []
#         for sample in self.datalist:
#             cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
#             cur_len = cur_len if 'image' in sample else -cur_len
#             length_list.append(cur_len)
#         return length_list

#     def __len__(self):
#         return len(self.datalist)

#     def __getitem__(self, i):
#         source = self.datalist[i]
        
#         images = []
#         if 'image' in source:
#             image_files = source['image']
#             if isinstance(image_files, str):
#                 image_files = [image_files]
            
#             for image_file in image_files:
#                 try:
#                     if not os.path.exists(image_file) and not image_file.startswith("http"):
#                         image_file = os.path.join(self.data_args.image_folder, image_file)
#                     image = Image.open(image_file).convert('RGB')
#                     images.append(image)
#                 except Exception as e:
#                     print(f"Error reading image {image_file}: {e}")
#                     images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

#         conversations = source['conversations']
#         # human_input = conversations[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').replace('<image>', '').strip()
#         gpt_response = conversations[1]['value'].strip()
#         # image_tokens = "<image>" * len(images)
        
#         # full_text = f"User:{image_tokens}{human_input}<end_of_utterance>Assistant:{gpt_response}<end_of_utterance>"
#         prompts = ""
#         for conversation in conversations:
#             if conversation["from"] == "human":
#                 content = []
#                 qs = conversation['value']
#                 qs = qs.replace("\n", " ").strip()
#                 if "<image> <image>" in qs:
#                     double_image = True
#                     chunks = [c.strip() for c in qs.split("<image> <image") if c.strip()]
#                     if not qs.startswith("<image> <image"):
#                         chunks = [" ".join(chunks[:2])] + chunks[2:]
#                 else:
#                     double_image = False
#                     chunks = [c.strip() for c in qs.split("<image>") if c.strip()]
#                     if not qs.startswith("<image>"):
#                         chunks = [" ".join(chunks[:2])] + chunks[2:]

#                 pairs = []

#                 for chunk in chunks:
#                     # Try to extract question and answer with flexible patterns
#                     # 1) With "Question:" and "Answer:"
#                     match = re.search(r"Question\s*[:：]\s*(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
#                     # 2) Fallback: look for last "Answer" keyword
#                     if not match:
#                         match = re.search(r"(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
                    
#                     if match:
#                         q, a = match.groups()
#                     else:
#                         q, a = chunk, ""
#                     pairs.append((q.strip(), a.strip()))
                
#                 if double_image:
#                     for pair_i, pair in enumerate(pairs):
#                         image1, image2 = image_files[pair_i*2], image_files[pair_i*2+1]
#                         if pair_i == len(pairs) - 1:
#                             prompts += f"User:<image><image>{pair[0]}<end_of_utterance>Assistant:{gpt_response}<end_of_utterance>\n"
#                         else:
#                             prompts += f"User:<image><image>{pair[0]}<end_of_utterance>Assistant:{pair[1]}<end_of_utterance>\n"
#                 else:
#                     for image_file_i, image_path in enumerate(image_files):
#                         pair = pairs[image_file_i]
#                         if image_file_i == len(image_files) - 1:
#                             prompts += f"User:<image>{pair[0]}<end_of_utterance>Assistant:{gpt_response}<end_of_utterance>\n"
#                         else:
#                             prompts += f"User:<image>{pair[0]}<end_of_utterance>Assistant:{pair[1]}<end_of_utterance>\n"
                            
#         full_text = prompts
#         processor_images = [images] if images else None
        
#         try:
#             inputs = self.processor(
#                 text=[full_text], 
#                 images=processor_images, 
#                 return_tensors="pt", 
#                 padding=False
#             )
#         except Exception as e:
#             print(f"Error processing prompts: {e}")
#             return self.__getitem__((i + 1) % len(self))

#         input_ids = inputs.input_ids[0]
#         attention_mask = inputs.attention_mask[0]
        
#         pixel_values = inputs.get("pixel_values")
#         if pixel_values is not None:
#             pixel_values = pixel_values[0]

#         labels = input_ids.clone()
        
#         eou_token_id = self.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
#         eou_indices = (input_ids == eou_token_id).nonzero(as_tuple=True)[0]
        
#         if len(eou_indices) > 0:
#             len_prompt = eou_indices[0].item() + 1
#             labels[:len_prompt] = IGNORE_INDEX
        
#         data_dict = dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=attention_mask,
#         )
        
#         if pixel_values is not None:
#             data_dict['pixel_values'] = pixel_values
            
#         return data_dict

@dataclass
class Idefics_DataCollatorForSupervisedDataset(object):
    """Collate examples for Idefics supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels, attention_mask = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "attention_mask"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0)

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        attention_mask = attention_mask[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'pixel_values' in instances[0]:
            pixel_values = [instance['pixel_values'] for instance in instances]
            if all(x is not None for x in pixel_values):
                 batch['pixel_values'] = torch.stack(pixel_values).to(torch.bfloat16)

        return batch
    
    
class Otter_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning of Otter."""

    def __init__(
        self,
        datalist,
        tokenizer,
        data_args,
        preprocess=False,
        is_mimic=False,
    ):
        super(Otter_LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        self.image_processor = self.data_args.image_processor
        self.is_mimic = is_mimic

    @property
    def lengths(self):
        length_list = []
        for sample in self.datalist:
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]))
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.datalist:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.datalist)

    def _build_source(self, source, mimic=False):
        source = copy.deepcopy(source)

        if mimic and len(source["conversations"]) > 0:
            first_val = source["conversations"][0]["value"].strip()

            if "<image> <image>" in first_val:
                query_question = first_val.split("<image> <image>")[-1].strip()
                source["conversations"][0]["value"] = "<image>\n" + query_question
            elif "<image>" in first_val:
                query_question = first_val.split("<image>")[-1].strip()
                source["conversations"][0]["value"] = "<image>\n" + query_question

        for conv in source["conversations"]:
            val = conv["value"].strip()
            val = val.replace("\nAnswer:", " GPT: <answer>")

            if "<image>" in val:
                parts = [p.strip() for p in val.split("<image>")]
                new_val = parts[0]
                for idx in range(1, len(parts)):
                    if idx == 1:
                        new_val += f"<image>User:{parts[idx]}"
                    else:
                        new_val += f"<|endofchunk|><image>User:{parts[idx]}"
                val = new_val

            val = val.replace("\nGPT: <answer>", " GPT: <answer>")
            conv["value"] = val

        return source

    def _load_pixel_values(self, source):
        if "image" in source:
            image_files = source["image"]
            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                try:
                    if not os.path.exists(image_file) and not image_file.startswith("http"):
                        image_file = os.path.join(self.data_args.image_folder, image_file)
                    image = Image.open(image_file).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"Error reading image {image_file}: {e}")

            try:
                if len(images) > 0:
                    pixel_values = torch.stack(
                        [
                            self.image_processor(img, return_tensors="pt")["pixel_values"][0]
                            for img in images
                        ]
                    )
                else:
                    pixel_values = torch.zeros(0)
            except Exception as e:
                print(f"Image processing error: {e}")
                pixel_values = torch.zeros(0)
        else:
            pixel_values = torch.zeros(0)

        return pixel_values

    def _process_single(self, source, mimic=False):
        built_source = self._build_source(source, mimic=mimic)
        pixel_values = self._load_pixel_values(built_source)

        wrapped_sources = [built_source]
        source_convs = [s["conversations"] for s in wrapped_sources]

        data_dict = preprocess_otter(
            source_convs,
            self.tokenizer,
            has_image=("image" in built_source),
        )

        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
        )
        data_dict["image"] = pixel_values

        return data_dict

    def __getitem__(self, i):
        try:
            source = self.datalist[i]

            data_dict = self._process_single(source, mimic=False)

            if self.is_mimic:
                query_data_dict = self._process_single(source, mimic=True)
                return data_dict, query_data_dict

            return data_dict

        except Exception as e:
            print(f"Error in sample {i}: {e}")
            return self.__getitem__((i + 1) % len(self))


# class Otter_LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning of Otter."""

#     def __init__(self, datalist,
#                  tokenizer,
#                  data_args,
#                  preprocess=False):
#         super(Otter_LazySupervisedDataset, self).__init__()

#         self.tokenizer = tokenizer
#         self.datalist = datalist
#         self.data_args = data_args
#         self.preprocess = preprocess
#         self.image_processor = self.data_args.image_processor

#     @property
#     def lengths(self):
#         length_list = []
#         for sample in self.datalist:
#             length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']))
#         return length_list

#     @property
#     def modality_lengths(self):
#         length_list = []
#         for sample in self.datalist:
#             cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
#             cur_len = cur_len if 'image' in sample else -cur_len
#             length_list.append(cur_len)
#         return length_list

#     def __len__(self):
#         return len(self.datalist)

#     def __getitem__(self, i):
#         sources = self.datalist[i]
#         if isinstance(i, int):
#             sources = [sources]
            
#         if 'image' in sources[0]:
#             image_files = sources[0]['image']
#             if isinstance(image_files, str):
#                 image_files = [image_files]
            
#             images = []
#             for image_file in image_files:
#                 try:
#                     image = Image.open(image_file).convert('RGB')
#                     images.append(image)
#                 except Exception as e:
#                     print(f"Error reading image {image_file}: {e}")
            
#             try:
#                 if len(images) > 0:
#                     pixel_values = torch.stack([self.image_processor(img, return_tensors='pt')['pixel_values'][0] for img in images])
#                 else:
#                     pixel_values = torch.zeros(0)
#             except Exception as e:
#                 print(f"Image processing error: {e}")

#         else:
#             pixel_values = torch.zeros(0)

#         for source in sources:
#             for conv in source["conversations"]:
#                 val = conv['value'].strip()
#                 val = val.replace("\nAnswer:", " GPT: <answer>")
                
#                 if "<image>" in val:
#                     parts = [p.strip() for p in val.split("<image>")]
#                     new_val = parts[0]
#                     for idx in range(1, len(parts)):
#                         if idx == 1:
#                             new_val += f"<image>User:{parts[idx]}"
#                         else:
#                             new_val += f"<|endofchunk|><image>User:{parts[idx]}"
#                     val = new_val
                
#                 val = val.replace("\nGPT: <answer>", " GPT: <answer>")
#                 conv['value'] = val

#         source_convs = [s["conversations"] for s in sources]
        
#         data_dict = preprocess_otter(
#             source_convs,
#             self.tokenizer,
#             has_image=('image' in sources[0])
#         )

#         if isinstance(i, int):
#             data_dict = dict(input_ids=data_dict["input_ids"][0],
#                              labels=data_dict["labels"][0])

#         data_dict['image'] = pixel_values
        
#         return data_dict

@dataclass
class Otter_DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning of Otter."""

    tokenizer: transformers.PreTrainedTokenizer

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
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            
        batch = dict(
            lang_x=input_ids,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            
            if all(x is not None and x.shape[0] != 0 for x in images):
                max_t_img = max(img.shape[0] for img in images) 
                
                padded_images = []
                for img in images:
                    pad_size = max_t_img - img.shape[0]
                    if pad_size > 0:
                        c, h, w = img.shape[1:]
                        padding = torch.zeros(pad_size, c, h, w, dtype=img.dtype)
                        padded_img = torch.cat([img, padding], dim=0)
                    else:
                        padded_img = img
                    padded_images.append(padded_img)
                
                batch['vision_x'] = torch.stack(padded_images).to(torch.bfloat16)
            else:
                batch['vision_x'] = [x.to(dtype=torch.bfloat16) for x in images]
            
            batch['vision_x'] = batch['vision_x'].unsqueeze(2)
        
        return batch    

def preprocess_otter(sources, tokenizer, has_image=False) -> Dict:
    conversations = []
    for source in sources:
        full_text = "".join([s["value"] for s in source])
        if not full_text.endswith("<|endofchunk|>"):
            full_text += "<|endofchunk|>"
        conversations.append(full_text)

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=min(tokenizer.model_max_length, 32767),
        truncation=True,
    ).input_ids

    targets = input_ids.clone()
    target_sep = "GPT: <answer>"

    for i, (conversation, target) in enumerate(zip(conversations, targets)):
        rou = conversation
        cur_len = 1 
        target[:cur_len] = IGNORE_INDEX
        
        parts = rou.split(target_sep)
        if len(parts) == 2:
            instruction_part = parts[0] + target_sep
            instruction_len = len(tokenizer(instruction_part, add_special_tokens=False).input_ids)
            
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            
            round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
            cur_len += round_len
            
        target[cur_len:] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)
    
def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # if DEFAULT_IMAGE_TOKEN in sentence['value']:
            #     sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            #     sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
            #     sentence['value'] = sentence['value'].strip()
            if "mmtag" in conversation_lib_llava.default_conversation.version:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


# llava preprocess
def preprocess_text_llava(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib_llava.default_conversation.sep_style == conversation_lib_llava.SeparatorStyle.PLAIN:
        return preprocess_plain_llava(sources, tokenizer)
    if conversation_lib_llava.default_conversation.sep_style == conversation_lib_llava.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib_llava.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib_llava.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib_llava.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib_llava.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib_llava.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib_llava.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib_llava.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib_llava.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib_llava.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain_llava(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib_llava.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib_llava.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib_llava.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len
