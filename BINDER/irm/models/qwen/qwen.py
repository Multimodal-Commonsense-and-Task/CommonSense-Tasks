import time
import os
import numpy as np
import torch
import pickle
import threading
import cv2
from datetime import datetime
from interface import BaseReasoner
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from models.qwen.processing_utils import convert_numpy_to_video
from qwen_vl_utils import process_vision_info
from PIL import Image
from vllm.lora.request import LoRARequest

class QwenReasoner(BaseReasoner):
    def __init__(self, lora_path: str = None):
        self.lora_path = lora_path
        self.enable_lora = lora_path is not None
        self.lora_request = LoRARequest(
            "adapter",
            1,
            self.lora_path
        ) if self.enable_lora else None
        super().__init__()
    
    def load_model(self):
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        # model_name = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
        self.llm = LLM(
            model = model_name,
            limit_mm_per_prompt={"image": 10, "video": 10},
            enable_lora=self.enable_lora,
            # max_model_len=8192,
            dtype="float16",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=64,
            stop_token_ids=[],
        )

    def run_from_dict(self, result_dict: dict, save: bool = True) -> str:
        image_batch = result_dict["images"]; image_batch_for_visualization = image_batch.copy()
        fps = result_dict["fps"]
        image_idx = result_dict["image_idx"] if "image_idx" in result_dict else -1
        start_frame = result_dict["start_frame"] if "start_frame" in result_dict else None
        end_frame = result_dict["end_frame"] if "end_frame" in result_dict else None
        post_time_instruction_prompt = result_dict["post_time_instruction_prompt"]
        print(f"Received video of shape {image_batch.shape}", flush=True)
        print(f"Received post_time_instruction_prompt: {post_time_instruction_prompt}", flush=True)
        start_time = time.time()

        if image_batch.shape[0] == 1:
            print(f"Running from single image", flush=True)
            single_image = image_batch[0]
            if 'system_prompt' in result_dict:
                generated_text = self.run_from_single_image(single_image, None, post_time_instruction_prompt, result_dict['system_prompt'])
            else:
                generated_text = self.run_from_single_image(single_image, None, post_time_instruction_prompt)
            if save:
                self.save_async(image_batch, post_time_instruction_prompt, generated_text, save_pickle=True, result_dict=result_dict)
            print(f"Result: {generated_text}", flush=True)
            return generated_text
        
        if image_idx != -1: # single image
            print(f"Running from single image at index {image_idx}", flush=True)
            single_image = image_batch[image_idx]
            if 'system_prompt' in result_dict:
                generated_text = self.run_from_single_image(single_image, None, post_time_instruction_prompt, result_dict['system_prompt'])
            else:
                generated_text = self.run_from_single_image(single_image, None, post_time_instruction_prompt)
            if save:
                self.save_async(image_batch, post_time_instruction_prompt, generated_text, save_pickle=True, result_dict=result_dict)
            print(f"Result: {generated_text}", flush=True)
            return generated_text
    
        if start_frame is not None and end_frame is not None:
            print(f"Running from video slice from {start_frame} to {end_frame}", flush=True)
            image_batch = image_batch[start_frame:end_frame + 1]
        
        # Preprocess
        prompt, video_inputs, video_kwargs = self.preprocess(image_batch, fps, post_time_instruction_prompt)
        preprocess_time = time.time()
        print(f"Preprocess time: {preprocess_time - start_time:.2f} seconds", flush=True)

        mm_data = {}
        mm_data['video'] = video_inputs

        llm_inputs = {
            'prompt': prompt,
            'multi_modal_data': mm_data, # ['video'][0].shape: torch.Size([32, 3, 644, 476])
            'mm_processor_kwargs': video_kwargs, # {'fps': [8]}
        }

        if self.enable_lora:
            print(f"Using LoRA: {self.lora_path}", flush=True)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params, lora_request=self.lora_request if self.enable_lora else None)
        generated_text = outputs[0].outputs[0].text
        generate_time = time.time()
        print(f"Generate time: {generate_time - preprocess_time:.2f} seconds", flush=True)

        if save:
            self.save_async(image_batch_for_visualization, post_time_instruction_prompt, generated_text, save_pickle=True, result_dict=result_dict)
        
        print(f"Total time taken: {time.time() - start_time:.2f} seconds", flush=True)
        print(f"Result: {generated_text}", flush=True)
        print("="*50, flush=True)
        response = {"description": generated_text, "status": "success"}

        return response

    def run_from_mp4(self, video_path: str, prompt: str) -> str:
        video_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video", 
                        "video": video_path,
                        "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28
                    }
                ]
            },
        ]

        prompt = self.processor.apply_chat_template(video_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(video_messages, return_video_kwargs=True)

        mm_data = {}
        mm_data['video'] = video_inputs

        llm_inputs = {
            'prompt': prompt,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params, lora_request=self.lora_request if self.enable_lora else None)
        generated_text = outputs[0].outputs[0].text

        return generated_text
    
    def run_from_single_image(self, image: np.ndarray, image_path: str, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        if image is not None:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            tmp_image_path = f"/tmp/tmp_image.png"
            cv2.imwrite(tmp_image_path, image_bgr)
        else:
            tmp_image_path = image_path

        image_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{tmp_image_path}",
                        "min_pixels": 240 * 320,
                        "max_pixels": 1280 * 28 * 28,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(image_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(image_messages, return_video_kwargs=True)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        if self.enable_lora:
            print(f"Using LoRA: {self.lora_path}", flush=True)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params, lora_request=self.lora_request if self.enable_lora else None)
        generated_text = outputs[0].outputs[0].text

        return generated_text

    def run_from_pickle_single_image(self, pickle_path: str, prompt: str, image_idx: int = 0, save: bool = True, system_prompt: str = "You are a helpful assistant.") -> str:
        with open(pickle_path, "rb") as f:
            result_dict = pickle.load(f)

        image_batch = result_dict["images"]
        single_image = image_batch[image_idx]

        generated_text = self.run_from_single_image(single_image, None, prompt, system_prompt)

        return generated_text
    
    def run_from_pickle(self, pickle_path: str, prompt: str, save: bool = True, start_frame: int = None, end_frame: int = None) -> str:
        with open(pickle_path, "rb") as f:
            result_dict = pickle.load(f)

        image_batch = result_dict["images"] # np.ndarray, shape: (B, 240, 320, 3), uint8, 0 ~ 255
        if start_frame is not None and end_frame is not None:
            print(f"Slicing image_batch from {start_frame} to {end_frame}", flush=True)
            image_batch = image_batch[start_frame:end_frame]
        
        fps = result_dict["fps"]
        post_time_instruction_prompt = prompt
        
        start_time = time.time()
        prompt, video_inputs, video_kwargs = self.preprocess(image_batch, fps, post_time_instruction_prompt)
        preprocess_time = time.time()
        print(f"Preprocess time: {preprocess_time - start_time:.2f} seconds", flush=True)
        # import pdb; pdb.set_trace()

        mm_data = {}
        mm_data['video'] = video_inputs

        llm_inputs = {
            'prompt': prompt,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params, lora_request=self.lora_request if self.enable_lora else None)
        generated_text = outputs[0].outputs[0].text
        generate_time = time.time()
        print(f"Generate time: {generate_time - preprocess_time:.2f} seconds", flush=True)

        if save:
            self.save_async(image_batch, post_time_instruction_prompt, generated_text, save_pickle=False)

        total_time = time.time()
        print(f"Total time taken: {total_time - start_time:.2f} seconds", flush=True)

        return generated_text

    def preprocess(self, image_batch: list[np.ndarray], fps: int, post_time_instruction_prompt: str) -> list[np.ndarray]:
        video_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                    {"type": "text", "text": f"{post_time_instruction_prompt}"},
                    {
                        "type": "video", 
                        "video": "dummy",
                        "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28
                    }
                ]
            },
        ]
        prompt = self.processor.apply_chat_template(video_messages, tokenize=False, add_generation_prompt=True)

        video_inputs = []; video_sample_fps_list = []
        video_input, video_sample_fps = convert_numpy_to_video(image_batch, fps)
        video_inputs.append(video_input)
        video_sample_fps_list.append(video_sample_fps)

        return prompt, video_inputs, {'fps': video_sample_fps_list}

    def save(self, image_batch, post_time_instruction_prompt, generated_text, save_pickle=True, result_dict=None):
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        log_dir = os.environ.get("IRM_LOG_DIR", "irm_log")
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, f"result_{timestamp}")
        self.save_video_summary(image_batch, post_time_instruction_prompt, generated_text, save_video=True, filename=filename)
        if save_pickle:
            with open(f"{filename}.pkl", "wb") as f:
                pickle.dump(result_dict, f)

    def save_async(self, *args, **kwargs):
        thread = threading.Thread(target=self.save, args=args, kwargs=kwargs)
        thread.start()