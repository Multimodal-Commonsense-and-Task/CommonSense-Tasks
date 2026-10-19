import time
import os
import numpy as np
import torch
import pickle
import threading
from models.smolvlm.smolvlm_utils import SmolVLMCustomProcessor
from transformers import AutoModelForImageTextToText
from datetime import datetime
from interface import BaseReasoner

class SmolVLMRReasoner(BaseReasoner):
    def __init__(self):
        super().__init__()
    
    def load_model(self):
        model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        self.processor = SmolVLMCustomProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2"
        ).to("cuda")

    def run_from_dict(self, result_dict: dict, save: bool = True) -> str:
        image_batch = result_dict["images"]
        fps = result_dict["fps"]
        post_time_instruction_prompt = result_dict["post_time_instruction_prompt"]
        image_idx = result_dict["image_idx"] if "image_idx" in result_dict else -1
        print(f"Received video of shape {image_batch.shape}")
        print(f"Received post_time_instruction_prompt: {post_time_instruction_prompt}")
        start_time = time.time()

        if image_idx != -1: # single image
            print(f"Running from single image at index {image_idx}", flush=True)
            single_image = image_batch[image_idx]
            generated_text = self.run_from_single_image(single_image, None, post_time_instruction_prompt)
            if save:
                self.save_async(image_batch, post_time_instruction_prompt, generated_text, save_pickle=True, result_dict=result_dict)
            print(f"Result: {generated_text}", flush=True)
            return generated_text
        
        # Preprocess
        inputs = self.preprocess(image_batch, fps, post_time_instruction_prompt)

        # Generate
        generated_text = self.generate(inputs)

        # Postprocess
        generated_text = self.postprocess(generated_text)

        # Save video
        if save:
            self.save_async(image_batch, post_time_instruction_prompt, generated_text, save_pickle=True, result_dict=result_dict)
        
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        print(f"Result: {generated_text}")
        print("="*50)
        response = {"description": generated_text, "status": "success"}

        return response

    def run_from_single_image(self, image: np.ndarray, image_path: str, prompt: str) -> str:
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        

    def run_from_mp4(self, video_path: str, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            target_fps=8,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"Result: {generated_text}")

        return generated_text
    
    def run_from_pickle(self, pickle_path: str, prompt: str, save: bool = True) -> str:
        with open(pickle_path, "rb") as f:
            result_dict = pickle.load(f)

        image_batch = result_dict["images"]
        fps = result_dict["fps"]
        post_time_instruction_prompt = prompt
        
        inputs = self.preprocess(image_batch, fps, post_time_instruction_prompt)
        generated_text = self.generate(inputs)
        generated_text = self.postprocess(generated_text)

        if save:
            self.save(image_batch, post_time_instruction_prompt, generated_text, save_pickle=False)

        return generated_text

    def preprocess(self, image_batch: list[np.ndarray], fps: int, post_time_instruction_prompt: str) -> list[np.ndarray]:
        inputs = self.processor.apply_chat_template(
            frames = image_batch,
            video_fps = fps,
            prompt = post_time_instruction_prompt,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
        ).to(self.model.device, dtype=torch.bfloat16)
        return inputs
    
    def generate(self, inputs: torch.Tensor) -> str:
        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    
    def postprocess(self, generated_text: str) -> str:
        generated_text = generated_text[generated_text.find("Assistant: ") + len("Assistant: "):]
        return generated_text

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