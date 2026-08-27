# Adapted from multimodal_rewardbench-style GPT judge calls.

import base64
import os
import time
from mimetypes import guess_type
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from openai import OpenAI

from models.base import BaseModel


# reference: https://github.com/facebookresearch/multimodal_rewardbench/blob/main/scripts/1_run_model_as_judge_gpt4o.py

class GPTModel(BaseModel):
    def __init__(self, model_name: str, max_try: int = 10):
        super().__init__(model_name)
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.max_try = max_try

    def local_image_to_data_url(self, image_path: str) -> str:
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
        with open(image_path, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def call_client(self, messages: list, temperature: float = 0.0) -> str:
        if "gpt-5" in self.model_name:
            response = self.client.responses.create(
                model=self.model_name,
                input=messages,
                reasoning={"effort": "minimal"},
                text={"verbosity": "medium"},
            )
        else:
            response = self.client.responses.create(
                model=self.model_name,
                input=messages,
                temperature=temperature,
            )
        return response.output_text

    def call_client_wrapper(self, messages: list, temperature: float = 0.0) -> str:
        count = 0
        while count < self.max_try:
            try:
                return self.call_client(messages, temperature=temperature)
            except Exception as e:
                print("Exception:", e)
                count += 1
                time.sleep(4 * count)
        raise RuntimeError(f"[GPTModel] Failed after {self.max_try} attempts")

    def get_messages(
        self, text: str, image: Union[str, List[str], None]
    ) -> list:
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
        ]
        if image is not None:
            image_list = image if isinstance(image, list) else [image]
            for image_path in image_list:
                if image_path is None:
                    continue
                image_url = self.local_image_to_data_url(image_path)
                messages[0]["content"].append(
                    {"type": "input_image", "image_url": image_url}
                )
        return messages

    def generate(
        self,
        text: str,
        image: Union[str, List[str], None],
        temperature: float = 0.0,
    ) -> str:
        messages = self.get_messages(text, image)
        return self.call_client_wrapper(messages, temperature=temperature)

    def generate_batch(
        self, batch: List[Dict[str, Any]], temperature: float = 0.0
    ) -> List[Dict[str, Any]]:
        def _one(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "idx": x["idx"],
                "response": self.generate(
                    x["Text"], x["Image"], temperature=temperature
                ),
                "prompt": x["Text"],
            }

        with ThreadPool(len(batch)) as pool:
            return list(pool.imap(_one, batch))
