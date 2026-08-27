"""OpenRouter model backend — uses the OpenAI-compatible chat completions API."""

import base64
import os
import time
from mimetypes import guess_type
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from openai import OpenAI

from models.base import BaseModel


class OpenRouterModel(BaseModel):
    def __init__(self, model_name: str, max_try: int = 10):
        super().__init__(model_name)
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. "
                "Add it to .env or export it as an environment variable."
            )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.max_try = max_try

    def _local_image_to_data_url(self, image_path: str) -> str:
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def _build_messages(
        self, text: str, image: Union[str, List[str], None]
    ) -> list:
        content: list = [{"type": "text", "text": text}]
        if image is not None:
            image_list = image if isinstance(image, list) else [image]
            for img_path in image_list:
                if img_path is None:
                    continue
                data_url = self._local_image_to_data_url(img_path)
                content.append(
                    {"type": "image_url", "image_url": {"url": data_url}}
                )
        return [{"role": "user", "content": content}]

    def _call_client(self, messages: list, temperature: float = 0.0) -> str:
        # Use OpenRouter's context-compression plugin to handle prompts
        # that exceed the model's context window (truncates from the middle).
        if sum(x["type"] == "image_url" for x in messages[-1]["content"]) > 1:
            messages[-1]["content"] = [
                x for x in messages[-1]["content"] if x["type"] != "image_url"
            ]
            
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            # extra_body={"plugins": [{"id": "context-compression"}]},
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    def _call_with_retry(self, messages: list, temperature: float = 0.0) -> str:
        for attempt in range(self.max_try):
            try:
                return self._call_client(messages, temperature=temperature)
            except Exception as e:
                print(f"[OpenRouterModel] attempt {attempt + 1}/{self.max_try}: {e}")
                time.sleep(4 * (attempt + 1))
        print(f"[OpenRouterModel] Failed after {self.max_try} attempts, returning empty string")
        return ""

    def generate(
        self,
        text: str,
        image: Union[str, List[str], None],
        temperature: float = 0.0,
    ) -> str:
        messages = self._build_messages(text, image)
        return self._call_with_retry(messages, temperature=temperature)

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
