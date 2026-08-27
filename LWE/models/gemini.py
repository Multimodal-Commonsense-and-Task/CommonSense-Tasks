"""Gemini model backend using google-genai."""

import os
import time
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Union

from dotenv import load_dotenv

from models.base import BaseModel


class GeminiModel(BaseModel):
    def __init__(self, model_name: str, max_try: int = 10):
        super().__init__(model_name)
        load_dotenv()
        self.max_try = max_try
        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY not set. Export it or add it to .env")

    def _load_image(self, image_path: str):
        """Open a local image as a PIL Image."""
        from PIL import Image
        return Image.open(image_path)

    def call_client(self, text: str, image: Union[str, List[str], None], temperature: float) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)

        contents: list = [text] if text else []
        if image is not None:
            image_list = image if isinstance(image, list) else [image]
            for img_path in image_list:
                if img_path is None:
                    continue
                contents.append(self._load_image(img_path))

        response = client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                candidate_count=1,
                max_output_tokens=65536,
                temperature=temperature,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ],
            ),
        )

        return response.text

    def generate(
        self,
        text: str,
        image: Union[str, List[str], None],
        temperature: float = 0.0,
    ) -> str:
        for attempt in range(self.max_try):
            try:
                return self.call_client(text, image, temperature)
            except Exception as e:
                print(f"[GeminiModel] attempt {attempt + 1}/{self.max_try} failed: {e}")
                time.sleep(20 * (attempt + 1))
        raise RuntimeError(f"[GeminiModel] Failed after {self.max_try} attempts")

    def generate_batch(
        self, batch: List[Dict[str, Any]], temperature: float = 0.0
    ) -> List[Dict[str, Any]]:
        def _one(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "idx": x["idx"],
                "response": self.generate(x["Text"], x["Image"], temperature=temperature),
                "prompt": x["Text"],
            }

        with ThreadPool(len(batch)) as pool:
            return list(pool.imap(_one, batch))
