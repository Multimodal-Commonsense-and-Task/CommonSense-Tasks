"""Claude (Anthropic) model backend."""

from __future__ import annotations

import base64
import io
import os
import time
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Union

from dotenv import load_dotenv

from models.base import BaseModel

_MAX_BYTES = 4 * 1024 * 1024  # 4 MB — Anthropic hard limit per image


class ClaudeModel(BaseModel):
    def __init__(self, model_name: str, max_try: int = 10, max_tokens: int = 62000):
        super().__init__(model_name)
        load_dotenv()
        self.max_try = max_try
        self.max_tokens = max_tokens

        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _image_to_base64(self, image_path: str) -> tuple[str, str]:
        """Return (base64_data, media_type). Compresses if needed."""
        from PIL import Image

        ext = image_path.lower()
        img_format = "JPEG" if ext.endswith((".jpg", ".jpeg")) else "PNG"
        media_type = "image/jpeg" if img_format == "JPEG" else "image/png"

        file_size = os.path.getsize(image_path)
        if file_size <= _MAX_BYTES:
            with open(image_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode("utf-8"), media_type

        # Compress to fit under 4 MB
        img = Image.open(image_path)
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")

        for quality in [90, 80, 70, 60, 50, 40, 30, 20, 10]:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= _MAX_BYTES:
                return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"

        raise RuntimeError(f"Cannot compress image under 4 MB: {image_path}")

    def _build_content(
        self, text: str, image: Union[str, List[str], None]
    ) -> list:
        content: list = []
        if text:
            content.append({"type": "text", "text": text})
        if image is not None:
            image_list = image if isinstance(image, list) else [image]
            for img_path in image_list:
                if img_path is None:
                    continue
                b64, mime = self._image_to_base64(img_path)
                content.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": b64},
                    }
                )
        return content

    def call_client(
        self, text: str, image: Union[str, List[str], None], temperature: float
    ) -> str:
        content = self._build_content(text, image)
        with self.client.messages.stream(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            return stream.get_final_text()

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
                print(f"[ClaudeModel] attempt {attempt + 1}/{self.max_try} failed: {e}")
                time.sleep(20 * (attempt + 1))
        raise RuntimeError(f"[ClaudeModel] Failed after {self.max_try} attempts")

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
