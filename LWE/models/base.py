from typing import Any, Dict, List


class BaseModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(
        self,
        text: str,
        image: str | List[str] | None,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError

    def generate_batch(
        self, batch: List[Dict[str, Any]], temperature: float = 0.0
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError
