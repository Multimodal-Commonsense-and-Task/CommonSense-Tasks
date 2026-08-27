import uvicorn, asyncio
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from fastapi import FastAPI, HTTPException
from concurrent.futures import ThreadPoolExecutor
from typing import Union, List


app = FastAPI(title="Multi-LLM API")


def load_text(name: str, device: str):
    return {
        "model": AutoModelForCausalLM.from_pretrained(
            name, device_map=device, trust_remote_code=True
        ),
        "tok": AutoTokenizer.from_pretrained(name),
    }

MODELS = {
    "0": load_text("mounKim/qwen3-tank-a", "auto"),
    "1": load_text("mounKim/qwen3-tank-b", "auto"),
    "2": load_text("mounKim/qwen3-tank-c", "auto"),
}


executor = ThreadPoolExecutor(max_workers=3)


class Query(BaseModel):
    prompt: Union[str, List]
    max_tokens: int = 512
    temperature: float = 0.7


async def _run_text(model_id: str, q: Query) -> str:
    cfg = MODELS[model_id]
    model = cfg["model"]
    tokenizer = cfg["tok"]
    messages = [
        {"role": "user", "content": q.prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    loop = asyncio.get_running_loop()
    generated_ids = await loop.run_in_executor(
        executor,
        lambda: model.generate(
            model_inputs.input_ids,
            max_new_tokens=q.max_tokens,
            do_sample=True,
            temperature=q.temperature,
        ),
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return response.replace("\n", "")


@app.post("/infer")
async def infer_all(q: Query):
    tasks = [_run_text(mid, q) for mid in ("0", "1", "2")]
    texts = await asyncio.gather(*tasks)
    response = (
        f"Suggestion A: '{texts[0]}',\n"
        f"Suggestion B: '{texts[1]}',\n"
        f"Suggestion C: '{texts[2]}',\n"
    )
    return {"text": response}


@app.post("/infer/{model_id}")
async def infer(model_id: str, q: Query):
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="unknown model_id")
    return {"model": model_id, "text": await _run_text(model_id, q)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8084, workers=1)
