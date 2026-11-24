from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import re
import json
import os
import sqlite3
import threading
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from evaluate import load
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from trl.rewards import think_format_reward
from functools import lru_cache
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import time
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# =========================================================
# Regex / caches
# =========================================================
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

_REWARD_CACHE = {
    "nli_tokenizer": None,
    "nli_model": None,
    "bertscore": None,
    "vg_processor": None,
    "vg_tokenizer": None,
    "vg_model": None,
    "device": None,
    "decomposer": None,
}

_IMAGE_TENSOR_CACHE = {}
_IMAGE_FEAT_CACHE = {}
_IMAGESIM_BATCH_CACHE = {}
_ENTAIL_BATCH_CACHE = {}
_VGV3_ENTAIL_BATCH_CACHE = {}

def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        logging.warning(
            "Invalid %s=%r; using %d",
            name,
            raw_value,
            default,
        )
        return default
    if value < 1:
        logging.warning(
            "%s must be positive; got %d, using %d",
            name,
            value,
            default,
        )
        return default
    return value


_PROPOSITION_PARALLELISM = _positive_int_env("PROPOSITION_PARALLELISM", 16)
_OPENAI_MAX_WORKERS = _PROPOSITION_PARALLELISM
_HF_PROPOSITION_BATCH_SIZE = _PROPOSITION_PARALLELISM
_PROPOSITION_PROMPT_VERSION = "vgv2-propositions-v2"
_PROPOSITION_MODEL_ALIASES = {
    "llama3_2-1b": ("hf", "meta-llama/Llama-3.2-1B-Instruct"),
    "llama3_2-3b": ("hf", "meta-llama/Llama-3.2-3B-Instruct"),
    "llama3_1-8b": ("hf", "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen3-1_7b": ("hf", "Qwen/Qwen3-1.7B"),
    "qwen3-0_6b": ("hf", "Qwen/Qwen3-0.6B"),
    "qwen3-4b": ("hf", "Qwen/Qwen3-4B"),
    "qwen3-8b": ("hf", "Qwen/Qwen3-8B"),
    "flan-t5-base": ("hf", "google/flan-t5-base"),
    "flan-t5-large": ("hf", "google/flan-t5-large"),
    "flan-t5-xl": ("hf", "google/flan-t5-xl"),
    "gpt": ("openai", "gpt-4o-mini"),
    # Timing baseline: use the captions themselves as propositions without
    # loading a decomposer model or making an API request.
    "precomputed": ("passthrough", "caption"),
    "passthrough": ("passthrough", "caption"),
}
_PROPOSITION_CONFIG = {
    "model": "llama3_2-1b",
    "cache_path": ".cache/vgv2/ground_truth_propositions.sqlite3",
    "device": "auto",
}


# =========================================================
# Basic text helpers
# =========================================================

PROPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "shared_proposition": {
            "type": "string",
            "description": "Single most informative proposition shared by both captions."
        },
        "caption_1_independent_propositions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "caption_2_independent_propositions": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "shared_proposition",
        "caption_1_independent_propositions",
        "caption_2_independent_propositions"
    ],
    "additionalProperties": False
}

def build_prompt(caption_1: str, caption_2: str) -> str:
    return f"""
        You are an expert semantic proposition extractor.

        Your task:

        1. Extract the SINGLE most informative shared proposition between the two captions.
        2. Extract the SINGLE best proposition unique to caption 1.
        3. Extract the SINGLE best proposition unique to caption 2.

        Definition of "most informative shared proposition":
        - Prefer the longest shared semantic relation.
        - Prefer predicate + argument structure.
        - Avoid trivial overlap like:
        "a person exists"
        "there is an object"
        - Example:

        Caption 1:
        "A person sits across from a sandwich"

        Caption 2:
        "A person sits across from a pizza"

        Correct shared proposition:
        "a person sits across from an object"

        NOT:
        "a person exists"

        Rules:
        - Keep propositions atomic
        - Do not hallucinate
        - Normalize wording if meaning is equivalent
        - Object identity differences are NOT shared
        - Return exactly one shared proposition

        Caption 1:
        {caption_1}

        Caption 2:
        {caption_2}

        Return only this JSON object:
        {{
          "shared_proposition": "one full sentence",
          "caption_1_independent_propositions": ["one full sentence"],
          "caption_2_independent_propositions": ["one full sentence"]
        }}
        """.strip()


def _resolve_proposition_model(model: str):
    model = (model or "llama3_2-1b").strip()
    alias = _PROPOSITION_MODEL_ALIASES.get(model.lower())
    if alias is not None:
        return alias
    if ":" in model:
        backend, model_name = model.split(":", 1)
        if backend.lower() in {"hf", "openai"} and model_name:
            return backend.lower(), model_name
    if model.lower().startswith(("gpt-", "o1", "o3", "o4")):
        return "openai", model
    return "hf", model


def configure_proposition_decomposer(
    model: str = "llama3_2-1b",
    cache_path: str = ".cache/vgv2/ground_truth_propositions.sqlite3",
    device: str = "auto",
):
    """Configure the vgv2 proposition model before reward functions are created."""
    new_config = {
        "model": model,
        "cache_path": cache_path,
        "device": device,
    }
    if new_config != _PROPOSITION_CONFIG:
        _PROPOSITION_CONFIG.update(new_config)
        _REWARD_CACHE["decomposer"] = None


def _normalize_proposition_output(value: Dict) -> Dict:
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object, got {type(value).__name__}")

    shared = value.get("shared_proposition", value.get("shared", ""))
    positive = value.get(
        "caption_1_independent_propositions", value.get("positive", [])
    )
    negative = value.get(
        "caption_2_independent_propositions", value.get("negative", [])
    )

    def as_string_list(item):
        if item is None:
            return []
        if not isinstance(item, (list, tuple)):
            item = [item]
        return [str(part).strip() for part in item if str(part).strip()]

    shared = str(shared or "").strip()
    if not shared:
        raise ValueError("The proposition result has an empty shared proposition")
    return {
        "shared_proposition": shared,
        "caption_1_independent_propositions": as_string_list(positive),
        "caption_2_independent_propositions": as_string_list(negative),
    }


def _parse_json_object(text: str) -> Dict:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        if start < 0:
            raise ValueError(f"Model did not return JSON: {text[:200]!r}")
        depth = 0
        in_string = False
        escaped = False
        end = None
        for index, char in enumerate(text[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            raise ValueError(f"Model returned incomplete JSON: {text[:200]!r}")
        text = text[start:end]
    return _normalize_proposition_output(json.loads(text))


class GroundTruthPropositionCache:
    """Process-safe persistent cache for invariant ground-truth decompositions."""

    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._thread_lock = threading.RLock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=60)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS propositions (
                    cache_key TEXT PRIMARY KEY,
                    backend TEXT NOT NULL,
                    model TEXT NOT NULL,
                    caption_1 TEXT NOT NULL,
                    caption_2 TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def exclusive(self):
        with self._thread_lock:
            with self.lock_path.open("a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def get(self, cache_key: str):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM propositions WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        return None if row is None else _normalize_proposition_output(json.loads(row[0]))

    def put(
        self,
        cache_key: str,
        backend: str,
        model: str,
        caption_1: str,
        caption_2: str,
        result: Dict,
    ):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO propositions
                    (cache_key, backend, model, caption_1, caption_2, result_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    backend,
                    model,
                    caption_1,
                    caption_2,
                    json.dumps(result, ensure_ascii=False, sort_keys=True),
                ),
            )


class PropositionDecomposer:
    def __init__(self, model: str, cache_path: str, device: str = "auto"):
        self.backend, self.model_name = _resolve_proposition_model(model)
        self.device_name = device
        self.cache = (
            None
            if self.backend == "passthrough"
            else GroundTruthPropositionCache(cache_path)
        )
        self._memory_cache = {}
        self._client = None
        self._tokenizer = None
        self._model = None
        self._is_encoder_decoder = False

    def _cache_key(self, caption_1: str, caption_2: str) -> str:
        payload = json.dumps(
            [
                _PROPOSITION_PROMPT_VERSION,
                self.backend,
                self.model_name,
                caption_1,
                caption_2,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_hf_model(self):
        if self._model is not None:
            return
        if self.device_name == "auto":
            device = torch.device(
                "cuda", torch.cuda.current_device()
            ) if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device(self.device_name)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        # InternLM2.5 supplies its Transformers implementation in the model
        # repository and therefore requires remote model code when loading.
        trust_remote_code = (
            self.model_name.casefold() == "internlm/internlm2_5-1_8b"
        )
        config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        self._is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Decoder-only models need left padding for batched generation; encoder-decoder
        # models (T5/FLAN-T5) use the encoder attention mask, so right padding is correct.
        self._tokenizer.padding_side = "right" if self._is_encoder_decoder else "left"
        model_cls = (
            AutoModelForSeq2SeqLM if self._is_encoder_decoder else AutoModelForCausalLM
        )
        self._model = model_cls.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(device).eval()
        self._device = device

    def _generate_openai(self, prompts: List[str]):
        if not prompts:
            return []

        if self._client is None:
            if OpenAI is None:
                raise RuntimeError(
                    "Install the 'openai' package when --proposition_model uses GPT"
                )
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY must be set when --proposition_model uses GPT"
                )
            self._client = OpenAI()

        def generate_one(prompt):
            response = self._client.responses.create(
                model=self.model_name,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "proposition_output",
                        "schema": PROPOSITION_SCHEMA,
                        "strict": True,
                    }
                },
            )
            return _normalize_proposition_output(json.loads(response.output_text))

        # [p40] Retry transient failures. The first call in a process can race inside the openai SDK
        # (lazy pydantic model rebuild -> AttributeError: __pydantic_core_schema__) when 16 threads
        # start at once, so the first prompt is sent alone before the parallel batch.
        def generate_one_retry(prompt, attempts=5):
            for attempt in range(attempts):
                try:
                    return generate_one(prompt)
                except Exception:
                    if attempt == attempts - 1:
                        raise
                    time.sleep(2 ** attempt)

        results = []
        if not getattr(self, "_p40_openai_warm", False):
            results.append(generate_one_retry(prompts[0]))
            self._p40_openai_warm = True
            prompts = prompts[1:]
        # Network-bound calls can run concurrently. executor.map preserves prompt
        # order, which keeps each decomposition aligned with its caption pair.
        if prompts:
            max_workers = min(_OPENAI_MAX_WORKERS, len(prompts))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results.extend(executor.map(generate_one_retry, prompts))
        return results

    def _render_chat(self, message):
        # No chat template (base models): fall back to a plain concatenated prompt.
        if getattr(self._tokenizer, "chat_template", None) is None:
            text = "\n\n".join(turn["content"] for turn in message)
            return text + "\n\nJSON:\n"
        # Qwen3 and other "thinking" models burn the token budget on <think>...</think>
        # before emitting JSON; disable it when the tokenizer supports the kwarg.
        try:
            return self._tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True,
            )

    @staticmethod
    def _fallback_proposition(caption_1: str, caption_2: str) -> Dict:
        # Small decomposers occasionally emit prose or malformed JSON. A single such
        # sample must not crash a multi-thousand-step run, so degrade gracefully by
        # treating each caption as its own proposition (lower-quality but valid).
        c1 = _normalize_text(caption_1)
        c2 = _normalize_text(caption_2)
        return {
            "shared_proposition": c1 or c2,
            "caption_1_independent_propositions": [c1] if c1 else [],
            "caption_2_independent_propositions": [c2] if c2 else [],
        }

    def _generate_hf(self, pairs):
        # Local model inference is parallelized as a tensor batch on the GPU.
        # Chunking bounds KV-cache memory while preserving input/output order.
        if len(pairs) > _HF_PROPOSITION_BATCH_SIZE:
            results = []
            for start in range(0, len(pairs), _HF_PROPOSITION_BATCH_SIZE):
                results.extend(
                    self._generate_hf(
                        pairs[start : start + _HF_PROPOSITION_BATCH_SIZE]
                    )
                )
            return results

        self._load_hf_model()
        prompts = [build_prompt(caption_1, caption_2) for caption_1, caption_2 in pairs]
        messages = [
            [
                {
                    "role": "system",
                    "content": (
                        "You extract semantic propositions. Follow the requested "
                        "schema exactly and return valid JSON only. Output nothing "
                        "before or after the JSON object: no prose, no markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        rendered = [self._render_chat(message) for message in messages]
        inputs = self._tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        ).to(self._device)
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        # Decoder-only models echo the prompt, so strip it; encoder-decoder models
        # (T5/FLAN-T5) return only freshly generated decoder tokens.
        if self._is_encoder_decoder:
            generated_ids = output_ids
        else:
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_texts = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        results = []
        for (caption_1, caption_2), text in zip(pairs, output_texts):
            try:
                results.append(_parse_json_object(text))
            except (ValueError, json.JSONDecodeError) as exc:
                logging.warning(
                    "Proposition decomposition failed for %s (%s); using caption "
                    "fallback. Raw output: %r",
                    self.model_name, exc, (text or "")[:200],
                )
                results.append(self._fallback_proposition(caption_1, caption_2))
        return results

    def _generate(self, pairs):
        if self.backend == "passthrough":
            return [
                self._fallback_proposition(caption_1, caption_2)
                for caption_1, caption_2 in pairs
            ]
        if self.backend == "openai":
            prompts = [build_prompt(c1, c2) for c1, c2 in pairs]
            return self._generate_openai(prompts)
        return self._generate_hf(pairs)

    def extract_many(self, pairs, persistent: bool = False):
        normalized_pairs = [
            (_normalize_text(caption_1), _normalize_text(caption_2))
            for caption_1, caption_2 in pairs
        ]
        if self.backend == "passthrough":
            return [
                self._fallback_proposition(caption_1, caption_2)
                for caption_1, caption_2 in normalized_pairs
            ]

        results = [None] * len(normalized_pairs)
        missing_pairs = []
        missing_keys = []
        missing_indices = {}
        for index, pair in enumerate(normalized_pairs):
            key = self._cache_key(*pair)
            if key in self._memory_cache:
                results[index] = self._memory_cache[key]
            else:
                if key not in missing_indices:
                    missing_indices[key] = []
                    missing_keys.append(key)
                    missing_pairs.append(pair)
                missing_indices[key].append(index)

        def assign_result(key, result):
            self._memory_cache[key] = result
            for index in missing_indices[key]:
                results[index] = result

        if persistent:
            pending_keys = []
            pending_pairs = []
            for key, pair in zip(missing_keys, missing_pairs):
                cached = self.cache.get(key)
                if cached is None:
                    pending_keys.append(key)
                    pending_pairs.append(pair)
                else:
                    assign_result(key, cached)

            if pending_pairs:
                # Recheck under the cross-process lock, then generate every
                # still-missing pair together. GPT uses concurrent requests;
                # local Hugging Face models use batched GPU inference.
                with self.cache.exclusive():
                    generate_keys = []
                    generate_pairs = []
                    for key, pair in zip(pending_keys, pending_pairs):
                        cached = self.cache.get(key)
                        if cached is None:
                            generate_keys.append(key)
                            generate_pairs.append(pair)
                        else:
                            assign_result(key, cached)

                    if generate_pairs:
                        generated = self._generate(generate_pairs)
                        for key, pair, result in zip(
                            generate_keys, generate_pairs, generated
                        ):
                            self.cache.put(
                                key,
                                self.backend,
                                self.model_name,
                                pair[0],
                                pair[1],
                                result,
                            )
                            assign_result(key, result)
        elif missing_pairs:
            generated = self._generate(missing_pairs)
            for key, result in zip(missing_keys, generated):
                self._memory_cache[key] = result
                for index in missing_indices[key]:
                    results[index] = result
        return results

    def extract(self, caption_1: str, caption_2: str, persistent: bool = False):
        pair = (_normalize_text(caption_1), _normalize_text(caption_2))
        if self.backend == "passthrough":
            return self._fallback_proposition(*pair)

        key = self._cache_key(*pair)
        if key in self._memory_cache:
            return self._memory_cache[key]
        if not persistent:
            return self.extract_many([pair], persistent=False)[0]

        cached = self.cache.get(key)
        if cached is not None:
            self._memory_cache[key] = cached
            return cached

        # Hold the cross-process lock across the second lookup and generation so
        # distributed workers cannot decompose the same ground truth repeatedly.
        with self.cache.exclusive():
            cached = self.cache.get(key)
            if cached is None:
                cached = self._generate([pair])[0]
                self.cache.put(
                    key,
                    self.backend,
                    self.model_name,
                    pair[0],
                    pair[1],
                    cached,
                )
        self._memory_cache[key] = cached
        return cached


def extract_props(
    caption_1: str,
    caption_2: str,
    persistent: bool = False,
) -> Dict:
    decomposer = get_reward_modules("decomposer")
    return decomposer.extract(caption_1, caption_2, persistent=persistent)


def extract_final_answer(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()

def _batched_entailment_scores_multi_dedup(
    tokenizer,
    model,
    pair_groups,
    device,
    max_length=256,
    batch_size=128,
):
    names = list(pair_groups.keys())

    all_pairs = []
    group_sizes = {}
    for name in names:
        premises, hypotheses = pair_groups[name]
        pairs = [(_normalize_text(p), _normalize_text(h)) for p, h in zip(premises, hypotheses)]
        group_sizes[name] = len(pairs)
        all_pairs.extend((name, p, h) for p, h in pairs)

    unique_pairs = []
    pair_to_idx = {}
    owner_pair_indices = {name: [] for name in names}

    for name, p, h in all_pairs:
        key = (p, h)
        if key not in pair_to_idx:
            pair_to_idx[key] = len(unique_pairs)
            unique_pairs.append(key)
        owner_pair_indices[name].append(pair_to_idx[key])

    flat_premises = [p for p, _ in unique_pairs]
    flat_hypotheses = [h for _, h in unique_pairs]

    all_scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flat_premises), batch_size):
            end = start + batch_size
            inputs = tokenizer(
                flat_premises[start:end],
                flat_hypotheses[start:end],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            all_scores.extend(probs[:, 2].float().cpu().tolist())

    all_scores = torch.tensor(all_scores, dtype=torch.float32)
    out = {}
    for name in names:
        idxs = torch.tensor(owner_pair_indices[name], dtype=torch.long)
        out[name] = all_scores[idxs]
    return out

def _normalize_text(value) -> str:
    """Normalize possibly nested/non-string values into a clean string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_normalize_text(v) for v in value]
        parts = [p for p in parts if p]
        return " ".join(parts).strip()
    return str(value).strip()


def _normalize_text_batch(values):
    return [_normalize_text(v) for v in values]


def extract_two_captions(caption: str):
    """
    Expected output format:
      caption1 || caption2
    Fallback:
      split by '. ' and take first / last
    """
    caption = _normalize_text(caption)
    captions = caption.split("||")
    if len(captions) == 1:
        captions = caption.split(". ")

    caption1 = captions[0].strip()
    caption2 = captions[-1].strip()
    return caption1, caption2


def _rouge_tokens(text: str) -> List[str]:
    """Tokenize text for dependency-free, case-insensitive ROUGE-L."""
    return re.findall(r"\w+", _normalize_text(text).lower(), flags=re.UNICODE)


def _lcs_length(left: List[str], right: List[str]) -> int:
    """Return the longest-common-subsequence length using O(len(right)) memory."""
    if len(left) < len(right):
        left, right = right, left

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_score(prediction: str, reference: str, beta: float = 1.2) -> float:
    """Compute the project-standard ROUGE-L F-score for one caption pair."""
    prediction_tokens = _rouge_tokens(prediction)
    reference_tokens = _rouge_tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0

    lcs = _lcs_length(prediction_tokens, reference_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    if precision == 0.0 or recall == 0.0:
        return 0.0

    beta_squared = beta * beta
    return (
        (1.0 + beta_squared)
        * precision
        * recall
        / (recall + beta_squared * precision)
    )


def rouge_reward(completions, golds, **kwargs):
    """Average aligned ROUGE-L for the generated positive/negative captions."""
    negative_golds = kwargs.get("gt_neg")
    if negative_golds is None:
        raise ValueError("The rouge reward requires a 'gt_neg' reference caption")
    if not (
        len(completions) == len(golds) == len(negative_golds)
    ):
        raise ValueError(
            "The rouge reward received mismatched completion/reference batch sizes"
        )

    rewards = []
    for completion, positive_gold, negative_gold in zip(
        completions, golds, negative_golds
    ):
        content = extract_final_answer(completion[0]["content"])
        positive_prediction, negative_prediction = extract_two_captions(content)
        positive_score = rouge_l_score(positive_prediction, positive_gold)
        negative_score = rouge_l_score(negative_prediction, negative_gold)
        rewards.append((positive_score + negative_score) / 2.0)
    return rewards


SYSTEM_PROMPT = """You decompose a positive/negative sentence pair into three full-sentence propositions.

Rules:
1. Output exactly three fields:
   - shared
   - positive
   - negative

2. Each field must be a single full sentence.

3. shared:
   - must contain only the semantic content common to both input sentences
   - must not include content that differs between the two sentences

4. positive:
   - must contain only the content specific to the positive sentence
   - must be a full sentence
   - must not repeat the shared content unless needed for grammaticality

5. negative:
   - must contain only the content specific to the negative sentence
   - must be a full sentence
   - must not repeat the shared content unless needed for grammaticality

6. Preserve meaning faithfully.
7. Do not add extra facts.
8. Prefer simple, literal wording.
9. Return valid JSON only.

Example 1:
Positive: A cat chases a mouse.
Negative: A dog chases a mouse.
Output:
{
  "shared": "An animal chases a mouse.",
  "positive": "The animal is a cat.",
  "negative": "The animal is a dog."
}

Example 2:
Positive: An elephant standing under a leaf filled tree.
Negative: A giraffe standing under a leaf filled tree.
Output:
{
  "shared": "An animal is standing under a leaf filled tree.",
  "positive": "The animal is an elephant.",
  "negative": "The animal is a giraffe."
}

Example 3:
Positive: An old person kisses a young person.
Negative: A young person kisses an old person.
Output:
{
  "shared": "An old person and a young person are involved in a kissing event.",
  "positive": "The old person kisses the young person.",
  "negative": "The young person kisses the old person."
}
"""


# =========================================================
# Model loading
# =========================================================
def get_reward_modules(reward_type):
    # [p40] REWARD_DEVICE=cpu keeps the NLI/CLIP reward models off the GPU
    device = torch.device(os.environ.get("REWARD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    _REWARD_CACHE["device"] = device

    if reward_type == "moacha":
        if _REWARD_CACHE["nli_tokenizer"] is None:
            nli_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
            nli_model = AutoModelForSequenceClassification.from_pretrained(
                "facebook/bart-large-mnli"
            ).to(device)
            nli_model.eval()

            nli_model.config.pad_token_id = nli_tokenizer.pad_token_id
            nli_model.model.shared.padding_idx = nli_tokenizer.pad_token_id
            nli_model.model.encoder.embed_tokens.padding_idx = nli_tokenizer.pad_token_id

            _REWARD_CACHE["nli_tokenizer"] = nli_tokenizer
            _REWARD_CACHE["nli_model"] = nli_model

        if _REWARD_CACHE["bertscore"] is None:
            _REWARD_CACHE["bertscore"] = load("bertscore")

        return (
            _REWARD_CACHE["nli_tokenizer"],
            _REWARD_CACHE["nli_model"],
            _REWARD_CACHE["bertscore"],
            _REWARD_CACHE["device"],
        )

    elif reward_type == "vg":
        if _REWARD_CACHE["vg_model"] is None:
            vg_model, _, vg_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="laion2b_s34b_b79k",
            )
            vg_model = vg_model.to(device).eval()
            vg_tokenizer = open_clip.get_tokenizer("ViT-B-32")

            _REWARD_CACHE["vg_model"] = vg_model
            _REWARD_CACHE["vg_processor"] = vg_preprocess
            _REWARD_CACHE["vg_tokenizer"] = vg_tokenizer

        return (
            _REWARD_CACHE["vg_processor"],
            _REWARD_CACHE["vg_tokenizer"],
            _REWARD_CACHE["vg_model"],
            _REWARD_CACHE["device"],
        )

    elif reward_type == "nli":
        if _REWARD_CACHE["nli_model"] is None:
            nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
            nli_model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-large-mnli"
            ).to(device).eval()

            _REWARD_CACHE["nli_tokenizer"] = nli_tokenizer
            _REWARD_CACHE["nli_model"] = nli_model

        return (
            _REWARD_CACHE["nli_tokenizer"],
            _REWARD_CACHE["nli_model"],
            _REWARD_CACHE["device"],
        )
    elif reward_type == "decomposer":
        if _REWARD_CACHE["decomposer"] is None:
            decomposer = PropositionDecomposer(**_PROPOSITION_CONFIG)
            _REWARD_CACHE["decomposer"] = decomposer

        return _REWARD_CACHE["decomposer"]

    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")


# =========================================================
# Cached image loading / encoding
# =========================================================
def load_image_tensor_cached(path, processor):
    if path not in _IMAGE_TENSOR_CACHE:
        img = Image.open(path).convert("RGB")
        _IMAGE_TENSOR_CACHE[path] = processor(img)
    return _IMAGE_TENSOR_CACHE[path]


def get_image_features_cached(image_paths, processor, model, device, batch_size=64):
    feats = [None] * len(image_paths)

    missing_paths = []
    missing_indices = []

    for i, p in enumerate(image_paths):
        if p in _IMAGE_FEAT_CACHE:
            feats[i] = _IMAGE_FEAT_CACHE[p]
        else:
            missing_paths.append(p)
            missing_indices.append(i)

    if missing_paths:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(missing_paths), batch_size):
                batch_paths = missing_paths[start:start + batch_size]
                batch_indices = missing_indices[start:start + batch_size]

                images = torch.stack(
                    [load_image_tensor_cached(p, processor) for p in batch_paths]
                ).to(device, non_blocking=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    batch_feats = model.encode_image(images)
                    batch_feats = F.normalize(batch_feats, dim=-1)

                batch_feats = batch_feats.detach().cpu()

                for idx, path, feat in zip(batch_indices, batch_paths, batch_feats):
                    _IMAGE_FEAT_CACHE[path] = feat
                    feats[idx] = feat

    return torch.stack(feats).to(device, non_blocking=True)


# =========================================================
# Batched entailment helpers
# =========================================================
def _batched_entailment_scores_multi(
    tokenizer,
    model,
    pair_groups,
    device,
    max_length=256,
    batch_size=128,
):
    """
    pair_groups: dict[name] = (premises, hypotheses)
    returns: dict[name] = tensor of entailment probs
    """
    names = list(pair_groups.keys())

    flat_premises = []
    flat_hypotheses = []
    owners = []

    for name in names:
        premises, hypotheses = pair_groups[name]
        assert len(premises) == len(hypotheses)
        for p, h in zip(premises, hypotheses):
            flat_premises.append(_normalize_text(p))
            flat_hypotheses.append(_normalize_text(h))
            owners.append(name)

    all_scores = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flat_premises), batch_size):
            end = start + batch_size

            inputs = tokenizer(
                flat_premises[start:end],
                flat_hypotheses[start:end],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(**inputs).logits

            probs = F.softmax(logits, dim=-1)
            # roberta-large-mnli: label 2 = entailment
            all_scores.extend(probs[:, 2].float().cpu().tolist())

    out = {name: [] for name in names}
    for owner, score in zip(owners, all_scores):
        out[owner].append(score)

    return {name: torch.tensor(vals) for name, vals in out.items()}


# =========================================================
# Merged proposition entailment reward
# =========================================================
def _batch_item(values, index):
    if values is None or index >= len(values):
        return None
    return values[index]


def _ground_truth_propositions(gts, gt_negs, kwargs):
    """Use dataset propositions when present; persistently cache missing ones."""
    provided_pos = kwargs.get("pos_props")
    provided_neg = kwargs.get("neg_props")
    provided_shared = kwargs.get("shared_proposition")
    decomposer = None
    pos_props, neg_props, shared_props = [], [], []
    missing_indices, missing_pairs = [], []

    for index, (pos_gt, neg_gt) in enumerate(zip(gts, gt_negs)):
        pos_prop = _normalize_text(_batch_item(provided_pos, index))
        neg_prop = _normalize_text(_batch_item(provided_neg, index))
        shared_prop = _normalize_text(_batch_item(provided_shared, index))

        if not (pos_prop and neg_prop and shared_prop):
            missing_indices.append(index)
            missing_pairs.append((pos_gt, neg_gt))

        pos_props.append(pos_prop)
        neg_props.append(neg_prop)
        shared_props.append(shared_prop)

    if missing_pairs:
        decomposer = get_reward_modules("decomposer")
        decomposed = decomposer.extract_many(missing_pairs, persistent=True)
        for index, result in zip(missing_indices, decomposed):
            independent_pos = result["caption_1_independent_propositions"]
            independent_neg = result["caption_2_independent_propositions"]
            pos_props[index] = (
                pos_props[index]
                or _normalize_text(independent_pos)
                or _normalize_text(gts[index])
            )
            neg_props[index] = (
                neg_props[index]
                or _normalize_text(independent_neg)
                or _normalize_text(gt_negs[index])
            )
            shared_props[index] = (
                shared_props[index] or result["shared_proposition"]
            )

    return pos_props, neg_props, shared_props


def prop_entail_reward_all(completions, gts, **kwargs):
    tokenizer, model, device = get_reward_modules("nli")

    contents = [_normalize_text(c[0]["content"]) for c in completions]
    preds = [extract_two_captions(x) for x in contents]
    pos_preds = [_normalize_text(x[0]) for x in preds]
    neg_preds = [_normalize_text(x[1]) for x in preds]

    neg_gts = kwargs["gt_neg"]
    pos_props, neg_props, shared_props = _ground_truth_propositions(
        gts, neg_gts, kwargs
    )
    
    scores = _batched_entailment_scores_multi_dedup(
        tokenizer=tokenizer,
        model=model,
        pair_groups={
            "pp": (pos_props, pos_preds),
            "nn": (neg_props, neg_preds),
            "pn": (pos_props, neg_preds),
            "np": (neg_props, pos_preds),
            "ps": (shared_props, pos_preds),
            "ns": (shared_props, neg_preds),
        },
        device=device,
        max_length=192,
        batch_size=256,
    )

    main_reward = torch.sigmoid(
        5.0 * (scores["pp"] - scores["np"] + scores["nn"] - scores["pn"])
    )
    shared_reward = scores["ps"] + scores["ns"]

    return main_reward.tolist(), shared_reward.tolist()


def prop_entail_reward_wrapped_all(completions, golds, **kwargs):
    key = id(completions)

    if key not in _ENTAIL_BATCH_CACHE:
        _ENTAIL_BATCH_CACHE[key] = prop_entail_reward_all(
            completions, golds, **kwargs
        )

    main_reward, _ = _ENTAIL_BATCH_CACHE[key]
    return main_reward


def prop_entail_reward_shared_wrapped_all(completions, golds, **kwargs):
    key = id(completions)

    if key not in _ENTAIL_BATCH_CACHE:
        _ENTAIL_BATCH_CACHE[key] = prop_entail_reward_all(
            completions, golds, **kwargs
        )

    _, shared_reward = _ENTAIL_BATCH_CACHE[key]
    del _ENTAIL_BATCH_CACHE[key]

    return shared_reward


# =========================================================
# Predicted-proposition entailment reward
# =========================================================
def predicted_prop_entail_reward_all(completions, gts, **kwargs):
    """Score predicted propositions against raw ground-truth captions.

    Unlike vgv2, this path never decomposes ground-truth captions or reads
    dataset-provided ground-truth propositions. Only the generated caption pair
    is sent to the configured proposition decomposer.
    """
    tokenizer, model, device = get_reward_modules("nli")
    decomposer = get_reward_modules("decomposer")

    contents = [_normalize_text(c[0]["content"]) for c in completions]
    preds = [extract_two_captions(x) for x in contents]
    pos_preds = [_normalize_text(x[0]) for x in preds]
    neg_preds = [_normalize_text(x[1]) for x in preds]

    decomposed = decomposer.extract_many(
        list(zip(pos_preds, neg_preds)),
        persistent=False,
    )
    pos_props, neg_props, shared_props = [], [], []
    for pos_pred, neg_pred, result in zip(pos_preds, neg_preds, decomposed):
        pos_props.append(
            _normalize_text(result["caption_1_independent_propositions"]) or pos_pred
        )
        neg_props.append(
            _normalize_text(result["caption_2_independent_propositions"]) or neg_pred
        )
        shared_props.append(
            _normalize_text(result["shared_proposition"]) or pos_pred or neg_pred
        )

    pos_gts = [_normalize_text(gt) for gt in gts]
    neg_gts = [_normalize_text(gt) for gt in kwargs["gt_neg"]]
    assert (
        len(pos_gts)
        == len(neg_gts)
        == len(pos_props)
        == len(neg_props)
        == len(shared_props)
    )

    scores = _batched_entailment_scores_multi_dedup(
        tokenizer=tokenizer,
        model=model,
        pair_groups={
            "pp": (pos_gts, pos_props),
            "nn": (neg_gts, neg_props),
            "pn": (pos_gts, neg_props),
            "np": (neg_gts, pos_props),
            "ps": (pos_gts, shared_props),
            "ns": (neg_gts, shared_props),
        },
        device=device,
        max_length=192,
        batch_size=256,
    )

    main_reward = torch.sigmoid(
        5.0 * (scores["pp"] - scores["np"] + scores["nn"] - scores["pn"])
    )
    shared_reward = scores["ps"] + scores["ns"]

    return main_reward.tolist(), shared_reward.tolist()


def predicted_prop_entail_reward_wrapped_all(completions, golds, **kwargs):
    key = id(completions)

    if key not in _VGV3_ENTAIL_BATCH_CACHE:
        _VGV3_ENTAIL_BATCH_CACHE[key] = predicted_prop_entail_reward_all(
            completions, golds, **kwargs
        )

    main_reward, _ = _VGV3_ENTAIL_BATCH_CACHE[key]
    return main_reward


def predicted_prop_entail_reward_shared_wrapped_all(
    completions, golds, **kwargs
):
    key = id(completions)

    if key not in _VGV3_ENTAIL_BATCH_CACHE:
        _VGV3_ENTAIL_BATCH_CACHE[key] = predicted_prop_entail_reward_all(
            completions, golds, **kwargs
        )

    _, shared_reward = _VGV3_ENTAIL_BATCH_CACHE[key]
    del _VGV3_ENTAIL_BATCH_CACHE[key]

    return shared_reward


# =========================================================
# Merged proposition-image similarity reward
# =========================================================
def prop_imagesim_reward_all(completions, gts, **kwargs):
    vg_process, vg_tokenizer, vg_model, device = get_reward_modules("vg")
    decomposer = get_reward_modules("decomposer")

    pos_imgs = kwargs["image"]
    neg_imgs = kwargs["neg_image"]

    contents = [_normalize_text(c[0]["content"]) for c in completions]
    preds = [extract_two_captions(x) for x in contents]
    pos_preds = [_normalize_text(x[0]) for x in preds]
    neg_preds = [_normalize_text(x[1]) for x in preds]

    decomposed = decomposer.extract_many(
        list(zip(pos_preds, neg_preds)),
        persistent=False,
    )
    pos_props, neg_props, shared_props = [], [], []
    for pos_pred, neg_pred, result in zip(pos_preds, neg_preds, decomposed):
        pos_props.append(
            _normalize_text(result["caption_1_independent_propositions"]) or pos_pred
        )
        neg_props.append(
            _normalize_text(result["caption_2_independent_propositions"]) or neg_pred
        )
        shared_props.append(result["shared_proposition"])

    assert len(pos_imgs) == len(neg_imgs) == len(pos_props) == len(neg_props) == len(shared_props)

    vg_model.eval()
    with torch.inference_mode():
        pos_image_features = get_image_features_cached(
            pos_imgs, vg_process, vg_model, device
        )
        neg_image_features = get_image_features_cached(
            neg_imgs, vg_process, vg_model, device
        )

        all_texts = pos_props + neg_props + shared_props
        text_tokens = vg_tokenizer(all_texts).to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            all_text_features = F.normalize(
                vg_model.encode_text(text_tokens), dim=-1
            )

        B = len(pos_props)
        pos_text_features = all_text_features[:B]
        neg_text_features = all_text_features[B:2 * B]
        shared_text_features = all_text_features[2 * B:3 * B]

        pp_scores = (pos_image_features * pos_text_features).sum(dim=-1)
        nn_scores = (neg_image_features * neg_text_features).sum(dim=-1)
        np_scores = (neg_image_features * pos_text_features).sum(dim=-1)
        pn_scores = (pos_image_features * neg_text_features).sum(dim=-1)

        main_reward = torch.sigmoid(pp_scores - np_scores + nn_scores - pn_scores)

        pshared_scores = (pos_image_features * shared_text_features).sum(dim=-1)
        nshared_scores = (neg_image_features * shared_text_features).sum(dim=-1)
        shared_reward = pshared_scores + nshared_scores

    return main_reward.cpu().tolist(), shared_reward.cpu().tolist()

def prop_imagesim_reward_wrapped_all(completions, golds, **kwargs):
    key = id(completions)

    if key not in _IMAGESIM_BATCH_CACHE:
        _IMAGESIM_BATCH_CACHE[key] = prop_imagesim_reward_all(
            completions, golds, **kwargs
        )

    main_reward, _ = _IMAGESIM_BATCH_CACHE[key]
    return main_reward


def prop_imagesim_reward_shared_wrapped_all(completions, golds, **kwargs):
    key = id(completions)

    if key not in _IMAGESIM_BATCH_CACHE:
        _IMAGESIM_BATCH_CACHE[key] = prop_imagesim_reward_all(
            completions, golds, **kwargs
        )

    _, shared_reward = _IMAGESIM_BATCH_CACHE[key]
    del _IMAGESIM_BATCH_CACHE[key]

    return shared_reward


# =========================================================
# ProCoRe proposition-level contrastive reward (paper Eq. 3-6)
# =========================================================
_PROCORE_BATCH_CACHE = {}


def procore_reward_all(completions, gts, **kwargs):
    """Return (R_I, R_T, R_s) per sample, following Algorithm 1 of the paper.

    R_I = sigma(simI(I1,P1) - simI(I1,P2) + simI(I2,P2) - simI(I2,P1))
    R_T = sigma(entail(t1,P1) - entail(t2,P1) + entail(t2,P2) - entail(t1,P2))
    R_s = sigma(simI(I1,Ps) + simI(I2,Ps) + entail(t1,Ps) + entail(t2,Ps))
    The weights lambda_I, lambda_T, lambda_s are applied via --reward_weight.
    """
    nli_tokenizer, nli_model, nli_device = get_reward_modules("nli")
    vg_process, vg_tokenizer, vg_model, vg_device = get_reward_modules("vg")
    decomposer = get_reward_modules("decomposer")

    contents = [_normalize_text(c[0]["content"]) for c in completions]
    preds = [extract_two_captions(x) for x in contents]
    pos_preds = [_normalize_text(x[0]) for x in preds]
    neg_preds = [_normalize_text(x[1]) for x in preds]

    # (P1, P2, Ps) <- PD(t1_hat, t2_hat); only predicted captions are decomposed.
    decomposed = decomposer.extract_many(
        list(zip(pos_preds, neg_preds)),
        persistent=False,
    )
    pos_props, neg_props, shared_props = [], [], []
    for pos_pred, neg_pred, result in zip(pos_preds, neg_preds, decomposed):
        pos_props.append(
            _normalize_text(result["caption_1_independent_propositions"]) or pos_pred
        )
        neg_props.append(
            _normalize_text(result["caption_2_independent_propositions"]) or neg_pred
        )
        shared_props.append(
            _normalize_text(result["shared_proposition"]) or pos_pred or neg_pred
        )

    pos_gts = [_normalize_text(gt) for gt in gts]
    neg_gts = [_normalize_text(gt) for gt in kwargs["gt_neg"]]
    pos_imgs = kwargs["image"]
    neg_imgs = kwargs["neg_image"]
    assert (
        len(pos_gts) == len(neg_gts) == len(pos_imgs) == len(neg_imgs)
        == len(pos_props) == len(neg_props) == len(shared_props)
    )

    # entail(t, P): ground-truth caption is the premise, proposition the hypothesis.
    ent = _batched_entailment_scores_multi_dedup(
        tokenizer=nli_tokenizer,
        model=nli_model,
        pair_groups={
            "t1_p1": (pos_gts, pos_props),
            "t2_p1": (neg_gts, pos_props),
            "t2_p2": (neg_gts, neg_props),
            "t1_p2": (pos_gts, neg_props),
            "t1_ps": (pos_gts, shared_props),
            "t2_ps": (neg_gts, shared_props),
        },
        device=nli_device,
        max_length=192,
        batch_size=256,
    )

    vg_model.eval()
    with torch.inference_mode():
        img1 = get_image_features_cached(pos_imgs, vg_process, vg_model, vg_device)
        img2 = get_image_features_cached(neg_imgs, vg_process, vg_model, vg_device)

        text_tokens = vg_tokenizer(pos_props + neg_props + shared_props).to(vg_device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(vg_device.type == "cuda"),
        ):
            text_feats = F.normalize(vg_model.encode_text(text_tokens), dim=-1)

        B = len(pos_props)
        p1, p2, ps = text_feats[:B], text_feats[B:2 * B], text_feats[2 * B:]

        def sim(img, txt):
            return (img * txt).sum(dim=-1).float().cpu()

        r_image = torch.sigmoid(
            sim(img1, p1) - sim(img1, p2) + sim(img2, p2) - sim(img2, p1)
        )
        sim_i1_ps = sim(img1, ps)
        sim_i2_ps = sim(img2, ps)

    r_text = torch.sigmoid(
        ent["t1_p1"] - ent["t2_p1"] + ent["t2_p2"] - ent["t1_p2"]
    )
    r_shared = torch.sigmoid(
        sim_i1_ps + sim_i2_ps + ent["t1_ps"] + ent["t2_ps"]
    )

    return r_image.tolist(), r_text.tolist(), r_shared.tolist()


def _procore_component(completions, golds, index, last, **kwargs):
    key = id(completions)
    if key not in _PROCORE_BATCH_CACHE:
        _PROCORE_BATCH_CACHE[key] = procore_reward_all(completions, golds, **kwargs)
    value = _PROCORE_BATCH_CACHE[key][index]
    if last:
        del _PROCORE_BATCH_CACHE[key]
    return value


def procore_image_reward(completions, golds, **kwargs):
    """R_I^(P), Eq. 3."""
    return _procore_component(completions, golds, 0, False, **kwargs)


def procore_text_reward(completions, golds, **kwargs):
    """R_T^(P), Eq. 4."""
    return _procore_component(completions, golds, 1, False, **kwargs)


def procore_shared_reward(completions, golds, **kwargs):
    """R_s^(P), Eq. 5. Must be registered last: it frees the batch cache."""
    return _procore_component(completions, golds, 2, True, **kwargs)


# =========================================================
# Existing vg reward
# =========================================================
def vg_reward(completions, moacha_gt: List[str], moacha_gt_neg: List[str], **kwargs):
    vg_process, vg_tokenizer, vg_model, device = get_reward_modules("vg")
    contents = [completion[0]["content"] for completion in completions]
    preds = [extract_final_answer(x).lower() for x in contents]

    pos_imgs = kwargs["image"]
    neg_imgs = kwargs["neg_image"]

    with torch.inference_mode():
        pos_image_features = get_image_features_cached(
            pos_imgs, vg_process, vg_model, device
        )
        neg_image_features = get_image_features_cached(
            neg_imgs, vg_process, vg_model, device
        )

        text_tokens = vg_tokenizer(preds).to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            text_features = vg_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

        pos_scores = (pos_image_features * text_features).sum(dim=-1)
        neg_scores = (neg_image_features * text_features).sum(dim=-1)

        vg_rewards = 5.0 * (pos_scores - neg_scores)
        vg_rewards = torch.clamp(vg_rewards, -1.0, 1.0).cpu().tolist()

    return vg_rewards


def vg_reward_wrapped(completions, golds, **kwargs):
    gt_neg = kwargs["gt_neg"]
    return vg_reward(
        completions=completions,
        moacha_gt=golds,
        moacha_gt_neg=gt_neg,
        **kwargs,
    )


# =========================================================
# Existing moacha reward
# =========================================================
def moacha_reward(
    completions,
    moacha_gt: List[str],
    **kwargs,
):
    nli_tokenizer, nli_model, bertscore, device = get_reward_modules("moacha")

    contents = [completion[0]["content"] for completion in completions]
    preds = [extract_final_answer(x).lower() for x in contents]
    refs = [(gt or "").strip().lower() for gt in moacha_gt]

    nli_rewards = []
    with torch.no_grad():
        for pred, truth in zip(preds, refs):
            toks = nli_tokenizer(
                truth,
                pred,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            logits = nli_model(**toks).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

            contradiction_prob = probs[0]
            reward = 1.0 - contradiction_prob
            reward = (reward - 0.5) * 2.0
            nli_rewards.append(float(reward))

    bert_result = bertscore.compute(
        predictions=preds,
        references=refs,
        lang="en",
        rescale_with_baseline=True,
        device=str(device),
    )
    bert_f1 = np.array(bert_result["f1"], dtype=np.float32)
    bert_rewards = np.clip((bert_f1 - 0.35) * 2.0, -1.0, 1.0).tolist()

    all_rewards = [(x + y) / 2.0 for x, y in zip(nli_rewards, bert_rewards)]
    return all_rewards


def moacha_reward_wrapped(completions, golds, **kwargs):
    return moacha_reward(completions=completions, moacha_gt=golds, **kwargs)


# =========================================================
# Reward selection
# =========================================================
def get_reward_func(reward_type):
    reward_fns = []

    if reward_type == "hall":
        reward_fns.append(moacha_reward_wrapped)

    elif reward_type == "vg":
        reward_fns.append(vg_reward_wrapped)

    elif reward_type == "hall+vg":
        reward_fns.append(moacha_reward_wrapped)
        reward_fns.append(vg_reward_wrapped)

    elif reward_type == "vgv2":
        reward_fns.append(prop_entail_reward_wrapped_all)
        reward_fns.append(prop_entail_reward_shared_wrapped_all)
        reward_fns.append(prop_imagesim_reward_wrapped_all)
        reward_fns.append(prop_imagesim_reward_shared_wrapped_all)

    elif reward_type == "procore":
        # Order must match --reward_weight (lambda_I, lambda_T, lambda_s).
        reward_fns.append(procore_image_reward)
        reward_fns.append(procore_text_reward)
        reward_fns.append(procore_shared_reward)

    elif reward_type in {"rouge", "vgpo", "diva", "pepo"}:
        # These methods use the same task-level ROUGE-L outcome reward.
        # The selector controls which advantage path the trainer applies.
        reward_fns.append(rouge_reward)

    elif reward_type in {"caprl", "caprlpp"}:
        # CapRL / CapRL++ QA-utility rewards. Imported lazily so runs that use
        # another reward never load the text-only answerer.
        from utils.caprl_reward import CAPRL_REWARD_FUNCTIONS

        reward_fns.extend(CAPRL_REWARD_FUNCTIONS[reward_type])

    elif reward_type == "fast":
        # FAST-GRPO difficulty-aware outcome/format/length reward, adapted to
        # the paired-caption task in utils.fast_reward.
        from utils.fast_reward import FAST_REWARD_FUNCTIONS

        reward_fns.extend(FAST_REWARD_FUNCTIONS)

    elif reward_type == "grit":
        # GRIT uses six separately logged and summed rewards. Their intrinsic
        # scales match the official release; the answer score is the local
        # paired-caption ROUGE-L outcome because the official GPT judge depends
        # on private Azure credentials.
        from methods.grit import GRIT_REWARD_FUNCTIONS

        reward_fns.extend(GRIT_REWARD_FUNCTIONS)

    else:
        raise ValueError(
            "--reward_type must be one of "
            "{'hall', 'vg', 'hall+vg', 'vgv2', 'procore', 'vgpo', 'diva', "
            "'pepo', 'rouge', 'grit', 'caprl', 'caprlpp', 'fast'}"
        )

    return reward_fns
