"""JSON-lines worker for legacy AllenNLP semantic-role-labeling models.

Legacy AllenNLP releases pin versions of PyTorch and Transformers that conflict
with the GRPO training environment.  This worker is therefore launched with the
Python interpreter named by PROPOSITION_SRL_PYTHON and keeps those dependencies
in an isolated environment.
"""

import argparse
from contextlib import redirect_stdout
import json
import sys


def _cuda_device(device_name: str, torch) -> int:
    device_name = (device_name or "auto").strip().lower()
    if device_name == "auto":
        return torch.cuda.current_device() if torch.cuda.is_available() else -1
    if device_name == "cpu":
        return -1
    if device_name == "cuda":
        return 0
    if device_name.startswith("cuda:"):
        return int(device_name.split(":", 1)[1])
    raise ValueError(
        f"Unsupported SRL device {device_name!r}; use auto, cpu, cuda, or cuda:N"
    )


def _load_predictor(model_source: str, cuda_device: int):
    # Keep library/download chatter off stdout because stdout is the protocol.
    protocol_stdout = sys.stdout
    with redirect_stdout(sys.stderr):
        import torch
        try:
            # AllenNLP >= 1 moved task models into this companion package.
            import allennlp_models  # noqa: F401
        except ImportError:
            # The official 2018 ELMo SRL archive runs on AllenNLP 0.9, where
            # the SRL model and predictor still lived in the core package.
            pretrained = None
        else:
            import allennlp_models.structured_prediction  # noqa: F401
            from allennlp_models import pretrained
        from allennlp.predictors.predictor import Predictor

        if model_source.startswith(("http://", "https://")):
            predictor = Predictor.from_path(
                model_source,
                cuda_device=cuda_device,
            )
        else:
            if pretrained is None:
                raise RuntimeError(
                    "AllenNLP model IDs require the allennlp-models package; "
                    "use an archive URL/path or install allennlp-models"
                )
            predictor = pretrained.load_predictor(
                model_source,
                cuda_device=cuda_device,
            )
    sys.stdout = protocol_stdout
    return predictor, torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Import torch in the isolated environment only.
    protocol_stdout = sys.stdout
    with redirect_stdout(sys.stderr):
        import torch
    sys.stdout = protocol_stdout

    predictor, _ = _load_predictor(
        args.model,
        _cuda_device(args.device, torch),
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
            captions = request["captions"]
            inputs = [{"sentence": str(caption)} for caption in captions]
            with redirect_stdout(sys.stderr):
                results = predictor.predict_batch_json(inputs)
            response = {"results": results}
        except Exception as exc:
            response = {
                "error": f"{type(exc).__name__}: {exc}",
            }
        protocol_stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol_stdout.flush()


if __name__ == "__main__":
    main()
