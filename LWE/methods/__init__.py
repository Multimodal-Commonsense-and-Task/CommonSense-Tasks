"""
Inference methods:

- vanilla: fixed judge prompt, one LLM call per sample.
- ssp: sample-specific prompt — static meta-instruction; per-sample rubric + judge.
- lwe: SSP + meta-eval + batch meta-prompt update.
- selective_lwe: vanilla -> LWE only on inconsistent samples.
"""
