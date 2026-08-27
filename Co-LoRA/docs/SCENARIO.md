# Federated Learning Scenarios

We load the FL configuration from `scenarios/scenario-{SCENARIO}.json` when you set `SCENARIO={SCENARIO}` in the training script.

---

## Client Model Configuration per Experiment

The table below summarizes some of configuration of each scenario JSON that we used in the paper with the number of clients per model family/size.
The LLaVA models with various sizes are trained by ourselves following [LLaVA-v1.5 training recipes](https://github.com/haotian-liu/LLaVA).
The models are available in [here](https://huggingface.co/collections/thkim0305/llava-variable-sizes).


### Multi-modal Experiments
set `IS_MULTIMODAL = True` to run multi-modal experiments
| Scenario | LLaVA-Llama3 1B | LLaVA-Llama3 3B | LLaVA-Llama3 8B | LLaVA-Qwen2.5 0.5B | LLaVA-Qwen2.5 1.5B | LLaVA-Qwen2.5 3B |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| `DRAKE_homo_llava_llama_3B` | 0 | 10 | 0 | 0 | 0 | 0 |
| `DRAKE_hetero_llava_llama_1B_3B` |  4 | 6 | 0 | 0 | 0 | 0 |
| `DRAKE_hetero_llava_llama_1B_3B_8B` | 3 | 5 | 2 | 0 | 0 | 0 |
| `DRAKE_homo_llava_qwen_1_5B` | 0 | 0 | 0 | 0 | 10 | 0 |
| `DRAKE_hetero_llava_qwen_0_5B_1_5B_3B` |  0 | 0 | 0 | 2 | 3 | 5 |
| `DRAKE_hetero_llava_llama_1B_3B_qwen_1_5B` | 2 | 5 | 0 | 0 | 3 | 0 |
| `DRAKE_hetero_llava_llama_3B_qwen_1_5B_3B` | 0 | 3 | 0 | 0 | 4 | 3 |
| `HFLB_homo_llava_llama_3B` | 0 | 9 | 0 | 0 | 0 | 0 |
| `HFLB_hetero_llava_llama_1B_3B` | 3 | 6 | 0 | 0 | 0 | 0 |

### Text-only Experiments
set `IS_MULTIMODAL = False` to run text-only experiments
| Scenario |Llama-3 1B | Llama-3 3B | Llama-3 8B |
|:---:|---|:---:|:---:|
| `Fed-LLM_hetero_llama_1B_3B` | 26 | 26 | 0 |
| `Fed-Scope_hetero_llama_3B_8B` | 0 | 3 | 2 |
| `Fed-aya_hetero_llama_1B_3B` |  4 | 4 | 0 |

---