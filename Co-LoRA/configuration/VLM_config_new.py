from dataclasses import dataclass, field
import transformers
from typing import Optional, Any

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None) #"liuhaotian/llava-v1.5-7b"
    model_type: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v1")
    max_new_tokens: Optional[int] = field(default=512)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'pad'
    model_name_for_dataarg: Optional[str] = field(default=None)
    
    get_prompt: bool = False

@dataclass
class TrainingConfig(transformers.TrainingArguments):
    # exp config
    mode: str = field(default="er") # method
    note: str = field(default=None) # decide output directory
    scenario: str = field(default='DRAKE_hetero_llava_llama_1B_3B') # read data config in scenarios/scenario-N.json
    
    # federated learning
    num_clients: int = 10
    num_rounds: float = 20
    num_tasks: int = 4
    num_iter:int = field(default=100) # iter per round
    state_dir: str = field(default="./client_states")
    final_lr: float = field(default=1e-6)
    mm_final_lr: float = field(default=1e-6)
    is_cross_model_series: bool = False
    
    # continual learning
    memory_size: int = 100000
    is_streamonly: bool = True
    is_continual: bool = True
    online_stream_T: float = 0.125
    online_stream_count_decay_ratio: float = 0.99
    online_iter: float = field(default=1.0)
    
    optim: str = field(default="adamw_torch")
    decay_ratio: float = field(default=1.0)
    save_optim: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    save_per_step: bool = False
    
    # fedmosaic config
    use_task_vector:bool = False
    grad_freq: int = 10
    load_pretrained_lora:bool = False # pre-trained lora weight for Co-LoRA A,B init
    softmax_temp: float = 0.5
    gradient_ratio: float = 0.4
    gradient_noise_type: str = "gaussian"
    gradient_noise_std: float = 1e-4
    
    AB_align_data_size: int = 5000
    num_blocks:int = 4
    
    # distillation config
    num_serverdistill: int = 80
    
    # eval config
    is_eval: bool = False
    round_to_eval: int = None
    eval_temp: float = 0.0
    unseen_task: bool = False
    eval_client: int = field(default=None)
    eval_iter: int = field(default=None)
    zeroshot: bool = False
    eval_all: bool = False
    eval_client_start: int = field(default=None)
    eval_client_end: int = field(default=None)
    eval_client_eval_start: int = field(default=None)
    eval_client_eval_end: int = field(default=None)
    set_state: str = field(default="gate")
    
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=8,
        metadata={"help": "How many bits to use."}
    )

    # lora config
    lora_enable: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = 2e-5
    group_by_modality_length: bool = field(default=True)