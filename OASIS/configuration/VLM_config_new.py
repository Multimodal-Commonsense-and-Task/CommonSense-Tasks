from dataclasses import dataclass, field
import transformers
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None) #"liuhaotian/llava-v1.5-7b"
    model_type: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
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
class TrainingArguments(transformers.TrainingArguments):
    # sampling
    selection_method: str = field(default=None)
    sample_ratio: float = 0.5
    embedding: str= field(default='representation')
    k_means: int = None
    
    eval_period: int = None
    min_z_score_threshold: float = 0.5
    min_z_score_threshold2: float = 0.5
    max_z_score_threshold: float = 0.5
    start_ema_update_step: int = None
    mutual_param: float = 0.5
    train_iter: int = 0
    
    pretrain_checkpoint: str = field(default=None)
    is_eval: bool = False
    is_prompt: bool = False
    round_to_eval: int = None
    checkpoint_path: Optional[str] = field(default=None)
    eval_temp: float = 0.2
    eval_server: bool = False
    unseen_task: bool = False
    eval_client: int = field(default=None)
    eval_iter: int = field(default=None)
    zeroshot: bool = False
    resume_train_round: int = field(default=None)
    
    num_iter:float = field(default=100)
    ema_update_mode: str = field(default=None)
    info_update_mode: str = field(default=None)
    softmax_update_temp: float = 1.0
    
    sampling: str = field(default=None)
    ema_average: bool = False
    ema_pool_size: int = field(default=0)

    setup: str = field(default=None)
    grad_ratio: float = field(default=1.0)
    
    # cl config
    mode: str = field(default="er")
    # dataset: str = field(default="cifar10")
    scenario: int = field(default=1)
    note: str = field(default=None)
    eval_period: int = field(default=100)
    online_iter: float = field(default=1.0)
    use_kornia: bool = True
    transform_on_gpu: bool = True
    transform_on_worker: bool = False
    topk: int = field(default=1)
    f_period: int = field(default=None)
    transforms: str= field(default='randaug')

    # federated learning
    num_clients: int = 5
    num_rounds: int = 20
    num_tasks: int = 4
    iter_per_round: int = 1
    state_dir: str = field(default="./model_states")
    final_lr: float = field(default=1e-6)
    mm_final_lr: float = field(default=1e-6)
    
    # continual learning
    memory_size: int = 100000
    is_streamonly: bool = True
    is_sar: bool = True
    is_sar_task: bool = True
    use_task_id: bool = False
    
    # prompt tuning args
    prompt_num: int = field(default=100)
    
    optim: str = field(default="adamw_torch")
    is_wsd: str = field(default=None)
    decay_ratio: float = field(default=1.0)
    save_optim: bool = field(default=False)
    
    temp_batchsize: int = field(default=2)

    cache_dir: Optional[str] = field(default=None)
    
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
    
    ia3_enable: bool = False
    
    
    # generator config
    generator_output_size: int = 256
    generator_hidden_dim: int = 32
    generator_hidden_feature: int = 16
    set_state:str = 'gate'
    ema_ratio:float = 0.01
    model_ema_ratio:float = 0.99
    key_embed_size:int = 64
    pool_size:int = 4
    prompt_top_k:int = 1
    
    use_task_vector:bool = False
    use_fisher:bool = False
    
    load_checkpoint:str = None
    fedours:bool = False
