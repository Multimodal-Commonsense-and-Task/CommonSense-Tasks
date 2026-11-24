from dataclasses import dataclass, field
import transformers
from typing import Optional, List

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
    
    model_name: Optional[str] = field(default=None)

    use_mtv: bool = field(default=False)
    mtv_path: Optional[str] = field(default=None)
    mtv_activation_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'pad'
    model_name_for_dataarg: Optional[str] = field(default=None)
    
    get_prompt: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # sampling
    k_means: int = None
    embedding: str = field(default=None)
    
    reward_type: str = field(default=None)
    proposition_model: str = field(
        default="llama3_2-1b",
        metadata={
            "help": (
                "Proposition decomposer: llama3_2-1b, llama3_2-3b, llama3_1-8b, "
                "qwen3-0_6b, qwen3-1_7b, qwen3-4b, qwen3-8b, flan-t5-base, "
                "flan-t5-large, flan-t5-xl, allennlp-srl, bert-base-srl, gpt, "
                "precomputed, openai:<model>, hf:<model>, "
                "allennlp_srl:<archive-or-model-id>, or a Hugging Face model ID."
            )
        },
    )
    proposition_cache_path: str = field(
        default=".cache/vgv2/ground_truth_propositions.sqlite3",
        metadata={"help": "Persistent cache for ground-truth proposition decompositions."},
    )
    proposition_device: str = field(
        default="auto",
        metadata={"help": "Device for local proposition models, e.g. auto, cpu, cuda:1."},
    )
    
    eval_period: int = None
    min_z_score_threshold: float = 0.5
    min_z_score_threshold2: float = 0.5
    max_z_score_threshold: float = 0.5
    start_ema_update_step: int = None
    warmup_sample_ratio: float = 0.5
    mutual_param: float = 0.5
    train_iter: int = 0
    
    pretrain_checkpoint: str = field(default=None)
    is_eval: bool = False
    is_prompt: bool = False
    round_to_eval: int = None
    eval_temp: float = 0.2
    eval_server: bool = False
    unseen_task: bool = False
    eval_client: int = field(default=None)
    eval_iter: int = field(default=None)
    zeroshot: bool = False
    resume_train_round: int = field(default=None)
    idn_un:float = field(default=0.9)
    num_iter:float = field(default=100)
    ema_update_mode: str = field(default=None)
    info_update_mode: str = field(default=None)
    softmax_update_temp: float = 1.0
    
    sampling: str = field(default=None)
    ema_average: bool = False
    ema_pool_size: int = field(default=0)

    setup: str = field(default=None)
    
    # cl config
    mode: str = field(default="er")
    # STV activation intervention. train_VLM.sh enables ``interv`` for METHOD=stv.
    intervention_mode: str = field(default="clean")
    intervention_topk: int = field(default=96)
    intervention_num_clusters: int = field(default=64)
    intervention_n_trials: int = field(default=100)
    intervention_num_shot: int = field(default=4)
    reinforce_n_epochs: int = field(default=100)
    reinforce_samples_per_epoch: int = field(default=32)
    reinforce_lr: float = field(default=0.1)
    reinforce_eval_every: int = field(default=50)
    reinforce_final_num_samples: int = field(default=10)
    # dataset: str = field(default="cifar10")
    scenario: int = field(default=1)
    train_scenario: int = field(default=1)
    note: str = field(default=None)
    eval_period: int = field(default=100)
    online_iter: float = field(default=1.0)
    use_kornia: bool = True
    transform_on_gpu: bool = True
    transform_on_worker: bool = False
    topk: int = field(default=1)
    f_period: int = field(default=None)
    transforms: str= field(default='randaug')
    # reward_weight: float = field(default=1.0)
    reward_weight: List[float] = field(default_factory=lambda: [1])
    grpo_num_gen: int = field(default=None)

    # DIVA-GRPO (Difficulty-Adaptive Variant Advantage)
    diva_variant_num: int = 1
    diva_weight_mode: str = "ZSCORENORM_WAN_RRBLOCAL_RRBGLOBAL"
    diva_score_ranges: str = "0.0:0.05,0.05:0.3,0.3:0.5,0.5:0.7,0.7:1.0"
    diva_difficulty_changes: str = "2,1,0,-1,-2"
    diva_min_difficulty: int = 1
    diva_max_difficulty: int = 9
    diva_initial_difficulty: int = 5
    diva_weighted_advantage_k: float = 0.1
    diva_rrb_threshold: float = 0.15
    diva_rrb_scale_factor: float = 0.1
    diva_local_advantage_weight: float = 1.0
    diva_global_advantage_weight: float = 1.0

    # PEPO (Perception-Exploration Policy Optimization)
    pepo_gate_alpha: float = 0.05
    pepo_gate_temperature: float = 1.8
    pepo_use_vision_weights: bool = True
    pepo_vision_score_type: str = "cosine"
    pepo_image_token: Optional[str] = None
    pepo_beta: float = 0.001

    # VGPO (Visually-Guided Policy Optimization)
    vgpo_use_intra_trajectory_reweighting: bool = True
    vgpo_use_inter_trajectory_reweighting: bool = True
    vgpo_use_visual_compensation: bool = True
    vgpo_use_gated_visual_compensation: bool = True
    vgpo_gated_visual_compensation_start_ratio: float = 0.5
    vgpo_visual_compensation_strength: float = 0.3
    vgpo_visual_attention_threshold: float = 0.2

    # CapRL / CapRL++ (QA-utility caption rewards)
    caprl_qa_source: str = field(
        default="contrastive",
        metadata={"help": "How the MCQ bank is built: 'contrastive' (from the "
                          "gold caption pair, no extra model) or 'llm'."},
    )
    caprl_qa_model: str = field(
        default="gpt",
        metadata={"help": "Question writer for --caprl_qa_source llm: gpt, "
                          "openai:<model>, qwen3-1_7b, hf:<model-id>, ..."},
    )
    caprl_qa_cache_path: str = field(
        default=".cache/caprl/caprl_qa.sqlite3",
        metadata={"help": "Persistent cache for LLM-generated CapRL questions."},
    )
    caprl_qa_num: int = field(
        default=4,
        metadata={"help": "Questions sampled per caption per step (CapRL qa_num)."},
    )
    caprl_qa_all: bool = field(
        default=False,
        metadata={"help": "Score every question instead of sampling qa_num."},
    )
    caprl_num_options: int = field(default=4)
    caprl_shuffle_qa: bool = field(default=True)
    caprl_answer_model: str = field(
        default="Qwen/Qwen3-1.7B",
        metadata={"help": "Text-only LLM that answers the MCQs from the caption."},
    )
    caprl_answer_device: str = field(default="auto")
    caprl_answer_batch_size: int = field(default=32)
    caprl_answer_temperature: float = field(default=0.6)
    caprl_acc_scale: float = field(
        default=2.0,
        metadata={"help": "CapRL multiplies mean QA accuracy by 2."},
    )
    caprl_format_mode: str = field(
        default="pair",
        metadata={"help": "CapRL++ R_format: 'pair' (a || b contract), "
                          "'timestamp' (CapRL++ video reward) or 'none'."},
    )
    caprl_format_weight: float = field(default=0.2)
    caprl_length_weight: float = field(default=0.2)
    caprl_length_l1: int = field(default=128)
    caprl_length_l2: int = field(default=256)
    caprl_length_tokenizer: str = field(default="")
    caprl_debug: bool = field(default=False)
    caprl_eval: bool = field(
        default=False,
        metadata={"help": "Eval only: also report CapRL/CapRL++ caption metrics."},
    )

    # FAST-GRPO difficulty-aware reward. ROUGE-L at or above the correctness
    # threshold counts as a successful paired-caption rollout for deciding
    # whether FAST should favor concise or exploratory responses.
    fast_correctness_threshold: float = 0.5
    fast_difficulty_percentile: float = 80.0
    fast_format_weight: float = 0.5
    fast_length_weight: float = 0.5

    # GRIT (Grounded Reasoning with Images and Text)
    grit_max_completion_length: int = 1000
    grit_clip_epsilon: float = 0.28
    
    shot: str = field(default=None)
    eval_shot: str = field(default=None)
    dataset_name: str = field(default=None)
    ordering: str = field(default=None)
    train_ordering: str = field(default=None)
    
    save_dir: str = field(default=None)
    
    eval_caption: str = field(default=None)
    
    use_vis_token_amplifier: bool = False
    vis_token_amplifier: float = 1.0
    use_pai: bool = False

    # federated learning
    num_clients: int = 1
    num_rounds: int = 1
    num_tasks: int = 4
    iter_per_round: int = 1
    state_dir: str = field(default="./client_states")
    final_lr: float = field(default=1e-6)
    mm_final_lr: float = field(default=1e-6)
    
    ccot: bool = False
    use_cotprompt: bool = False
    cotprompt_ver: int = 1
    use_icot: bool = False

    # SPARC (Selective Progressive Attention ReCalibration) — eval-time only.
    # Requires --attn_impl eager. See utils/sparc.py.
    use_sparc: bool = field(default=False)
    sparc_alpha: float = field(default=1.1)
    sparc_beta: float = field(default=0.1)
    sparc_tau: float = field(default=1.5)
    sparc_selected_layer: int = field(default=20)
    sparc_start_layer: int = field(default=0)
    sparc_end_layer: int = field(default=31)
    
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
    per_gpu_train_batch_size: int = field(default=2)

    cache_dir: Optional[str] = field(default=None)
    attn_impl : str = field(default="flash_attention_2")
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
    
    output_dir: str = "none"
    
    
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
