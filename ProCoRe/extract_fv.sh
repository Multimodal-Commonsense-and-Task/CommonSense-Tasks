MODE="sft"
MODEL_ARCHS=("internvl3_5-8b" "qwen3-8b" "onevision1_5-8b") # "onevision-7b" "idefics-8b" "internvl2_5-8b" "qwen-7b" "otter-9b" "internvl3_5-8b" "qwen3-8b" "onevision1_5-8b"
SHOT="original"
EVAL_SHOTS=("2-shot") # "0-shot" "1-shot" "2-shot" "3-shot" "4-shot"
# EVAL_SHOTS=("12-shot" "16-shot" "2-shot" "8-shot" "4-shot")
DATASET_NAME="flintstonesv3" #mic vlicl truemicl isekai stanford10000 stanford10000_5 stanford10000_+1 pororo flintstones story_visual_ICL storystream movie101 moviegraphs memor
ORDERS=("random") # "mmices_asc" "multimodal" "random"

EPOCH=1
METHOD=None # Finetune None dara
TRAIN_DATASET_NAME="coco" # openchair mhal-detect haloquest inpaintcoco winoground
TRAIN_SCENARIO=1
LR=1e-3

# fed args
SCENARIOS=(1)

# Don't Change Arguments Below-----------------------------------------------------------------
USE_VIS_TOKEN_AMPLIFIER=False
USE_PAI=False
if [ "$USE_VIS_TOKEN_AMPLIFIER" == "True" ]; then
    VIS_TOKEN_AMPLIFIERS=(0.1 0.3 0.5 0.7 1.0 1.5 2.0)
elif [ "$USE_PAI" == "True" ]; then
    VIS_TOKEN_AMPLIFIERS=(0.1)
else
    VIS_TOKEN_AMPLIFIERS=(0.0)
fi

USE_ICOT=False
USE_COTPROMPT=False
COTPROMPT_VER=2
if [ "$USE_ICOT" == "True" ]; then
    USE_COTPROMPT=True
fi

NUM_ROUNDS=1
NUM_TASKS=1
NUM_CLIENTS=1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
MODEL_MAX_LEN=40000
MAX_NEW_TOKENS=512

ROUND_TO_EVALS=(1)
ITER_TO_EVAL=0
# eval_onevision.py
# eval_VLM_CL.py
# CUDA_VISIBLE_DEVICES=$1
for ((ind5=0; ind5<${#VIS_TOKEN_AMPLIFIERS[@]}; ind5++)); do
    VIS_TOKEN_AMPLIFIER=${VIS_TOKEN_AMPLIFIERS[$ind5]}
    for ((ind3=0; ind3<${#EVAL_SHOTS[@]}; ind3++)); do
        EVAL_SHOT=${EVAL_SHOTS[$ind3]}
        for ((ind4=0; ind4<${#MODEL_ARCHS[@]}; ind4++)); do
            MODEL_ARCH=${MODEL_ARCHS[$ind4]}
            if [ "$MODEL_ARCH" == "llava" ]; then
                MODEL_NAME="liuhaotian/llava-v1.5-7b"
                MODEL_NAME_OFF="llava-7b"
                VERSION="v1"
                VISION_TOWER="openai/clip-vit-large-patch14-336"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "llava13b" ]; then
                MODEL_NAME="liuhaotian/llava-v1.5-13b"
                MODEL_NAME_OFF="llava-13b"
                VERSION="v1"
                VISION_TOWER="openai/clip-vit-large-patch14-336"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-8b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-8B"
                MODEL_NAME_OFF="internvl2_5-8b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-4b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-4B"
                MODEL_NAME_OFF="internvl2_5-4b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-2b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-2B"
                MODEL_NAME_OFF="internvl2_5-2b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-1b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-1B"
                MODEL_NAME_OFF="internvl2_5-1b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3-8b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3-8B"
                MODEL_NAME_OFF="internvl3-8b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3-1b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3-1B"
                MODEL_NAME_OFF="internvl3-1b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3-2b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3-2B"
                MODEL_NAME_OFF="internvl3-2b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3-9b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3-9B"
                MODEL_NAME_OFF="internvl3-9b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3-14b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3-14B"
                MODEL_NAME_OFF="internvl3-14b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3_5-8b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3_5-8B"
                MODEL_NAME_OFF="internvl3_5-8b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3_5-14b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3_5-14B"
                MODEL_NAME_OFF="internvl3_5-14b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3_5-4b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3_5-4B"
                MODEL_NAME_OFF="internvl3_5-4b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3_5-2b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3_5-2B"
                MODEL_NAME_OFF="internvl3_5-2b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3_5-1b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3_5-1B"
                MODEL_NAME_OFF="internvl3_5-1b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-8b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-8B"
                MODEL_NAME_OFF="internvl2_5-8b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-4b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-4B"
                MODEL_NAME_OFF="internvl2_5-4b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-2b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-2B"
                MODEL_NAME_OFF="internvl2_5-2b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-1b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-1B"
                MODEL_NAME_OFF="internvl2_5-1b"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "qwen-3b" ]; then
                MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
                MODEL_NAME_OFF="qwen-3b"
                VERSION="v1"
                VISION_TOWER="openai/clip-vit-large-patch14-336"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "qwen-7b" ]; then
                MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
                MODEL_NAME_OFF="qwen-7b"
                VERSION="v1"
                VISION_TOWER="openai/clip-vit-large-patch14-336"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "onevision-0_5b" ]; then
                MODEL_NAME="lmms-lab/llava-onevision-qwen2-0.5b-ov"
                MODEL_NAME_OFF="onevision-0_5b"
                VERSION="llama"
                VISION_TOWER="google/siglip-so400m-patch14-384"
                MODEL_TYPE="llama3-8b"
                BITS=16
            elif [ "$MODEL_ARCH" == "onevision-7b" ]; then
                MODEL_NAME="lmms-lab/llava-onevision-qwen2-7b-ov"
                MODEL_NAME_OFF="onevision-7b"
                VERSION="llama"
                VISION_TOWER="google/siglip-so400m-patch14-384"
                MODEL_TYPE="llama3-8b"
                BITS=16
            elif [ "$MODEL_ARCH" == "onevision1_5-8b" ]; then
                MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
                MODEL_NAME_OFF="onevision1_5-8b"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            elif [ "$MODEL_ARCH" == "onevision1_5-4b" ]; then
                MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
                MODEL_NAME_OFF="onevision1_5-4b"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            elif [ "$MODEL_ARCH" == "idefics-8b" ]; then
                MODEL_NAME="HuggingFaceM4/Idefics3-8B-Llama3"
                MODEL_NAME_OFF="idefics-8b"
                VERSION="llama"
                VISION_TOWER="google/siglip-so400m-patch14-384"
                MODEL_TYPE="llama3-8b"
                BITS=16
            elif [ "$MODEL_ARCH" == "openflamingo-9b" ]; then
                MODEL_NAME="openflamingo/openflamingo-9b"
                MODEL_NAME_OFF="openflamingo-9b"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            elif [ "$MODEL_ARCH" == "otter-9b" ]; then
                MODEL_NAME="luodian/otter-9b-hf"
                MODEL_NAME_OFF="otter-9b"
                VERSION="otter"
                VISION_TOWER="openai/clip-vit-large-patch14"
                MODEL_TYPE="otter"
                BITS=16
            elif [ "$MODEL_ARCH" == "qwen3-8b" ]; then
                MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
                MODEL_NAME_OFF="qwen3-8b"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            elif [ "$MODEL_ARCH" == "qwen3-4b" ]; then
                MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"
                MODEL_NAME_OFF="qwen3-4b"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            elif [ "$MODEL_ARCH" == "qwen3-2b" ]; then
                MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
                MODEL_NAME_OFF="qwen3-2b"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            else
                echo "Undefined setting"
                exit 1
            fi
        
            ZEROSHOT=True
            ENABLE_LORA=False
            USE_MTV=False
            if [ "$METHOD" == "Finetune" ]; then
                ZEROSHOT=False
                ENABLE_LORA=True
                if [ "$USE_VIS_TOKEN_AMPLIFIER" == "True" ]; then
                    MODEL_NAME_OFF=FINAL_VIS_AMPLIFIERx${VIS_TOKEN_AMPLIFIER}_${MODEL_NAME_OFF}_finetune_${TRAIN_DATASET_NAME}_${EPOCH}
                elif [ "$USE_PAI" == "True" ]; then
                    MODEL_NAME_OFF=PAIx${VIS_TOKEN_AMPLIFIER}_${MODEL_NAME_OFF}_finetune_${TRAIN_DATASET_NAME}_${EPOCH}
                else
                    MODEL_NAME_OFF=${MODEL_NAME_OFF}_finetune_${TRAIN_DATASET_NAME}_${EPOCH}
                fi
            elif [ "$METHOD" == "dara" ]; then
                ZEROSHOT=False
                TRAIN_SCENARIO=$(echo "$EVAL_SHOT" | sed 's/[^0-9]//g')
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_dara_${TRAIN_DATASET_NAME}_${TRAIN_SCENARIO}_${EPOCH}_${LR}
            elif [ "$METHOD" == "mtv" ]; then
                USE_MTV=True
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_mtv
                TRAIN_DATASET_NAME=$DATASET_NAME
            else
                TRAIN_DATASET_NAME=$DATASET_NAME
            fi

            if [ "$USE_VIS_TOKEN_AMPLIFIER" == "True" ]; then
                MODEL_NAME_OFF=FINAL_VIS_AMPLIFIERx${VIS_TOKEN_AMPLIFIER}_${MODEL_NAME_OFF}
            elif [ "$USE_PAI" == "True" ]; then
                MODEL_NAME_OFF=PAIx${VIS_TOKEN_AMPLIFIER}_${MODEL_NAME_OFF}
            fi

            if [ "$USE_ICOT" == "True" ]; then
                MODEL_NAME_OFF=ICOT_VER${COTPROMPT_VER}_${MODEL_NAME_OFF}
            elif [ "$USE_COTPROMPT" == "True" ]; then
                MODEL_NAME_OFF=COTPROMPT_VER${COTPROMPT_VER}_${MODEL_NAME_OFF}
            fi
            for ((ind2=0; ind2<${#ORDERS[@]}; ind2++)); do
                ORDER=${ORDERS[$ind2]}
                for ((ind=0; ind<${#SCENARIOS[@]}; ind++)); do
                    SCENARIO=${SCENARIOS[$ind]}
                    MTV_PATH="./MTV/storage/${MODEL_ARCH}/${DATASET_NAME}/mtv_${SCENARIO}.pt"
                    MTV_ACT_PATH="./MTV/storage/${MODEL_ARCH}/${DATASET_NAME}/mtv_activation_${SCENARIO}.pt"
                    if [ "$METHOD" == "dara" ]; then
                        MODE=dara
                        NOTE=${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EPOCH}_${SHOT}_scen${TRAIN_SCENARIO}_${LR}
                        ATTEN_IMPLE="eager"
                    elif [ "$METHOD" == "only" ]; then
                        MODE=only
                        NOTE=${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EPOCH}_${SHOT}_scen${TRAIN_SCENARIO}
                        ATTEN_IMPLE="eager"
                    elif [ "$METHOD" == "mtv" ]; then
                        NOTE=${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EVAL_SHOT}_${ORDER}_scen_${SCENARIO}_${TRAIN_SCENARIO}_mtv_${USE_MTV}
                        ATTEN_IMPLE="flash_attention_2"
                    elif [ "$METHOD" == "Finetune" ]; then
                        NOTE=${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EPOCH}_${SHOT}_scen${TRAIN_SCENARIO}
                        ATTEN_IMPLE="flash_attention_2"
                        if [ "$USE_VIS_TOKEN_AMPLIFIER" == "True" ]; then
                            NOTE=VIS_AMPLIFIERx${VIS_TOKEN_AMPLIFIER}_${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EPOCH}_${SHOT}_scen${TRAIN_SCENARIO}
                            ATTEN_IMPLE="eager"
                        fi
                    else
                        NOTE=${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EVAL_SHOT}_${ORDER}_scen${SCENARIO}
                        ATTEN_IMPLE="flash_attention_2"
                        if [ "$USE_VIS_TOKEN_AMPLIFIER" == "True" ]; then
                            NOTE=VIS_AMPLIFIERx${VIS_TOKEN_AMPLIFIER}_${NOTE}
                            ATTEN_IMPLE="eager"
                        elif [ "$USE_PAI" == "True" ]; then
                            NOTE=PAIx${VIS_TOKEN_AMPLIFIER}_${NOTE}
                            ATTEN_IMPLE="eager"
                        fi

                        if [ "$USE_ICOT" == "True" ]; then
                            NOTE=ICOT_${COTPROMPT_VER}_${NOTE}
                            ATTEN_IMPLE="eager"
                        elif [ "$USE_" == "True" ]; then
                            NOTE=COTPROMPT_VER${COTPROMPT_VER}_${NOTE}
                        fi
                    fi

                    for ((index=0; index<${#ROUND_TO_EVALS[@]}; index++)); do
                        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                        CUDA_VISIBLE_DEVICES=$1 python extract_fv.py \
                            --is_eval True \
                            --model_name_or_path $MODEL_NAME \
                            --model_name_for_dataarg $MODEL_NAME \
                            --model_name $MODEL_NAME_OFF \
                            --model_type $MODEL_TYPE \
                            --version $VERSION \
                            --scenario $SCENARIO \
                            --train_scenario $TRAIN_SCENARIO \
                            --num_rounds $NUM_ROUNDS \
                            --num_tasks $NUM_TASKS \
                            --num_clients $NUM_CLIENTS \
                            --model_max_length $MODEL_MAX_LEN \
                            --max_new_tokens $MAX_NEW_TOKENS \
                            --vision_tower $VISION_TOWER \
                            --bits $BITS \
                            --bf16 True \
                            --tf32 False \
                            --note $NOTE \
                            --mode $MODE \
                            --eval_server True \
                            --unseen_task True \
                            --zeroshot $ZEROSHOT \
                            --lora_enable $ENABLE_LORA \
                            --generator_output_size 512 \
                            --generator_hidden_dim 8 \
                            --generator_hidden_feature 8 \
                            --key_embed_size 64 \
                            --prompt_top_k 1 \
                            --pool_size 40 \
                            --set_state "gate" \
                            --is_prompt False \
                            --use_task_vector False \
                            --round_to_eval ${ROUND_TO_EVALS[$index]} \
                            --eval_iter $ITER_TO_EVAL \
                            --dataset_name $DATASET_NAME \
                            --setup "" \
                            --ordering $ORDER \
                            --shot $SHOT \
                            --eval_shot $EVAL_SHOT \
                            --attn_impl $ATTEN_IMPLE \
                            --use_mtv $USE_MTV \
                            --mtv_path $MTV_PATH \
                            --use_pai $USE_PAI \
                            --mtv_activation_path $MTV_ACT_PATH \
                            --use_vis_token_amplifier $USE_VIS_TOKEN_AMPLIFIER \
                            --vis_token_amplifier $VIS_TOKEN_AMPLIFIER \
                            --use_cotprompt $USE_COTPROMPT \
                            --cotprompt_ver $COTPROMPT_VER \
                            --use_icot $USE_ICOT \
                            --output_dir "./nohup" #> ./nohup2/fv_extract_${NOTE}.log 2>&1
                    done
                done
            done
        done
    done
done
