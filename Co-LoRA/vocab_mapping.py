import json
import editdistance
import tqdm
import multiprocessing
import logging
from transformers import AutoTokenizer
# from fate_llm.algo.fedmkt.token_alignment.spectal_token_mapping import TOKENIZER_TO_SPECIAL_TOKEN

logger = logging.getLogger(__name__)

import transformers


TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: '▁',
    transformers.LlamaTokenizerFast: '▁',
    transformers.GPTNeoXTokenizerFast: 'Ġ',
    transformers.GPT2TokenizerFast: 'Ġ',
    transformers.GPT2Tokenizer: 'Ġ',
    transformers.BloomTokenizerFast: 'Ġ',
}

def get_tokenizer(
    tokenizer_name_or_path,
    trust_remote_code=False,
    padding_side=None,
    pad_token=None,
    bos_token=None,
    eos_token=None,
    pad_token_id=None,
    bos_token_id=None,
    eos_token_id=None,
    add_eos_token=True,
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=trust_remote_code,
        add_eos_token=add_eos_token,
        use_fast=False,
    )
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    if pad_token is not None:
        tokenizer.add_special_tokens({'pad_token': pad_token})
    if bos_token is not None:
        tokenizer.add_special_tokens({'bos_token': bos_token})
    if eos_token is not None:
        tokenizer.add_special_tokens({"eos_token": eos_token})
    if pad_token_id is not None:
        tokenizer.pad_token_id = pad_token_id
    if bos_token_id is not None:
        tokenizer.bos_token_id = bos_token_id
    if eos_token_id is not None:
        tokenizer.eos_token_id = eos_token_id

    # if "llama" in tokenizer_name_or_path.lower() or "gpt2" in tokenizer_name_or_path.lower():
    #     tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def find_best_mapping(x, base_tokens, blending_model_special_token, base_model_special_token, best_one=True):
    """code refer to https://github.com/fanqiwan/FuseAI/blob/main/FuseLLM/src/utils/vocab_mapping.py#L82"""
    tmp_x = x.replace(blending_model_special_token, base_model_special_token)
    if tmp_x in base_tokens:
        return tmp_x, tmp_x
    else:
        if best_one:
            return tmp_x, min([(y, editdistance.eval(tmp_x, y)) for y in base_tokens], key=lambda d: d[1])[0]
        else:
            token_and_distance = [(y, editdistance.eval(tmp_x, y)) for y in base_tokens]
            min_distance = min(item[1] for item in token_and_distance)
            shortest_distance_tokens = [item[0] for item in token_and_distance if item[1] == min_distance]
            return tmp_x, shortest_distance_tokens


def get_vocab_mappings(model_name_or_path, candidate_model_name_or_path, vocab_mapping_save_path, num_processors=8):
    ori_tokenizer = get_tokenizer(model_name_or_path)
    candidate_tokenizer = get_tokenizer(candidate_model_name_or_path)

    ori_special_tok = 'Ġ'#TOKENIZER_TO_SPECIAL_TOKEN[ori_tokenizer.__class__]
    candidate_special_tok = 'Ġ'#TOKENIZER_TO_SPECIAL_TOKEN[candidate_tokenizer.__class__]

    candidate_tokens = list(candidate_tokenizer.get_vocab().keys())
    with multiprocessing.Pool(num_processors) as process_pool:
        func_args = [(tok, candidate_tokens, ori_special_tok, candidate_special_tok) for tok in ori_tokenizer.get_vocab()]

        vocab_mappings = dict(tqdm.tqdm(process_pool.starmap(find_best_mapping, func_args)),
                              total=len(ori_tokenizer.get_vocab()))

    with open(vocab_mapping_save_path, "w") as fout:
        json.dump(vocab_mappings, fout)

    return vocab_mappings

get_vocab_mappings(
    model_name_or_path="thkim0305/llama3.2_3B_vl",
    candidate_model_name_or_path="thkim0305/qwen2.5_0.5B_vl",
    # model_name_or_path="thkim0305/qwen2.5_0.5B_vl",
    # candidate_model_name_or_path="thkim0305/llama3.2_3B_vl",
    vocab_mapping_save_path="./llama2qwen_vocab_mapping.json",
    num_processors=8
)