import torch
from typing import Dict, List, Optional, Sequence, Literal, Union

import logging
import transformers
# import editdistance
import numpy as np

from typing import Dict, List

logger = logging.getLogger(__name__)


def dtw(series_1, series_2, norm_func=np.linalg.norm):
    """code refer to: https://github.com/fanqiwan/FuseAI/blob/main/FuseLLM/src/utils/others.py#L318"""

    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(matrix[i, j + 1], matrix[i + 1, j], matrix[i, j])
    matrix = matrix[1:, 1:]
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    matches = []
    mappings_series_1 = [list() for v in range(matrix.shape[0])]
    mappings_series_2 = [list() for v in range(matrix.shape[1])]
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix


def greedy_dynamic_matching(base_model_tokens, blending_model_tokens, base_model_sp_t, blending_model_sp_t):
    l1 = len(base_model_tokens)
    l2 = len(blending_model_tokens)

    base_model_tokens = [token.replace(base_model_sp_t, "") for token in base_model_tokens]
    blending_model_tokens = [token.replace(blending_model_sp_t, "") for token in blending_model_tokens]

    dp = np.full((l1 + 1, l2 + 1), -1000000000, dtype="int32")
    matched_left = np.full((l1, l2), -1, dtype="int32")
    matched_right = np.full((l1, l2), -1, dtype="int32")
    trans_left = np.full((l1 + 1, l2 + 1), -1, dtype="int32")
    trans_right = np.full((l1 + 1, l2 + 1), -1, dtype="int32")

    # this can be optimizer use suffix data structure, but naive implemented for fast trial , it will be optimize later.
    for i in range(l1):
        for j in range(l2):
            if base_model_tokens[i] == blending_model_tokens[j]:
                matched_left[i][j] = 1
                matched_right[i][j] = 1
                continue

            i2, j2 = i, j
            t1 = ""
            t2 = ""
            sq_l1, sq_l2 = 0, 0
            while i2 >= 0 and j2 >= 0:
                if len(t1) > len(t2):
                    t2 = blending_model_tokens[j2] + t2
                    sq_l2 += 1
                    j2 -= 1
                elif len(t1) < len(t2):
                    t1 = base_model_tokens[i2] + t1
                    sq_l1 += 1
                    i2 -= 1
                else:
                    if sq_l1 == 0:
                        sq_l1 += 1
                        sq_l2 += 1
                        t1 += base_model_tokens[i2]
                        t2 += blending_model_tokens[j2]
                        i2 -= 1
                        j2 -= 1
                        continue
                    if t1 == t2:
                        matched_left[i][j] = sq_l1
                        matched_right[i][j] = sq_l2
                    break

    """
    always shortest matching
    """
    for i in range(0, l1 + 1):
        dp[i][0] = 0

    for j in range(0, l2 + 1):
        dp[0][j] = 1

    for i in range(0, l1):
        for j in range(0, l2):
            if matched_left[i][j] == -1:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
                if dp[i + 1][j + 1] == dp[i + 1][j]:
                    trans_right[i + 1][j + 1] = j
                else:
                    trans_left[i + 1][j + 1] = i
            else:
                l_len = matched_left[i][j]
                r_len = matched_right[i][j]
                dp[i + 1][j + 1] = max(max(dp[i + 1][j], dp[i][j + 1]), dp[i + 1 - l_len][j + 1 - r_len] + l_len)
                if dp[i + 1][j + 1] == dp[i + 1 - l_len][j + 1 - r_len] + l_len:
                    trans_left[i + 1][j + 1] = i + 1 - l_len
                    trans_right[i + 1][j + 1] = j + 1 - r_len
                    assert l_len > 0 and r_len > 0
                elif dp[i + 1][j + 1] == dp[i + 1][j]:
                    trans_right[i + 1][j + 1] = j
                else:
                    trans_left[i + 1][j + 1] = i

    i, j = l1, l2
    matches = []
    while i > 0 and j > 0:
        if trans_left[i][j] != -1 and trans_right[i][j] != -1:
            l = trans_left[i][j]
            r = trans_right[i][j]
            matches.append([(l, i - 1), (r, j - 1)])
            i, j = l, r
        elif trans_left[i][j] < 0:
            j -= 1
        else:
            i -= 1

    matches.reverse()
    return matches

# ----------------------------------------------
# 1.  Alignment helper (DTW / greedy_dp)
# ----------------------------------------------
def _build_alignment(
    base_tokens: Sequence[str],
    blend_tokens: Sequence[str],
    *,                          # keyword‑only
    base_special: str,
    blend_special: str,
    strategy: Literal["dtw", "greedy_dp"] = "greedy_dp",
):
    """
    Return `base_to_blending`, a list where base_to_blending[i] is a *list*
    of positions in the blending sequence that correspond to base token *i*.
    """
    if strategy == "dtw":
        import editdistance

        def dist(a, b):
            return editdistance.eval(
                a.replace(blend_special, ""),
                b.replace(base_special, ""),
            )

        _d, _cost, _acc, base2blend, _path = dtw(
            blend_tokens, base_tokens, norm_func=dist
        )
        return base2blend

    elif strategy == "greedy_dp":
        matches = greedy_dynamic_matching(
            base_tokens, blend_tokens, base_special, blend_special
        )

        base2blend: List[List[int]] = [[] for _ in range(len(base_tokens))]
        for (b0, b1), (l0, l1) in matches:
            for b in range(b0, b1 + 1):
                base2blend[b].extend(range(l0, l1 + 1))
        return base2blend

    else:
        raise ValueError(strategy + " not implemented")


# ----------------------------------------------
# 2.  The main routine
# ----------------------------------------------
def align_hidden(
    hidden: Union[
        torch.Tensor,       # shape (layers, seq, dim) *or* (batch, layers, seq, dim)
        Sequence[torch.Tensor],  # list of (batch?, seq, dim)
    ],
    base2blend,
    multi_token_reduce: Literal["mean", "sum", "last"] = "mean",
):
    """
    Align *all requested layers* of `blending_model_hidden_states` to the
    base‑model tokenization **using one shared mapping**.

    Returns
    -------
    torch.Tensor
        shape = (n_layers, seq_len_base, hidden_out)
        where hidden_out = projection.out_features if projection else hidden_dim_blend
    """

    seq_blend, hidden_dim_blend = hidden.shape

    pooled: List[torch.Tensor] = []
    discard_idx = []
    for idx, blend_pos_list in enumerate(base2blend):
        if not blend_pos_list:
            pooled.append(torch.zeros(hidden_dim_blend, device=hidden.device))
            discard_idx.append(idx)
        elif len(blend_pos_list) == 1:
            pooled.append(hidden[blend_pos_list[0]])
        else:
            sub = hidden[blend_pos_list]      # (k, D)
            if multi_token_reduce == "mean":
                pooled.append(sub.mean(dim=0))
            elif multi_token_reduce == "sum":
                pooled.append(sub.sum(dim=0))
            elif multi_token_reduce == "last":
                pooled.append(sub[-1])
            else:
                raise ValueError(multi_token_reduce)
    layer_out = torch.stack(pooled, dim=0)        # (S_base, D)
    keep_idx = [
        idx for idx in range(len(layer_out)) if idx not in discard_idx
    ]  # indices of kept tokens
    return layer_out, keep_idx

def transform_step_logits(base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
                          blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
                          base_model_vocab: Dict[str, int],
                          base_model_input_ids: List[int],
                          blending_model_input_ids: List[int],
                          blending_model_per_step_logits: List[List[float]],
                          blending_model_per_step_indices: List[List[int]],
                          blending_to_base_mapping: Dict[str, str] = None,
                          align_strategy: str = "dtw"
                          ):
    """modified from https://github.com/fanqiwan/FuseAI/blob/main/FuseLLM/src/utils/others.py#L364"""
    """Align blending model per step logits & indices with base model."""
    base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
    blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(blending_model_input_ids)
    base_model_special_token = 'Ġ'#TOKENIZER_TO_SPECIAL_TOKEN[base_model_tokenizer.__class__]
    blending_model_special_token = 'Ġ'#TOKENIZER_TO_SPECIAL_TOKEN[blending_model_tokenizer.__class__]

    aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = [], []
    if align_strategy == "dtw":
        def dist_fn(a, b):
            """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
            return editdistance.eval(a.replace(blending_model_special_token, ''),
                                     b.replace(base_model_special_token, ''))

        _, _, _, base_to_blending, _ = dtw(blending_model_tokens, base_model_tokens, norm_func=dist_fn)
        for i, blending_idx in enumerate(base_to_blending):
            aligned_blending_model_per_step_logit = []
            aligned_blending_model_per_step_index = []
            if len(blending_idx) == 1:  # one base token map to one blending token
                j = blending_idx[0]
                base_token = base_model_tokens[i]
                blending_token = blending_model_tokens[j].replace(blending_model_special_token,
                                                                  base_model_special_token)
                if (
                    blending_model_tokenizer.__class__ == transformers.GPTNeoXTokenizerFast
                    or blending_model_tokenizer.__class__ == transformers.GPT2TokenizerFast) and i == 0 and base_token.startswith(
                    base_model_special_token) and not blending_token.startswith(base_model_special_token):
                    blending_token = base_model_special_token + blending_token  # special case for mpt

                if (base_token == blending_token) or (
                        blending_token in blending_to_base_mapping and base_token == blending_to_base_mapping[
                    blending_token]):  # find the aligned mapping, use the corresponding logits
                    # the logits and indices at this step
                    for blending_logit, blending_index in zip(blending_model_per_step_logits[j],
                                                              blending_model_per_step_indices[j]):
                        # the token corresponds to the logit and indices
                        blending_t = blending_model_tokenizer.convert_ids_to_tokens([blending_index])[0].replace(
                            blending_model_special_token, base_model_special_token)
                        blending_t = blending_to_base_mapping[blending_t]
                        if blending_t in base_model_vocab:
                            aligned_index = base_model_vocab[blending_t]  # the index of the token in base model vocab
                            if aligned_index not in aligned_blending_model_per_step_index:
                                aligned_blending_model_per_step_index.append(aligned_index)
                                aligned_blending_model_per_step_logit.append(blending_logit)
                        else:
                            logger.warning(f"blending_t: {blending_t} not in base_model_vocab!")
                else:  # find error aligned mapping, use the one-hot logits
                    aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                    aligned_blending_model_per_step_logit.append(1.0)
            else:  # one base token map to multiple blending token, in this case only fit base token. use the one-hot logits
                base_token = base_model_tokens[i]
                aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                aligned_blending_model_per_step_logit.append(1.0)
            aligned_blending_model_per_step_indices.append(aligned_blending_model_per_step_index)
            aligned_blending_model_per_step_logits.append(aligned_blending_model_per_step_logit)
    elif align_strategy == "greedy_dp":
        matches = greedy_dynamic_matching(base_model_tokens, blending_model_tokens, base_model_special_token, blending_model_special_token)
        fusion_logits = [[] for _ in range(len(matches))]
        fusion_indices = [[] for _ in range(len(matches))]
        match_pos = [-1] * len(base_model_tokens)
        used = [False] * len(matches)

        for idx, ((start_pos_1, end_pos_1), (start_pos_2, end_pos_2)) in enumerate(matches):
            fusion_dict = dict()
            fusion_counter_dict = dict()
            for blending_pos in range(start_pos_2, end_pos_2 + 1):
                for blending_logit, blending_index in zip(blending_model_per_step_logits[blending_pos],
                                                          blending_model_per_step_indices[blending_pos]):
                    if blending_index not in fusion_dict:
                        fusion_dict[blending_index] = 0
                        fusion_counter_dict[blending_index] = 0
                    fusion_dict[blending_index] += blending_logit
                    fusion_counter_dict[blending_index] += 1

            for j in range(start_pos_1, end_pos_1 + 1):
                match_pos[j] = idx

            for token_index, token_logit in fusion_dict.items():
                fusion_logits[idx].append(token_logit / fusion_counter_dict[token_index])
                fusion_indices[idx].append(token_index)

        for i in range(len(base_model_tokens)):
            aligned_blending_model_per_step_logit = []
            aligned_blending_model_per_step_index = []
            if match_pos[i] == -1 or used[match_pos[i]]:
                base_token = base_model_tokens[i]
                aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                aligned_blending_model_per_step_logit.append(1.0)
            else:
                pos = match_pos[i]
                used[pos] = True
                for blending_logit, blending_index in zip(fusion_logits[pos],
                                                          fusion_indices[pos]):
                    # the token corresponds to the logit and indices
                    blending_t = blending_model_tokenizer.convert_ids_to_tokens([blending_index])[0].replace(
                        blending_model_special_token, base_model_special_token)
                    blending_t = blending_to_base_mapping[blending_t]
                    if blending_t in base_model_vocab:
                        aligned_index = base_model_vocab[blending_t]  # the index of the token in base model vocab
                        if aligned_index not in aligned_blending_model_per_step_index:
                            aligned_blending_model_per_step_index.append(aligned_index)
                            aligned_blending_model_per_step_logit.append(blending_logit)
                    else:
                        logger.warning(f"blending_t: {blending_t} not in base_model_vocab!")
            aligned_blending_model_per_step_indices.append(aligned_blending_model_per_step_index)
            aligned_blending_model_per_step_logits.append(aligned_blending_model_per_step_logit)
    else:
        raise ValueError(f"{align_strategy} not implemented yet.")

    return aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices
