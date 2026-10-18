# -*- encoding: utf-8 -*-
'''
@File    :   coglm_strategy.py
@Time    :   2021/10/08 22:22:42
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import numpy as np
import torch.nn.functional as F


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-65504):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits

def get_masks_and_position_ids_coglm(seq, context_length):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros(len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[:context_length])
    torch.arange(512, 512 + len(seq) - context_length, 
            out=position_ids[context_length:]
    )

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids

class CoglmStrategy:
    def __init__(self, invalid_slices=[], temperature=1., top_k=200, eps=1e-4, top_p=0.0, end_tokens=None, top_k_cluster=1.):
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = top_k
        self.top_p = top_p
        self.eps = eps
        if end_tokens is None:
            end_tokens = []
        self.end_tokens = end_tokens
        self._is_done = False
        self.outlier_count_down = 5
        self.vis_list = [[]for i in range(16)]
        self.cluster_labels = torch.tensor(np.load('CogView2/cluster_label.npy'), device='cuda', dtype=torch.long)
        self.top_k_cluster = top_k_cluster

    @property
    def is_done(self) -> bool:
        return self._is_done

    def forward(self, logits, tokens, mems, temperature=None):
        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
            
        rprobs = F.softmax(logits.float(), dim=-1)
        c = self.cluster_labels.expand(*rprobs.shape)
        cprobs = torch.zeros(logits.shape[0], 500, device=logits.device).scatter_add_(1, c, rprobs)
        best_scores, best_clusters = cprobs.topk(self.topk)
        bz = logits.shape[0]
        for i in range(bz):
            best_scores[i] = best_scores[i] #** 0.2
            selected_cluster = best_clusters[i][torch.multinomial(best_scores[i] / best_scores[i].sum(), num_samples=1)]
            logits[i, self.cluster_labels != selected_cluster] = -65504
            
        probs = F.softmax(logits.float()/self.top_k_cluster, dim=-1)  # float is essetial, due to a bug in Pytorch
        pred = torch.multinomial(probs, num_samples=1)
        
        if pred.numel() == 1 and pred.item() in self.end_tokens:
            self._is_done = True
        tokens = torch.cat((tokens, pred.view(tokens.shape[0], 1)), dim=1)
        return tokens, mems

    def finalize(self, tokens, mems):
        self._is_done = False
        return tokens, mems

def get_recipe(name):
    r = {
        'attn_plus': 1.4,
        'temp_all_gen': 1.15,
        'topk_gen': 16,
        'temp_cluster_gen': 1.,

        'temp_all_dsr': 1.5,
        'topk_dsr': 100,
        'temp_cluster_dsr': 0.89,

        'temp_all_itersr': 1.3,
        'topk_itersr': 16,
        'query_template': '{}<start_of_image>'
    }
    if name == 'none':
        pass
    elif name == 'mainbody':
        r['query_template'] = '{} 高清摄影 隔绝<start_of_image>'
        
    elif name == 'photo':
        r['query_template'] = '{} 高清摄影<start_of_image>'
        
    elif name == 'flat':
        r['query_template'] = '{} 平面风格<start_of_image>'
        # r['attn_plus'] = 1.8
        # r['temp_cluster_gen'] = 0.75
        r['temp_all_gen'] = 1.1
        r['topk_dsr'] = 5
        r['temp_cluster_dsr'] = 0.4

        r['temp_all_itersr'] = 1
        r['topk_itersr'] = 5
    elif name == 'comics':
        r['query_template'] = '{} 漫画 隔绝<start_of_image>'
        r['topk_dsr'] = 5
        r['temp_cluster_dsr'] = 0.4
        r['temp_all_gen'] = 1.1
        r['temp_all_itersr'] = 1
        r['topk_itersr'] = 5
    elif name == 'oil':
        r['query_template'] = '{} 油画风格<start_of_image>'
        pass
    elif name == 'sketch':
        r['query_template'] = '{} 素描风格<start_of_image>'
        r['temp_all_gen'] = 1.1
    elif name == 'isometric':
        r['query_template'] = '{} 等距矢量图<start_of_image>'
        r['temp_all_gen'] = 1.1
    elif name == 'chinese':
        r['query_template'] = '{} 水墨国画<start_of_image>'
        r['temp_all_gen'] = 1.12
    elif name == 'watercolor':
        r['query_template'] = '{} 水彩画风格<start_of_image>'
    return r