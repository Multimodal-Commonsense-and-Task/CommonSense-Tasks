import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import ACT2FN

class LlamaEVOIA3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x,query_embeds=None):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x), query_embeds=query_embeds)

        return down_proj