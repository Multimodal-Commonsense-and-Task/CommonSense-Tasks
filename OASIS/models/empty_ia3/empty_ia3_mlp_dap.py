import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import ACT2FN

class LlamaEmptyIA3DAPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        
        self.lang_prompt_downsample = nn.Linear(576, 1)
        nn.init.zeros_(self.lang_prompt_downsample.weight)
        nn.init.zeros_(self.lang_prompt_downsample.bias)
        # self.lang_prompt_film = nn.Linear(config.key_embed_size, self.intermediate_size * 2)
        self.lang_prompt_film = nn.Sequential(
            nn.Linear(config.key_embed_size, config.generator_hidden_feature, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(config.generator_hidden_feature, self.intermediate_size*2, bias=False),
        )
        self.lang_prompt_norm = nn.LayerNorm(self.intermediate_size, eps=1e-6)

    def forward(self, x, image_feature_indices=None, task_id_estimated_emb=None):
        bsz = x.shape[0]
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        
        if image_feature_indices is not None:
            image_features = []
            for i in range(bsz):
                img_feat = torch.stack(torch.split(x[i,image_feature_indices[i], :], 576, dim=0), dim=0)
                img_feat = torch.mean(img_feat, dim=0)
                image_features.append(img_feat)
                
            image_features = torch.stack(image_features, dim=0)
            img_norm = self.lang_prompt_norm(image_features)
            img_norm = torch.transpose(img_norm, 2, 1)
            down = self.lang_prompt_downsample(img_norm)
            
            film = self.lang_prompt_film(task_id_estimated_emb)
            gamma4 = film[:, :self.intermediate_size]
            beta4 = film[:, self.intermediate_size:]
            gamma_norm = gamma4.norm(p=2, dim=1, keepdim=True).detach()
            beta_norm = beta4.norm(p=2, dim=1, keepdim=True).detach()

            gamma4 = gamma4.div(gamma_norm).view(film.size(0), -1, 1)
            beta4 = beta4.div(beta_norm).view(film.size(0), -1, 1)
            down = gamma4 * down + beta4
            query_embeds = torch.transpose(down, 2, 1)
        else:
            query_embeds = None
        down_proj = self.down_proj(x, query_embeds=query_embeds)

        return down_proj