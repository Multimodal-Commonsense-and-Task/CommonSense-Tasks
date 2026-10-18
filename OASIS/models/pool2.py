import torch
import torch.nn as nn

class Pool2(nn.Module):
    def __init__(self, length=1, embed_dim=1024, key_dim=1024, embedding_key='cls', prompt_init='one', prompt_pool=True, 
                 prompt_key=True, pool_size=10, top_k=5, batchwise_prompt=True, prompt_key_init='uniform',):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.prompt_pool = prompt_pool
        self.prompt_init = prompt_init
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'one':
                self.prompt = nn.Parameter(torch.ones(prompt_pool_shape))
                # print(self.prompt)
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                # nn.init.uniform_(self.prompt, -1, 1)
                with torch.no_grad():
                    self.prompt.uniform_(-1, 1)
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, prompt_mask):
        # x_embed = img encoder output
        # improvement - add text embedding to choose keys
        out = dict()
        if self.prompt_pool:
            idx = prompt_mask
            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                with torch.no_grad():
                    self.prompt.uniform_(-1, 1)
            batched_prompt = self.prompt.unsqueeze(0).expand(prompt_mask.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['batched_prompt'] = batched_prompt

        return batched_prompt, out