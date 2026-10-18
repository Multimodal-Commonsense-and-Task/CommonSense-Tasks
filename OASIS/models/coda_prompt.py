import torch
import torch.nn as nn
import copy

class CodaPrompt(nn.Module):
    def __init__(self, length=4, embed_dim=1024, key_dim=1024, embedding_key='cls', prompt_init='uniform', prompt_pool=True, 
                 prompt_key=True, pool_size=100, top_k=5, batchwise_prompt=True, prompt_key_init='uniform',n_tasks=4):
        super().__init__()
        self.task_count=-1
        self.n_tasks=n_tasks
        self.length = length
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        p = tensor_prompt(self.pool_size, length, embed_dim) #, ones=True
        k = tensor_prompt(self.pool_size, self.key_dim)
        a = tensor_prompt(self.pool_size, self.key_dim)
        # p = self.gram_schmidt(p)
        # k = self.gram_schmidt(k)
        # a = self.gram_schmidt(a)
        self.P = p
        self.K = k
        self.A = a
    
    def process_task_count(self, task_id):
        self.task_count = task_id

        self.K = self.gram_schmidt(self.K)
        self.A = self.gram_schmidt(self.A)
        self.P = self.gram_schmidt(self.P)
    
    def forward(self, x_embed):
        # x_embed = img encoder output
        # improvement - add text embedding to choose keys
        K = self.K
        A = self.A
        p = self.P
        
        pt = int(self.pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        
        # freeze/control past tasks
        if self.training:
            if self.task_count > 0:
                K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
            else:
                K = K[s:f]
                A = A[s:f]
                p = p[s:f]
        # else:
        #     K = K[0:f]
        #     A = A[0:f]
        #     p = p[0:f]

        # with attention and cosine sim
        # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
        a_querry = torch.einsum('bd,kd->bkd', x_embed, A)
        # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
        n_K = nn.functional.normalize(K, dim=1)
        q = nn.functional.normalize(a_querry, dim=2)
        aq_k = torch.einsum('bkd,kd->bk', q, n_K)
        # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
        P_ = torch.einsum('bk,kld->bld', aq_k, p)
        P_ += 1.0
        return P_
    
    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ones=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ones:
        nn.init.ones_(p)
    else:
        nn.init.uniform_(p)
    return p    
