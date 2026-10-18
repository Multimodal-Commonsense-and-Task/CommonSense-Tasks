# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, List, Optional

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils import transpose
from models.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
import torch.nn.functional as F

def init_weights_to_zero(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class PromptMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 8,
        bias: bool = False,
        dropout: float = 0.0,
        activation: str = "relu",
        is_forward: bool = False,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.len = 1
        self.is_forward = is_forward

        # non_linearity = nn.ReLU(inplace=True)
        # if activation == "sigmoid":
        #     non_linearity = nn.Sigmoid()
        # elif activation == "attention":
        #     non_linearity = nn.Softmax(dim=-1)
        non_linearity = nn.SiLU(inplace=True)

        self.block = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features, bias=bias),
            non_linearity,
            nn.Linear(self.hidden_features, self.out_features * self.len, bias=bias),
        )
        if dropout > 0.0:
            self.block[1].register_forward_hook(
                lambda m, inp, out: F.dropout(out, p=dropout, training=m.training)
            )
        # self.block[0].apply(init_weights_to_zero)
        # self.block[0].weight.data = torch.randn(self.block[0].weight.size()) * 0.01
        
        # Initialize fc2 to output zeros initially
        self.block[2].weight.data.fill_(0)

    def forward(self, x: torch.Tensor):
        bsz = x.size(0)
        out = self.block(x)
        if self.is_forward:
            out = out.reshape(bsz, self.len, self.out_features)
        else:
            out = out.reshape(bsz, self.out_features, self.len)

        return out

class DualEVOIA3Layer2(BaseTunerLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("ia3_generator_1","ia3_generator_2")

    def __init__(self, base_layer: nn.Module, is_feedforward: bool, **kwargs) -> None:
        self.base_layer = base_layer
        self.ia3_generator_1 = nn.ParameterDict({})
        self.ia3_generator_2 = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.is_feedforward = is_feedforward

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        self.in_features = in_features
        self.out_features = out_features
        
        self.active_state = 'lora1'

    def update_layer(self, adapter_name, init_ia3_weights):
        # This code works for linear layers, override for other layer types
        # Actual trainable parameters
        # self.ia3_generator_1[adapter_name] = PromptMLP(512,self.in_features,hidden_features=32, is_forward=self.is_feedforward)
        # self.ia3_generator_2[adapter_name] = PromptMLP(512,self.in_features,hidden_features=32, is_forward=self.is_feedforward)
        
        # self.to(self.get_base_layer().weight.device)
        # self.set_adapter(self.active_adapters)
        pass

    def reset_ia3_parameters(self, adapter_name):
        if adapter_name in self.ia3_l.keys():
            # initialize learned vector with torch.ones
            nn.init.constant_(self.ia3_l[adapter_name], 1.0)

    def set_state(self, state):
        assert state in ['lora1', 'lora2', 'gate'], state
        self.active_state = state

    def activate_all(self):
        for p in self.ia3_generator_1.parameters():
            p.requires_grad = True
        for p in self.ia3_generator_2.parameters():
            p.requires_grad = True

    def activate_lora1(self):
        for p in self.ia3_generator_1.parameters():
            p.requires_grad = True
        for p in self.ia3_generator_2.parameters():
            p.requires_grad = False
    
    def activate_lora2(self):
        for p in self.ia3_generator_1.parameters():
            p.requires_grad = False
        for p in self.ia3_generator_2.parameters():
            p.requires_grad = True

class Linear(nn.Module, DualEVOIA3Layer2):
    # (IA)^3 implemented in a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_feedforward: bool = False,  # Set to True if the layer is treated as a feedforward layer
        is_target_conv_1d_layer: bool = False,  # whether target module is a conv1d layer. useful while unloading later
        init_ia3_weights: bool = True,  # whether to initialize IA3 weights
        **kwargs,
    ) -> None:
        super().__init__()
        DualEVOIA3Layer2.__init__(self, base_layer, is_feedforward=is_feedforward)
        self.fan_in_fan_out = fan_in_fan_out
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, init_ia3_weights)
        
        self.prev_ia3_scaling = None

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.ia3_l.keys():
                base_layer = self.get_base_layer()
                ia3_l = transpose(self.ia3_l[active_adapter].mean(dim=0).data, self.fan_in_fan_out)
                if safe_merge:
                    orig_weights = base_layer.weight.data
                    orig_weights = torch.mul(orig_weights, ia3_l)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = torch.mul(base_layer.weight.data, ia3_l)

                if not self.is_feedforward and (base_layer.bias is not None):
                    scaling = self.ia3_l[active_adapter].mean(dim=0).reshape(base_layer.bias.shape)
                    base_layer.bias.data = torch.mul(base_layer.bias.data, scaling.data)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        warnings.warn("Unmerge result can be inaccurate for (IA)^3.")
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.ia3_l.keys():
                base_layer = self.get_base_layer()
                # Add tolerace to avoid division by zero
                ia3_l = transpose(self.ia3_l[active_adapter].data, self.fan_in_fan_out) + 1e-8
                base_layer.weight.data = torch.div(base_layer.weight.data, ia3_l)

                if not self.is_feedforward and (base_layer.bias is not None):
                    scaling = self.ia3_l[active_adapter].reshape(base_layer.bias.shape)
                    base_layer.bias.data = torch.div(base_layer.bias.data, scaling.data + 1e-8)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        dtype = previous_dtype = x.dtype
        ia3_scaling = ia3_delta_2 = None
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x)#, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x)#, *args, **kwargs)
        else:
            query_embeds = kwargs['query_embeds'] if 'query_embeds' in kwargs.keys() else None
            ia3_scaling = 1
            # for active_adapter in self.active_adapters:
            #     if active_adapter not in self.ia3_generator_1.keys():
            #         continue
                # dtype = self.ia3_generator_1[active_adapter].block[0].weight.dtype

            bs = x.shape[0]
            if query_embeds is not None:
                query_embeds_1, query_embeds_2 = query_embeds
                if self.active_state == 'lora1':
                    # ia3_delta = self.ia3_generator_1[active_adapter](query_embeds_1)
                    ia3_delta = ia3_delta_2 = query_embeds_1
                elif self.active_state == 'lora2':
                    # ia3_delta = self.ia3_generator_2[active_adapter](query_embeds_2)
                    ia3_delta = ia3_delta_2 = query_embeds_2
                elif self.active_state == 'gate':
                    # ia3_delta_1 = self.ia3_generator_1[active_adapter](query_embeds_1)
                    ia3_delta_1 = query_embeds_1

                    # ia3_delta_2 = self.ia3_generator_2[active_adapter](query_embeds_2)
                    ia3_delta_2 = query_embeds_2
                    
                    # FIXME: gumbel softmax combining
                    ia3_delta = (ia3_delta_1 + ia3_delta_2)/2
                    # ia3_delta = selective_masking(ia3_delta_1, ia3_delta_2)
                    # ia3_delta, gumbel_out = create_mask_gumbel(ia3_delta_1, ia3_delta_2, is_training = self.training)
                
                if self.is_feedforward:
                    weight = torch.ones((bs,1, self.in_features)).to(x.device)
                    ia3_delta = ia3_delta.reshape((bs,1,self.in_features))
                else:
                    weight = torch.ones((bs,self.in_features, 1)).to(x.device)
                    ia3_delta = ia3_delta.reshape((bs,self.in_features,1))
                
                ia3 = weight + ia3_delta
                
                ia3_scaling *= ia3.reshape(bs, -1)
                
                self.prev_ia3_scaling = ia3_scaling.clone()
            else:
                ia3_scaling = self.prev_ia3_scaling
                ia3_delta_2 = self.prev_ia3_scaling
                # self.prev_ia3_scaling = None

            if self.is_feedforward:
                x = x.to(dtype)
                # TODO: weight.dtype can be != self.ia3_l[self.active_adapters].dtype
                # e.g. bf16 vs fp32. Is that okay?
                interm = (x * ia3_scaling.unsqueeze(1)).to(self.get_base_layer().weight.dtype)
                result = self.base_layer(interm)
            else:
                result = self.base_layer(x)
                result = result.to(dtype) * ia3_scaling.unsqueeze(1)
                
            ia3_delta_2 = ia3_delta_2.reshape(bs, -1)
        
        result = result.to(previous_dtype)
        # return result, (ia3_scaling, ia3_delta_2.reshape(bs, -1))
        return result, (ia3_scaling, ia3_delta_2)

import torch.nn.functional as F
def create_mask_gumbel(tensor1, tensor2, tau=1.0, hard=True, is_training=False):
    # Initialize logits for each condition
    logits = torch.zeros(tensor1.size(0), tensor1.size(1), 3).to(tensor1.device)  # 3 categories: tensor1, tensor2, average
    
    # Condition masks
    both_greater_than_1 = (tensor1 >= 0) & (tensor2 >= 0)
    both_smaller_than_1 = (tensor1 <= 0) & (tensor2 <= 0)
    one_greater_one_smaller = (tensor1 > 0) & (tensor2 < 0) | (tensor1 < 0) & (tensor2 > 0)
    
    # For both_greater_than_1
    indices = both_greater_than_1.nonzero(as_tuple=True)
    logits[indices[0], indices[1], 0] = (tensor1[indices[0], indices[1]] >= tensor2[indices[0], indices[1]]).float()
    logits[indices[0], indices[1], 1] = (tensor2[indices[0], indices[1]] >= tensor1[indices[0], indices[1]]).float()
    
    # For both_smaller_than_1
    indices = both_smaller_than_1.nonzero(as_tuple=True)
    logits[indices[0], indices[1], 0] = (tensor1[indices[0], indices[1]] <= tensor2[indices[0], indices[1]]).float()
    logits[indices[0], indices[1], 1] = (tensor2[indices[0], indices[1]] <= tensor1[indices[0], indices[1]]).float()
    
    # For one_greater_one_smaller 
    indices = one_greater_one_smaller.nonzero(as_tuple=True)
    logits[indices[0], indices[1], 2] = 1.0  # Average choice for mixed cases
    # logits[..., 2] = 1.0
    
    # Apply Gumbel-Softmax to get the mask
    if is_training:
        gumbel_out = F.gumbel_softmax(logits.to(torch.bfloat16), tau=tau, hard=hard)#.to(torch.bfloat16)
    else:
        gumbel_out = logits.to(torch.bfloat16)
    # Calculate the final result based on gumbel_out                                            
    # result = gumbel_out[..., 0] * tensor1 + gumbel_out[..., 1] * tensor2 + gumbel_out[..., 2] * (0.5 * (tensor1 + tensor2))
    result = gumbel_out[..., 0] * tensor1 + gumbel_out[..., 1] * tensor2 + gumbel_out[..., 2] * tensor2 #-> follow local
    
    return result, gumbel_out

def selective_masking(tensor1, tensor2):
    
    # mask sign conflict
    same_sign = (tensor1*tensor2) >= 0
    
    
    result = tensor1*same_sign + tensor2
    return result