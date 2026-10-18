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
import copy

class DualLAEIA3Layer(BaseTunerLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("ia3_l_1_1","ia3_l_1_2", "ia3_l_2_1","ia3_l_2_2")

    def __init__(self, base_layer: nn.Module, is_feedforward: bool, **kwargs) -> None:
        self.base_layer = base_layer
        self.ia3_l_1_1 = nn.ParameterDict({})
        self.ia3_l_1_2 = nn.ParameterDict({})
        self.ia3_l_2_1 = nn.ParameterDict({})
        self.ia3_l_2_2 = nn.ParameterDict({})
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
        
        self.active_lae_state = 'lora2'
        self.active_state = 'lora2'

    def update_layer(self, adapter_name, init_ia3_weights):
        # This code works for linear layers, override for other layer types
        # Actual trainable parameters
        if self.is_feedforward:
            weight = torch.ones((1, self.in_features))
        else:
            weight = torch.ones((self.out_features, 1))
        self.ia3_l_1_1[adapter_name] = nn.Parameter(weight)
        self.ia3_l_1_2[adapter_name] = copy.deepcopy(self.ia3_l_1_1[adapter_name])
        self.ia3_l_2_1[adapter_name] = copy.deepcopy(self.ia3_l_1_1[adapter_name])
        self.ia3_l_2_2[adapter_name] = copy.deepcopy(self.ia3_l_1_1[adapter_name])
        if init_ia3_weights:
            self.reset_ia3_parameters(adapter_name)
        self.to(self.get_base_layer().weight.device)
        self.set_adapter(self.active_adapters)
        
        

    def reset_ia3_parameters(self, adapter_name):
        if adapter_name in self.ia3_l_1_1.keys():
            # initialize learned vector with torch.ones
            init_weight = torch.empty_like(self.ia3_l_1_1[adapter_name].data)
            nn.init.constant_(init_weight, 1.0)
            self.ia3_l_1_1[adapter_name].data.copy_(init_weight)
            self.ia3_l_1_2[adapter_name].data.copy_(init_weight)
            self.ia3_l_2_1[adapter_name].data.copy_(init_weight)
            self.ia3_l_2_2[adapter_name].data.copy_(init_weight)

    def set_lae_state(self, state):
        assert state in ['lora1', 'lora2', 'gate'], state
        self.active_lae_state = state
    
    def set_state(self, state):
        assert state in ['lora1', 'lora2', 'gate'], state
        self.active_state = state

    def activate_all(self):
        for p in self.ia3_l_1_1.parameters():
            p.requires_grad = True
        for p in self.ia3_l_1_2.parameters():
            p.requires_grad = True
        for p in self.ia3_l_2_1.parameters():
            p.requires_grad = True
        for p in self.ia3_l_2_2.parameters():
            p.requires_grad = True
            

    def activate_lora1(self):
        for p in self.ia3_l_1_1.parameters():
            p.requires_grad = True
        for p in self.ia3_l_1_2.parameters():
            p.requires_grad = True
        for p in self.ia3_l_2_1.parameters():
            p.requires_grad = False
        for p in self.ia3_l_2_2.parameters():
            p.requires_grad = False
    
    def activate_lora2(self):
        for p in self.ia3_l_1_1.parameters():
            p.requires_grad = False
        for p in self.ia3_l_1_2.parameters():
            p.requires_grad = False
        for p in self.ia3_l_2_1.parameters():
            p.requires_grad = True
        for p in self.ia3_l_2_2.parameters():
            p.requires_grad = True

class Linear(nn.Module, DualLAEIA3Layer):
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
        DualLAEIA3Layer.__init__(self, base_layer, is_feedforward=is_feedforward)
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

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            # idx = kwargs['idx'] if 'idx' in kwargs.keys() else None
            ia3_scaling = 1
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ia3_l_1_1.keys():
                    continue
                dtype = self.ia3_l_1_1[active_adapter].dtype
                if self.active_state == 'lora1' and self.active_lae_state == 'lora1':
                    ia3_scaling *= self.ia3_l_1_1[active_adapter].flatten()
                elif self.active_state == 'lora1' and self.active_lae_state == 'lora2':
                    ia3_scaling *= self.ia3_l_1_2[active_adapter].flatten()
                elif self.active_state == 'lora1' and self.active_lae_state == 'gate':
                    ia3_scaling *= ((self.ia3_l_1_1[active_adapter]+self.ia3_l_1_2[active_adapter])/2).flatten()
                    
                elif self.active_state == 'lora2' and self.active_lae_state == 'lora1':
                    ia3_scaling *= self.ia3_l_2_1[active_adapter].flatten()
                elif self.active_state == 'lora2' and self.active_lae_state == 'lora2':
                    ia3_scaling *= self.ia3_l_2_2[active_adapter].flatten()
                elif self.active_state == 'lora2' and self.active_lae_state == 'gate':
                    ia3_scaling *= ((self.ia3_l_2_1[active_adapter]+self.ia3_l_2_2[active_adapter])/2).flatten()

                elif self.active_state == 'gate' and self.active_lae_state == 'lora1':
                    ia3_scaling *= ((self.ia3_l_1_1[active_adapter]+self.ia3_l_2_1[active_adapter])/2).flatten()
                elif self.active_state == 'gate' and self.active_lae_state == 'lora2':
                    ia3_scaling *= ((self.ia3_l_1_2[active_adapter]+self.ia3_l_2_2[active_adapter])/2).flatten()
                elif self.active_state == 'gate' and self.active_lae_state == 'gate':
                    ia3_scaling *= ((self.ia3_l_1_1[active_adapter]+self.ia3_l_1_2[active_adapter]+self.ia3_l_2_1[active_adapter]+self.ia3_l_2_2[active_adapter])/4 ).flatten()


            if self.is_feedforward:
                x = x.to(dtype)
                # TODO: weight.dtype can be != self.ia3_l[self.active_adapters].dtype
                # e.g. bf16 vs fp32. Is that okay?
                interm = (x * ia3_scaling).to(self.get_base_layer().weight.dtype)
                result = self.base_layer(interm)
            else:
                result = self.base_layer(x)
                result = result.to(dtype) * ia3_scaling

        result = result.to(previous_dtype)
        return result
