#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from typing import List, Optional, Tuple, Union, Dict
from transformers.processing_utils import Unpack
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
# from transformers.modeling_flash_attention_utils import FlashAttention

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import *
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer
import copy
# from transformers.masking_utils import create_causal_mask


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    img_pos_ranges: Optional[object] = None,
    use_dara: bool = False,
    dara_amplifier: Optional[nn.Module] = None,
    **kwargs
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    # DARA (logits column scaling) — only if explicitly enabled
    if use_dara:
        triples = []
        if isinstance(img_pos_ranges, (list, tuple)) and len(img_pos_ranges) > 0:
            next_slot = 0
            for rng in img_pos_ranges:
                if len(rng) == 2:
                    s, e = int(rng[0]), int(rng[1])
                    triples.append((0, s, e, next_slot))  # auto slot 0,1,2,...
                    next_slot += 1
                elif len(rng) == 3:
                    s, e, sl = int(rng[0]), int(rng[1]), int(rng[2])
                    triples.append((0, s, e, sl))
                else:
                    raise ValueError("img_pos_ranges element must be (s,e) or (s,e,slot)")
            img_pos_tensor = torch.as_tensor(triples, device=attn_weights.device, dtype=torch.long)

            finite_mask = torch.isfinite(attn_weights)  # defensive
            attn_weights = dara_amplifier(attn_weights, img_pos_tensor)
            attn_weights = torch.where(finite_mask, attn_weights, attn_weights.new_full((), float("-inf")))


    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class AmplifiedAttentionHeadFactor(nn.Module):
    def __init__(self, attention_amplify_head_ind, max_images: int = 0):
        """
        attention_amplify_head_ind: iterable of head indices to amplify.
        max_images: number of columns for image slots (amplify_factor's 2nd dim).
        """
        super().__init__()
        # Store selected heads as a buffer (compile-friendly; not a Parameter)
        head_idx = list(map(int, attention_amplify_head_ind))
        self.register_buffer(
            "head_idx_tensor",
            torch.as_tensor(head_idx, dtype=torch.long),
            persistent=False,
        )
        if max_images > 0:
            # Learnable amplification matrix: [#selected_heads, max_images], init to 1.0
            self.amplify_factor = nn.Parameter(
                torch.ones(len(head_idx), max_images, dtype=torch.float32),
                requires_grad=True,
            )

    @torch.no_grad()
    def _interval_mask_single(self, B, K, b: int, s: int, e: int, device, dtype):
        """Return a [B,1,1,K] mask for a single inclusive interval [s, e] on batch b."""
        s = max(0, min(int(s), K - 1))
        e = max(0, min(int(e), K - 1))
        if e < s:
            s, e = e, s
        mask = torch.zeros((B, 1, 1, K), device=device, dtype=dtype)
        mask[b, 0, 0, s:e + 1] = 1
        return mask

    def forward(self, attention_scores: torch.Tensor, img_pos: torch.Tensor):
        """
        attention_scores: [B, H, Q, K] attention logits BEFORE mask & softmax.
        img_pos: LongTensor [N, 4], each row = (batch_idx, start, end, slot), inclusive [start, end].
                - slot selects the column in amplify_factor (modulo max_images if desired).
                - If multiple rows share same slot for the same batch, their masks add up (they are unioned).
        """
        B, H, Q, K = attention_scores.shape
        device = attention_scores.device
        dtype  = attention_scores.dtype

        if img_pos is None or img_pos.numel() == 0:
            return attention_scores

        img_pos = img_pos.to(device=device)  # <-- device only!
        # expected shape [N, 4]
        if img_pos.size(1) != 4:
            raise ValueError("img_pos must be of shape [N, 4] with (batch_idx, start, end, slot).")

        # Start from ones so columns outside any image keep scale 1.
        # We accumulate per-row contribution: combined = 1 + Σ_row ( delta(head, slot) * mask_row )
        combined = torch.ones((B, H, 1, K), device=device, dtype=dtype)

        # Prebuild a "ones per head" baseline and the selected-heads index once.
        heads_all_ones = torch.ones((H,), device=device, dtype=dtype)

        # Loop over rows (N = #image intervals across batch). Typically small.
        for i in range(img_pos.size(0)):
            b = int(img_pos[i, 0].item())
            s = int(img_pos[i, 1].item())
            e = int(img_pos[i, 2].item())
            raw_slot = int(img_pos[i, 3].item())

            # Map slot into [0, max_images-1]
            col = raw_slot % self.max_images

            # Build the interval mask for this row: [B,1,1,K]
            mask_row = self._interval_mask_single(B, K, b, s, e, device, dtype)

            # Build per-head scale for this row.
            # Default = 1 for all heads; selected heads use learnable factors at column `col`.
            scales_H = heads_all_ones.clone()
            if self.head_idx_tensor.numel() > 0:
                # NOTE: fix the earlier bug — use dtype=... not device=...
                learnable_col = self.amplify_factor[:, col].to(dtype=dtype)  # [#selected_heads]
                scales_H.scatter_(0, self.head_idx_tensor, learnable_col)

            # delta = (scale - 1) per head → [1,H,1,1] for broadcast.
            delta = (scales_H - 1.0).view(1, H, 1, 1)

            # Accumulate: 1 + delta * mask_row
            combined = combined + delta * mask_row  # broadcast over B,H,K

        # Multiply attention logits in-place (safe for non-leaf tensors)
        attention_scores.mul_(combined)  # [B,H,Q,K] *= [B,H,1,K]

        return attention_scores
    

class My_Qwen2ForCausalLM(Qwen2ForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, max_images=1):
        super().__init__(config)
        self.model = My_Qwen2Model(config, max_images=max_images)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        img_pos_ranges: Optional[object] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class My_Qwen2Attention(Qwen2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int, use_dara: bool=False, max_images: int=0):
        super().__init__(config, layer_idx)
        self.max_images = max_images
        self.num_heads = config.num_attention_heads
        if use_dara:
            # Amplifier used by DARA (select all heads by default)
            self._amp_head_idx = list(range(self.num_heads))
            self.amplifier = AmplifiedAttentionHeadFactor(self._amp_head_idx, max_images=max_images)
        self.use_dara = use_dara
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        img_pos_ranges: Optional[object] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        print("output_attn2", kwargs["output_attentions"])
        
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
                
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=sliding_window,  # main diff with Llama
                **kwargs,
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=sliding_window,  # main diff with Llama
                img_pos_ranges=img_pos_ranges,
                use_dara=self.use_dara,
                dara_amplifier=getattr(self, "amplifier", None),
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights



class My_Qwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int, use_dara: bool=False, max_images: int=0):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.self_attn = My_Qwen2Attention(config=config, layer_idx=layer_idx, use_dara=use_dara, max_images=max_images)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        print("output_attention3", output_attentions)

        return outputs

class My_Qwen2Model(Qwen2Model):
    def __init__(self, config: Qwen2Config, use_dara=False, max_images=0):
        super().__init__(config)

        if use_dara:
            self.layers = nn.ModuleList([
                My_Qwen2DecoderLayer(
                    config,
                    layer_idx,
                    use_dara=use_dara if layer_idx == 0 else False,
                    max_images=max_images if layer_idx == 0 else 0,
                )
                for layer_idx in range(config.num_hidden_layers)
            ])
        else:
            self.layers = nn.ModuleList(
                [My_Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
   
class LlavaQwenModel(LlavaMetaModel, My_Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config, use_dara=False, max_images=0):
        super(LlavaQwenModel, self).__init__(config, use_dara=use_dara, max_images=max_images)


class LlavaQwenForCausalLM(My_Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config, use_dara=False, max_images=0):
        # super(Qwen2ForCausalLM, self).__init__(config)
        My_Qwen2ForCausalLM.__init__(self, config, max_images=max_images)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        self.model = LlavaQwenModel(config, use_dara=use_dara, max_images=max_images)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        img_pos_ranges=None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, img_pos_ranges) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, img_pos_ranges)
        
        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                img_pos_ranges=img_pos_ranges
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, img_pos_ranges) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities)
            kwargs["img_pos_ranges"] = img_pos_ranges
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
