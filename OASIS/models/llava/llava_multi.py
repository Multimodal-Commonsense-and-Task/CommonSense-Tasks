from transformers import LlavaForConditionalGeneration, AutoModel, AutoModelForCausalLM
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast, LlavaMultiModalProjector, LlavaConfig
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers.utils import logging

# from models.dual_ia3.dual_ia3_layer import DualIA3Layer
# from models.duallora.dualloralayer import DualLoraLayer
# from models.duallora_moe.dualmoeloralayer import DualMOELoraLayer
# from models.dual_pqlora.dual_pqloralayer import PQLoraLayer
# from models.dual_pqlora_freeze.dual_pqloralayer_freeze import PQLoraFreezeLayer
# from models.dual_pqlora_freezeA.dual_pqloralayer_freezeA import PQLoraFreezeALayer
# from models.dual_pqlora_freeze_full.dual_pqloralayer_freeze_full import PQLoraFullFreezeLayer
# from models.dual_pqlora_freeze_full_moe.dual_pqloralayer_freeze_full_moe import PQMOELoraFullFreezeLayer
# from models.dual_pqlora_freezeA_full.dual_pqloralayer_freezeA_full import PQLoraFullFreezeALayer
# from models.dual_pqlora_LIL_freeze_full.dual_pqloraLILlayer_freeze_full import PQLoraLILFullFreezeLayer
# from models.dual_plora_freeze_full.dual_ploralayer_freeze_full import PLoraFullFreezeLayer

# from models.quadralora.quadraloralayer import QuadraLoraLayer
# from models.quadralora_moe.quadramoeloralayer import QuadraMOELoraLayer
# from models.quadra_pqlora_freeze_full.quadra_pqloralayer_freeze_full import QuadraPQLoraFullFreezeLayer
# from models.quadra_pqlora_freeze_full_moe.quadra_pqloralayer_freeze_full_moe import QuadraPQMOELoraFullFreezeLayer
# from models.hexalora.hexaloralayer import HexaLoraLayer
# from models.hexalora_moe.hexamoeloralayer import HexaMOELoraLayer

# from models.triplelora.tripleloralayer import TripleLoraLayer
# from models.triple_pqlora_freeze_full.triple_pqloralayer_freeze_full import TriplePQLoraFullFreezeLayer

logger = logging.get_logger(__name__)

class LlavaMultiForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: LlavaConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config, use_flash_attention_2=True)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()
        
        self.active_state = 'gate'
    
    def set_state(self, state):
        assert state in ['lora1', 'lora2', 'gate'], state
        self.active_state = state
        
        for name, module in self.named_modules():
            if isinstance(module, DualLoraLayer) or isinstance(module, DualIA3Layer) or isinstance(module, DualMOELoraLayer) or isinstance(module, TripleLoraLayer) or isinstance(module, TriplePQLoraFullFreezeLayer) \
                or isinstance(module, PQLoraLayer) or isinstance(module, PQLoraFreezeLayer) or isinstance(module, PQLoraFreezeALayer) or isinstance(module, PQLoraLILFullFreezeLayer) \
                or isinstance(module, PQLoraFullFreezeLayer) or isinstance(module, PQLoraFullFreezeALayer) or isinstance(module, PQMOELoraFullFreezeLayer) or isinstance(module, PLoraFullFreezeLayer) \
                or isinstance(module, QuadraLoraLayer) or isinstance(module, QuadraMOELoraLayer) or isinstance(module, QuadraPQLoraFullFreezeLayer) or isinstance(module, QuadraPQMOELoraFullFreezeLayer) \
                or isinstance(module, HexaLoraLayer) or isinstance(module, HexaMOELoraLayer):
                module.set_state(state)

    def activate_all(self):
        for name, module in self.named_modules():
            if isinstance(module, DualLoraLayer) or isinstance(module, DualIA3Layer) or isinstance(module, DualMOELoraLayer) or isinstance(module, TripleLoraLayer) or isinstance(module, TriplePQLoraFullFreezeLayer) \
                or isinstance(module, PQLoraLayer) or isinstance(module, PQLoraFreezeLayer) or isinstance(module, PQLoraFreezeALayer) or isinstance(module, PQLoraLILFullFreezeLayer) \
                or isinstance(module, PQLoraFullFreezeLayer) or isinstance(module, PQLoraFullFreezeALayer) or isinstance(module, PQMOELoraFullFreezeLayer) or isinstance(module, PLoraFullFreezeLayer) \
                or isinstance(module, QuadraLoraLayer) or isinstance(module, QuadraMOELoraLayer) or isinstance(module, QuadraPQLoraFullFreezeLayer) or isinstance(module, QuadraPQMOELoraFullFreezeLayer) \
                or isinstance(module, HexaLoraLayer) or isinstance(module, HexaMOELoraLayer):
                module.activate_all()

    def activate_lora1(self):
        for name, module in self.named_modules():
            if isinstance(module, DualLoraLayer) or isinstance(module, DualIA3Layer) or isinstance(module, DualMOELoraLayer) or isinstance(module, TripleLoraLayer) or isinstance(module, TriplePQLoraFullFreezeLayer) \
                or isinstance(module, PQLoraLayer) or isinstance(module, PQLoraFreezeLayer) or isinstance(module, PQLoraFreezeALayer) or isinstance(module, PQLoraLILFullFreezeLayer) \
                or isinstance(module, PQLoraFullFreezeLayer) or isinstance(module, PQLoraFullFreezeALayer) or isinstance(module, PQMOELoraFullFreezeLayer) or isinstance(module, PLoraFullFreezeLayer) \
                or isinstance(module, QuadraLoraLayer) or isinstance(module, QuadraMOELoraLayer) or isinstance(module, QuadraPQLoraFullFreezeLayer) or isinstance(module, QuadraPQMOELoraFullFreezeLayer) \
                or isinstance(module, HexaLoraLayer) or isinstance(module, HexaMOELoraLayer):
                module.activate_lora1()
    
    def activate_lora2(self):
        for name, module in self.named_modules():
            if isinstance(module, DualLoraLayer) or isinstance(module, DualIA3Layer) or isinstance(module, DualMOELoraLayer) or isinstance(module, TripleLoraLayer) or isinstance(module, TriplePQLoraFullFreezeLayer) \
                or isinstance(module, PQLoraLayer) or isinstance(module, PQLoraFreezeLayer) or isinstance(module, PQLoraFreezeALayer) or isinstance(module, PQLoraLILFullFreezeLayer) \
                or isinstance(module, PQLoraFullFreezeLayer) or isinstance(module, PQLoraFullFreezeALayer) or isinstance(module, PQMOELoraFullFreezeLayer) or isinstance(module, PLoraFullFreezeLayer) \
                or isinstance(module, QuadraLoraLayer) or isinstance(module, QuadraMOELoraLayer) or isinstance(module, QuadraPQLoraFullFreezeLayer) or isinstance(module, QuadraPQMOELoraFullFreezeLayer) \
                or isinstance(module, HexaLoraLayer) or isinstance(module, HexaMOELoraLayer):
                module.activate_lora2()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            if type(pixel_values) is list or pixel_values.ndim == 5:
                if type(pixel_values) is list:
                    pixel_values = [x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values]
                concat_images = torch.cat([image for image in pixel_values], dim=0)
                image_features = self.get_image_features(
                    pixel_values=concat_images,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )
                n_image_features = image_features.shape[0] * image_features.shape[1]
                split_sizes = [image.shape[0] for image in pixel_values]
                image_features = torch.split(image_features, split_sizes, dim=0)
            else:
                image_features = self.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )

        # TODO: @raushan retain only the new behavior after v4.47
        if image_features is not None:
            n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
            if not isinstance(image_features, tuple):
                n_image_features = image_features.shape[0] * image_features.shape[1]

            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features: {n_image_features}"
                )
            # Split inputs_embeds and special_image_mask into individual samples
            batch_size = input_ids.shape[0]
            updated_inputs_embeds = []
            
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )

            for i in range(batch_size):
                single_inputs_embeds = inputs_embeds[i]
                single_special_image_mask = special_image_mask[i]
                image_feature = image_features[i]
                
                image_feature = image_feature.to(single_inputs_embeds.device, single_inputs_embeds.dtype)
  
                # Apply masked_scatter for this sample
                single_inputs_embeds = single_inputs_embeds.masked_scatter(single_special_image_mask, image_feature)
                updated_inputs_embeds.append(single_inputs_embeds)

            # Concatenate the updated embeddings
            inputs_embeds = torch.stack(updated_inputs_embeds, dim=0)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )