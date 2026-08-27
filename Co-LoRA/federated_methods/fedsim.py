import torch
from torch import nn
from federated_methods.fedavg import LLaVATrainerFEDAVG
from transformers.trainer import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, _is_peft_model

def fedsim_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    training_args.max_seq_length = training_args.model_max_length
    training_args.packing=False
    trainer = LLaVATrainerFEDSIM(model=model,
        tokenizer=tokenizer,
        args=training_args,
        client_id = extra_state_dict_dict['client_id'],
        curr_round = extra_state_dict_dict['curr_round'],
        test_datalist=extra_state_dict_dict['test_datalist'],
        processor=extra_state_dict_dict['processor'],
        data_args=extra_state_dict_dict['data_args'],
        task_vector=extra_state_dict_dict['task_vector'] if 'task_vector' in extra_state_dict_dict else None,
        fisher_old=extra_state_dict_dict['fisher_old'] if 'fisher_old' in extra_state_dict_dict else None,
        fisher_freq=extra_state_dict_dict['fisher_freq'] if 'fisher_freq' in extra_state_dict_dict else 5,
        model2=extra_state_dict_dict['model2'] if 'model2' in extra_state_dict_dict else None,
        **data_module,
        )
    return trainer

class LLaVATrainerFEDSIM(LLaVATrainerFEDAVG):
    # loss for t5-model evaluation
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.training is False:
            with torch.no_grad():
                # global forward
                model.set_state('lora1')
                if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
                    labels = inputs.pop("labels")
                else:
                    labels = None
                if self.model_accepts_loss_kwargs:
                    loss_kwargs = {}
                    if num_items_in_batch is not None:
                        loss_kwargs["num_items_in_batch"] = num_items_in_batch
                    inputs = {**inputs, **loss_kwargs}
                outputs = model(**inputs)
                # Save past state if it exists
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index]

                if labels is not None:
                    unwrapped_model = self.accelerator.unwrap_model(model)
                    if _is_peft_model(unwrapped_model):
                        model_name = unwrapped_model.base_model.model._get_name()
                    else:
                        model_name = unwrapped_model._get_name()
                    # User-defined compute_loss function
                    if self.compute_loss_func is not None:
                        loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
                    elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                        loss = self.label_smoother(outputs, labels, shift_labels=True)
                    else:
                        loss = self.label_smoother(outputs, labels)
                else:
                    if isinstance(outputs, dict) and "loss" not in outputs:
                        raise ValueError(
                            "The model did not return a loss from the inputs, only the following keys: "
                            f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                        )
                    # We don't use .loss here since the model may return tuples instead of ModelOutput.
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
                    loss *= self.accelerator.num_processes

                # local forward
                #second 
                model.set_state('lora2')
                if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
                    labels = inputs.pop("labels")
                else:
                    labels = None
                if self.model_accepts_loss_kwargs:
                    loss_kwargs = {}
                    if num_items_in_batch is not None:
                        loss_kwargs["num_items_in_batch"] = num_items_in_batch
                    inputs = {**inputs, **loss_kwargs}
                local_outputs = model(**inputs)
                # Save past state if it exists
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = local_outputs[self.args.past_index]

                if labels is not None:
                    unwrapped_model = self.accelerator.unwrap_model(model)
                    if _is_peft_model(unwrapped_model):
                        model_name = unwrapped_model.base_model.model._get_name()
                    else:
                        model_name = unwrapped_model._get_name()
                    # User-defined compute_loss function
                    if self.compute_loss_func is not None:
                        loss = self.compute_loss_func(local_outputs, labels, num_items_in_batch=num_items_in_batch)
                    elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                        loss = self.label_smoother(local_outputs, labels, shift_labels=True)
                    else:
                        loss = self.label_smoother(local_outputs, labels)
                else:
                    if isinstance(local_outputs, dict) and "loss" not in local_outputs:
                        raise ValueError(
                            "The model did not return a loss from the inputs, only the following keys: "
                            f"{','.join(local_outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                        )
                    # We don't use .loss here since the model may return tuples instead of ModelOutput.
                    loss = local_outputs["loss"] if isinstance(local_outputs, dict) else local_outputs[0]

                if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
                    loss *= self.accelerator.num_processes
                
                final_logits = outputs['logits'] + local_outputs['logits']
                labels = inputs['labels']
                # Shift so that tokens < n predict n
                shift_logits = final_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels) 
                
                outputs['logits'] = final_logits
                outputs['loss'] = loss
                
                return (loss, outputs) if return_outputs else loss
        # global forward
        model.module.set_state('lora1')
        _, outputs = super(LLaVATrainerFEDSIM, self).compute_loss(model, inputs, return_outputs=True,num_items_in_batch=num_items_in_batch)     
        # local forward
        model.module.set_state('lora2')
        _, local_outputs = super(LLaVATrainerFEDSIM, self).compute_loss(model, inputs, return_outputs=True,num_items_in_batch=num_items_in_batch) 
        
        final_logits = outputs['logits'] + local_outputs['logits']
        labels = inputs['labels']
        # Shift so that tokens < n predict n
        shift_logits = final_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels) 
        
        return (loss, outputs) if return_outputs else loss