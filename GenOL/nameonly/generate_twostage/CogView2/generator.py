import os
import sys
import math
import random
import torch
import argparse
from functools import partial
import numpy as np
from argparse import Namespace
from PIL import Image

from SwissArmyTransformer import get_args, get_tokenizer
from SwissArmyTransformer.model import CachedAutoregressiveModel
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence, evaluate_perplexity
from SwissArmyTransformer.generation.utils import timed_name, save_multiple_images, generate_continually

from CogView2.utils import CoglmStrategy, get_recipe
from CogView2.utils import get_masks_and_position_ids_coglm
from CogView2.sr_pipeline import SRGroup

from icetk import icetk as tokenizer
tokenizer.add_special_tokens(['<start_of_image>', '<start_of_english>', '<start_of_chinese>'])

class InferenceModel(CachedAutoregressiveModel):
    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(logits_parallel.float(), self.transformer.word_embeddings.weight[:20000].float())
        return logits_parallel

class Cogview2:
    def __init__(self, img_size=160, style='photo', batch_size=1, max_inference_batch_size=1):
        args_list = ['--mode', 'inference', '--fp16', '--input-source', 'input.txt', '--output-path', 
                    'samples_sat_v0.2', '--batch-size', f"{batch_size}", '--max-inference-batch-size', 
                    f"{max_inference_batch_size}", '--input-source', 'input.txt']
        
        self.args = get_args(args_list)
        self.args.img_size = img_size
        self.args.only_first_stage = False
        self.args.inverse_prompt = False
        self.args.style = style

        self.args = Namespace(**vars(self.args), **get_recipe(self.args.style))
        
        # Load the model using the arguments
        print(f"Loading InferenceModel - coglm")
        self.model, self.args = InferenceModel.from_pretrained(self.args, 'coglm')

        print(f"Loading CachedAutoregressiveModel")
        self.text_model = CachedAutoregressiveModel(self.args, transformer=self.model.transformer)
        self.query_template = self.args.query_template

        invalid_slices = [slice(tokenizer.num_image_tokens, None)]
        self.strategy = CoglmStrategy(invalid_slices,
                                temperature=self.args.temp_all_gen, top_k=self.args.topk_gen, top_k_cluster=self.args.temp_cluster_gen)

        if not self.args.only_first_stage:
            print(f"Loading cogview2-dsr, cogview2-itersr")
            self.srg = SRGroup(self.args)
        
    def generate_images(self, prompt, batch_size=1):
        text = self.query_template.format(prompt)
        seq = tokenizer.encode(text)
        if len(seq) > 110:
            raise ValueError("The input text is too long. Please use a shorter prompt.")
        txt_len = len(seq) - 1
        seq = torch.tensor(seq + [-1]*400, device=self.args.device)
        log_attention_weights = torch.zeros(len(seq), len(seq), 
            device=self.args.device, dtype=torch.half if self.args.fp16 else torch.float32)
        log_attention_weights[:, :txt_len] = self.args.attn_plus

        # Generation
        mbz = self.args.max_inference_batch_size
        assert self.args.batch_size < mbz or self.args.batch_size % mbz == 0
        get_func = partial(get_masks_and_position_ids_coglm, context_length=txt_len)
        output_list, score_list = [], []

        for tim in range(max(self.args.batch_size // mbz, 1)): # 16 // 8 = 2
            self.strategy.start_pos = txt_len + 1
            with torch.no_grad():
                print(f"Filling sequence...")
                coarse_samples = filling_sequence(self.model, seq.clone(),
                        batch_size=min(self.args.batch_size, mbz),
                        strategy=self.strategy,
                        log_attention_weights=log_attention_weights,
                        get_masks_and_position_ids=get_func
                        )[0] # torch.Size([8, 413])
            
            # get ppl for inverse prompting
            if self.args.inverse_prompt:
                image_text_seq = torch.cat(
                    (
                        coarse_samples[:, -400:],
                        torch.tensor([tokenizer['<start_of_chinese>']]+tokenizer.encode(prompt), device=self.args.device).expand(coarse_samples.shape[0], -1)
                    ), dim=1)
                seqlen = image_text_seq.shape[1]
                attention_mask = torch.zeros(seqlen, seqlen, device=self.args.device, dtype=torch.long)
                attention_mask[:, :400] = 1
                attention_mask[400:, 400:] = 1
                attention_mask[400:, 400:].tril_()
                position_ids = torch.zeros(seqlen, device=self.args.device, dtype=torch.long)
                torch.arange(513, 513+400, out=position_ids[:400])
                torch.arange(0, seqlen-400, out=position_ids[400:])
                loss_mask = torch.zeros(seqlen, device=self.args.device, dtype=torch.long)
                loss_mask[401:] = 1
                scores = evaluate_perplexity(
                    self.text_model, image_text_seq, attention_mask,
                    position_ids, loss_mask#, invalid_slices=[slice(0, 20000)], reduction='mean'
                )
                score_list.extend(scores.tolist())
                # ---------------------
            
            output_list.append(
                    coarse_samples
                )
        output_tokens = torch.cat(output_list, dim=0) # torch.Size([16, 413])
        order_list = range(output_tokens.shape[0])

        imgs, txts = [], []
        if self.args.only_first_stage:
            for i in order_list:
                seq = output_tokens[i]
                with torch.no_grad():
                    decoded_img = tokenizer.decode(image_ids=seq[-400:])
                    decoded_img = torch.nn.functional.interpolate(decoded_img, size=(480, 480))
                imgs.append(decoded_img) # only the last image (target)
        
        if not self.args.only_first_stage: 
            with torch.no_grad():
                iter_tokens = self.srg.sr_base(output_tokens[:, -400:], seq[:txt_len])
                for seq in iter_tokens: # [16, 3600]
                    decoded_img = tokenizer.decode(image_ids=seq[-3600:])
                    decoded_img = torch.nn.functional.interpolate(decoded_img, size=(480, 480))
                    imgs.append(decoded_img) # only the last image (target)
        
        # Convert image tensors to PIL images
        results = []
        for img in imgs:
            img = img[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            img = Image.fromarray(img)
            results.append(img)
        
        return results[0]
        