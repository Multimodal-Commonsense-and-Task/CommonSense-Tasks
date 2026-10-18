import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import RandomSampler
from packaging import version
from torch import nn
from utils.train_utils import load_deepspeed
from models.llava.llava_trainer import LLaVATrainer
# from transformers.utils import logging
import logging
import sys, os, time, shutil
import math
from typing import Optional, Dict, Union, Any
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_utils import (
    HPSearchBackend, 
    TrainOutput,
    has_length,
    speed_metrics,
)
from typing import List, Tuple, Union
from transformers.trainer_pt_utils import get_model_param_count, get_dataloader_sampler, reissue_pt_warnings
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import bitsandbytes
from transformers.trainer import (
    is_sagemaker_mp_enabled, 
    _is_peft_model, 
    TRAINER_STATE_NAME,
    is_torch_xla_available,
    is_accelerate_available,
    is_deepspeed_available,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    SCHEDULER_NAME
)
import warnings
from transformers.integrations import hp_params
from transformers.trainer_callback import TrainerState
from transformers.training_args import ParallelMode
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from torch.nn.functional import normalize
import h5py
from typing import List
import copy
import random
import faiss
from qpsolvers import solve_qp

from selection_methods.fedavg import (
    LLaVATrainerFEDAVG,
    _capture_backprops,
    _capture_output_activations,
    _layer_type,
)
from collections import OrderedDict
from deepspeed.utils import safe_get_full_grad
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedType
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]
    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_torch_xla_available():
    from torch_xla.core import xla_model as xm
    from torch_xla.debug import metrics as met
    from torch_xla.distributed import xla_multiprocessing as smp

try:
    from apex import amp
except ImportError:
    amp = None

# logger = logging.get_logger(__name__)
logger = logging.getLogger()
_supported_layers = ['Linear']

def infoCluster_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    model_to_load = global_state_dict
    with torch.no_grad():
        # if 'zero3' in training_args.deepspeed:
        #     load_deepspeed(model_to_load, model, strict=False)
        # else:
        model.load_state_dict(model_to_load, strict=False)  

def infoCluster_aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    for key in global_state_dict.keys():
        global_state_dict[key] = local_state_dict_list[0][key]

def infoCluster_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, prev_all_inputs=None, prev_all_input_inds=None,return_task_vectors=False):
    trainer = LLaVATrainerInfoCluster(model=model,
        ema_ratio=training_args.ema_ratio,
        model_ema_ratio=training_args.model_ema_ratio,
        k_means=training_args.k_means,
        mode=training_args.mode,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        client_id = extra_state_dict_dict['client_id'],
        curr_round = extra_state_dict_dict['curr_round'],
        task_vector=extra_state_dict_dict['task_vector'] if 'task_vector' in extra_state_dict_dict else None,
        fisher_old=extra_state_dict_dict['fisher_old'] if 'fisher_old' in extra_state_dict_dict else None,
        fisher_freq=extra_state_dict_dict['fisher_freq'] if 'fisher_freq' in extra_state_dict_dict else 5,
        return_embedding=training_args.embedding,
        return_task_vectors=return_task_vectors,
        selection_method=training_args.selection_method,
        kmeans=training_args.k_means,
        sample_ratio=training_args.sample_ratio,
        prev_all_inputs=prev_all_inputs,
        prev_all_input_inds=prev_all_input_inds,
        eval_period=training_args.eval_period,
        **data_module,
        )
    return trainer

class LLaVATrainerInfoCluster(LLaVATrainerFEDAVG):
    def __init__(self, ema_ratio, model_ema_ratio, k_means,mode,client_id, curr_round, task_vector=None, fisher_old=None, fisher_freq=5, return_embedding=None, return_task_vectors=False, selection_method=None, kmeans=None, sample_ratio=0, prev_all_inputs=None,prev_all_input_inds=None, eval_period=8000, **kwargs):
        super(LLaVATrainerInfoCluster, self).__init__(client_id=client_id, curr_round=curr_round, task_vector=task_vector, fisher_old=fisher_old, fisher_freq=fisher_freq, return_embedding=return_embedding, return_task_vectors=return_task_vectors, selection_method=selection_method, kmeans=kmeans, **kwargs)
        self.ema_ratio = ema_ratio
        self.model_ema_ratio = model_ema_ratio
        self.mode = mode
        self.round_info = []
        self.selected_info = []
        self.selected_info_ind = []
        
        self.sample_ratio = sample_ratio
        
        self.model_ema_update_period = 5000
        
        self.k_means = k_means
        self.cluster_exists = False
        self.selected_inputs = []
        
        self.bs = 1
        self.eval_period = eval_period
        
        self.target_layers = [4,10,16,22,28]
            
    
    def update_ema_lora(self, model):
        with torch.no_grad():
            model_params = OrderedDict(model.named_parameters())
            for name, param in model_params.items():
                if 'lora1' in name:
                    target_name = name.replace('lora1', 'lora2')
                    model_params[target_name].sub_((1. - self.model_ema_ratio) * (model_params[target_name] - param))
            
    def get_mwp_score(self, scores, selected_idxs):
        entropies = []
        score_keys = list(scores.keys())
        for k in score_keys:
            score_tensor = scores[k]
            score_indices = selected_idxs.to(score_tensor.device)
            score_list = []
            for ind in score_indices:
                score_list.append(float(score_tensor[ind].cpu()))
            entropies.append(self.get_dist_entropy(score_list))
        for k, e in zip(score_keys, entropies):
            print(f"{k}: {e}")
        print("Selecting ", score_keys[np.argmax(entropies)])
        selected_score = scores[score_keys[np.argmax(entropies)]]
        return selected_score[selected_idxs.to(selected_score.device)].to(selected_idxs.device)


    def get_dist_entropy(self, x, nbins=50, verbose=False):
        """
        x is assumed to be an (nsignals, nsamples) array containing integers between
        0 and n_unique_vals
        """

        # normalize with mean and std dev
        nan_count = np.count_nonzero(np.isnan(x))
        if verbose:
            x = x.astype(np.float32)
            print("Replacing %s nan values in input vector" % nan_count)
        x = np.nan_to_num(x)
        if len(x) == 0:
            return 0.0
        
        x_std = np.std(x)
        if not np.isfinite(x_std) or x_std == 0:
            return 0.0
        x = (x - np.mean(x))/x_std

        counts = np.zeros(nbins)
        bins = np.linspace(-4*x_std, 4*x_std, nbins)
        for i in range(1, nbins):
            counts[i] = np.sum((x > bins[i-1]) & (x <= bins[i]))

        epsilon = np.finfo(float).eps
        # divide by number of columns to get the probability of each unique value
        count_sum = np.sum(counts)
        if count_sum == 0:
            return 0.0
        p = counts / count_sum
        # Replace zeros with epsilon
        p = np.where(p == 0, epsilon, p)
        nan_count = np.count_nonzero(np.isnan(p))
        p = np.nan_to_num(p)

        # compute Shannon entropy in bits
        return -np.sum(p * np.log2(p))
    
    def set_outlier_threshold(self, scores):

        scores = np.sort(scores)
        low_thresh = scores[int(len(scores)*0.05)]
        high_thresh = scores[int(len(scores)*0.95)]
        print("Setting outlier threshold to %s (low) and %s (high)" % (low_thresh, high_thresh))
        return low_thresh, high_thresh
    
    
    def get_ccs_samples(self, scores, bins, budget):

        # low, high = set_outlier_threshold(list(scores.values()))
        low, high = self.set_outlier_threshold(scores)
        right = high

        remaining = budget
        budget = math.ceil(budget/bins)

        selected_idxs = []
        for i in tqdm(range(bins), desc="Getting CCS samples"):
            left = right - ((high-low)/bins)
            # candidates = [d for d in data.keys() if d in scores and scores[d] <= right and scores[d] >= left]
            candidates = [j for j, score in enumerate(scores) if score > left and score <= right]
            if len(candidates) < budget:
                selected_idxs.extend(candidates)
                if (i+1) < bins:
                    budget = math.ceil((remaining - len(candidates))/(bins-(i+1)))
                    remaining -= len(candidates)
            else:
                selected_idxs.extend(random.sample(candidates, k=budget))
                remaining -= budget
            print("Selected %s samples from between %s and %s values" % (len(selected_idxs), left, right))
            right = left
        
        return np.array(selected_idxs)
    
    
    def perform_multiway(self, sorted_clusters_list, multiway_sampling="min"):
        selected_ids_list = []
        nonempty_clusters = [cluster_i for cluster_i in sorted_clusters_list if len(cluster_i)]
        if not nonempty_clusters:
            return selected_ids_list
        base, remainder = divmod(self.sample_size_perbatch, len(nonempty_clusters))
        for k,n in self.score_dict.items():
            self.score_dict[k] = torch.stack(n)
        for cluster_idx, cluster_i in enumerate(nonempty_clusters):
            cluster_i = np.array(cluster_i)
            cluster_size = cluster_i.shape[0]
            cluster_ids = cluster_i[:, 0].astype("int32").tolist()
            cluster_ids = torch.tensor([torch.tensor(ind) for ind in cluster_ids]).cuda()
            cluster_scores = self.get_mwp_score(self.score_dict, cluster_ids)
            budget = min(base + (cluster_idx < remainder), cluster_size)
            if budget == 0:
                continue
            
            if multiway_sampling == "min":
                selected_ids = cluster_ids[torch.argsort(cluster_scores)[:budget]]
            elif multiway_sampling == "random":
                selected_ids = cluster_ids[torch.randperm(cluster_size, device=cluster_ids.device)[:budget]]
            else:
                selected_ids = cluster_ids[self.get_ccs_samples(cluster_scores, budget, budget)]
            
            selected_ids_list.extend(selected_ids)
        return selected_ids_list
        
    
    def perform_selection(self):

        dist_df = pd.DataFrame(
        {
            # "paths_list": paths_list,
            "nearest_cent": self.nearest_cent,
            "dist_to_cent": self.dist_to_cent,
        }
        )
        sorted_clusters_list = rank_within_cluster(
            self.batch_features,
            dist_df,
            self.kmeans_centroids,
            "cosine",
            True, # sort cluster items in descending order by the similarity to cluster centroid
            True, # spherical
            self.cluster_ids
        )
        if self.selection_method == "selfsup":
            sampled_indices = self.perform_selfsup(sorted_clusters_list)
        # elif self.selection_method == "semdedup":
            # sampled_indices, _ = self.perform_semdedup(sorted_clusters_list)
        elif self.selection_method == "coincide":
            cosine_sim = cosine_similarity(self.kmeans_centroids, self.kmeans_centroids)
            knn_cluster_indices = np.argsort(cosine_sim, axis=-1)[:,::-1][:,:self.k_means+1]
            # knn_cluster_similarity = cosine_sim[np.arange(len(cosine_sim))[:,None], knn_cluster_indices]
            mask = cosine_sim > 0.97
            cosine_sim[mask] = 0
            if 0 in list((~mask).sum(axis=-1)):
                sampled_indices = list(random.sample(list(range(len(self.batch_features))), k=self.sample_size_perbatch))
            else:
                transfer = cosine_sim.sum(axis=-1) / (~mask).sum(axis=-1)
                sampled_indices = self.perform_coincide(self.batch_features, transfer, cluster_indices=self.nearest_cent)
        elif self.selection_method == "dbp":
            sampled_indices1, _ = self.perform_semdedup(sorted_clusters_list, selection="dbp")
            sampled_indices = self.perform_dbp(sampled_indices1)
        elif self.selection_method == "multiway":
            sampled_indices = self.perform_multiway(sorted_clusters_list)
            
        # Selection implementations return a mix of lists, ndarrays, and CUDA
        # tensors. Normalize here before topping up the batch.
        sampled_indices = [int(index) for index in sampled_indices]
        target_size = min(self.sample_size_perbatch, len(self.batch_features))
        if len(sampled_indices) > target_size:
            sampled_indices = random.sample(sampled_indices, k=target_size)
        elif len(sampled_indices) < target_size:
            selected = set(int(index) for index in sampled_indices)
            candidates = list(set(range(len(self.batch_features))) - selected)
            sampled_indices.extend(random.sample(candidates, k=min(target_size - len(sampled_indices), len(candidates))))
        real_sampled_indices = [self.batch_input_inds[ind] for ind in sampled_indices]
        
        if len(real_sampled_indices) > 0:
            self.selected_info_ind.extend(real_sampled_indices)  
        else:      
            self.selected_info_ind.append(real_sampled_indices)        
        sampled_datalist = []
        for ind in sampled_indices:
            sampled_datalist.append(self.batch_inputs[int(ind)])
        
        return sampled_datalist
    
    def greedy_mmd_selection(self, K, M):

        n = len(K)

        indices = np.arange(n)
        selected = np.array([], dtype=int)

        K_XX = K.mean()

        for i in range(M):

            candidates = np.setdiff1d(indices, selected)

            temp_select = np.tile(selected, (len(candidates),1))
            temp_select = np.concatenate([temp_select, candidates[:,np.newaxis]], axis=1)  # Assume that each candidate index is selected

            candidates = torch.from_numpy(candidates).cuda()
            temp_select = torch.from_numpy(temp_select).cuda()

            K_XY = K[:, temp_select]
            K_XY = K_XY.mean(dim=(0,2))

            K_YY = K[temp_select[:,:,None], temp_select[:,None,:]]
            K_YY = K_YY.mean(dim=(1,2))

            MMD = K_XX + K_YY - 2 * K_XY

            best_idx = torch.argmin(MMD)
            best_idx = candidates[best_idx]
            selected = np.append(selected, best_idx.cpu().numpy())

        return selected
                
    def perform_coincide(self, cluster_embedding, transfer, cluster_indices):
        clusters = np.unique(cluster_indices)
        num_clusters = len(clusters)
        avg_num_samples = len(cluster_embedding) / num_clusters
        
        # Calculate the average number of samples per cluster
        # target_num_samples = int(sample_ratio * len(cluster_embedding))
        target_num_samples = self.sample_size_perbatch
        remainings = target_num_samples

        selected_indices = []
        count = 0

        K_list = []
        # ``transfer`` has one entry per fitted centroid, while ``clusters``
        # may omit empty centroids. Work in the compact observed-cluster
        # coordinate system to keep the vectors aligned.
        transfer = np.asarray(transfer, dtype='float64')[clusters]
        cluster_density = np.zeros(len(clusters), dtype='float64')
        for compact_idx, cluster_idx in enumerate(clusters):
            i_cluster_indices = np.where(cluster_indices == cluster_idx)[0]
            i_cluster_embeddings = cluster_embedding[i_cluster_indices]
            # Note that the float64 and float16 result is highly different, due to the accuracy.
            i_cluster_embeddings = torch.from_numpy(i_cluster_embeddings).cuda().to(torch.float64)

            i_K = self.gaussian_kernel(i_cluster_embeddings, i_cluster_embeddings, 1) # gamma=1
            density = i_K.mean()
            cluster_density[compact_idx] = density.item()

            K_list.append(i_K)
        cluster_score = transfer / np.maximum(cluster_density, np.finfo(float).eps)

        # Use the density to select the number of samples in each cluster
        ratio = np.exp(cluster_score / 1) / np.sum(np.exp(cluster_score / 1)) # temp=1
        ratio_sort_indices = np.argsort(ratio)[::-1]  # Sort in descending order
        ratio = ratio[ratio_sort_indices]
        cluster_score = cluster_score[ratio_sort_indices]

        for idx, compact_idx in enumerate(ratio_sort_indices):

            cluster_idx = clusters[compact_idx]

            i_cluster_indices = np.where(cluster_indices == cluster_idx)[0]
            i_K = K_list[compact_idx]
            i_target_num_samples = round(remainings * ratio[idx])

            if i_target_num_samples > len(i_cluster_indices):
                i_selected_indices = i_cluster_indices
            else:
                i_proto_indices = self.greedy_mmd_selection(i_K, i_target_num_samples)
                i_selected_indices = i_cluster_indices[i_proto_indices]
            selected_indices.append(i_selected_indices)
            count = len(i_selected_indices)
            # If not sufficient amount of samples were selected, toss it to the next selections.
            # We do this to satisfy select target_num_samples amounts of sample (if not, less than target_num_samples is sampled).
            remainings = remainings - count
            if idx + 1 < len(ratio):
                tail = np.exp(cluster_score[idx+1:])
                ratio[idx+1:] = tail / tail.sum()
        selected_indices = list(np.concatenate(selected_indices))
        return selected_indices 
    
    
    def get_shifted_logits_labels(self,batch, outputs):

        batch_size = batch['input_ids'].shape[0]
        target_length = batch['input_ids'].shape[1]
        outputs['logits'] = outputs['logits'][:, -target_length:, :]


        # Shift so that tokens < n predict n
        if batch['attention_mask'] is not None:
            shift_attention_mask = batch['attention_mask'][..., 1:]
            if batch_size == 1:
                shift_logits = outputs['logits'][..., :-1, :][shift_attention_mask.to(outputs['logits'].device) != 0].contiguous()
                shift_labels = batch['labels'][..., 1:][shift_attention_mask.to(batch['labels'].device) != 0].contiguous()
            else:
                shift_logits = [outputs['logits'][i, :-1, :][shift_attention_mask[i].to(outputs['logits'].device) != 0].contiguous() for i in range(batch_size)]
                shift_labels = [batch['labels'][i, 1:][shift_attention_mask[i].to(outputs['logits'].device) != 0].contiguous() for i in range(batch_size)]

        else:
            shift_logits = outputs['logits'][..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()

        return shift_logits.cuda(), shift_labels.cuda()
    
    def update_features(self, model):
        from itertools import zip_longest


        updated_features = []
        if self.selection_method == "multiway":
            score_dict = defaultdict(list)
            org_weights ={}
            for b_dict in self.batch_inputs:
                b_dict = self._prepare_inputs(b_dict)
                self.model.set_state('lora2')
                self.model.activate_lora2()
                ema_output = self.model(**b_dict, output_hidden_states=True)
                ema_output_loss = ema_output.loss
                
                gradient_list = []
                for n, p in self.model.base_model.model.model.layers[-1].named_parameters():
                    if p.requires_grad == True:
                        grad = torch.autograd.grad(ema_output_loss, p, retain_graph=True)[0].clone().detach()
                        grad = F.normalize(grad)
                        gradient_list.append(grad.view(-1))
                gradient_list = torch.cat(gradient_list).cpu().to(dtype=torch.float16)
                updated_features.append(gradient_list)
                model.zero_grad()
                
                #entropy
                ema_reps = ema_output.logits
                ema_reps2 = ema_reps.mean(dim=1).detach()
                ema_probs = torch.nn.Softmax(dim=-1)(ema_reps2)
                entropy = -1 * ema_probs * torch.log(ema_probs + 1e-10)
                entropy = torch.mean(torch.sum(entropy, dim=-1), dim=-1).detach().squeeze()
                score_dict["entropy"].append(entropy.to(torch.float))
                
                #el2n
                shift_logits, shift_labels =self.get_shifted_logits_labels(b_dict, ema_output)
                vocab_size = shift_logits.shape[-1]
                indices = torch.where(shift_labels != -100)[0]
                probs = torch.nn.Softmax(dim=-1)(shift_logits)[indices]
                label_onehot = torch.nn.functional.one_hot(shift_labels[indices], num_classes=vocab_size).to(probs.device)
                l2_values = torch.pow(label_onehot-probs, 2)
                l2_values = l2_values.sum(dim=1)
                l2_values = torch.sqrt(l2_values)
                el2n = torch.mean(l2_values).detach()
                score_dict["el2n"].append(el2n.to(torch.float))
                
                #IG
                shift_labels = shift_labels[indices]
                shift_logits = shift_logits[indices]
                loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')

                # Calculate perplexity per sample
                if shift_labels.size(0) == 0:
                    ppl_with_image = torch.exp(loss.mean()).cpu().detach()
                else:
                    sample_perplexities = torch.exp(loss.view(shift_labels.size(0), -1).mean(dim=1)).mean().squeeze().cpu()
                    ppl_with_image = sample_perplexities.detach()
                    
                # LLaVA batches use ``images``; Qwen batches do not. For a
                # text-only/Qwen batch image ablation is undefined, so use a
                # neutral IG score rather than aborting the whole round.
                if 'images' in b_dict:
                    no_image_batch = dict(b_dict)
                    no_image_batch['images'] = torch.zeros_like(no_image_batch['images'])
                    with torch.inference_mode():
                        no_image_batch = self._prepare_inputs(no_image_batch)
                        loss_no_image = model(**no_image_batch).loss
                    ppl_wo_image = torch.exp(loss_no_image).detach()
                    IG = (ppl_wo_image / ppl_with_image).to(torch.float)
                else:
                    IG = torch.ones_like(ppl_with_image, dtype=torch.float)
                score_dict["IG"].append(IG)
                    
            self.score_dict = score_dict
            for k, v in self.score_dict.items():
                v = torch.stack(v)
            self.batch_features = pad_sequence(updated_features, batch_first=True)
            # self.batch_features = torch.stack(updated_features)
        elif self.selection_method == "coincide":
            for b_dict in self.batch_inputs:
                with torch.no_grad():
                    with torch.inference_mode():
                        self.model.set_state('lora2')
                        self.model.deactivate_all()
                        b_dict = self._prepare_inputs(b_dict)
                        ema_output = self.model(**b_dict, output_hidden_states=True).hidden_states
                        embeddings = []
                        for target_layer in self.target_layers:
                            embedding = F.normalize(ema_output[target_layer].detach()).view(-1).cpu().to(dtype=torch.float16)
                            embeddings.append(embedding)
                        reps = torch.cat(embeddings, axis=-1)
                        updated_features.append(reps)

            self.batch_features = pad_sequence(updated_features, batch_first=True)        
        else:
            combined_dicts = []

            for b_dict in self.batch_inputs:
                with torch.no_grad():
                    with torch.inference_mode():
                        self.model.set_state('lora2')
                        self.model.deactivate_all()
                        b_dict = self._prepare_inputs(b_dict)
                        ema_output = self.model(**b_dict, output_hidden_states=True)#.los
                        
                        reps = ema_output.logits
                        reps = F.normalize(reps.detach())
                        updated_features.append(reps.view(-1).cpu().to(dtype=torch.float16))
            # self.batch_features = torch.stack(updated_features)
            self.batch_features = pad_sequence(updated_features, batch_first=True)
        self.batch_features = self.batch_features.numpy()

    def gaussian_kernel(self, X, Y, sigma):
        X_norm = X.pow(2).sum(1).view(-1, 1)
        Y_norm = Y.pow(2).sum(1).view(1, -1)
        pairwise_dists = X_norm + Y_norm - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))

        # pairwise_dists = cdist(X, Y, 'sqeuclidean')
        K = torch.exp(-pairwise_dists / (2 * sigma ** 2))
        return K

    
    def update_k_means_cluster(self, model):
        self.update_features(model)
        d = self.batch_features.shape[1]
        kmeans = faiss.Kmeans(
            d,
            self.k_means,
            niter=20, #100
            verbose=True,
            seed=1234,
            spherical=True,
            gpu=True
        )
        kmeans.train(self.batch_features)
        dist_to_cent, nearest_cent = kmeans.index.search(self.batch_features, 1)
        self.dist_to_cent, self.nearest_cent = dist_to_cent.squeeze(1), nearest_cent.squeeze(1)
        self.kmeans_index = kmeans.index
        self.kmeans_centroids = kmeans.centroids
        self.cluster_ids = list(range(self.k_means))
        
    
    def compute_loss_org(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        ##############################################################################################################
        if 'prompt' in inputs:
            text_prompt = inputs.pop('prompt')
        else:
            text_prompt = None
        
        outputs = model(**inputs, prompt=text_prompt) if text_prompt is not None else model(**inputs)
        ##############################################################################################################
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        
        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
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

        
        return (loss, outputs) if return_outputs else loss
    
    
    def compute_loss(self, model, inputs, return_outputs=False, update_adapter='lora1'):
        model_to_update = model.module if hasattr(model, "module") else model
        # global forward
        if update_adapter == 'lora1':
            model_to_update.set_state('lora1')
            model_to_update.activate_lora1()
            loss_global, outputs = self.compute_loss_org(model, inputs, return_outputs=True)    

            return (loss_global, outputs) if return_outputs else loss_global
    
        elif update_adapter == 'lora2':

            model_to_update.set_state('lora2')
            model_to_update.deactivate_all()
            loss_local, local_outputs = self.compute_loss_org(model, inputs, return_outputs=True) 

            return (loss_local, local_outputs) if return_outputs else loss_local
    
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], update_adapter) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, output = self.compute_loss(model, inputs, return_outputs=True, update_adapter=update_adapter)
            
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # if update_adapter == "lora1" and self.selection_method != "multiway":
        if update_adapter == "lora1":
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
        elif self.selection_method == "multiway":
            loss.backward()
        # elif self.selection_method == "multiway":
        #     if self.use_apex:
        #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #             scaled_loss.backward()
        #     else:
        #         self.accelerator.backward(loss)
       
        return loss.detach() / self.args.gradient_accumulation_steps, output
    
    def add_hooks(self, model: nn.Module) -> None:
        """
        Adds hooks to model to save activations and backprop values.

        The hooks will
        1. save activations into param.activations during forward pass
        2. append backprops to params.backprops_list during backward pass.

        Call "remove_hooks(model)" to disable this.

        Args:
            model:
        """

        global _hooks_disabled
        _hooks_disabled = False

        handles = []
        for layer in model.modules():
            if _layer_type(layer) in _supported_layers:
                handles.append(layer.register_forward_hook(_capture_output_activations))
                handles.append(layer.register_backward_hook(_capture_backprops))

        model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)
    
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        '''
        most lines are original transformer trainer code
        lines surrounded with
        ##############################################################################################################
        are added lines
        '''
        all_info = None
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        # self.is_deepspeed_enabled = False
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)
      
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size *self.bs
        self.sample_size_perbatch = int(total_train_batch_size*self.sample_ratio)
        self.warmup_period = ((self.k_means//total_train_batch_size)+1)*total_train_batch_size
        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            # num_update_steps_per_epoch = len_dataloader // (args.gradient_accumulation_steps*self.bs/self.sample_size_perbatch)
            # num_update_steps_per_epoch = len_dataloader // (args.gradient_accumulation_steps*self.bs)
            num_update_steps_per_epoch = len_dataloader // (args.gradient_accumulation_steps)
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False
        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        # if self.selection_method == "coincide":
        #     self.layers = self.model.base_model.model.model.layers
        #     target_layers = [3,7,11,15,19]
        #     for i in target_layers:
        #         self.add_hooks(layers[i].self_attn.o_proj)
        #     self.num_layers = len(layers)
        #     num_target_layers = len(target_layers)
        #     sim_act = nn.Tanh()
        #     msa_task_emb = np.zeros((self.bs, model.config.hidden_size * num_target_layers * 2), dtype='float16')
       
        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        if model is not self.model:
            self.model_wrapped = model

        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        #############################################################################################################

        if self.args.save_optim and self.curr_round > 0:
            output_dir = f'model_states_{self.args.note}/client_{self.client_id}/'
            os.makedirs(output_dir, exist_ok=True)
            self._load_optimizer_and_scheduler(output_dir)
            
        ##############################################################################################################
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # # Train!
        # logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {num_examples:,}")
        # logger.info(f"  Num Epochs = {num_train_epochs:,}")
        # logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        # if self.args.per_device_train_batch_size != self._train_batch_size:
        #     logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        # logger.info(f"  Total optimization steps = {max_steps:,}")
        # logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= (args.gradient_accumulation_steps)
            else:
                steps_trained_in_current_epoch = 0

            # logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            # logger.info(f"  Continuing training from epoch {epochs_trained}")
            # logger.info(f"  Continuing training from global step {self.state.global_step}")
            # if not args.ignore_data_skip:
            #     logger.info(
            #         f"  Will skip the first {epochs_trained} epochs then the first"
            #         f" {steps_trained_in_current_epoch} batches in the first epoch."
            #     )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        batch_inputs = []

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps *self.bs
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            all_reps = []
            selected_info = []
            accumulated_batches, accul_threshold = [], 5
            add_accul = True
            start = 0
            
            self.batch_inputs, self.batch_features, self.batch_input_inds = [], [], []
            
            for step, inputs in enumerate(epoch_iterator):
                copy_inputs = copy.deepcopy(inputs)
                self.batch_inputs.append(copy_inputs)
                self.batch_input_inds.append(step)
    
                if step % self.model_ema_update_period == 0:
                    self.update_ema_lora(model)
                    
                total_batched_samples += 1
                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_device = inputs[main_input_name].device
                        self.state.num_input_tokens_seen += torch.sum(
                            self.accelerator.gather(
                                torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64)
                            )
                        ).item()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % (args.gradient_accumulation_steps*self.bs) == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= (args.gradient_accumulation_steps*self.bs) and (step + 1) == steps_in_epoch
                    )

                if (
                    total_batched_samples % (args.gradient_accumulation_steps*self.bs) == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)
                        
                    self.update_k_means_cluster(model)
                    selected_inputs = self.perform_selection()
                    self.batch_inputs, self.batch_features, self.batch_input_inds = [], [], []

                    for d in selected_inputs:
                   
                        with self.accelerator.accumulate(model):
                        
                            tr_loss_step, output = self.training_step(model, d, update_adapter='lora1')

                        
                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_xla_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                        else:
                            if tr_loss.device != tr_loss_step.device:
                                raise ValueError(
                                    f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                                )
                            tr_loss += tr_loss_step

                        self.current_flos += float(self.floating_point_ops(d))    
                            
                        

                        # # Gradient clipping
                        # if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        #     # deepspeed does its own clipping

                        #     if is_sagemaker_mp_enabled() and args.fp16:
                        #         _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        #     elif self.use_apex:
                        #         # Revert to normal clipping otherwise, handling Apex or full precision
                        #         _grad_norm = nn.utils.clip_grad_norm_(
                        #             amp.master_params(self.optimizer),
                        #             args.max_grad_norm,
                        #         )
                        #     else:
                        #         _grad_norm = self.accelerator.clip_grad_norm_(
                        #             model.parameters(),
                        #             args.max_grad_norm,
                        #         )

                        #     if (
                        #         is_accelerate_available()
                        #         and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        #     ):
                        #         grad_norm = model.get_global_grad_norm()
                        #         # In some cases the grad norm may not return a float
                        #         if hasattr(grad_norm, "item"):
                        #             grad_norm = grad_norm.item()
                        #     else:
                        #         grad_norm = _grad_norm

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                        
                    # compute fisher online
                    if step % self.fisher_freq == 0:
                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    
                    # wsd
                    if self.args.is_wsd == 'WSD' and math.ceil(self.state.epoch*steps_in_epoch) == math.ceil(self.args.decay_ratio*steps_in_epoch):
                        self.global_weight = {k: t.detach().cpu().clone() for k, t in self.model.named_parameters() if t.requires_grad}
                    
                    # # save client model
                    # if (step+1) % self.eval_period == 0:
                    #     output_dir = os.path.join(self.args.state_dir, f"{self.client_id}_client_model_round{self.curr_round+1}_itr{step+1}.pth")
                    #     state_dict = {k: t.detach().cpu().clone() for k, t in self.model.named_parameters() if t.requires_grad}
                        
                    #     if (self.args.local_rank == 0 or self.args.local_rank == -1):
                    #         torch.save(state_dict, output_dir)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval,0)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                # # save client model
                # if (step+1) % self.eval_period == 0:
                #     output_dir = os.path.join(self.args.state_dir, f"{self.client_id}_client_model_round{self.curr_round+1}_itr{step+1}.pth")
                #     state_dict = {k: t.detach().cpu().clone() for k, t in self.model.named_parameters() if t.requires_grad}
                    
                #     if (self.args.local_rank == 0 or self.args.local_rank == -1):
                #         torch.save(state_dict, output_dir)
                        
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
                
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval,0)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
        
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        ##############################################################################################################
        if self.args.save_optim:
            output_dir = f'model_states_{self.args.note}/client_{self.client_id}/'
            os.makedirs(output_dir, exist_ok=True)
            self._save_optimizer_and_scheduler(output_dir)
        ##############################################################################################################
        
        
        # if self.return_embedding == "gradient" and self.return_task_vectors:
        #     self.fisher_old = ((self.fisher_cur/self.fisher_cnt) + self.fisher_old) / 2 if self.fisher_old is not None else (self.fisher_cur/self.fisher_cnt)
        #     self.task_vector = self.fisher_old.detach().cpu()
        # elif "EMA" in self.mode:
        #     self.all_info = all_info
        #     self.selected_info = selected_info
        # elif self.return_embedding == "representation" and self.return_task_vectors:
        #     all_reps = torch.cat(all_reps).numpy()
        #     self.task_vector = all_reps
        
        self.model.activate_all()
        
        return TrainOutput(self.state.global_step, train_loss, metrics)

    def perform_selfsup(self, sorted_clusters_list):
        selected_ids_list = []
        # 1) store cluster size
        for cluster_i in sorted_clusters_list:
            if not cluster_i:
                continue
            cluster_i = np.array(cluster_i)
            cluster_size = cluster_i.shape[0]

            cluster_ids = cluster_i[:, 0].astype("int32").tolist()
            # The remainder is filled by perform_selection, so never select more
            # than the requested total budget here.
            quota = max(0, self.sample_size_perbatch // self.k_means)
            if quota:
                selected_ids_list.extend(cluster_ids[-quota:])

        return selected_ids_list
    
    def perform_semdedup(self, sorted_clusters_list, selection="semdedup"):
        selected_ids_list = []
        selected_inputs_list = []
        # 1) store cluster size
        for cluster_i in sorted_clusters_list:
            if not cluster_i:
                # Keep the positional correspondence with centroids for DBP.
                if selection == "dbp":
                    selected_ids_list.append([])
                continue
            cluster_i = np.array(cluster_i)
            cluster_size = cluster_i.shape[0]

            ## -- By default, we keep hard examples from groups
            # clutser_items_indices = list(range(cluster_size))

            ## -- indices for cluster items in the dataset
            cluster_ids = cluster_i[:, 0].astype("int32").tolist()
            cluster_reps = [torch.from_numpy(embedding) for i, embedding in enumerate(self.batch_features) if i in cluster_ids]
            cluster_reps = torch.stack(cluster_reps)

            M = self.semdedup(cluster_i, cluster_reps)
            
            if selection == "semdedup":
                _, max_index = torch.max(M, dim=0)
                selected_ids_list.append(cluster_ids[max_index])
                # selected_inputs_list.append(self.memory_inputs[max_index])

            elif selection == "dbp":
                threshold = torch.quantile(M, self.sample_ratio)
                eps_points_to_keep = M <= threshold
                selected_ids_list.append([cluster_ids[i] for i, boo in enumerate(eps_points_to_keep) if boo])

            # selected_ids_list = np.concatenate(selected_ids_list)
            # selected_features_list = np.vstack(selected_features_list)
        return selected_ids_list, selected_inputs_list


    def semdedup(self, cluster, cluster_reps):
        ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
        # cluster_reps.cuda()
        cluster_reps = F.normalize(cluster_reps, dim=1)
        pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
        del cluster_reps
        pair_w_sim_matrix.fill_diagonal_(0.0)
        assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

        ## -- get paths to cluster i images
        # image_urls = cluster[:, 0]

        ## -- make sure all the paths are unique this ensure that the duplicates are really stored many time times on memory
        # assert not self._contains_duplicates(image_urls)

        ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the compinations not permutations
        triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)
        M = torch.max(triu_sim_mat, dim=0)[0].cpu().float()
        return M
    
    
    def pruning_qp(self,
        probabilites,
        number_of_items_in_cluster,
        pruned_dataset_size,
        num_centroids,
    ):

        if num_centroids == 0:
            return np.array([], dtype=int)
        capacities = np.asarray(number_of_items_in_cluster, dtype=int)
        probabilites = np.asarray(probabilites, dtype=float)
        target = min(int(pruned_dataset_size), int(capacities.sum()))
        if target == 0:
            return np.zeros(num_centroids, dtype=int)
        min_samples = min(int(target / num_centroids), int(capacities.min()))
        P = np.eye(num_centroids)
        q = -probabilites * target
        A = np.array([1.0] * num_centroids)
        b = np.array([pruned_dataset_size])
        bounds = np.array(
            [
                (
                    min(min_samples, capacities[i]),
                    capacities[i],
                )
                for i in range(num_centroids)
            ]
        )

        x = None
        if np.all(np.isfinite(q)):
            try:
                x = solve_qp(P=P, q=q, A=A, b=np.array([target]), lb=bounds[:, 0], ub=bounds[:, 1], solver="osqp")
            except Exception as exc:
                logging.warning("DBP QP solve failed; using bounded fallback: %s", exc)
        if x is None or not np.all(np.isfinite(x)):
            # OSQP can decline tiny or degenerate cluster problems. Use a
            # bounded proportional allocation so the training round can run.
            weights = np.nan_to_num(np.asarray(probabilites, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            weights = np.maximum(weights, 0)
            if weights.sum() == 0:
                weights = np.ones(num_centroids)
            x = np.minimum(np.floor(weights / weights.sum() * target).astype(int), capacities)
            while x.sum() < target:
                available = np.flatnonzero(x < capacities)
                if not len(available):
                    break
                pick = available[np.argmax(weights[available] / (x[available] + 1))]
                x[pick] += 1
        else:
            x = np.clip(np.rint(x).astype(int), 0, capacities)
        # assert sum((x < 0)) == 0

        return x
    
    def perform_dbp(self, sorted_clusters_list):
        active_clusters = [(idx, cluster) for idx, cluster in enumerate(sorted_clusters_list) if len(cluster)]
        if not active_clusters:
            return []
        selected_ids_list = []
        avg_distance_to_cent_list = []
        # 1) Calculate dintra
        for i, cluster_ids in active_clusters:
            cluster_embeddings = [embedding for i, embedding in enumerate(self.batch_features) if i in cluster_ids]
            avg_distance_to_cent_list.append(np.mean(np.linalg.norm(cluster_embeddings - self.kmeans_centroids[i], axis=1)))
        
        # 2) Calculate dinter
        active_centroids = self.kmeans_centroids[[i for i, _ in active_clusters]]
        mean_dist = self.average_cosine_similarities(active_centroids)
        
        # 3) Compute density
        d_intra = np.array(avg_distance_to_cent_list)
        d_inter = np.array(mean_dist)
        assert d_inter.shape == d_intra.shape
        d_i = d_inter * d_intra
        # remove nans. nans can arise if there are only 1-2 items per cluster
        indices_nan = np.argwhere(np.isnan(d_i))
        assert sum((d_i < 0)) < 10

        for item in indices_nan:
            d_i[item] = [np.nanmean(d_i)]
        
        ### turn d_i into a probability
        temperature=0.5
        softmax = torch.nn.Softmax()
        probs = torch.Tensor(d_i) / temperature
        probs = softmax(probs)
        number_of_items_in_cluster = list()
        for _, cluster_i in active_clusters:
            cluster_i = np.array(cluster_i)
            number_of_items_in_cluster.append(cluster_i.shape[0])
            
        # calculate the number of items per cluster according to QP pruning
        num_of_items_qp = list()
        pruned_dataset_size = self.sample_size_perbatch
        number_of_items_in_cluster_pruned = self.pruning_qp(
            probs.data.numpy(),
            number_of_items_in_cluster,
            pruned_dataset_size,
            len(active_clusters)
        )
        i = 0
        for _, cluster_all_paths in active_clusters:
            # cluster_i = np.array(cluster_i)
            # cluster_all_paths = cluster_i[:, 0].astype("<U32").tolist()
            kept_data = np.array(cluster_all_paths)[
                    : int(number_of_items_in_cluster_pruned[i])
            ]
            kept_data = [int(float(x)) for x in kept_data]
            selected_ids_list.append(kept_data)
            i+=1
        selected_ids_list = np.concatenate(selected_ids_list)
        return selected_ids_list
    
    
    def average_cosine_similarities(self, centroids):
        normalized_centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        
        cosine_similarities = np.dot(normalized_centroids, normalized_centroids.T)
        
        np.fill_diagonal(cosine_similarities, np.nan)
        
        avg_similarities = np.nanmean(cosine_similarities, axis=1)
        
        return avg_similarities
    
    
    def k_means_update(self, point):
        k = self.k_means
        cluster_distances = np.zeros(k)
        point = np.squeeze(point)
        for cluster in range(k):
            cluster_distances[cluster] = np.sum(np.dot(point, self.kmeans_centroids[cluster]))
        c = np.argmax(cluster_distances)
        i_cluster_indices = np.where(self.nearest_cent == c)[0]
        self.dist_to_cent = np.append(self.dist_to_cent, np.sum(np.dot(point, self.kmeans_centroids[c])))
        self.nearest_cent = np.append(self.nearest_cent, c)
        self.kmeans_centroids[c] += 1.0/(len(i_cluster_indices)+1)*(point - self.kmeans_centroids[c])


def rank_within_cluster(
    data: Union[np.memmap, np.ndarray],
    dist_df: pd.DataFrame,
    centroids: np.ndarray,
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    spherical: bool = False,
    cluster_ids: List[int] = range(50000),
) -> List[List[Tuple[str, int, float, int]]]:

    assert sim_metric in [
        "cosine",
        "l2",
    ], "sim_metric should be one of ['cosine', 'l2']"

    sorted_clusters_list = []
    for cluster_c in cluster_ids:

        cluster_df = dist_df.loc[dist_df["nearest_cent"] == cluster_c]

        cluster_items = list(cluster_df.index)  ## -- ids of examples in cluster c
        if sim_metric == "cosine":
            if spherical:
                cluster_dists_to_cent = list(1 - cluster_df["dist_to_cent"])
            else:
                cluster_c_centroid = torch.Tensor(centroids[cluster_c])
                sim_to_cent = torch.nn.CosineSimilarity(dim=1)(
                    torch.Tensor(data[cluster_items]), cluster_c_centroid
                )
                cluster_dists_to_cent = (1 - sim_to_cent).tolist()

        elif sim_metric == "l2":  # -- get l2 distance from "dist_to_cent" array
            cluster_dists_to_cent = list(cluster_df["dist_to_cent"])

        cluster_label = np.full((len(cluster_df)), cluster_c).tolist()
        # example_paths = list(cluster_df["paths_list"])
        sort_descending = keep_hard
        cluster_sorted = sorted(
            zip(cluster_items, cluster_dists_to_cent, cluster_label),
            key=lambda x: x[1],
            reverse=sort_descending,
        )  # -- sort_descending = True for descending sort
        sorted_clusters_list.append(
            cluster_sorted
        )  # -- Descending dists. list of of list of tuples (example, dist). The ith list of tuples corresponds to cluster i
        # sorted_cluster_file_path = f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"
        # np.save(sorted_cluster_file_path, cluster_sorted)
    return sorted_clusters_list

def dicts_are_equal(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True
