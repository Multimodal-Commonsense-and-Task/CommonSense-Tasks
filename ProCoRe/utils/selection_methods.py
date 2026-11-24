import shutil
import os
import json
import numpy as np
import torch
import random
import transformers
from utils.data_loader_VLM import GenerationDataset, DataCollatorForGenerationDataset
from transformers import StoppingCriteria, StoppingCriteriaList
from torch.utils.data import DataLoader
from models.llava import conversation as conversation_lib_llava
from models.bunny import conversation as conversation_lib_bunny
from models.llava.mm_utils import KeywordsStoppingCriteria
from transformers import RobertaModel

import faiss
import pickle
import tqdm
from typing import List, Tuple, Union
import pandas as pd
import h5py
from qpsolvers import solve_qp
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KernelDensity
from itertools import combinations
from sklearn.metrics import mutual_info_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import zscore

def semdedup(embeddings, cluster_labels, num_clusters, sample_ratio=0.9, sort=False, cluster_centers=None, visualize=False):

    assert type(keep_samples) == int

    all_idxs = []
    all_Ms = []

    cluster_sizes = [np.where(cluster_labels == i)[0].shape[0] for i in range(num_clusters)]
    keep_budget_by_cluster = assign_budget_to_cluster(cluster_sizes, keep_samples, num_clusters)
    print("Budget for SemDeDup")
    for i, (old, new) in enumerate(zip(cluster_sizes, keep_budget_by_cluster)):
        print(f"cluster {i}: {old} --> {new}")

    for i in tqdm(range(num_clusters), desc="Running SemDeDup over clusters"):
    
        # print("Cluster: %s" % i)
        cluster_idxs = np.where(cluster_labels == i)[0]
        # print(cluster_idxs)
        cluster_i_embeddings = torch.tensor(embeddings[cluster_idxs]).type(torch.float32)
        cluster_i_embeddings = torch.nn.functional.normalize(cluster_i_embeddings, p=2, dim=-1)
        # dist_from_centroid = np.linalg.norm(embeddings[cluster_idxs] - cluster_centers[i], axis=-1)
        # sorted_idxs = np.argsort(dist_from_centroid)[::-1]
    
        # We use descending = True / False for keeping examples with low/ high similarity to cluster centroids . We ignore this step for keeping random examples from each group of similar examples . See Appendix D for more details about this step .
        # Compute the pairwise cosine similarity between embeddings
    
        pairwise_sim_matrix = cluster_i_embeddings @ cluster_i_embeddings.T
        triu_sim_matrix = torch.triu(pairwise_sim_matrix, diagonal = 1)
        M = torch.max(triu_sim_matrix, dim =0)[0]
        sorted_idxs = torch.argsort(M)
        points_to_keep = sorted_idxs[:keep_budget_by_cluster[i]].numpy()
        points_to_keep = cluster_idxs[points_to_keep]
        all_idxs.append(points_to_keep)
        all_Ms.append(M)
        # Check if the maximum similarity <= the threshold

    all_idxs = np.concatenate(all_idxs, axis=0)
    all_Ms = torch.cat(all_Ms, dim=0)

    print(f"Selected {points_to_keep.shape[0]} samples to retain after deduplication")
    return all_idxs, all_Ms.numpy()


def sample_selection(model, datalist, selection_method, data_args, training_args, model_args, tokenizer, sample_ratio=0.5, device='cuda', train_sample_vectors=None, embedding='representation', k=10):
    model = model.cpu()
    if selection_method == "no_sampling":
        sampled_datalist = datalist
    elif selection_method == "random":
        sampled_datalist = random.sample(datalist, k=int(len(datalist)*sample_ratio))
    elif selection_method == "MI":
        embeddings = train_sample_vectors
        sampled_indices = perform_MI(embeddings, total_data_len=len(embeddings), sample_ratio=sample_ratio)
    elif selection_method == "spanningMI_minmax":
        embeddings = train_sample_vectors
        sampled_indices = perform_spanningMI(embeddings, sample_ratio=sample_ratio, scenario="mst")
        sampled_datalist = []
        for ind in sampled_indices:
            sampled_datalist.append(datalist[int(ind)])
    elif selection_method == "spanningMI_maxmin":
        embeddings = train_sample_vectors
        sampled_indices = perform_spanningMI(embeddings, sample_ratio=sample_ratio, scenario="maxst")
        sampled_datalist = []
        for ind in sampled_indices:
            sampled_datalist.append(datalist[int(ind)])
    elif selection_method in ["semdedup", "dbp", "coincide", "selfsup", "cluster_MI"]:
        '''
        print("here1")
        # emb_array = np.memmap(f"embedding_{selection_method}_k{k}_debug.npy", dtype='float16', mode='w+', shape=(30000,4213, 4096))
        # for i in range(10000):
        #     emb_array[i] = np.random.randn(1,4213,4096).astype(np.float16)
        cluster_embedding = np.memmap(f"embedding_{selection_method}_k{k}_debug.npy", dtype="float16", mode="w+", shape=(30000, 4213, 4096))
        print("type1", type(cluster_embedding))
        cluster_embedding = cluster_embedding.reshape(len(cluster_embedding), -1)
        cluster_embedding = np.nan_to_num(cluster_embedding)
        print("type2", type(cluster_embedding))
        # cluster_embedding = np.memmap(f"embedding_{selection_method}_k{k}.npy", dtype="float16", mode="r", shape=(len(datalist), 4213, 4096))
        # cluster_embedding = cluster_embedding.reshape(len(cluster_embedding), -1).astype(np.float16)
        # cluster_embedding = torch.rand(10000, 17256448, dtype=torch.float16)
        # cluster_embedding = np.random.randn(10000, 17256448).astype(np.float16)
        mbkm = MiniBatchKMeans(n_clusters=k, batch_size=10000)  # Take a good look at the docstring and set options here
        mbkm.fit(cluster_embedding)
        cluster_labels = mbkm.labels_
        sampled_indices, pairwise_sims = semdedup(cluster_embedding, cluster_labels, k, sample_ratio)
        sampled_datalist = []
        for ind in sampled_indices:
            sampled_datalist.append(datalist[ind])
        '''
        ### CLUSTERING
        cluster_embedding = train_sample_vectors
        print("start selection")
        # cluster_embedding = np.memmap(f"embedding_{selection_method}_k{k}.npy", dtype="float16", mode="r", shape=(len(datalist), 4213, 4096))
        # cluster_embedding = np.memmap(f"embedding_{selection_method}_k{k}_debug.npy", dtype="float16", mode="r", shape=(30000, 4213, 4096))
        # with h5py.File(f"embedding_{selection_method}.h5", 'r') as f:
        #     cluster_embedding = f["embeddings"][:].astype(np.float32)
        # cluster_embedding = cluster_embedding.reshape(len(cluster_embedding), -1)
        
        d = cluster_embedding.shape[1]
        kmeans = faiss.Kmeans(
            d,
            k,
            niter=20, #100
            verbose=True,
            seed=1234,
            spherical=True,
            gpu=True
        )
        
        # chunk_size = 1000
        # chunks = [cluster_embedding[i:i + chunk_size] for i in range(0, cluster_embedding.shape[0], chunk_size)]
        # c_dist_to_cent_list, c_nearest_cent_list = [], []
        # print("here1")
        # for chunk in chunks:
        kmeans.train(cluster_embedding)
        # print("here2")
        # for chunk in chunks:
        #     c_dist_to_cent, c_nearest_cent = kmeans.index.search(chunk, 1)
        #     c_dist_to_cent, c_nearest_cent = c_dist_to_cent.squeeze(1), c_nearest_cent.squeeze(1)
        #     c_dist_to_cent_list.extend(c_dist_to_cent)
        #     c_nearest_cent_list.extend(c_nearest_cent)
        # for chunk in chunks:
        #     kmeans.train(chunk)
        # res = faiss.StandardGpuResources()
        # kmeans_gpu_index = faiss.index_cpu_to_gpu(res, 0, kmeans.index)
        dist_to_cent, nearest_cent = kmeans.index.search(cluster_embedding, 1)
        dist_to_cent, nearest_cent = dist_to_cent.squeeze(1), nearest_cent.squeeze(1)
        kmeans_index = kmeans.index
        # dist_to_cent, nearest_cent = kmeans.index.search(cluster_embedding, 1)
        # dist_to_cent, nearest_cent = dist_to_cent.squeeze(1), nearest_cent.squeeze(1)
        kmeans_centroids = kmeans.centroids
        ### Sort Cluster
        dist_df = pd.DataFrame(
            {
                # "paths_list": paths_list,
                "nearest_cent": nearest_cent,
                "dist_to_cent": dist_to_cent,
            }
        )
        
        if selection_method == "selfsup":
            # dist_to_cent = torch.tensor(dist_to_cent)
            # sampled_indices = torch.topk(dist_to_cent, int(len(dist_to_cent)*sample_ratio)).indices
            cluster_ids = list(range(k))
            ### SEMDEDUP
            sorted_clusters_list = rank_within_cluster(
                cluster_embedding,
                dist_df,
                kmeans_centroids,
                "cosine",
                True, # sort cluster items in descending order by the similarity to cluster centroid
                True, # spherical
                cluster_ids,
            )
            sampled_indices = perform_selfsup(sorted_clusters_list, sample_ratio=sample_ratio)
        elif selection_method in ["semdedup"]:
            cluster_ids = list(range(k))
            ### SEMDEDUP
            sorted_clusters_list = rank_within_cluster(
                cluster_embedding,
                dist_df,
                kmeans_centroids,
                "cosine",
                True, # sort cluster items in descending order by the similarity to cluster centroid
                True, # spherical
                cluster_ids,
            )
            sampled_indices = perform_semdedup(sorted_clusters_list, cluster_embedding, sample_ratio=sample_ratio)
            # sampled_datalist = datalist[sampled_indices]
        elif selection_method == "cluster_MI":
            cluster_ids = list(range(k))
            ### SEMDEDUP
            sorted_clusters_list = rank_within_cluster(
                cluster_embedding,
                dist_df,
                kmeans_centroids,
                "cosine",
                True, # sort cluster items in descending order by the similarity to cluster centroid
                True, # spherical
                cluster_ids,
            )
            sampled_indices = perform_cluster_MI(sorted_clusters_list, cluster_embedding, sample_ratio=sample_ratio)
            # sampled_datalist = datalist[sampled_indices]
        elif selection_method == "dbp":
            cluster_ids = list(range(k))
            sorted_clusters_list = rank_within_cluster(
                cluster_embedding,
                dist_df,
                kmeans_centroids,
                "cosine",
                True, # sort cluster items in descending order by the similarity to cluster centroid
                True, # spherical
                cluster_ids,
            )
            ### DBP
            sampled_indices = perform_dbp(cluster_embedding,sorted_clusters_list, kmeans_centroids, d, k, total_data_len=len(cluster_embedding), sample_ratio=sample_ratio)
            # sampled_datalist = datalist[sampled_indices]
        elif selection_method == "coincide":
            ### COINCIDE
            
            # Cal Transferability
            # kmeans_centroids = kmeans_centroids.reshape(-1, 5, 4096)
            # kmeans_centroids = kmeans_centroids[:,:,2048:]
            # kmeans_centroids = kmeans_centroids.reshape(-1, 5*2048)
            cosine_sim = cosine_similarity(kmeans_centroids, kmeans_centroids)
            knn_cluster_indices = np.argsort(cosine_sim, axis=-1)[:,::-1][:,:k+1]
            knn_cluster_similarity = cosine_sim[np.arange(len(cosine_sim))[:,None], knn_cluster_indices]
            mask = cosine_sim > 0.95
            cosine_sim[mask] = 0
            transfer = cosine_sim.sum(axis=-1) / (~mask).sum(axis=-1)
            
            sampled_indices = perform_coincide(cluster_embedding, transfer, cluster_indices=nearest_cent, sample_ratio=sample_ratio)
            
            
                    
        sampled_datalist = []
        for ind in sampled_indices:
            sampled_datalist.append(datalist[int(ind)])
        del cluster_embedding
    
    elif selection_method == "info":
        train_sample_vectors = torch.tensor(train_sample_vectors)
        sampled_indices = torch.topk(train_sample_vectors, int(len(datalist)*sample_ratio)).indices
        
        # z_scores = zscore(train_sample_vectors)
        # sampled_indices = [i for i, z in enumerate(z_scores) if z > 1.5]
        
        sampled_datalist = []
        sampled_info = []
        for ind in sampled_indices:
            sampled_datalist.append(datalist[ind])
            sampled_info.append(train_sample_vectors[ind])    
            
        model = model.cuda()
        return sampled_datalist, sampled_info
    
    else:
        # TODO embedding vs gradient
        # 이전 태스트 data도 계속 datalist에 포함?
        
        '''
        dataset = GenerationDataset(datalist, tokenizer, data_args)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
        if 'llava' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
        elif 'bunny' in model_args.model_name_or_path.lower():
            conv = conversation_lib_bunny.default_conversation
        repeat_criteria = CustomStoppingCriteria()
        stop_str = conv.sep2
        keywords = [stop_str]
        print("data length", len(dataset))
        for i, batch in enumerate((dataloader)): #tqdm
            input_ids, imgs, golds, prompts, img_files = batch['input_ids'], batch['images'], batch['gold'], batch['prompt'], batch['image_file']
            attention_mask = batch['attention_mask'].to(device=device)
            input_ids = input_ids.to(device=device, non_blocking=True)
            if imgs is not None:
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                image_sizes = [x.shape[-2:] for x in imgs]
            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
            
            input_ids,position_ids,attention_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                attention_mask,
                None,
                None,
                imgs,
                image_sizes=image_sizes
            )
            with torch.inference_mode():
                if isinstance(model, RobertaModel):
                    reps = model(input_ids=input_ids,
                                attention_mask=attention_mask, output_hidden_states=True, return_dict=True).pooler_output
                else:
                    hidden_states = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        output_hidden_states=True)
                    ids = torch.arange(len(input_ids), device=input_ids.device)
                    # pos = attention_mask.sum(dim=1) - 1
                    # reps = hidden_states[-1][ids, pos]
                    reps = hidden_states[-1]
                print("reps shape", reps.shape())
                # all_reps.append(reps.cpu())
                all_reps.append(reps)
        if len(all_reps) > 0:
            all_reps = torch.cat(all_reps)
        '''
        cluster_embedding = np.memmap(f"embedding_{selection_method}_k{k}.npy", dtype="float16", mode="r", shape=(len(datalist), 4213, 4096))
        # with h5py.File(f"embedding_{selection_method}.h5", 'r') as f:
        #     train_sample_vectors = f["embeddings"][:]
        all_reps = train_sample_vectors
        all_reps = torch.from_numpy(all_reps)
        if selection_method == "entropy":
            score = obtain_entropy_score(all_reps)
        elif selection_method == "grand":
            score = obtain_grand_score(all_reps)
            
        # shift_logits, shift_labels = get_shifted_logits_labels(batch, outputs)
        elif selection_method == "el2n":
            score = obtain_el2n_score(shift_logits, shift_labels, eval_batch_size)
        elif selection_method == "perplexity":
            score = obtain_perplexity_score(all_reps)
        elif selection_method == "ig_score":
            score = obtain_ig_score(all_reps)
        sampled_indices = torch.topk(score, int(len(datalist)*sample_ratio)).indices
        sampled_datalist = []
        for ind in sampled_indices:
            sampled_datalist.append(datalist[ind])
        # sampled_datalist = datalist[sampled_indices]
        # sampled_datalist = [data for i, data in enumerate(datalist) if i in sampled_indices.tolist()]
                    
    model = model.cuda()
    return sampled_datalist


def perform_spanningMI(embeddings, sample_ratio=0.2, scenario="mst", n_bins=10):
    sample_num = int(len(embeddings)*sample_ratio)
    n_samples = embeddings.shape[0]
    # Discretize embeddings into bins
    discrete_embeddings = np.digitize(
        embeddings, bins=np.linspace(embeddings.min(), embeddings.max(), n_bins)
    )
    # Initialize the MI matrix
    mi_matrix = np.zeros((n_samples, n_samples))
    # Compute MI for each pair of samples
    for i in range(n_samples):
        for j in range(i + 1, n_samples):  # Only compute upper triangle
            mi_matrix[i, j] = mutual_info_score(discrete_embeddings[i], discrete_embeddings[j])
            mi_matrix[j, i] = mi_matrix[i, j]  # Symmetric matrix
    mi_matrix
    n_samples = mi_matrix.shape[0]

    # Convert MI to edge weights for MST or MaxST
    if scenario == "mst":
        # Use negative MI for MST (to minimize weights)
        edge_weights = -mi_matrix
    elif scenario == "maxst":
        # Use positive MI for MaxST (to maximize weights)
        edge_weights = mi_matrix
    else:
        raise ValueError("Scenario must be 'mst' or 'maxst'")

    # Create a sparse graph for MST calculation
    sparse_graph = csr_matrix(edge_weights)

    # Compute the Minimum or Maximum Spanning Tree
    if scenario == "mst":
        spanning_tree = minimum_spanning_tree(sparse_graph).toarray()
    else:  # MaxST: Flip the signs back after MST calculation
        spanning_tree = -minimum_spanning_tree(-sparse_graph).toarray()

    # Extract edges and their weights from the spanning tree
    edges = []
    for i in range(n_samples):
        for j in range(n_samples):
            if spanning_tree[i, j] != 0:
                edges.append((i, j, mi_matrix[i, j]))  # Original MI as weight

    # Sort edges by weight (highest MI for MST, lowest MI for MaxST)
    edges = sorted(edges, key=lambda x: x[2], reverse=(scenario == "mst"))

    # Select unique sample indices from the top-k edges
    selected_samples = set()
    for i, j, _ in edges[:sample_num]:
        selected_samples.add(i)
        selected_samples.add(j)
        if len(selected_samples) >= sample_num:
            break

    return list(selected_samples)
    

from collections import Counter

def mutual_information(embeddings, n_bins=10):
    """
    Compute mutual information for all pairwise samples in a batch embedding,
    and identify the top-k samples contributing to the highest MI pairs.

    Args:
        embeddings (numpy.ndarray): Array of shape (batch_size, embedding_dim).
        n_bins (int): Number of bins for discretizing embeddings.
        top_k (int): Number of samples to return.

    Returns:
        top_samples (list): List of tuples (sample_index, cumulative_MI)
                            for the top-k samples contributing to highest MI pairs.
    """
    batch_size, embedding_dim = embeddings.shape

    # Normalize embeddings for similarity computation (optional)
    embeddings = (embeddings - embeddings.mean(axis=1, keepdims=True)) / embeddings.std(axis=1, keepdims=True)

    # Discretize embeddings into bins for MI calculation
    discrete_embeddings = np.digitize(embeddings, bins=np.linspace(embeddings.min(), embeddings.max(), n_bins))

    # Compute pairwise MI
    mi_matrix = np.zeros((batch_size, batch_size))  # MI matrix
    for i in range(batch_size):
        for j in range(i + 1, batch_size):  # Compute only upper triangle
            mi_matrix[i, j] = mutual_info_score(discrete_embeddings[i], discrete_embeddings[j])

    # Compute cumulative MI contribution for each sample
    sample_mi_contributions = np.zeros(batch_size)
    for i in range(batch_size):
        sample_mi_contributions[i] = np.sum(mi_matrix[i, :]) + np.sum(mi_matrix[:, i])

    # Get top-k samples with the highest cumulative MI contributions
    top_samples = sorted(enumerate(sample_mi_contributions), key=lambda x: x[1], reverse=True)

    return top_samples

def gaussian_kernel(X, Y, sigma):
    X_norm = X.pow(2).sum(1).view(-1, 1)
    Y_norm = Y.pow(2).sum(1).view(1, -1)
    pairwise_dists = X_norm + Y_norm - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))

    # pairwise_dists = cdist(X, Y, 'sqeuclidean')
    K = torch.exp(-pairwise_dists / (2 * sigma ** 2))
    return K

def perform_coincide(cluster_embedding, transfer, cluster_indices, sample_ratio=None):
    clusters = np.unique(cluster_indices)
    num_clusters = len(clusters)
    avg_num_samples = len(cluster_embedding) / num_clusters
    
    # Calculate the average number of samples per cluster
    target_num_samples = int(sample_ratio * len(cluster_embedding))
    remainings = target_num_samples

    selected_indices = []
    count = 0

    K_list = []
    cluster_density = np.zeros(len(clusters), dtype='float64')
    for cluster_idx in clusters:
        i_cluster_indices = np.where(cluster_indices == cluster_idx)[0]
        i_cluster_embeddings = cluster_embedding[i_cluster_indices]
        # Note that the float64 and float16 result is highly different, due to the accuracy.
        i_cluster_embeddings = torch.from_numpy(i_cluster_embeddings).cuda().to(torch.float64)

        i_K = gaussian_kernel(i_cluster_embeddings, i_cluster_embeddings, 1) # gamma=1
        density = i_K.mean()
        cluster_density[cluster_idx] = density.item()

        K_list.append(i_K)
    cluster_score = (1 / cluster_density) * transfer

    # Use the density to select the number of samples in each cluster
    ratio = np.exp(cluster_score / 1) / np.sum(np.exp(cluster_score / 1)) # temp=1
    ratio_sort_indices = np.argsort(ratio)[::-1]  # Sort in descending order
    ratio = ratio[ratio_sort_indices]
    cluster_score = cluster_score[ratio_sort_indices]
    
    for idx, cluster_idx in enumerate(ratio_sort_indices):

        i_cluster_indices = np.where(cluster_indices == cluster_idx)[0]
        i_K = K_list[cluster_idx]
        i_target_num_samples = round(remainings * ratio[idx])

        if i_target_num_samples > len(i_cluster_indices):
            i_selected_indices = i_cluster_indices
        else:
            i_proto_indices = greedy_mmd_selection(i_K, i_target_num_samples)
            i_selected_indices = i_cluster_indices[i_proto_indices]

        selected_indices.append(i_selected_indices)
        count = len(i_selected_indices)
        # If not sufficient amount of samples were selected, toss it to the next selections.
        # We do this to satisfy select target_num_samples amounts of sample (if not, less than target_num_samples is sampled).
        remainings = remainings - count
        ratio[idx+1:] = np.exp(cluster_score[idx+1:] / 1) / np.sum(np.exp(cluster_score[idx+1:] / 1)) # temp=1

    selected_indices = np.concatenate(selected_indices)
    return selected_indices

def get_shifted_logits_labels(batch, outputs):

    batch_size = batch['input_ids'].shape[0]
    target_length = batch['input_ids'].shape[1]
    outputs['logits'] = outputs['logits'][:, -target_length:, :]

    # print(batch['input_ids'].shape, outputs['logits'].shape)

    # Shift so that tokens < n predict n
    if batch['attention_mask'] is not None:
        shift_attention_mask = batch['attention_mask'][..., 1:]
        if batch_size == 1:
            # print(outputs['logits'].shape, shift_attention_mask.shape)
            shift_logits = outputs['logits'][..., :-1, :][shift_attention_mask.to(outputs['logits'].device) != 0].contiguous()
            shift_labels = batch['labels'][..., 1:][shift_attention_mask.to(batch['labels'].device) != 0].contiguous()
            # print('attention', shift_attention_mask[0, :], 'logits', shift_logits[0], 'labels', shift_labels[0])
        else:
            shift_logits = [outputs['logits'][i, :-1, :][shift_attention_mask[i].to(outputs['logits'].device) != 0].contiguous() for i in range(batch_size)]
            shift_labels = [batch['labels'][i, 1:][shift_attention_mask[i].to(outputs['logits'].device) != 0].contiguous() for i in range(batch_size)]
            # print('attention', shift_attention_mask[0, :], 'logits', shift_logits[0][0], 'labels', shift_labels[0])

    else:
        shift_logits = outputs['logits'][..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()

    return shift_logits, shift_labels

def obtain_entropy_score(outputs, eval_batch_size=None):
    probs = torch.nn.Softmax(dim=-1)(outputs)
    entropy = -1 * probs * torch.log(probs + 1e-10)
    entropy = torch.mean(torch.sum(entropy, dim=-1), dim=-1).detach().cpu().squeeze()
    return entropy

def obtain_grand_score(gradients, eval_batch_size=None):
    # L2-Norm of the gradient vector; we consider the gradients that we get (see https://arxiv.org/pdf/2403.09559)
    grand = torch.linalg.norm(gradients, dim=-1, ord=2).detach().cpu()
    # print('grand', grand.shape, grand)
    return grand

def obtain_ig_score(model, batch, ppl_with_image, eval_batch_size): #TODO: Fix

    if eval_batch_size == 1:
        batch['images'] = torch.zeros_like(batch['images'])
        # print(batch.keys())
        with torch.inference_mode():
            loss_no_image = model(**batch).loss
        ppl_wo_image = torch.exp(loss_no_image)
        ig_score = ppl_wo_image/ppl_with_image
        # print('ig_score', ig_score.shape, ig_score)
        return ig_score
    else:
        raise NotImplementedError
    
def obtain_perplexity_score(shift_logits, shift_labels):
# def obtain_perplexity_score(outputs, eval_batch_size, shift_logits=None, shift_labels=None):
    # if eval_batch_size == 1:
    #     return torch.exp(outputs.loss).detach().cpu().item()
    # else:
    assert shift_logits is not None and shift_labels is not None
    indices = torch.where(shift_labels != -100)[0]
    shift_labels = shift_labels[indices]
    shift_logits = shift_logits[indices]
    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')

    # Calculate perplexity per sample
    if shift_labels.size(0) == 0:
        return torch.exp(loss.mean())
    else:
        sample_perplexities = torch.exp(loss.view(shift_labels.size(0), -1).mean(dim=1)).mean().squeeze()
        # print('ppl', indices.shape, shift_logits.shape, shift_labels.shape, loss.shape, sample_perplexities.shape)
        return sample_perplexities


def obtain_el2n_score(shift_logits, shift_labels, eval_batch_size):
    # L2 Norm of the error vector, averaged over all tokens to adapt for sequences (see https://arxiv.org/pdf/2403.09559)

    # print(type(shift_logits), type(shift_labels))
    # print([label.shape for label in shift_labels], [logit.shape for logit in shift_logits])
    # print(shift_logits.shape)
    # vocab_size = shift_logits.shape[-1]
    # indices = torch.where(shift_labels != -100)[0]
    # print(indices.shape)
    # probs = torch.nn.Softmax(dim=-1)(shift_logits)[indices]
    # print(probs.shape)
    # # print(probs)
    # # print(shift_labels)
    
    # label_onehot = torch.nn.functional.one_hot(shift_labels[indices], num_classes=vocab_size).to(probs.device)
    
    if eval_batch_size == 1:
        assert type(shift_logits) != list
        vocab_size = shift_logits.shape[-1]
        indices = torch.where(shift_labels != -100)[0]
        probs = torch.nn.Softmax(dim=-1)(shift_logits)[indices]
        # print('Probabilities', probs, probs.sum(dim=-1))
        label_onehot = torch.nn.functional.one_hot(shift_labels[indices], num_classes=vocab_size).to(probs.device)
        # print(probs.shape, label_onehot.shape)
        # _ = label_onehot - probs
        # l2_values = torch.nn.functional.mse_loss(label_onehot, probs, reduction='none')
        # print(label_onehot, probs)
        l2_values = torch.pow(label_onehot-probs, 2)
        # print(l2_values)
        l2_values = l2_values.sum(dim=1)
        # print(l2_values)
        # print(l2_values.shape)
        # print(l2_values)
        l2_values = torch.sqrt(l2_values)
        # print(l2_values)
        el2n_score = torch.mean(l2_values).detach().cpu()
        # print('el2n', el2n_score.shape, el2n_score)
        # print(el2n_score)
    else:
        assert type(shift_logits) == list
        # print([logits.shape for logits in shift_logits], [labels.shape for labels in shift_labels])
        vocab_size = shift_logits[0].shape[-1]
        indices = [torch.where(labels != -100)[0] for labels in shift_labels]
        probs = [torch.nn.Softmax(dim=-1)(logits)[indic] for logits, indic in zip(shift_logits, indices)]
        
        label_onehot = [torch.nn.functional.one_hot(labels[indic], num_classes=vocab_size).to(probs[0].device) for labels, indic in zip(shift_labels, indices)]
        # print('Probabilities', [prob for prob in probs], [prob.sum(dim=-1) for prob in probs])
        # print([prob.shape for prob in probs], [onehot.shape for onehot in label_onehot])
        l2_values = [torch.pow(label_onehot_i-probs_i, 2) for label_onehot_i, probs_i in zip(label_onehot, probs)]
        print(probs[0][0], probs[0][0].sum(), l2_values[0][0], l2_values[0][0].sum())
        # print(l2_values)
        l2_values = [val.sum(dim=1) for val in l2_values]
        # print(l2_values)
        l2_values = [torch.sqrt(val) for val in l2_values]
        # print(l2_values)
        el2n_score = [torch.mean(val).detach().cpu().item() for val in l2_values]
        print(el2n_score)
    
    return el2n_score

def obtain_ig_score(model, batch, ppl_with_image, eval_batch_size): #TODO: Fix

    if eval_batch_size == 1:
        batch['images'] = torch.zeros_like(batch['images'])
        # print(batch.keys())
        with torch.inference_mode():
            loss_no_image = model(**batch).loss
        ppl_wo_image = torch.exp(loss_no_image)
        ig_score = ppl_wo_image/ppl_with_image
        # print('ig_score', ig_score.shape, ig_score)
        return ig_score
    else:
        raise NotImplementedError
    
def rank_within_cluster(
    data: Union[np.memmap, np.ndarray],
    dist_df: pd.DataFrame,
    centroids: np.ndarray,
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    spherical: bool = False,
    cluster_ids: List[int] = range(50000),
) -> List[List[Tuple[str, int, float, int]]]:
    """
    Sorts each cluster items by the distance to the cluster centroid.
    Cluster is represented as list of tuples. Each tuple has 4 values:
        example_path: unique path to the example/image/text doc, for imagenet it could be something like "n04235860_14959.JPEG",
        example_id_in_dataset: int between 0 and cluster_size-1
        dist_to_cent: cosine distance to cluster centroid
        cluster_id: cluster number (from 0 to number of clusters)

    Arguments:
    data -- the data for which the clusters were created (np.ndarray or np.memmap)
    dist_df -- DataFrame with the distances between the data points and the centroids, nearest centroid for each example, and path to each example.
    centroids -- np.ndarray with the centroids for each cluster.
    sim_metric -- the similarity metric used to compute distances, should be one of ["cosine", "l2"]
    keep_hard -- a boolean ehen True, we sort cluster items in descending order by the similarity to cluster centroid. Defaults to True.
    spherical -- a boolean True means spherical was used for computing centroids (used for cosine similarity).
    cluster_ids -- a list of cluster ids to process. Each slurm job will process part of the clusters.
    sorted_clusters_file_loc -- the location to save the sorted clusters.

    Returns:
    A list of cluster representations, where each representation is a list of tuples with 4 values.
    -- exampel for a cluster (the list bellow is sorted by dist_to_cent in descending order)
        [
          [example_name, example_id_in_dataset, dist_to_cent, cluster_label],
          [example_name, example_id_in_dataset, dist_to_cent, cluster_label],
                                        .
                                        .
                                        .
                                                                    ]
    """

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
        print("cluster_items", cluster_items)

        sorted_clusters_list.append(
            cluster_sorted
        )  # -- Descending dists. list of of list of tuples (example, dist). The ith list of tuples corresponds to cluster i
        # sorted_cluster_file_path = f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"
        # np.save(sorted_cluster_file_path, cluster_sorted)
    return sorted_clusters_list

def perform_selfsup(sorted_clusters_list, sample_ratio=0.2):
    selected_ids_list = []
    # 1) store cluster size
    for cluster_i in sorted_clusters_list:
        cluster_i = np.array(cluster_i)
        cluster_size = cluster_i.shape[0]
        print("cluster_size: ", cluster_size)

        ## -- By default, we keep hard examples from groups
        # clutser_items_indices = list(range(cluster_size))

        ## -- indices for cluster items in the dataset
        cluster_ids = cluster_i[:, 0].astype("int32").tolist()
        # cluster_reps = [torch.from_numpy(embedding) for i, embedding in enumerate(cluster_embedding) if i in cluster_ids]
        # cluster_reps = torch.stack(cluster_reps)
        
        selected_ids_list.extend(cluster_ids[:int(len(cluster_ids)*sample_ratio)])
        

        # M = semdedup(cluster_i, cluster_reps)

        # ## -- 5) We need to keep a point from the dataset when its pairwise similarity to other point is < 1-ebs
        # # eps_points_to_keep = cluster_ids[torch.where(M < 1 - eps)[0]]
        # threshold = torch.quantile(M, sample_ratio)
        # eps_points_to_keep = M <= threshold
        # selected_ids_list.append([cluster_ids[i] for i, boo in enumerate(eps_points_to_keep) if boo])
        # selected_ids_list.append(cluster_ids[eps_points_to_keep])
    print("number of selected", len(selected_ids_list))  
    return selected_ids_list


def perform_semdedup(sorted_clusters_list, cluster_embedding, sample_ratio=0.2, eps=0.1):
    selected_ids_list = []
    # 1) store cluster size
    for cluster_i in sorted_clusters_list:
        cluster_i = np.array(cluster_i)
        cluster_size = cluster_i.shape[0]
        print("cluster_size: ", cluster_size)

        ## -- By default, we keep hard examples from groups
        # clutser_items_indices = list(range(cluster_size))

        ## -- indices for cluster items in the dataset
        cluster_ids = cluster_i[:, 0].astype("int32").tolist()
        cluster_reps = [torch.from_numpy(embedding) for i, embedding in enumerate(cluster_embedding) if i in cluster_ids]
        cluster_reps = torch.stack(cluster_reps)

        M = semdedup(cluster_i, cluster_reps)

        ## -- 5) We need to keep a point from the dataset when its pairwise similarity to other point is < 1-ebs
        # eps_points_to_keep = cluster_ids[torch.where(M < 1 - eps)[0]]
        threshold = torch.quantile(M, sample_ratio)
        eps_points_to_keep = M <= threshold
        selected_ids_list.append([cluster_ids[i] for i, boo in enumerate(eps_points_to_keep) if boo])
        # selected_ids_list.append(cluster_ids[eps_points_to_keep])
    selected_ids_list = np.concatenate(selected_ids_list)
    print("number of selected", len(selected_ids_list))  
    return selected_ids_list

def perform_cluster_MI(sorted_clusters_list, cluster_embedding, sample_ratio=0.2, eps=0.1):
    selected_ids_list = []
    # 1) store cluster size
    for cluster_i in sorted_clusters_list:
        cluster_i = np.array(cluster_i)
        cluster_size = cluster_i.shape[0]
        print("cluster_size: ", cluster_size)

        ## -- By default, we keep hard examples from groups
        # clutser_items_indices = list(range(cluster_size))

        ## -- indices for cluster items in the dataset
        cluster_ids = cluster_i[:, 0].astype("int32").tolist()
        cluster_reps = [torch.from_numpy(embedding) for i, embedding in enumerate(cluster_embedding) if i in cluster_ids]
        cluster_reps = torch.stack(cluster_reps)

        M = MI(cluster_reps, sample_ratio)

        ## -- 5) We need to keep a point from the dataset when its pairwise similarity to other point is < 1-ebs
        # eps_points_to_keep = cluster_ids[torch.where(M < 1 - eps)[0]]
        # threshold = torch.quantile(M, sample_ratio)
        # eps_points_to_keep = M <= threshold
        # print("eps_points", eps_points_to_keep, threshold)
        selected_ids_list.extend([ind for ind, sample in M])
        # selected_ids_list.append([cluster_ids[i] for i, boo in enumerate(eps_points_to_keep) if boo])
        # selected_ids_list.append(cluster_ids[eps_points_to_keep])
    # selected_ids_list = np.concatenate(selected_ids_list)
    print("number of selected", len(selected_ids_list))  
    return selected_ids_list


def perform_MI(all_embeddings, total_data_len, sample_ratio=0.2):
    selected_ids_list = []
    # # 1) store cluster size
    # for cluster_i in sorted_clusters_list:
    #     cluster_i = np.array(cluster_i)
    #     cluster_size = cluster_i.shape[0]
    #     print("cluster_size: ", cluster_size)

    #     ## -- By default, we keep hard examples from groups
    #     # clutser_items_indices = list(range(cluster_size))

    #     ## -- indices for cluster items in the dataset
    #     cluster_ids = cluster_i[:, 0].astype("int32").tolist()
    #     cluster_reps = [torch.from_numpy(embedding) for i, embedding in enumerate(cluster_embedding) if i in cluster_ids]
    #     cluster_reps = torch.stack(cluster_reps)

    M = MI(all_embeddings, sample_ratio)
    selected_ids_list.extend([ind for ind, sample in M])
    #     ## -- 5) We need to keep a point from the dataset when its pairwise similarity to other point is < 1-ebs
    #     # eps_points_to_keep = cluster_ids[torch.where(M < 1 - eps)[0]]
    # threshold = torch.quantile(M, sample_ratio)
    # eps_points_to_keep = M >= threshold
    # print("eps_points", eps_points_to_keep)
    # selected_ids_list = [cluster_ids[i] for i, boo in enumerate(eps_points_to_keep) if boo]
        # selected_ids_list.append(cluster_ids[eps_points_to_keep])
    # selected_ids_list = np.concatenate(selected_ids_list)
    print("number of selected", len(selected_ids_list))  
    return selected_ids_list

def MI(cluster_reps, sample_ratio):
    ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
    # cluster_reps.cuda()
    # print("cluster_reps", cluster_reps.shape, cluster_reps[0])
    cluster_reps = F.normalize(cluster_reps, dim=1)
    # print("cluster_reps2", cluster_reps.shape, cluster_reps[0])
    # pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
    pair_w_sim_matrix = mutual_information(cluster_reps)
    pair_w_sim_matrix = pair_w_sim_matrix[:int(len(cluster_reps)*sample_ratio)]
    
    del cluster_reps
    # pair_w_sim_matrix = np.fill_diagonal(pair_w_sim_matrix, 0.0)
    pair_w_sim_matrix = torch.tensor(pair_w_sim_matrix)
    # pair_w_sim_matrix.fill_diagonal_(0.0)
    # assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]
    # print("pair_w_sim_matrix", pair_w_sim_matrix)

    ## -- get paths to cluster i images
    # image_urls = cluster[:, 0]

    ## -- make sure all the paths are unique this ensure that the duplicates are really stored many time times on memory
    # assert not self._contains_duplicates(image_urls)

    ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the compinations not permutations
    # triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)
    # pair_w_sim_matrix = torch.tensor(pair_w_sim_matrix)
    # M = torch.max(pair_w_sim_matrix, dim=0)[0].cpu().float()
    return pair_w_sim_matrix

def semdedup(cluster, cluster_reps):
    ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
    # cluster_reps.cuda()
    # print("cluster_reps", cluster_reps.shape, cluster_reps[0])
    cluster_reps = F.normalize(cluster_reps, dim=1)
    # print("cluster_reps2", cluster_reps.shape, cluster_reps[0])
    pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
    del cluster_reps
    pair_w_sim_matrix.fill_diagonal_(0.0)
    assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]
    # print("pair_w_sim_matrix", pair_w_sim_matrix)

    ## -- get paths to cluster i images
    # image_urls = cluster[:, 0]

    ## -- make sure all the paths are unique this ensure that the duplicates are really stored many time times on memory
    # assert not self._contains_duplicates(image_urls)

    ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the compinations not permutations
    triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)
    M = torch.max(triu_sim_mat, dim=0)[0].cpu().float()
    return M

def average_cosine_similarities(centroids):
    # Normalize centroids to unit vectors (required for cosine similarity)
    normalized_centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    # Compute pairwise cosine similarities
    cosine_similarities = np.dot(normalized_centroids, normalized_centroids.T)
    
    # Exclude self-similarities by setting diagonal to NaN
    np.fill_diagonal(cosine_similarities, np.nan)
    
    # Compute the average cosine similarity for each centroid to others
    avg_similarities = np.nanmean(cosine_similarities, axis=1)
    
    return avg_similarities

def perform_dbp(cluster_embedding_all, sorted_clusters_list, centroids, d, k, total_data_len=0, sample_ratio=0.2):
    selected_ids_list = []
    avg_distance_to_cent_list = []
    # 1) Calculate dintra
    for i,cluster_i in enumerate(sorted_clusters_list):
        cluster_i = np.array(cluster_i)
        print("number of cent", len(centroids))
        cluster_size = cluster_i.shape[0]
        cluster_ids = cluster_i[:, 0].astype("int32").tolist()
        cluster_embeddings = [embedding for i, embedding in enumerate(cluster_embedding_all) if i in cluster_ids]
        # avg_distance_to_cent_list.append((cluster_i[:, CLUSTER_SCHEMA['distance_to_centroid']['id']].astype("float32")).mean())
        avg_distance_to_cent_list.append(np.mean(np.linalg.norm(cluster_embeddings - centroids[i], axis=1)))
        # print("cluster_size: ", cluster_size)
    
    # 2) Calculate dinter
    # index = faiss.IndexFlatIP(d)
    # index.add(centroids)
    # Dist, _ = index.search(centroids, k + 1)
    # mean_dist = np.mean(1 - Dist[:, 1:], axis=1)
    mean_dist = average_cosine_similarities(centroids)
    assert len(mean_dist) == len(centroids)
    # print("index", index)
    # print("centroids", centroids)
    # print("Dist", Dist)
    
    # 3) Compute density
    ## load d_inter and d_intra first
    d_intra = np.array(avg_distance_to_cent_list)
    d_inter = np.array(mean_dist)
    assert d_inter.shape == d_intra.shape
    d_i = d_inter * d_intra
    # print("d_intra", d_intra)
    # print("d_inter", d_inter)
    # print("d_i", d_i)

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
    for cluster_i in sorted_clusters_list:
        cluster_i = np.array(cluster_i)
        number_of_items_in_cluster.append(cluster_i.shape[0])
        
    # calculate the number of items per cluster according to QP pruning
    num_of_items_qp = list()
    pruned_dataset_size = int(sample_ratio * total_data_len)
    number_of_items_in_cluster_pruned = pruning_qp(
        probs.data.numpy(),
        number_of_items_in_cluster,
        pruned_dataset_size,
        k
    )
    i = 0
    for cluster_i in sorted_clusters_list:
        cluster_i = np.array(cluster_i)
        cluster_all_paths = cluster_i[:, 0].astype("<U32").tolist()
        kept_data = np.array(cluster_all_paths)[
                : int(number_of_items_in_cluster_pruned[i])
        ]
        kept_data = [int(float(x)) for x in kept_data]
        selected_ids_list.append(kept_data)
        i+=1
    selected_ids_list = np.concatenate(selected_ids_list)
    return selected_ids_list


def pruning_qp(
    probabilites,
    number_of_items_in_cluster,
    pruned_dataset_size,
    num_centroids,
):

    min_samples = int(pruned_dataset_size / num_centroids)
    P = np.eye(num_centroids)
    q = -probabilites * pruned_dataset_size
    A = np.array([1.0] * num_centroids)
    b = np.array([pruned_dataset_size])
    bounds = np.array(
        [
            (
                min(min_samples, number_of_items_in_cluster[i] - 1),
                number_of_items_in_cluster[i],
            )
            for i in range(num_centroids)
        ]
    )

    x = solve_qp(P=P, q=q, A=A, b=b, lb=bounds[:, 0], ub=bounds[:, 1], solver="osqp")
    x = np.rint(x).astype(int)
    # assert sum((x < 0)) == 0

    return x


def greedy_mmd_selection(K, M):

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


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, repeat_len = 2):
      self.n = repeat_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        should_stop =False
        if input_ids.shape[1] > self.n*3:
            last_n_ids = input_ids[0][-self.n:]		# 마지막으로 생성한 n개의 토큰
            lastlast_n_ids = input_ids[0][-self.n*2:-self.n]
            lastlastlast_n_ids = input_ids[0][-self.n*2:-self.n]
            for i in range(self.n):
                if lastlastlast_n_ids[i] != lastlast_n_ids[i] or lastlast_n_ids[i] != last_n_ids[i]: # stop sequence와 비교
                    should_stop = False
                    break
                else :
                    should_stop = True
        return should_stop