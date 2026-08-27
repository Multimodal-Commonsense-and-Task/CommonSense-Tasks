import numpy as np
import torch

def mean_absolute_error(predicted, ground_truth):
    # predicted   : [batch_size, H, W]
    # ground_truth: [batch_size, H, W]
    
    batch_mae = 0.0
    for pred, gt in zip(predicted, ground_truth):
        # Iterate for image
        # pred: [H, W]
        # gt  : [H, W]
        mask = gt > 0.0
    
        pred, gt = pred[mask], gt[mask]
        error = torch.abs(pred - gt)
        mae = error.mean()
        batch_mae += mae
    
    return batch_mae / predicted.size(0)

def n_pixel_error(predicted, ground_truth, n=1):
    # predicted   : [batch_size, H, W]
    # ground_truth: [batch_size, H, W]
    
    batch_npe = 0.0
    for pred, gt in zip(predicted, ground_truth):
        # Iterate for image
        # pred: [H, W]
        # gt  : [H, W]
        mask = gt > 0.0
    
        pred, gt = pred[mask], gt[mask]
        error = torch.abs(pred - gt)
        error_mask = error > n
        error_mask = error_mask.to(torch.float)
        npe = error_mask.mean() * 100
        batch_npe += npe
        
    return batch_npe / predicted.size(0)

def root_mean_square_error(predicted, ground_truth):
    # predicted   : [batch_size, H, W]
    # ground_truth: [batch_size, H, W]
    
    batch_rmse = 0.0
    for pred, gt in zip(predicted, ground_truth):
        # Iterate for image
        # pred: [H, W]
        # gt  : [H, W]
        mask = gt > 0.0
    
        pred, gt = pred[mask], gt[mask]
        rmse = ((pred - gt) ** 2).mean().sqrt()
        batch_rmse += rmse
    
    return batch_rmse / predicted.size(0)
