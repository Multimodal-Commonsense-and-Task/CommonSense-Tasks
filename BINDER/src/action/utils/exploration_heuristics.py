import numpy as np
import torch
from scipy.ndimage import maximum_filter
from stretch.utils.morphology import get_edges
from stretch.utils.voxel import scatter3d


def compute_reachable_map(obstacles, planner, xyt):
    if len(xyt) == 3:
        xyt = xyt[:2]

    reachable_points = planner.get_reachable_points(planner.to_pt(xyt))
    reachable_xs, reachable_ys = zip(*reachable_points)
    reachable_xs = torch.tensor(reachable_xs)
    reachable_ys = torch.tensor(reachable_ys)

    reachable_map = torch.zeros_like(obstacles)
    reachable_map[reachable_xs, reachable_ys] = 1
    reachable_map = reachable_map.to(torch.bool)

    edges = get_edges(reachable_map)
    outside_frontier = edges & ~reachable_map

    return reachable_map, outside_frontier


def compute_time_heuristic(history_soft, outside_frontier, time_smooth=0.1, time_threshold=50):
    history_soft = np.ma.masked_array(history_soft, ~outside_frontier)
    time_heuristics = history_soft.max() - history_soft
    time_heuristics = 1 / (1 + np.exp(-time_smooth * (time_heuristics - time_threshold)))
    return time_heuristics


def get_2d_alignment_heuristics(voxel_map, text):
    xyz, _, _, _ = voxel_map.semantic_memory.get_pointcloud()
    xyz = xyz.detach().cpu()
    if xyz is None:
        xyz = torch.zeros((0, 3))

    xyz = ((xyz / voxel_map.grid_resolution) + voxel_map.grid_origin).long()
    xyz[xyz[:, -1] < 0, -1] = 0

    max_height = int(voxel_map.obs_max_height / voxel_map.grid_resolution)
    grid_size = voxel_map.grid_size + [max_height]

    obs_mask = xyz[:, -1] < max_height
    xyz = xyz[obs_mask, :]

    alignments = voxel_map.find_alignment_over_model(text)[0].detach().cpu()
    alignments = alignments[obs_mask][:, None]

    alignment_heuristics = scatter3d(xyz, alignments, grid_size, "max")
    alignment_heuristics = torch.max(alignment_heuristics, dim=-1).values
    alignment_heuristics = torch.from_numpy(
        maximum_filter(alignment_heuristics.numpy(), size=5)
    )

    return alignment_heuristics


def min_max_normalize(x, mask=None):
    if mask is not None:
        x_masked = np.ma.masked_array(x, ~mask)
        min_val = x_masked.min()
        max_val = x_masked.max()
    else:
        min_val = np.min(x)
        max_val = np.max(x)
    return (x - min_val) / (max_val - min_val + 1e-6)


def compute_combined_heuristics(time_heuristics, alignment_heuristics, alignment_weight=0.7):
    time_normalized = min_max_normalize(time_heuristics)
    alignment_normalized = min_max_normalize(alignment_heuristics)
    return time_normalized + alignment_weight * alignment_normalized
