import numpy as np
import torch
from utils import compute_obstacle_density, T_global_to_robot

from .exploration_heuristics import (
    compute_reachable_map,
    compute_time_heuristic,
    compute_combined_heuristics,
    get_2d_alignment_heuristics,
)


def select_points_with_distance_constraint(
    order,
    grid_width,
    planner,
    xyt,
    k,
    min_distance_from_other_points,
    min_distance_from_robot,
    grid_to_xy_resolution=0.1,
):
    if len(xyt) == 3:
        xyt = xyt[:2]

    min_dist_grid = min_distance_from_other_points / grid_to_xy_resolution
    min_robot_dist_grid = min_distance_from_robot / grid_to_xy_resolution
    xy_grid = np.array(planner.to_pt(xyt))[::-1]  # (x, y) order

    res_xyz = []
    res_grid = []

    for idx in order:
        if len(res_xyz) >= k:
            break
        gy, gx = divmod(idx, grid_width)
        gpt = np.array([gx, gy])

        too_close_to_others = any(
            np.linalg.norm(gpt - sg) < min_dist_grid for sg in res_grid
        )
        too_close_to_robot = np.linalg.norm(xy_grid - gpt) < min_robot_dist_grid

        if too_close_to_others or too_close_to_robot:
            continue

        world_pt = planner.to_xy(torch.tensor([gx, gy]))
        res_xyz.append(world_pt)
        res_grid.append(gpt)

    return res_xyz, res_grid


def sample_k_points_from_heuristics(
    space,
    planner,
    xyt,
    target_object,
    k=3,
    grid_to_xy_resolution=0.1,
    min_distance_from_other_points=2.0,
    min_distance_from_robot=0.5,
    alignment_smooth=15,
    alignment_threshold=0.13,
):
    obstacles, _, history_soft = space.voxel_map.get_2d_map(return_history_id=True)
    _, W = obstacles.shape

    if len(xyt) == 3:
        xyt = xyt[:2]

    reachable_map, outside_frontier = compute_reachable_map(obstacles, planner, xyt)

    time_heuristics = compute_time_heuristic(history_soft, outside_frontier)
    alignment_heuristics_raw = get_2d_alignment_heuristics(space.voxel_map, target_object)

    alignments = np.ma.masked_array(alignment_heuristics_raw, ~outside_frontier)
    alignment_heuristics = 1 / (
        1 + np.exp(-alignment_smooth * (alignments - alignment_threshold))
    )

    total_heuristics = compute_combined_heuristics(time_heuristics, alignment_heuristics)

    total_h = total_heuristics.filled(-1)
    flat = total_h.ravel()
    obstacle_density_map = compute_obstacle_density(obstacles, kernel_size=5)
    flat_density = obstacle_density_map.ravel()
    order = np.lexsort((flat_density, -flat))

    world_points, grid_points = select_points_with_distance_constraint(
        order=order,
        grid_width=W,
        planner=planner,
        xyt=xyt,
        k=k,
        min_distance_from_other_points=min_distance_from_other_points,
        min_distance_from_robot=min_distance_from_robot,
        grid_to_xy_resolution=grid_to_xy_resolution,
    )

    if len(grid_points) == 0:
        top_idx = order[0]
        gy, gx = divmod(top_idx, W)
        gpt = np.array([gx, gy])
        world_pt = planner.to_xy(torch.tensor([gx, gy]))
        world_points.append(world_pt)
        grid_points.append(gpt)

    return {
        "world_points": world_points,
        "grid_points": grid_points,
        "time_heuristics": time_heuristics,
        "alignment_heuristics": alignment_heuristics,
        "total_heuristics": total_heuristics,
        "outside_frontier": outside_frontier,
        "obstacles": obstacles,
        "history_soft": history_soft,
    }


def find_optimal_look_order(target_xyz_list, robot_pose):
    robot_x, robot_y, robot_theta = robot_pose
    T_robot_global = T_global_to_robot(robot_x, robot_y, robot_theta)

    head_only = []
    requires_base = []

    for i, target_xyz in enumerate(target_xyz_list):
        target_x, target_y, target_z = target_xyz
        target_robot = T_robot_global @ np.array([target_x, target_y, 1])
        target_robot_theta = np.arctan2(target_robot[1], target_robot[0])

        if target_robot_theta > np.pi / 4 or target_robot_theta < -np.pi:
            requires_base.append(i)
        else:
            head_only.append(i)

    return head_only + requires_base


def grid_points_to_xyt(grid_points, voxel_map):
    return [
        voxel_map.grid_coords_to_xyt(torch.tensor([pt[1], pt[0]]))
        for pt in grid_points
    ]