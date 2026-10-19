import numpy as np
import torch
import matplotlib.cm as cm
from PIL import Image

from utils import crop_nonzero_with_margin, draw_marker_on_zoom, get_line_points, show_maps


def create_heuristics_image(heuristics, cmap_name='viridis'):
    h_min = heuristics.min()
    h_max = heuristics.max()
    h_range = h_max - h_min
    if h_range > 0:
        h_normalized = (heuristics - h_min) / h_range
    else:
        h_normalized = np.zeros_like(heuristics)

    cmap = cm.get_cmap(cmap_name)
    return cmap(h_normalized).astype(np.float32)[..., :3]


def generate_visual_prompt(
    space,
    planner,
    candidate_points,
    previous_trajectories,
    base_pose,
    save_path=None,
):
    obstacles, _, _ = space.voxel_map.get_2d_map(return_history_id=True)

    reachable_points = planner.get_reachable_points(planner.to_pt(base_pose))
    reachable_xs, reachable_ys = zip(*reachable_points)
    reachable_xs = torch.tensor(reachable_xs)
    reachable_ys = torch.tensor(reachable_ys)

    reachable_map = torch.zeros_like(obstacles)
    reachable_map[reachable_xs, reachable_ys] = 1
    reachable_map = reachable_map.to(torch.bool)

    obstacles_np = obstacles.cpu().numpy().astype(bool)
    reachable_map_np = reachable_map.cpu().numpy().astype(bool)

    H, W = obstacles_np.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[reachable_map_np] = [0, 1, 0]
    img[obstacles_np] = [1, 0, 0]

    if len(previous_trajectories) > 0:
        previous_trajectories_pt = [
            [planner.to_pt(point) for point in trajectory]
            for trajectory in previous_trajectories
        ]
        for traj_pt in previous_trajectories_pt:
            traj_pt_line = get_line_points(traj_pt, step=2)
            for x, y in traj_pt_line:
                if 0 <= x < H and 0 <= y < W:
                    img[x, y] = [0.6, 0.6, 0.6]

    img_zoom, crop_origin, crop_size = crop_nonzero_with_margin(img, margin=10)
    base_pose_pt = planner.to_pt(base_pose)

    marked_image = draw_marker_on_zoom(
        img_zoom, base_pose_pt, crop_origin, crop_size,
        color=(255, 255, 0), label='robot', marker_type="square"
    )

    for i, point in enumerate(candidate_points):
        point_pt = planner.to_pt(point)
        marked_image = draw_marker_on_zoom(
            marked_image, point_pt, crop_origin, crop_size,
            color=(0, 0, 255), label=f'candidate {i + 1}',
            marker_type="circle", label_color=(255, 0, 0)
        )

    if save_path:
        Image.fromarray(marked_image).save(save_path)

    return marked_image


def save_heuristics_visualization(
    history_soft,
    time_heuristics,
    alignment_heuristics,
    total_heuristics,
    outside_frontier,
    grid_points,
    robot_grid_pos,
    save_path,
):
    cmap = cm.get_cmap('viridis')

    h_min = history_soft.min()
    h_max = history_soft.max()
    h_range = h_max - h_min
    if h_range > 0:
        h_normalized = (history_soft - h_min) / h_range
    else:
        h_normalized = np.zeros_like(history_soft)
    history_img = cmap(h_normalized).astype(np.float32)[..., :3]
    history_img[outside_frontier] = [0, 0, 1]

    time_img = create_heuristics_image(time_heuristics)
    alignment_img = create_heuristics_image(alignment_heuristics)
    total_img = create_heuristics_image(total_heuristics)

    for x, y in grid_points:
        total_img[int(y), int(x)] = [1, 0, 0]

    rx, ry = robot_grid_pos
    total_img[int(ry), int(rx)] = [1, 1, 0]

    show_maps(
        [history_img, time_img, alignment_img, total_img],
        titles=["history_soft", "time_heuristics", "alignment_heuristics", "total_heuristics"],
        zoom=9.0,
        save_path=save_path,
    )

    return total_img
