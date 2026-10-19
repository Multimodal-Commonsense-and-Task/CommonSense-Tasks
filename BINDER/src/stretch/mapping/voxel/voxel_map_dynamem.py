# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from stretch.motion import Footprint
from stretch.utils.morphology import binary_dilation, get_edges

from .voxel_dynamem import SparseVoxelMap
from .voxel_map import SparseVoxelMapNavigationSpace as SparseVoxelMapNavigationSpaceBase
from logging_utils import get_logger

logger = get_logger(__name__)

class SparseVoxelMapNavigationSpace(SparseVoxelMapNavigationSpaceBase):

    # Used for making sure we do not divide by zero anywhere
    tolerance: float = 1e-8

    def __init__(
        self,
        voxel_map: SparseVoxelMap,
        step_size: float = 0.1,
        rotation_step_size: float = 0.5,
        use_orientation: bool = False,
        orientation_resolution: int = 64,
        dilate_frontier_size: int = 12,
        dilate_obstacle_size: int = 2,
        extend_mode: str = "separate",
    ):
        super().__init__(
            voxel_map=voxel_map,
            robot=None,
            step_size=step_size,
            rotation_step_size=rotation_step_size,
            use_orientation=use_orientation,
            orientation_resolution=orientation_resolution,
            dilate_frontier_size=dilate_frontier_size,
            dilate_obstacle_size=dilate_obstacle_size,
            extend_mode=extend_mode,
        )
        self.create_collision_masks(orientation_resolution)
        self.traj = None

    def extract_outer_edges(self, tensor_mask: torch.Tensor) -> torch.Tensor:
        """
        reachable_map: (H, W) torch.Tensor (bool) on CUDA or CPU
        Returns: torch.Tensor (bool) of same shape, with only outer edge
        """

        # 1. Convert to numpy (uint8) for OpenCV
        mask_np = tensor_mask.cpu().numpy().astype(np.uint8) * 255

        # 2. Find only external contours
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. Create empty image and draw only outer contour
        outer_edge = np.zeros_like(mask_np)
        cv2.drawContours(outer_edge, contours, -1, color=255, thickness=1)

        # 4. Convert back to torch.Tensor (bool)
        return torch.from_numpy(outer_edge > 0).to(tensor_mask.device)

    def create_collision_masks(self, orientation_resolution: int):
        """Create a set of orientation masks

        Args:
            orientation_resolution: number of bins to break it into
        """
        self._footprint = Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1)
        self._orientation_resolution = 64
        self._oriented_masks = []

        for i in range(orientation_resolution):
            theta = i * 2 * np.pi / orientation_resolution
            mask = self._footprint.get_rotated_mask(
                self.voxel_map.grid_resolution, angle_radians=theta
            )
            self._oriented_masks.append(mask)

    def compute_theta(self, cur_x, cur_y, end_x, end_y):
        theta = 0
        if end_x == cur_x and end_y >= cur_y:
            theta = np.pi / 2
        elif end_x == cur_x and end_y < cur_y:
            theta = -np.pi / 2
        else:
            theta = np.arctan((end_y - cur_y) / (end_x - cur_x))
            if end_x < cur_x:
                theta = theta + np.pi
            if theta > np.pi:
                theta = theta - 2 * np.pi
            if theta < -np.pi:
                theta = theta + 2 * np.pi
        return theta


    def sample_point_for_recovery(
        self,
        start: torch.Tensor,
        target_point: np.ndarray,
        planner,
        min_distance_from_robot: float = 0.5,
    ) -> np.ndarray:
        """ Sample a point for recovery - localize target point again with small margin

        Sampled point should be:
        1. In the line connecting target_point and robot's current position
        2. Farther from the target than the robot is (i.e., farther from the target point than the robot is)
        3. Must be in free space & reachable

        Returns:
            [px, py, theta] or None if no valid point found
        """

        obstacles, explored = self.voxel_map.get_2d_map()
        start_pt = planner.to_pt(start)
        reachable_points = planner.get_reachable_points(start_pt)
        if len(reachable_points) == 0:
            print("No target point found; maybe no point is reachable.")
            return None

        reachable_xs, reachable_ys = zip(*reachable_points)
        reachable_xs = torch.tensor(reachable_xs)
        reachable_ys = torch.tensor(reachable_ys)
        reachable = torch.empty(obstacles.shape, dtype=torch.bool).fill_(False)
        reachable[reachable_xs, reachable_ys] = True

        obstacles, explored = self.voxel_map.get_2d_map()
        reachable = reachable & ~obstacles
        start_xy = np.array([start[0].item(), start[1].item()])
        target_xy = np.array([target_point[0], target_point[1]])

        robot_to_target = target_xy - start_xy
        dist_robot_target = np.linalg.norm(robot_to_target)

        if dist_robot_target < 1e-8:
            print("Robot and target are in nearly the same position or invalid.")
            return None

        direction = start_xy - target_xy
        direction_norm = np.linalg.norm(direction)
        direction /= direction_norm

        t_candidates = np.linspace(0.5, 2.0, num=20)

        for t in t_candidates:
            px, py = start_xy + t * direction

            dist_pt_target = np.linalg.norm([px - target_xy[0], py - target_xy[1]])
            if dist_pt_target <= dist_robot_target:
                continue

            gx, gy = planner.to_pt([px, py])
            if not (0 <= gx < obstacles.shape[0] and 0 <= gy < obstacles.shape[1]):
                continue
            if not reachable[gx, gy]:
                continue

            dx = target_xy[0] - px
            dy = target_xy[1] - py
            theta = np.arctan2(dy, dx)
            return np.array([px, py, theta], dtype=np.float32)

        print("No valid recovery point found on the line from robot to target.")
        return None
    
    def sample_point_for_recovery_rectangular(
        self,
        start: torch.Tensor,
        target_point: np.ndarray,
        planner,
        min_distance_from_robot: float = 0.5,
        max_distance_from_robot: float = 2.0,
        lateral_extent: float = 0.5,
        num_longitudinal_samples: int = 20,
        num_lateral_samples: int = 11,
        rerun_visualizer = None
    ) -> np.ndarray:
        obstacles, explored = self.voxel_map.get_2d_map()
        start_pt = planner.to_pt(start)
        reachable_points = planner.get_reachable_points(start_pt)
        if len(reachable_points) == 0:
            print("[sample_point_for_recovery] No reachable point from robot's current position.")
            return None

        reachable_xs, reachable_ys = zip(*reachable_points)
        reachable_xs = torch.tensor(reachable_xs)
        reachable_ys = torch.tensor(reachable_ys)
        reachable_mask = torch.empty(obstacles.shape, dtype=torch.bool).fill_(False)
        reachable_mask[reachable_xs, reachable_ys] = True

        obstacles, explored = self.voxel_map.get_2d_map()
        reachable_mask = reachable_mask & ~obstacles

        start_xy = np.array([start[0].item(), start[1].item()])
        target_xy = np.array([target_point[0], target_point[1]])
        target_to_robot = start_xy - target_xy
        dist_robot_target = np.linalg.norm(target_to_robot)

        if dist_robot_target < 1e-8:
            print("[sample_point_for_recovery] Robot and target are in nearly the same position.")
            return None

        direction = target_to_robot / dist_robot_target
        perp = np.array([-direction[1], direction[0]])

        longitudinal_dists = np.linspace(min_distance_from_robot, max_distance_from_robot, 
                                        num_longitudinal_samples)
        lateral_offsets = np.linspace(-lateral_extent, lateral_extent, num_lateral_samples)

        best_score = -1.0
        best_point = None

        rerun_visualizer_samples = []; sample_scores = []

        for ld in longitudinal_dists:
            for lo in lateral_offsets:
                px_py = start_xy + ld * direction + lo * perp  # (px, py)
                
                dist_pt_target = np.linalg.norm(px_py - target_xy)
                if dist_pt_target <= dist_robot_target:
                    continue

                gx, gy = planner.to_pt(px_py)
                if not (0 <= gx < obstacles.shape[0] and 0 <= gy < obstacles.shape[1]):
                    continue
                if not reachable_mask[gx, gy]:
                    continue

                dx = target_xy[0] - px_py[0]
                dy = target_xy[1] - px_py[1]
                theta = np.arctan2(dy, dx)

                score_longitudinal = np.exp(-((ld - 0.5) ** 2) / 0.09)
                score_lateral = np.exp(-(lo ** 2) / 0.25)
                score = score_longitudinal * score_lateral

                if rerun_visualizer is not None:
                    rerun_visualizer_samples.append(px_py); sample_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_point = (px_py[0], px_py[1], theta)

        if best_point is None:
            return None
            
        if rerun_visualizer is not None:
            scores = np.array(sample_scores)
            score_min = scores.min()
            score_max = scores.max()
            score_range = score_max - score_min if (score_max > score_min) else 1e-6
            for i, (sample, score) in enumerate(zip(rerun_visualizer_samples, sample_scores)):
                norm_score = (score - score_min) / score_range
                R = 1.0 - norm_score
                G = norm_score
                B = 0.0
                color = torch.tensor([R, G, B], dtype=torch.float32)
                rerun_visualizer.log_custom_pointcloud(f"world/recovery_samples/{i}", [sample[0], sample[1], 1.5], color, 0.03)

            rerun_visualizer.log_custom_pointcloud("world/recovery_best_point", [best_point[0], best_point[1], 1.5], torch.Tensor([0, 0, 1]), 0.1)
        
        return np.array(best_point, dtype=np.float32)
    
    def sample_target_point(
        self, start: torch.Tensor, point: torch.Tensor, planner, exploration: bool = False, edge_margin: float = 0.0,
        rerun_visualizer = None, obstacle_margin: int = 3, is_exploration: bool = False, close_threshold: float = 0.7
    ) -> Optional[np.ndarray]:
        obstacles, explored = self.voxel_map.get_2d_map()
        start_pt = planner.to_pt(start)
        reachable_points = planner.get_reachable_points(start_pt)
        if len(reachable_points) == 0:
            print("No target point find, maybe no point is reachable")
            return None
        reachable_xs, reachable_ys = zip(*reachable_points)
        reachable_xs = torch.tensor(reachable_xs)
        reachable_ys = torch.tensor(reachable_ys)

        reachable_map = torch.zeros_like(obstacles)
        reachable_map[reachable_xs, reachable_ys] = 1
        reachable_map = reachable_map.to(torch.bool)
        edges = self.extract_outer_edges(reachable_map)

        reachable = torch.empty(obstacles.shape, dtype=torch.bool).fill_(False)
        reachable[reachable_xs, reachable_ys] = True

        obstacles, explored = self.voxel_map.get_2d_map()
        reachable = reachable & ~obstacles

        target_x, target_y = planner.to_pt(point)

        xs, ys = torch.where(reachable)
        if len(xs) < 1:
            print("No target point find, maybe no point is reachable")
            return None

        target_xy = torch.tensor([target_x, target_y], dtype=torch.float32)
        start_xy = torch.tensor(start, dtype=torch.float32)
        all_pts = torch.stack([xs, ys], dim=-1).float()

        all_pts_world = torch.tensor([planner.to_xy([x.item(), y.item()]) for x, y in all_pts], dtype=torch.float32)
        target_xy_world = torch.tensor(planner.to_xy((target_x, target_y)), dtype=torch.float32)
        start_xy_world = torch.tensor([start[0], start[1]], dtype=torch.float32)
        dist_to_target = torch.norm(all_pts_world - target_xy_world, dim=-1)
        dist_to_robot = torch.norm(all_pts_world - start_xy_world, dim=-1) if not is_exploration else 0.0
        
        preferred_dist = 0.5 if not is_exploration else 0.0
        penalty = torch.abs(dist_to_target - preferred_dist)
        score = -penalty - 0.15 * dist_to_robot
        sorted_indices = torch.argsort(score, descending=True)

        if rerun_visualizer is not None:
            filter_threshold = 0.8
            filtered_pts = all_pts[dist_to_target <= filter_threshold]; filtered_score = score[dist_to_target <= filter_threshold]

            if len(filtered_pts) > 0:
                filtered_scores = np.array(filtered_score)
                filtered_score_min = filtered_scores.min()
                filtered_score_max = filtered_scores.max()
                filtered_score_range = filtered_score_max - filtered_score_min if (filtered_score_max > filtered_score_min) else 1e-6
                for i, (sample, score) in enumerate(zip(filtered_pts, filtered_score)):
                    norm_score = (score - filtered_score_min) / filtered_score_range  # [0, 1]
                    R = 1.0 - norm_score
                    G = norm_score
                    B = 0.0
                    color = torch.tensor([R, G, B], dtype=torch.float32)
                    visualize_point = planner.to_xy([sample[0], sample[1]])
                    rerun_visualizer.log_custom_pointcloud(f"world/target_point_score/{i}", [visualize_point[0], visualize_point[1], 1.5], color, 0.03)

        selected_targets = all_pts[sorted_indices].to(torch.long)
        
        edge_indices = torch.stack(torch.where(edges)).T
        edges_xy_global = np.array([planner.to_xy((i.item(), j.item())) for i, j in edge_indices])
        edge_tree = cKDTree(edges_xy_global)

        for selected_target in selected_targets:
            selected_x, selected_y = planner.to_xy([selected_target[0], selected_target[1]])
            if not is_exploration:
                theta = self.compute_theta(selected_x, selected_y, point[0], point[1]) + np.pi / 2
            else:
                theta = self.compute_theta(selected_x, selected_y, point[0], point[1])
            if theta > np.pi:
                theta -= 2 * np.pi
            elif theta < -np.pi:
                theta += 2 * np.pi

            target_is_valid = self.is_valid(np.array([selected_x, selected_y, theta]))
            if not target_is_valid:
                continue
            if not is_exploration and np.linalg.norm([selected_x - point[0], selected_y - point[1]]) <= 0.35:
                continue
            elif np.linalg.norm([selected_x - point[0], selected_y - point[1]]) <= 0.5:
                i = (point[0] - selected_target[0]) // abs(point[0] - selected_target[0])
                j = (point[1] - selected_target[1]) // abs(point[1] - selected_target[1])
                index_i = int(selected_target[0].int() + i)
                index_j = int(selected_target[1].int() + j)
                if obstacles[index_i][index_j]:
                    target_is_valid = False

            if obstacles[selected_target[0] - obstacle_margin:selected_target[0] + obstacle_margin + 1, selected_target[1] - obstacle_margin:selected_target[1] + obstacle_margin + 1].any():
                target_is_valid = False

            if not target_is_valid:
                continue
                
            distance_to_target = np.linalg.norm([selected_x - point[0], selected_y - point[1]])
            is_close_to_target = distance_to_target <= close_threshold

            if not is_close_to_target and edge_margin > 0.0:
                distance_to_edge, _ = edge_tree.query([selected_x, selected_y], k=1)
                if distance_to_edge < edge_margin:
                    continue

            return np.array([selected_x, selected_y, theta]), is_close_to_target, distance_to_target

        return None
    
    

    def sample_exploration(
        self,
        xyt,
        planner,
        use_alignment_heuristics=True,
        text=None,
        debug=False,
    ):
        obstacles, explored, history_soft = self.voxel_map.get_2d_map(return_history_id=True)
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
        kernel = None
        if kernel is not None:
            expanded_frontier = binary_dilation(
                edges.float().unsqueeze(0).unsqueeze(0),
                kernel,
            )[0, 0].bool()
        else:
            expanded_frontier = edges
        outside_frontier = expanded_frontier & ~reachable_map
        time_heuristics = self._time_heuristic(history_soft, outside_frontier, debug=debug)
        if (
            use_alignment_heuristics
            and len(self.voxel_map.semantic_memory._points) > 0
            and text != ""
            and text is not None
        ):
            alignments_heuristics = self.voxel_map.get_2d_alignment_heuristics(text)
            alignments_heuristics = self._alignment_heuristic(
                alignments_heuristics, outside_frontier, debug=debug
            )
            total_heuristics = time_heuristics
        else:
            alignments_heuristics = None
            total_heuristics = time_heuristics

        rounded_heuristics = np.ceil(total_heuristics * 200) / 200
        max_heuristic = rounded_heuristics.max()
        indices = np.column_stack(np.where(rounded_heuristics == max_heuristic))
        closest_index = np.argmax(np.linalg.norm(indices - np.asarray(planner.to_pt(xyt)), axis=-1))
        index = indices[closest_index]
        if debug:
            from matplotlib import pyplot as plt

            plt.subplot(221)
            plt.imshow(obstacles.int() * 5 + outside_frontier.int() * 10)
            plt.subplot(222)
            plt.imshow(explored.int() * 5)
            plt.subplot(223)
            plt.imshow(total_heuristics)
            plt.scatter(index[1], index[0], s=15, c="g")
            plt.subplot(224)
            plt.imshow(history_soft)
            plt.scatter(index[1], index[0], s=15, c="g")
            plt.show()
        return index, time_heuristics, alignments_heuristics, total_heuristics

    def _alignment_heuristic(
        self,
        alignments,
        outside_frontier,
        alignment_smooth=15,
        alignment_threshold=0.13,
        debug=False,
    ):
        alignments = np.ma.masked_array(alignments, ~outside_frontier)
        alignment_heuristics = 1 / (
            1 + np.exp(-alignment_smooth * (alignments - alignment_threshold))
        )
        index = np.unravel_index(np.argmax(alignment_heuristics), alignments.shape)
        if debug:
            plt.clf()
            plt.title("alignment")
            plt.imshow(alignment_heuristics)
            plt.scatter(index[1], index[0], s=15, c="g")
            plt.show()
        return alignment_heuristics

    def _time_heuristic(
        self, history_soft, outside_frontier, time_smooth=0.1, time_threshold=50, debug=False
    ):
        history_soft = np.ma.masked_array(history_soft, ~outside_frontier)
        time_heuristics = history_soft.max() - history_soft
        time_heuristics[history_soft < 1] = float("inf")
        time_heuristics = 1 / (1 + np.exp(-time_smooth * (time_heuristics - time_threshold)))
        index = np.unravel_index(np.argmax(time_heuristics), history_soft.shape)
        if debug:
            plt.title("time")
            plt.imshow(time_heuristics)
            plt.scatter(index[1], index[0], s=15, c="r")
            plt.show()
        return time_heuristics

    def to_pt(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        xy = np.array([xy[0], xy[1]])  # type: ignore
        pt = self.voxel_map.xy_to_grid_coords(xy)  # type: ignore
        return int(pt[0]), int(pt[1])

    def to_xy(self, pt: Tuple[int, int]) -> Tuple[float, float]:
        pt = np.array([pt[0], pt[1]])  # type: ignore
        xy = self.voxel_map.grid_coords_to_xy(pt)  # type: ignore
        return float(xy[0]), float(xy[1])

    def sample_navigation(self, start, planner, point, mode="navigation", edge_margin=0.0, rerun_visualizer=None, is_exploration=False, obstacle_margin=3, close_threshold=0.7):
        import matplotlib
        matplotlib.use("Agg")
        plt.clf()
        if point is None:
            start_pt = self.to_pt(start)
            return None

        goal, is_close_to_target, distance_to_target = self.sample_target_point(start, point, planner, exploration=mode != "navigation", edge_margin=edge_margin,
                                                            rerun_visualizer=rerun_visualizer, is_exploration=is_exploration, obstacle_margin=obstacle_margin, close_threshold=close_threshold)
        
        obstacles, explored = self.voxel_map.get_2d_map()
        plt.imshow(obstacles)
        start_pt = self.to_pt(start)
        plt.scatter(start_pt[1], start_pt[0], s=15, c="b")
        point_pt = self.to_pt(point)
        plt.scatter(point_pt[1], point_pt[0], s=15, c="r")
        if goal is not None:
            goal_pt = self.to_pt(goal)
            plt.scatter(goal_pt[1], goal_pt[0], s=10, c="g")
        return goal, is_close_to_target, distance_to_target

    def sample_frontier(self, planner, start_pose=[0, 0, 0], text=None):
        if text is not None and text != "":
            (
                index,
                time_heuristics,
                alignments_heuristics,
                total_heuristics,
            ) = self.sample_exploration(
                start_pose,
                planner,
                use_alignment_heuristics=True,
                text=text,
                debug=False,
            )
        else:
            index, time_heuristics, _, total_heuristics = self.sample_exploration(
                start_pose,
                planner,
                use_alignment_heuristics=False,
                text=None,
                debug=False,
            )
            alignments_heuristics = time_heuristics

        obstacles, explored = self.voxel_map.get_2d_map()
        return self.voxel_map.grid_coords_to_xyt(torch.tensor([index[0], index[1]]))
