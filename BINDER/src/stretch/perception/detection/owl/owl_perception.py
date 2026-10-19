# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# (c) 2024 Hello Robot by Atharva Pusalkar
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
import torch
import torchvision
from transformers import AutoProcessor, Owlv2ForObjectDetection
from PIL import Image
from stretch.utils.image import Camera, camera_xyz_to_global_xyz

default_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

class OwlPerception:
    def __init__(
        self,
        version="owlv2-L-p14-ensemble",
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = 0.2,
    ):
        """Load trained OWL model for inference.

        Arguments:
            version: owlv2 version, currently supporting google/owlv2-large-patch14-ensemble and google/owlv2-base-patch16-ensemble
            device: which device you want to run the model,
        """
        self.confidence_threshold = confidence_threshold
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if version == "owlv2-L-p14-ensemble":
            configuration = "google/owlv2-large-patch14-ensemble"
        elif version == "owlv2-B-p16-ensemble":
            configuration = "google/owlv2-base-patch16-ensemble"
        elif version == "owlv2-B-p16":
            configuration = "google/owlv2-base-patch16"
        else:
            raise ValueError("Owlv2 version not implemented yet!")

        self.processor = AutoProcessor.from_pretrained(configuration)
        self.model = Owlv2ForObjectDetection.from_pretrained(configuration).to(self.device)

        print(f"Loaded owl model from {configuration}")

    def detect_object(
        self,
        rgb: torch.Tensor,
        text: str,
        confidence_threshold: Optional[float] = None,
    ):
        """Try to find target objects given one text query.
        Arguments:
            rgb: ideally of shape (C, H, W), the pixel value should be integer between [0, 255]
        """
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.Tensor(rgb).to(torch.uint8)
        rgb = rgb.squeeze()
        if rgb.shape[0] != 3:
            rgb = rgb.permute(2, 0, 1)
        inputs = self.processor(text=[["a photo of a " + text]], images=rgb, return_tensors="pt")
        for input in inputs:
            inputs[input] = inputs[input].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([rgb.size()[-2:]]).to(self.device)
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )[0]

        return results["scores"], results["boxes"]
    
    def compute_obj_coord(
        self,
        text: str,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        camera_K: torch.Tensor,
        camera_pose: torch.Tensor,
        confidence_threshold: Optional[float] = None,
        depth_threshold: float = 3.0
    ):
        height, width = depth.squeeze().shape
        camera = Camera.from_K(np.array(camera_K), width=width, height=height)
        camera_xyz = camera.depth_to_xyz(np.array(depth))  # [H, W, 3]
        xyz = torch.Tensor(camera_xyz_to_global_xyz(camera_xyz, np.array(camera_pose)))  # [H, W, 3]

        scores, boxes = self.detect_object( # scores: [N], boxes: [N, 4], text='black toy'
            rgb=rgb, text=text, confidence_threshold=confidence_threshold
        )
        if len(scores) == 0:
            return None, (rgb, rgb), None
        tl_x, tl_y, br_x, br_y = boxes[0]
        drawn_rgb = rgb.clone() # [H, W, 3], torch.float32, 0 ~ 255.0
        H, W = depth.squeeze().shape
        # result = torchvision.utils.draw_bounding_boxes(drawn_rgb.to(torch.uint8).permute(2, 0, 1), torch.tensor([int(tl_x), int(tl_y), int(br_x), int(br_y)]).unsqueeze(0), labels=[text], colors="red", width=4, font=default_font_path, font_size=20)
        # Image.fromarray(result.permute(1, 2, 0).cpu().numpy()).save("test.png")

        for idx, (score, bbox) in enumerate(
            sorted(zip(scores, boxes), key=lambda x: x[0], reverse=True)
        ):
            tl_x, tl_y, br_x, br_y = bbox
            
            # Normalize coordinates
            x1 = int(max(0, tl_x.item()))
            y1 = int(max(0, tl_y.item()))
            x2 = int(min(W, br_x.item()))
            y2 = int(min(H, br_y.item()))
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop
            m = 8
            xi1 = max(0, x1 + m); yi1 = max(0, y1 + m)
            xi2 = min(W, x2 - m); yi2 = min(H, y2 - m)
            if xi2 <= xi1 or yi2 <= yi1:
                xi1, yi1, xi2, yi2 = x1, y1, x2, y2

            # Get RoI
            roi_depth = depth[yi1:yi2, xi1:xi2]
            roi_xyz = xyz[yi1:yi2, xi1:xi2].reshape(-1, 3)

            # Get valid depth mask (without 0/NaN/Inf)
            valid = torch.isfinite(roi_depth).reshape(-1) & (roi_depth.reshape(-1) > 0)
            valid_ratio = valid.float().mean().item()
            if valid_ratio < 0.2:
                continue
            roi_xyz_valid = roi_xyz[valid]

            # Remove outliers using z percentile
            z = roi_xyz_valid[:, 2]
            lo = torch.quantile(z, 0.10)
            hi = torch.quantile(z, 0.90)
            band = (z >= lo) & (z <= hi)
            roi_xyz_robust = roi_xyz_valid[band]
            if roi_xyz_robust.shape[0] < 10:
                continue

            # Final object coordinate
            obj_coord = roi_xyz_robust.median(dim=0).values
            print(f"[{text}] score={float(score):.3f} | valid={valid_ratio*100:.1f}% | obj_coord={obj_coord.tolist()}")

            # Draw bbox with text
            drawn_rgb_with_text = torchvision.utils.draw_bounding_boxes(
                drawn_rgb.to(torch.uint8).permute(2, 0, 1),
                torch.tensor([x1, y1, x2, y2]).unsqueeze(0),
                labels=[text],
                colors="red",
                width=4,
                font=default_font_path,
                font_size=20
            )

            # Draw bbox only
            margin = 10
            margin_tl_x = max(0, x1 - margin)
            margin_tl_y = max(0, y1 - margin)
            margin_br_x = min(W, x2 + margin)
            margin_br_y = min(H, y2 + margin)
            drawn_rgb_bbox_only = torchvision.utils.draw_bounding_boxes(
                drawn_rgb.to(torch.uint8).permute(2, 0, 1),
                torch.tensor([margin_tl_x, margin_tl_y, margin_br_x, margin_br_y]).unsqueeze(0),
                colors="red",
                width=2
            )
            
            return obj_coord, (drawn_rgb_with_text.permute(1, 2, 0), drawn_rgb_bbox_only.permute(1, 2, 0)), score # [H, W, 3]

        return None, (rgb, rgb), None
    # def compute_obj_coord(
    #     self,
    #     text: str,
    #     rgb: torch.Tensor,
    #     depth: torch.Tensor,
    #     camera_K: torch.Tensor,
    #     camera_pose: torch.Tensor,
    #     confidence_threshold: Optional[float] = None,
    #     depth_threshold: float = 3.0
    # ):
    #     height, width = depth.squeeze().shape
    #     camera = Camera.from_K(np.array(camera_K), width=width, height=height)
    #     camera_xyz = camera.depth_to_xyz(np.array(depth))  # [H, W, 3]
    #     xyz = torch.Tensor(camera_xyz_to_global_xyz(camera_xyz, np.array(camera_pose)))  # [H, W, 3]

    #     scores, boxes = self.detect_object( # scores: [N], boxes: [N, 4], text='black toy'
    #         rgb=rgb, text=text, confidence_threshold=confidence_threshold
    #     )
    #     if len(scores) == 0:
    #         return None, (rgb, rgb), None
    #     tl_x, tl_y, br_x, br_y = boxes[0]
    #     drawn_rgb = rgb.clone() # [H, W, 3], torch.float32, 0 ~ 255.0
    #     # result = torchvision.utils.draw_bounding_boxes(drawn_rgb.to(torch.uint8).permute(2, 0, 1), torch.tensor([int(tl_x), int(tl_y), int(br_x), int(br_y)]).unsqueeze(0), labels=[text], colors="red", width=4, font=default_font_path, font_size=20)
    #     # Image.fromarray(result.permute(1, 2, 0).cpu().numpy()).save("test.png")

    #     for idx, (score, bbox) in enumerate(
    #         sorted(zip(scores, boxes), key=lambda x: x[0], reverse=True)
    #     ):
    #         tl_x, tl_y, br_x, br_y = bbox
    #         w, h = depth.shape
    #         tl_x, tl_y, br_x, br_y = (
    #             int(max(0, tl_x.item())),
    #             int(max(0, tl_y.item())),
    #             int(min(h, br_x.item())),
    #             int(min(w, br_y.item())),
    #         )

    #         if torch.min(depth[tl_y:br_y, tl_x:br_x].reshape(-1)) < depth_threshold: # 3.0
    #             # Draw bounding box with text
    #             drawn_rgb_with_text = torchvision.utils.draw_bounding_boxes(
    #                 drawn_rgb.to(torch.uint8).permute(2, 0, 1), # [3, H, W]
    #                 torch.tensor([int(tl_x), int(tl_y), int(br_x), int(br_y)]).unsqueeze(0), # [1, 4]
    #                 labels=[text],
    #                 colors="red",
    #                 width=4,
    #                 font=default_font_path,
    #                 font_size=20
    #             ) # [3, H, W]
                
    #             # Add margin for bounding box without text
    #             margin = 10  # pixels
    #             margin_tl_x = max(0, int(tl_x) - margin)
    #             margin_tl_y = max(0, int(tl_y) - margin)
    #             margin_br_x = min(h, int(br_x) + margin)
    #             margin_br_y = min(w, int(br_y) + margin)
                
    #             # Draw bounding box only (without text) with margin
    #             drawn_rgb_bbox_only = torchvision.utils.draw_bounding_boxes(
    #                 drawn_rgb.to(torch.uint8).permute(2, 0, 1), # [3, H, W]
    #                 torch.tensor([margin_tl_x, margin_tl_y, margin_br_x, margin_br_y]).unsqueeze(0), # [1, 4]
    #                 colors="red",
    #                 width=2
    #             ) # [3, H, W]

    #             obj_coord = torch.median(xyz[tl_y:br_y, tl_x:br_x].reshape(-1, 3), dim=0).values
    #             return obj_coord, (drawn_rgb_with_text.permute(1, 2, 0), drawn_rgb_bbox_only.permute(1, 2, 0)), score # [H, W, 3]

    #     return None, (rgb, rgb), None
