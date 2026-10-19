import re
import numpy as np
import torch
import base64
import time
import sys
import termios
import tty
from typing import TYPE_CHECKING
from typing import Dict, Set, List
from PIL import Image, ImageDraw
from io import BytesIO
from matplotlib import pyplot as plt
from pathlib import Path
from skimage.draw import line
from scipy.ndimage import convolve
from logging_utils import get_logger

if TYPE_CHECKING:
    from stretch.visualization.rerun import RerunVisualizer

BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

logger = get_logger(__name__)

def compute_tilt(camera_xyz, target_xyz):
    if not isinstance(camera_xyz, np.ndarray):
        camera_xyz = np.array(camera_xyz)
    if not isinstance(target_xyz, np.ndarray):
        target_xyz = np.array(target_xyz)
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))

def match_targets(model, report: str, targets, threshold: float = 0.6) -> List[str]:
    mentions = [x.strip() for x in re.split(r"[,.]", report.lower()) if len(x.strip()) > 1]
    targets = list(dict.fromkeys(targets))
    if not mentions or not targets:
        return []
    emb = model.encode(mentions + [t.lower() for t in targets], normalize_embeddings=True, show_progress_bar=False)
    sim = emb[:len(mentions)] @ emb[len(mentions):].T
    best = sim.argmax(axis=1)
    return [t for j, t in enumerate(targets) if (sim[best == j, j] > threshold).any()]

def normalize_depth_for_matplotlib(depth, clip_min=None, clip_max=None):
    """Normalize depth image for matplotlib and return used min/max"""
    valid = depth > 0
    if np.sum(valid) == 0:
        return np.zeros_like(depth), 0.0, 1.0

    d_min = np.min(depth[valid]) if clip_min is None else clip_min
    d_max = np.max(depth[valid]) if clip_max is None else clip_max

    norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0.0, 1.0)
    norm[~valid] = 0.0
    return norm, d_min, d_max

def visualize_sample_scores_with_rerun(
    rerun_visualizer: "RerunVisualizer",
    samples: List[np.ndarray],
    scores: List[float],
    identity: str,
    base_height: float = 1.5,
    point_radius: float = 0.05,
    best_point_radius: float = 0.1

) -> None:
    rerun_visualizer.clear_identity(f"{identity}")

    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    score_min = scores.min()
    score_max = scores.max()
    score_range = score_max - score_min if (score_max > score_min) else 1e-6
    for i, (sample, score) in enumerate(zip(samples, scores)):
        norm_score = (score - score_min) / score_range # [0, 1]
        R = 1.0 - norm_score
        G = norm_score
        B = 0.0
        color = torch.tensor([R, G, B], dtype=torch.float32)
        rerun_visualizer.log_custom_pointcloud(f"{identity}/{i}", [sample[0], sample[1], base_height], color, point_radius)

    best_score, best_point = max(zip(scores, samples), key=lambda x: x[0])
    rerun_visualizer.log_custom_pointcloud(f"{identity}/best", [best_point[0], best_point[1], base_height], torch.Tensor([0, 0, 1]), best_point_radius)

def visualize_points_with_rerun(
    rerun_visualizer: "RerunVisualizer",
    samples: torch.Tensor,
    identity: str,
    color: torch.Tensor = torch.tensor([0.0, 1.0, 0.0]),  # default: green
    base_height: float = 1.5,
    point_radius: float = 0.05,
) -> None:
    rerun_visualizer.clear_identity(f"{identity}")

    for i, sample in enumerate(samples):
        x = sample[0].item(); y = sample[1].item()
        rerun_visualizer.log_custom_pointcloud(
            f"{identity}/{i}",
            [x, y, base_height],
            color,
            point_radius,
        )

def crop_nonzero_with_margin(img, margin=16, ignore_external_border=30):
    import numpy as np
    from PIL import Image

    H, W = img.shape[:2]
    if img.ndim == 3:
        mask = (img > 0.1).any(axis=2)
    else:
        mask = img > 0.1

    mask[:ignore_external_border, :] = False
    mask[-ignore_external_border:, :] = False
    mask[:, :ignore_external_border] = False
    mask[:, -ignore_external_border:] = False

    if not np.any(mask):
        start_h, end_h = 0, H
        start_w, end_w = 0, W
    else:
        ys, xs = np.where(mask)
        start_h = max(ys.min() - margin, 0)
        end_h = min(ys.max() + margin, H-1)
        start_w = max(xs.min() - margin, 0)
        end_w = min(xs.max() + margin, W-1)

    img_crop = img[start_h:end_h+1, start_w:end_w+1, ...]
    crop_h, crop_w = img_crop.shape[:2]

    img_crop_pil = Image.fromarray(
        (img_crop*255).astype(np.uint8) if img_crop.max() <= 1.0 else img_crop.astype(np.uint8)
    )
    img_zoom_pil = img_crop_pil.resize((W, H), resample=Image.NEAREST)
    return np.array(img_zoom_pil), (start_h, start_w), (crop_h, crop_w)

def draw_marker_on_zoom(
    img_zoom, orig_xy, crop_origin, crop_size, 
    color=(0,0,255), r=10, label=None, label_color=(0,0,0), font_size=18, font_offset=(16,-12),
    marker_type="circle", marker_width=0
):
    import math
    H, W = img_zoom.shape[:2]
    crop_w, crop_h = crop_size
    x_new = (orig_xy[0] - crop_origin[0]) * (W / crop_w)
    y_new = (orig_xy[1] - crop_origin[1]) * (H / crop_h)
    x_new, y_new = y_new, x_new

    if img_zoom.dtype == np.float32 or img_zoom.max() <= 1.0:
        img_zoom = (img_zoom * 255).clip(0,255).astype(np.uint8)
    img_pil = Image.fromarray(img_zoom)
    draw = ImageDraw.Draw(img_pil)

    if marker_type == "circle":
        draw.ellipse([x_new-r, y_new-r, x_new+r, y_new+r], fill=color, outline="white", width=marker_width)
    elif marker_type == "square":
        draw.rectangle([x_new-r, y_new-r, x_new+r, y_new+r], fill=color, outline="white", width=marker_width)
    elif marker_type == "cross":
        draw.line([x_new-r, y_new-r, x_new+r, y_new+r], fill=color, width=marker_width)
        draw.line([x_new-r, y_new+r, x_new+r, y_new-r], fill=color, width=marker_width)
    elif marker_type == "plus":
        draw.line([x_new, y_new-r, x_new, y_new+r], fill=color, width=marker_width)
        draw.line([x_new-r, y_new, x_new+r, y_new], fill=color, width=marker_width)
    elif marker_type == "triangle":
        points = [
            (x_new, y_new - r),
            (x_new - r * math.sin(math.pi/3), y_new + r/2),
            (x_new + r * math.sin(math.pi/3), y_new + r/2)
        ]
        draw.polygon(points, fill=color, outline="white")
    elif marker_type == "star":
        points = []
        for i in range(5):
            angle = i * (2 * math.pi / 5) - math.pi/2
            x = x_new + r * math.cos(angle)
            y = y_new + r * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color, outline="white")
    else:
        draw.ellipse([x_new-r, y_new-r, x_new+r, y_new+r], fill=color, outline="white", width=marker_width)

    if label is not None:
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
            bbox = font.getbbox(str(label))
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except Exception:
            print("DejaVuSans-Bold.ttf not found")
            font = None
            text_width = len(str(label)) * 8
            text_height = 12
        text_xy = (x_new + font_offset[0], y_new + font_offset[1])
        bg_rect = [
            text_xy[0] - 1.5,
            text_xy[1] - 1.5,
            text_xy[0] + text_width + 5,
            text_xy[1] + text_height + 8,
        ]
        temp_img = Image.new('RGBA', (int(bg_rect[2] - bg_rect[0]), int(bg_rect[3] - bg_rect[1])), (255, 255, 255, 200))
        draw.bitmap((int(bg_rect[0]), int(bg_rect[1])), temp_img)
        draw.text(text_xy, str(label), fill=label_color, font=font)
    return np.array(img_pil)

def pil_to_base64_str(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode('utf-8')

def show_maps(maps, titles=None, zoom=1.0, save_path=None):
    if isinstance(maps, (torch.Tensor, np.ndarray)):
        maps = [maps]
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for i, bm in enumerate(maps):
        if isinstance(bm, torch.Tensor):
            bm = bm.cpu().numpy()
        if zoom > 1.0:
            H, W = bm.shape[:2]
            crop_h = int(H / zoom)
            crop_w = int(W / zoom)
            start_h = (H - crop_h) // 2
            start_w = (W - crop_w) // 2
            bm = bm[start_h:start_h+crop_h, start_w:start_w+crop_w, ...]
        if bm.ndim == 2:
            axes[i].imshow(bm, cmap='gray', interpolation='nearest')
        else:
            axes[i].imshow(bm, interpolation='nearest')
        if titles is not None:
            axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved to {save_path.resolve()}")
    else:
        plt.show()

def get_line_points(points, step=2):
    trajectory_line_points = []
    
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        
        rr, cc = line(start_point[1], start_point[0], end_point[1], end_point[0])
        line_points = list(zip(cc, rr))
        
        dotted_points = []
        for idx, point in enumerate(line_points):
            if idx == 0 or idx == len(line_points) - 1 or idx % step == 0:
                dotted_points.append(point)
        
        trajectory_line_points.extend(dotted_points)
    
    trajectory_line_points = list(set(trajectory_line_points))
    return trajectory_line_points

def T_global_to_robot(x_r, y_r, theta_r):
    c, s = np.cos(theta_r), np.sin(theta_r)
    T = np.array([[ c,  s, -(x_r*c + y_r*s)],
                  [-s,  c,  (x_r*s - y_r*c)],
                  [ 0,  0,               1 ]])
    return T

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def get_grid_image(images: List[Image.Image], cols=4, border=4, num_max_images=None):
    if num_max_images is not None and len(images) > num_max_images:
        indices = np.linspace(0, len(images) - 1, num_max_images, dtype=int)
        images = [images[i] for i in indices]
    
    img_w, img_h = images[0].size
    rows = len(images) // cols
    total_w = cols * img_w + (cols - 1) * border
    total_h = rows * img_h + (rows - 1) * border
    combined = Image.new('RGB', (total_w, total_h), color='black')
    for i, img in enumerate(images):
        row, col = divmod(i, cols)
        x = col * (img_w + border)
        y = row * (img_h + border)
        combined.paste(img, (x, y))
    return combined


def compute_obstacle_density(obstacle_tensor, kernel_size=3):
    obstacle_array = obstacle_tensor.numpy().astype(np.float32)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    density = convolve(obstacle_array, kernel, mode='constant', cval=0)
    return density
