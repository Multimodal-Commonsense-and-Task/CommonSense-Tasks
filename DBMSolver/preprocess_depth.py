from PIL import Image
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2


from joblib import Parallel, delayed
from tqdm import tqdm
import csv

SPLIT = "train"
img_size = 256
data_dir = "/root/data/DIODE/data_list/train_outdoor.csv"


def plot_depth_map(dm, validity_mask, name):
    validity_mask = validity_mask > 0
    MIN_DEPTH = 0.5
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    dm = np.log(dm, where=validity_mask)

    dm = np.ma.masked_where(~validity_mask, dm)

    cmap = plt.cm.get_cmap("jet")
    cmap.set_bad(color="black")
    norm = plt.Normalize(vmin=0, vmax=np.log(MAX_DEPTH + 1.01))
    image = cmap(norm(dm))
    plt.imsave(name, np.clip(image, 0.0, 1.0))


def plot_normal_map(normal_map, name):
    normal_viz = normal_map[:, ::, :]

    normal_viz = normal_viz + np.equal(np.sum(normal_viz, 2, keepdims=True), 0.0).astype(np.float32) * np.min(
        normal_viz
    )

    normal_viz = (normal_viz - np.min(normal_viz)) / 2.0
    plt.axis("off")
    plt.imsave(name, np.clip(normal_viz, 0.0, 1.0))


all_files = []
with open(data_dir, newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")

    for row in reader:
        if row[-1] == "Unavailable":
            continue
        all_files.append(row[0].split("/")[-1])


def process(file):
    scene_id, scan_id = file.split("_")[0], file.split("_")[1]
    base_path = f"/root/data/DIODE/{SPLIT}/outdoor/scene_{scene_id}/scan_{scan_id}"
    path = os.path.join(base_path, file)
    pil_image = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BICUBIC)

    path2 = os.path.join(base_path, file[:-4] + "_depth.npy")
    depth = np.load(path2).squeeze()
    depth = depth.astype(np.float32)

    path3 = os.path.join(base_path, file[:-4] + "_depth_mask.npy")
    depth_mask = np.load(path3)
    depth_mask = depth_mask.astype(np.float32)

    path4 = os.path.join(base_path, file[:-4] + "_normal.npy")
    normal = np.load(path4)
    normal = normal.astype(np.float32)

    image_depth = cv2.resize(depth, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)
    image_depth_mask = cv2.resize(depth_mask, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)

    normal = cv2.resize(normal, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)

    name = os.path.join(target_dir, file)
    pil_image.save(name)
    plot_depth_map(image_depth, image_depth_mask, name[:-4] + "_depth.png")
    plot_normal_map(normal, name[:-4] + "_normal.png")


target_dir = f"/root/data/DIODE-{img_size}/{SPLIT}"
os.makedirs(target_dir, exist_ok=True)
Parallel(n_jobs=1)(delayed(process)(name) for name in tqdm(all_files))
