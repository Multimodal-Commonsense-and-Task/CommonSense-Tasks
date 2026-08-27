# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import requests
from tqdm import tqdm

import torch

ADM_IMG256_COND_CKPT = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt"
I2SB_IMG256_COND_CKPT = "256x256_diffusion_fixedsigma.pt"


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def download_adm_image256_cond_ckpt(ckpt_dir):
    ckpt_pt = os.path.join(ckpt_dir, I2SB_IMG256_COND_CKPT)
    if os.path.exists(ckpt_pt):
        return

    adm_ckpt = os.path.join(ckpt_dir, os.path.basename(ADM_IMG256_COND_CKPT))

    print("Downloading ADM checkpoint to {} ...".format(adm_ckpt))
    download(ADM_IMG256_COND_CKPT, adm_ckpt)
    ckpt_state_dict = torch.load(adm_ckpt, map_location="cpu")

    # pt: remove the sigma prediction and add concat module
    ckpt_state_dict["out.2.weight"] = ckpt_state_dict["out.2.weight"][:3]
    ckpt_state_dict["out.2.bias"] = ckpt_state_dict["out.2.bias"][:3]
    ckpt_state_dict["input_blocks.0.0.weight"] = torch.cat(
        [ckpt_state_dict["input_blocks.0.0.weight"], ckpt_state_dict["input_blocks.0.0.weight"]], dim=1
    )
    torch.save(ckpt_state_dict, ckpt_pt)

    print(f"Saved adm cond pretrain models at {ckpt_pt}!")


def download_ckpt(ckpt_dir="assets/ckpts"):
    os.makedirs(ckpt_dir, exist_ok=True)
    download_adm_image256_cond_ckpt(ckpt_dir=ckpt_dir)


download_ckpt()
