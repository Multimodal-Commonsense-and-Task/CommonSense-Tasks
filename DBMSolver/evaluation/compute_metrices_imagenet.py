# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import argparse
import random
from pathlib import Path
import json

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from logger import Logger

from evaluation.resnet import build_resnet50
from evaluation import fid_util

RESULT_DIR = Path("results")
ADM_IMG256_FID_TRAIN_REF_CKPT = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

class NumpyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        img_np = self.data[index]
        y = self.targets[index]

        if img_np.dtype == "uint8":
            # transform gives [0,1]
            img_t = self.transform(img_np) * 2 - 1
        elif img_np.dtype == "float32":
            # transform gives [0,255]
            img_t = self.transform(img_np) / 127.5 - 1

        # img_t: [-1,1]
        return img_t, y

    def __len__(self):
        return len(self.data)

@torch.no_grad()
def compute_accu(opt, numpy_arr, numpy_label_arr, batch_size=256):
    dataset = NumpyDataset(numpy_arr, numpy_label_arr)
    loader = DataLoader(dataset,
        batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    resnet = build_resnet50().to(opt.device)
    correct = total = 0
    for (x,y) in loader:
        pred_y = resnet(x.to(opt.device))

        _, predicted = torch.max(pred_y.cpu(), 1)
        correct += (predicted==y).sum().item()
        total += y.size(0)

    accu = correct / total
    return accu

def convert_to_numpy(t):
    # t: [-1,1]
    out = (t + 1) * 127.5
    out = out.clamp(0, 255)
    out = out.to(torch.uint8)
    out = out.permute(0, 2, 3, 1) # batch, 256, 256, 3
    out = out.contiguous()
    return out.cpu().numpy() # [0, 255]


def build_numpy_data(opt, log):
    arr = []
    ckpt_path = opt.ckpt
    numpy_arr = np.load(ckpt_path)['arr_0']
    label_path = opt.label
    label_arr = np.load(label_path)['arr_0']

    # converet to numpy
    return numpy_arr, label_arr

def build_ref_opt(opt, ref_fid_fn):
    split = ref_fid_fn.name[:-4].split("_")[-1]
    image_size = int(ref_fid_fn.name[:-4].split("_")[-2])
    assert opt.image_size == image_size
    return edict(
        mode=opt.mode,
        split=split,
        image_size=image_size,
        dataset_dir=opt.dataset_dir,
    )

def get_ref_fid(opt, log):
    # get ref fid npz file

    ref_fid_fn = Path("assets/stats/fid_imagenet_256_val.npz")
    if not ref_fid_fn.exists():
        log.info(f"Generating {ref_fid_fn=} (this can take a while ...)")
        ref_opt = build_ref_opt(opt, ref_fid_fn)
        fid_util.compute_fid_ref_stat(ref_opt, log)

    # load npz file
    ref_fid = np.load(ref_fid_fn)
    ref_mu, ref_sigma = ref_fid['mu'], ref_fid['sigma']
    return ref_fid_fn, ref_mu, ref_sigma

def log_metrices(opt):
    # setup
    set_seed(opt.seed)
    if opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
    log = Logger(0, ".log")

    log.info(f"======== Compute metrices: {opt.ckpt=}, {opt.mode=} ========")

    # find all recon pt files
    # recon_imgs_pts = find_recon_imgs_pts(opt, log)
    # log.info(f"Found {len(recon_imgs_pts)} pt files={[pt.name for pt in recon_imgs_pts]}")

    # build torch array
    numpy_arr, numpy_label_arr  = build_numpy_data(opt, log)
    log.info(f"Collected {numpy_arr.shape=}!")

    # compute accu
    accu = compute_accu(opt, numpy_arr, numpy_label_arr)
    log.info(f"Accuracy={accu:.3f}!")

    # load ref fid stat
    ref_fid_fn, ref_mu, ref_sigma = get_ref_fid(opt, log)
    log.info(f"Loaded FID reference statistics from {ref_fid_fn}!")

    # compute fid
    fid = fid_util.compute_fid_from_numpy(numpy_arr, ref_mu, ref_sigma, mode=opt.mode)
    log.info(f"FID(w.r.t. {ref_fid_fn=})={fid:.2f}!")

    res_dict = {
        "accu": accu,
        "fid": fid,
    }

    # save to file
    res_dir = opt.ckpt.split("/")[:-1]
    res_dir = Path("/".join(res_dir)) 
    with open(res_dir / "res.json", "w") as f:
        json.dump(res_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,  default=0)
    parser.add_argument("--gpu",         type=int,  default=None,             help="set only if you wish to run on a particular device")
    parser.add_argument("--ckpt",        type=str,  default=None,             help="the checkpoint name for which we wish to compute metrices")
    parser.add_argument("--label",       type=str,  default=None)
    parser.add_argument("--mode",        type=str,  default="legacy_pytorch", help="the FID computation mode used in clean-fid")
    parser.add_argument("--dataset-dir", type=Path, default="/dataset",       help="path to LMDB dataset")
    parser.add_argument("--sample-dir",  type=Path, default=None,             help="directory where samples are stored")
    parser.add_argument("--image-size",  type=int,  default=256)

    arg = parser.parse_args()

    opt = edict(
        device="cuda",
    )
    opt.update(vars(arg))

    log_metrices(opt)
