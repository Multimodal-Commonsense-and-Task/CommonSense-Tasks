# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import io

from PIL import Image
import lmdb

import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset

from corruption.inpaint import build_inpaint_freeform, build_inpaint_center

# from ipdb import set_trace as debug


def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    # print(path)
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode())
    img = Image.open(io.BytesIO(bytedata))
    return img.convert("RGB")


def _build_lmdb_dataset(root, transform=None, target_transform=None, loader=lmdb_loader):
    """
    You can create this dataloader using:
    train_data = _build_lmdb_dataset(traindir, transform=train_transform)
    valid_data = _build_lmdb_dataset(validdir, transform=val_transform)
    """

    root = str(root)
    if root.endswith("/"):
        root = root[:-1]
    pt_path = os.path.join(root + "_faster_imagefolder.lmdb.pt")
    lmdb_path = os.path.join(root + "_faster_imagefolder.lmdb")

    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        # log.info('[Dataset] Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        # log.info('[Dataset] Saving pt to {}'.format(pt_path))
        # log.info('[Dataset] Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for _path, class_index in data_set.imgs:
                with open(_path, "rb") as f:
                    data = f.read()
                txn.put(_path.encode("ascii"), data)
    data_set.lmdb_data = lmdb.open(lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)

    return data_set


def build_train_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),  # [0,1] --> [-1, 1]
        ]
    )


def build_test_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),  # [0,1] --> [-1, 1]
        ]
    )


def build_lmdb_dataset(dataset_dir, image_size, train, transform=None):
    """resize -> crop -> to_tensor -> norm(-1,1)"""
    fn = os.path.join(dataset_dir, "train" if train else "val")

    if transform is None:
        build_transform = build_train_transform if train else build_test_transform
        transform = build_transform(image_size)

    dataset = _build_lmdb_dataset(fn, transform=transform)
    # log.info(f"[Dataset] Built Imagenet dataset {fn=}, size={len(dataset)}!")
    return dataset


def readlines(fn):
    file = open(fn, "r").readlines()
    return [line.strip("\n\r") for line in file]


def build_lmdb_dataset_val10k(dataset_dir, image_size, transform=None):
    fn = os.path.join(dataset_dir, "val")
    fn_10k = readlines("assets/datasets/val_faster_imagefolder_10k_fn.txt")
    label_10k = readlines("assets/datasets/val_faster_imagefolder_10k_label.txt")

    if transform is None:
        transform = build_test_transform(image_size)
    dataset = _build_lmdb_dataset(fn, transform=transform)
    dataset.samples = [(os.path.join(fn, name), int(label)) for name, label in zip(fn_10k, label_10k)]

    assert len(dataset) == 10_000
    # log.info(f"[Dataset] Built Imagenet val10k, size={len(dataset)}!")
    return dataset


class InpaintingVal10kSubset(Dataset):
    def __init__(self, dataset_dir, image_size, corrupt_type, transform=None):
        super(InpaintingVal10kSubset, self).__init__()
        self.dataset = build_lmdb_dataset_val10k(dataset_dir, image_size, transform)

        if corrupt_type == "center":
            self.corrupt = build_inpaint_center(corrupt_type, image_size)
        elif "freeform" in corrupt_type:
            self.corrupt = build_inpaint_freeform(corrupt_type)
        else:
            raise NotImplementedError()

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img_clean, label = self.dataset[index]
        img_corrupt, mask = self.corrupt(img_clean)
        return img_clean, img_corrupt, (index, mask, label)


class ImageNetInpaintingDataset(Dataset):
    def __init__(self, dataset_dir, image_size, corrupt_type, train, transform=None):
        super(ImageNetInpaintingDataset, self).__init__()
        self.dataset = build_lmdb_dataset(dataset_dir, image_size, train, transform)

        if corrupt_type == "center":
            self.corrupt = build_inpaint_center(corrupt_type, image_size)
        elif "freeform" in corrupt_type:
            self.corrupt = build_inpaint_freeform(corrupt_type)
        else:
            raise NotImplementedError()

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img_clean, label = self.dataset[index]
        img_corrupt, mask = self.corrupt(img_clean)
        return img_clean, img_corrupt, (index, mask, label)
