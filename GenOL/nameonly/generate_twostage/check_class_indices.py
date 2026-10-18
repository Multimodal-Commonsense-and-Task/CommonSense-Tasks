import os
import argparse
from tqdm import tqdm
from classes import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset')
parser.add_argument('-r', '--source_dir')
parser.add_argument('-s', '--start_idx', type=int)
parser.add_argument('-e', '--end_idx', type=int)
parser.add_argument('-t', '--threshold', type=int)
parser.add_argument('--threshold_ratio', type=float)
parser.add_argument('--num_devices', type=int, default=None)
args = parser.parse_args()

cls_num_dict = count_dict[args.dataset]

classes = os.listdir(args.source_dir)
cls_dir_count_dict = {cls: len(os.listdir(os.path.join(args.source_dir, cls))) for cls in classes
                      if os.path.isdir(os.path.join(args.source_dir, cls))}

cls_not_generated = []
for cls in tqdm(list(cls_num_dict.keys())):
    if cls not in cls_dir_count_dict.keys():
        cls_not_generated.append((cls, 0))
    else:
        cls_dir_count = cls_dir_count_dict[cls]
        if args.threshold is not None:
            if cls_dir_count < args.threshold:
                cls_not_generated.append((cls, cls_dir_count))
        elif args.threshold_ratio is not None:
            if cls_dir_count < cls_num_dict[cls] * args.threshold_ratio:
                cls_not_generated.append((cls, cls_dir_count))
        elif cls_dir_count < cls_num_dict[cls]:
            cls_not_generated.append((cls, cls_dir_count))

# Find indices
class_list = list(cls_num_dict.keys())
result_list = []
for cls, count in cls_not_generated:
    index = class_list.index(cls)
    if args.start_idx is not None:
        if index < args.start_idx or index > args.end_idx:
            continue
    result_list.append((f"{index}-{cls}:{count}"))

print(result_list)
print(len(result_list))
print([result.split('-')[0] for result in result_list])

if args.num_devices is not None:
    num_devices = args.num_devices
    num_classes = len(result_list)

    # Sort result_list by class index
    sorted_result_list = sorted(result_list, key=lambda x: int(x.split('-')[0]))

    num_classes_per_device = num_classes // num_devices
    for i in range(num_devices):
        start_idx = i * num_classes_per_device
        end_idx = (i + 1) * num_classes_per_device
        if i == num_devices - 1:
            end_idx = num_classes
        print(f"Device {i}:")
        print([result.split('-')[0] for result in sorted_result_list[start_idx:end_idx]])
