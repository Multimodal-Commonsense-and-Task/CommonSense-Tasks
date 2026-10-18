import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", type=str)
parser.add_argument("--start_class", type=int, default=0)
parser.add_argument("--end_class", type=int)
args = parser.parse_args()

SOURCE_PATH = args.source_path
# If end of the SOURCE_PATH is /, remove it
if SOURCE_PATH[-1] == '/':
    SOURCE_PATH = SOURCE_PATH[:-1]
TARGET_PATH = SOURCE_PATH + "_resized"
SIZE = (224, 224)
classes = os.listdir(SOURCE_PATH)
classes = [cls for cls in classes if not cls.startswith('.')]

if args.end_class is None:
    args.end_class = len(classes) - 1
    print(f"End class not specified. Setting end class to {args.end_class}")

num_resized = 0
# [0:11], [11:22], [22:33], [33:44], [44:55], [55:66], [66:77], [77:88], [88:101]
for cls in tqdm(classes[args.start_class:args.end_class + 1]):
    os.makedirs(os.path.join(TARGET_PATH, cls), exist_ok=True)
    images = os.listdir(os.path.join(SOURCE_PATH, cls))

    for img in images:
        if img.startswith('.'):
            continue
        image = Image.open(os.path.join(SOURCE_PATH, cls, img))
        image = image.resize(SIZE)
        num_resized += 1
        image.save(os.path.join(TARGET_PATH, cls, img))

# Remove old directory
print(f"Resized {num_resized} images")
if num_resized > 0:
    print(f"Removing old directory {SOURCE_PATH}")
    shutil.rmtree(SOURCE_PATH)
    print(f"Renaming {TARGET_PATH} to {SOURCE_PATH}")
    os.rename(TARGET_PATH, SOURCE_PATH)