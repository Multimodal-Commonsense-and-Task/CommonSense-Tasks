import os
import glob
import argparse
import random
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, required=True, help='Directory to postprocess')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save image')
parser.add_argument('--max_images', type=int, required=True, help='Max number of images to rename')
parser.add_argument('--start_class', type=int, default=0, help='Start class index')
parser.add_argument('--end_class', type=int, default=9, help='End class index')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


def rename_images(directory, max_images, output_dir):
    extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']
    images = []

    # Find all image files in the directory
    for ext in extensions:
        images.extend(glob.glob(f"{directory}/*.{ext}"))

    random.shuffle(images)
    moved_images = 0
    image_number = 0

    while moved_images < max_images and image_number < len(images):
        image_path = images[image_number]
        new_name = f"{moved_images:06}.png"
        new_path = os.path.join(output_dir, new_name)

        # Resize and save image with new name
        with Image.open(image_path) as img:
            img = img.resize((256, 256))
            try:
                img.save(new_path)
                moved_images += 1
            except:
                print(f"Failed to save {new_path}")

        image_number += 1

    print(f"Moved {moved_images} images")


if __name__ == '__main__':
    class_names = sorted(os.listdir(args.dataset_dir))
    for class_name in class_names[args.start_class:args.end_class + 1]:
        print(f"Processing {class_name}")
        image_dir = os.path.join(args.dataset_dir, class_name, '00000')
        output_dir = os.path.join(args.output_dir, class_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        rename_images(image_dir, args.max_images, output_dir)
