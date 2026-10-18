import os
import argparse
import shutil

def generate_unique_filename(directory, base_filename):
    filename, extension = os.path.splitext(base_filename)
    counter = 1

    new_filename = base_filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{filename}_{counter}{extension}"
        counter += 1
    return new_filename

parser = argparse.ArgumentParser(description='Merge images')
parser.add_argument('-s', '--source', type=str, required=True, help='source directory')
parser.add_argument('-t', '--destination', type=str, required=True, help='destination directory')
args = parser.parse_args()


# Get source images
source_images = os.listdir(args.source)

# Copy images to destination, if same name exists, add _1, _2, ...
for source_image in source_images:
    source_image_path = os.path.join(args.source, source_image)
    destination_image_path = os.path.join(args.destination, source_image)
    if os.path.exists(destination_image_path):
        destination_image_path = os.path.join(args.destination, generate_unique_filename(args.destination, source_image))
    shutil.copy(source_image_path, destination_image_path)