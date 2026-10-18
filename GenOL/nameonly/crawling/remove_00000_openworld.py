import os
import shutil

source_path = './datasets/openworld'
target_path = './datasets/openworld_processed'
type = 'neg'

urls = os.listdir(source_path)
print(f"Length of url list: {len(urls)}")

image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
for url in urls:
    url_path_with_zero = os.path.join(source_path, url, type, '00000')
    images = os.listdir(url_path_with_zero)
    images = [file for file in images if file.lower().endswith(image_extensions)]
    images = [os.path.join(url_path_with_zero, image) for image in images]

    target_url_path = os.path.join(target_path, url, type)
    os.makedirs(target_url_path, exist_ok=True)

    for image in images:
        new_image_name = os.path.basename(image)
        shutil.copy(image, os.path.join(target_url_path, new_image_name))
