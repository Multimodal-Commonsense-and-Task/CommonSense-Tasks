import os

target_path = './datasets/openworld'
threshold = 10

urls = [os.path.join(target_path, path) for path in os.listdir(target_path)]
print(f"Length of url list: {len(urls)}")

image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
for url_path in urls:
    url_path = os.path.join(url_path, 'neg', '00000')
    images = os.listdir(url_path)
    images = [file for file in images if file.lower().endswith(image_extensions)]
    if len(images) < threshold:
        print(f"path - {url_path}, num: {len(images)}")