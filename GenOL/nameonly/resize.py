import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process, Queue
import argparse
import shutil

def resize_images(classes, source_path, target_path, queue, size=(224, 224)):
    num_resized = 0
    for cls in tqdm(classes):
        os.makedirs(os.path.join(target_path, cls), exist_ok=True)
        images = os.listdir(os.path.join(source_path, cls))

        # Copy images
        for img in images:
            if img.startswith('.'):
                continue
            try:
                image = Image.open(os.path.join(source_path, cls, img))
                image = image.resize(size)
            except Exception as e:
                print(f"Error occured while processing {cls}/{img}")
                print(e)
                continue
            num_resized += 1
            try:
                image.save(os.path.join(target_path, cls, img))
            except Exception as e:
                print(f"Error occured while saving {cls}/{img}")
                print(e)
                continue
    
    queue.put(num_resized)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--source_path", type=str)
    parser.add_argument('-p', "--num_process", default=1, type=int)
    args = parser.parse_args()

    SOURCE_PATH = args.source_path
    # If end of the SOURCE_PATH is /, remove it
    if SOURCE_PATH[-1] == '/':
        SOURCE_PATH = SOURCE_PATH[:-1]
    TARGET_PATH = SOURCE_PATH + "_resized"
    SIZE = (224, 224)
    classes = os.listdir(SOURCE_PATH)
    classes = [cls for cls in classes if not cls.startswith('.')]

    # Create dir if not exists
    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH, exist_ok=True)
    
    queue = Queue()
    # Single process
    if args.num_process == 1:
        resize_images(classes, source_path=SOURCE_PATH, target_path=TARGET_PATH, queue=queue)
    else:
        chunk_size = len(classes) // args.num_process
        processes = []
        for i in range(args.num_process):
            start = i * chunk_size
            end = (start + chunk_size - 1) if i != args.num_process - 1 else len(classes) - 1
            target_classes = classes[start:end + 1]
            process = Process(target=resize_images, args=(target_classes, SOURCE_PATH, TARGET_PATH, queue))
            processes.append(process)
            process.start()
        
        total_resized = 0
        for process in processes:
            process.join()
        
        while not queue.empty():
            total_resized += queue.get()
        
        print(f"All procceses have completed. Total number of resized images: {total_resized}")
        print(f"Removing old data...")
        if total_resized > 0:
            print(f"Removing old directory {SOURCE_PATH}")
            shutil.rmtree(SOURCE_PATH)
            print(f"Renaming {TARGET_PATH} to {SOURCE_PATH}")
            os.rename(TARGET_PATH, SOURCE_PATH)


if __name__ == "__main__":
    main()