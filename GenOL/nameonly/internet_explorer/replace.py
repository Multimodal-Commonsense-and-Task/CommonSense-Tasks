import os
import shutil

def replace_folders(A: str, B: str):
    for folder in os.listdir(B):
        source_path = os.path.join(B, folder)
        target_path = os.path.join(A, folder)
        
        if os.path.isdir(source_path):
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            
            shutil.move(source_path, target_path)

# 사용 예시
A = "ImageNet_internet_explorer"
B = "ImageNet_internet_explorer_missing"
replace_folders(A, B)

