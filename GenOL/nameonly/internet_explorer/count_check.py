import os

def check_folder_file_count(A: str):
    for folder in os.listdir(A):
        folder_path = os.path.join(A, folder)
        
        if os.path.isdir(folder_path) and folder.isdigit():
            file_count = len(os.listdir(folder_path))
            if file_count <= 1000:
                print(folder, file_count)

# 사용 예시
A = "ImageNet_internet_explorer_missing"
check_folder_file_count(A)
