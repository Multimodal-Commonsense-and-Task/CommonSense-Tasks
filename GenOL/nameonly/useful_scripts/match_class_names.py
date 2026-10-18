import os

source_path = '/workspace/home/user/web_dataset/officehome_filtered_x10'
source_class_names = os.listdir(source_path)

target_path = '/workspace/home/user/web_dataset/office_homes'
target_class_names = os.listdir(target_path)

mapping_dict = {}

for source_class_name in source_class_names:
    # Ignore lower case, and underscores
    source_class_name_converted = source_class_name.lower().replace('_', ' ')
    for target_class_name in target_class_names:
        target_class_name_converted = target_class_name.lower().replace('_', ' ')
        if source_class_name_converted in target_class_name_converted:
            mapping_dict[source_class_name] = target_class_name
            break

reverse_mapping_dict = {v: k for k, v in mapping_dict.items()} # target to source

for class_name in target_class_names:
    if class_name not in mapping_dict.values():
        print(f"Class {class_name} not found in source classes")
    
    source_class_name = reverse_mapping_dict[class_name]
    print(f"Renaming {class_name} to {source_class_name}")
    os.rename(os.path.join(target_path, class_name), os.path.join(target_path, source_class_name))