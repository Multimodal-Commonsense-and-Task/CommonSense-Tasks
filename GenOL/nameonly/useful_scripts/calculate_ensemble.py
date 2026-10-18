from classes import *
def split_class_count(class_count_dict, split_ratio):

    total_ratio = sum(split_ratio)
    
    split_result = {key: [] for key in class_count_dict.keys()}
    
    for class_key, count in class_count_dict.items():
        remaining_count = count
        for ratio in split_ratio:
            split_count = round((ratio / total_ratio) * count)    
            split_count = min(split_count, remaining_count)
            split_result[class_key].append(split_count)
            
            remaining_count -= split_count
        
        # Add the remaining count to the last split
        split_result[class_key][-1] += remaining_count
    
    # Sanity check (sum of each class should be equal to the original count)
    for class_key, split_counts in split_result.items():
        assert sum(split_counts) == class_count_dict[class_key], f"Sum of split counts for class {class_key} does not match the original count."
    
    return split_result

split_ratio = (4, 3, 0, 0)
class_count_dict = pacs_count


# Test
split_result = split_class_count(class_count_dict, split_ratio)
result_sdxl = {k:v[0] for k, v in split_result.items()}
result_dalle2 = {k:v[1] for k, v in split_result.items()}
result_deepfloyd = {k:v[2] for k, v in split_result.items()}
result_cogview2 = {k:v[3] for k, v in split_result.items()}
print(f"Split result: {split_result}")
print(f"Result SDXL: {result_sdxl}")
print(f"Result DALLE2: {result_dalle2}")
print(f"Result DeepFloyd: {result_deepfloyd}")
print(f"Result CogView2: {result_cogview2}")
