# High RMD가 High rarity를 갖고, Low RMD가 Low rarity를 갖는다는 것을 위해 qualitative figure를 만들 때 필요한 것들
# Rarity score를 구해서 저장해야 함. PACS 대상으로.
# 이건 이미 저장된 RMD, Rarity score pickle을 읽어서 high rmd, high rarity 등을 class별로 보기 위한 것.
import pickle
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

save_path = './RMD_Rarity_visualize'
rmd_path = './RMD_scores/RMD_scores_PACS_ablation.pkl'
rarity_path = './RMD_scores/Rarity_scores_PACS.pkl'
ratio = 0.3

with open(rmd_path, 'rb') as f:
    rmd_scores_raw = pickle.load(f)

with open(rarity_path, 'rb') as f:
    rarity_scores = pickle.load(f)

classes = list(rarity_scores.keys()) # WARNING: Rarity scores do not contain 'person' class!
# Convert rmd scores dictionary keys: [('PACS_full', 'dog') -> ('dog')]
rmd_scores = {}
for (model_name, class_name), scores in rmd_scores_raw.items():
    rmd_scores[class_name] = scores

# Get top / bottom RMD / Rarity scores for each class
rmd_average_dict = {}; rarity_average_dict = {}
for cls in classes:
    rmd_scores_class = rmd_scores[cls]; rmd_scores_path_to_score = {k:v for k, v in rmd_scores_class}
    rarity_scores_class = rarity_scores[cls]; rarity_scores_path_to_score = {k:v for k, v in rarity_scores_class}
    
    # Remove zero score element
    rmd_scores_class = [sample for sample in rmd_scores_class if sample[1] > 0.0]
    rarity_scores_class = [sample for sample in rarity_scores_class if sample[1] > 0.0]
    
    rmd_scores_list = [sample[1] for sample in rmd_scores_class]
    rarity_scores_list = [sample[1] for sample in rarity_scores_class]
    rmd_average_dict[cls] = sum(rmd_scores_list) / len(rmd_scores_list)
    rarity_average_dict[cls] = sum(rarity_scores_list) / len(rarity_scores_list)
    
    # # 여기에서부터는 scatter plot을 위함
    # rmd_paths = [sample[0] for sample in rmd_scores_class]; rarity_paths = [sample[0] for sample in rarity_scores_class]
    # intersect_paths = list(set(rmd_paths) & set(rarity_paths))
    # rmd_scores_intersect = np.array([rmd_scores_path_to_score[path] for path in intersect_paths])
    # rarity_scores_intersect = np.array([rarity_scores_path_to_score[path] for path in intersect_paths])
    # correlation_matrix = np.corrcoef(rmd_scores_intersect, rarity_scores_intersect)
    # correlation_coefficient = correlation_matrix[0, 1]
    # plt.scatter(rmd_scores_intersect, rarity_scores_intersect)
    # plt.xlabel('RMD Score')
    # plt.ylabel('Rarity Score')
    # plt.title(f"Scatter Plot of RMD vs Rarity - Corr: {correlation_coefficient}")
    # plt.savefig('./test.png')
    
    # # 여기서부터는 이제 RMD, Rarity를 top ratio만큼 뽑아서 겹치는 것들을 저장, 점수 저장
    # # Get top-ratio, bottom-ratio samples
    # rmd_scores_class.sort(key=lambda x: x[1], reverse=True) # Descending
    # rarity_scores_class.sort(key=lambda x: x[1], reverse=True)
    
    # rmd_ratio_num = int(len(rmd_scores_class) * ratio)
    # rarity_ratio_num = int(len(rarity_scores_class) * ratio)
    # rmd_scores_top = rmd_scores_class[:rmd_ratio_num]; rmd_scores_bottom = rmd_scores_class[rmd_ratio_num:]
    # rarity_scores_top = rarity_scores_class[:rarity_ratio_num]; rarity_scores_bottom = rarity_scores_class[rarity_ratio_num:]
    
    # # Get intersection of top / bottom elements of RMD and rarity scores
    # rmd_top_paths = [sample[0] for sample in rmd_scores_top]; rmd_bottom_paths = [sample[0] for sample in rmd_scores_bottom]
    # rarity_top_paths = [sample[0] for sample in rarity_scores_top]; rarity_bottom_paths = [sample[0] for sample in rarity_scores_bottom]
    
    # top_intersect = set(rmd_top_paths) & set(rarity_top_paths)
    # bottom_intersect = set(rmd_bottom_paths) & set(rarity_bottom_paths)
    
    # # With scores
    # top_intersect_with_score = [(path, {'rmd':rmd_scores_path_to_score[path], 'rarity':rarity_scores_path_to_score[path]}) for path in top_intersect]
    # bottom_intersect_with_score = [(path, {'rmd':rmd_scores_path_to_score[path], 'rarity':rarity_scores_path_to_score[path]}) for path in bottom_intersect]
        
    # cls_save_path = os.path.join(save_path, cls)
    # if os.path.exists(cls_save_path):
    #     shutil.rmtree(cls_save_path)
    # os.makedirs(cls_save_path)
    # os.makedirs(os.path.join(cls_save_path, 'top_intersect'))
    # os.makedirs(os.path.join(cls_save_path, 'bottom_intersect'))
        
    
    # # Copy elements
    # for sample in top_intersect_with_score:
    #     source_path = sample[0]
    #     rmd_score = sample[1]['rmd']; rarity_score = sample[1]['rarity']
    #     sample_name = os.path.basename(source_path)
    #     dst_path = os.path.join(cls_save_path, 'top_intersect', sample_name)
    #     shutil.copy(source_path, dst_path)
    
    # for sample in bottom_intersect_with_score:
    #     source_path = sample[0]
    #     rmd_score = sample[1]['rmd']; rarity_score = sample[1]['rarity']
    #     sample_name = os.path.basename(source_path)
    #     dst_path = os.path.join(cls_save_path, 'bottom_intersect', sample_name)
    #     shutil.copy(source_path, dst_path)
    
    # # Write text file
    # txt_path = os.path.join(cls_save_path, 'scores.txt')
    # with open(txt_path, 'w', encoding='utf-8') as f:
    #     f.write(f"[Top RMD & Rarity samples, ratio: {ratio}]\n")
    #     for sample in top_intersect_with_score:
    #         source_path = sample[0]
    #         rmd_score = sample[1]['rmd']; rarity_score = sample[1]['rarity']
    #         sample_name = os.path.basename(source_path)
    #         f.write(f"{sample_name} - RMD: {rmd_score}, Rarity: {rarity_score}\n")
        
    #     f.write(f"\n[Bottom RMD & Rarity samples, ratio: {ratio}]\n")
    #     for sample in bottom_intersect_with_score:
    #         source_path = sample[0]
    #         rmd_score = sample[1]['rmd']; rarity_score = sample[1]['rarity']
    #         sample_name = os.path.basename(source_path)
    #         f.write(f"{sample_name} - RMD: {rmd_score}, Rarity: {rarity_score}\n")
            

breakpoint()