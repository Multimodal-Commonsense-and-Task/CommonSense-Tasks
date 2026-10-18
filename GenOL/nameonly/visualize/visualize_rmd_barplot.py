# RMD score pickle을 넣어줬을 때 각 class마다 각 model의 RMD score 평균을 구하고, 이를 visualize하는 목적 (ablation에 쓰임)
import pickle
import numpy as np
import matplotlib.pyplot as plt
import statistics
from utils import count_top_samples

pickle_path = 'RMD_scores/RMD_scores_cifar10_ablation_equalsize_without_meta.pkl'
ratio = 0.3

with open(pickle_path, 'rb') as f:
    dataset_dict = pickle.load(f)

# Count the number of samples of each model
model_class_dict = {}
for k, v in dataset_dict.items():
    model_name, class_name = k
    if model_name not in model_class_dict.keys():
        model_class_dict[model_name] = {}
    
    model_class_dict[model_name][class_name] = len(v)

for model_name, class_count_dict in model_class_dict.items():
    total = sum(class_count_dict.values())
    print(f"{model_name}: {total}")

# class -> model -> list of scores
cls_model_scores_dict = {}
for k, v in dataset_dict.items():
    model_name, class_name = k
    if class_name not in cls_model_scores_dict:
        cls_model_scores_dict[class_name] = {}
    cls_model_scores_dict[class_name][model_name] = v

# Calculate the average RMD scores
cls_model_average_dict = {}
for k, v in dataset_dict.items():
    model_name, class_name = k
    if class_name not in cls_model_average_dict:
        cls_model_average_dict[class_name] = {}
    
    # Calculate the average RMD score for class, model
    scores_list = [sample[1] for sample in v]
    cls_model_average_dict[class_name][model_name] = sum(scores_list) / len(scores_list)

cls_model_count_dict = {} # Dictionary to store the number of top-k% samples included for each model
# Now the mapping is {cls1: {model1: [], model2: []}, cls2: {}, ...}, each item: (PATH, score)
for class_name in cls_model_scores_dict:
    model_scores_dict = cls_model_scores_dict[class_name]

    result_dict = count_top_samples(model_scores_dict, ratio=ratio)
    if class_name not in cls_model_count_dict:
        cls_model_count_dict[class_name] = {}
    cls_model_count_dict[class_name] = result_dict

# 여기는 그냥 class 원래 있던거에 순서 맞춰주는 용도
cls_mapping_list = ['automobile', 'dog', 'cat', 'airplane', 'horse', 'bird', 'deer', 'frog', 'ship', 'truck']
base_count = [cls_model_count_dict[cls]['cifar10_base_1x'][0] for cls in cls_mapping_list]
base_average = [cls_model_count_dict[cls]['cifar10_base_1x'][1] for cls in cls_mapping_list]
# meta_count = [cls_model_count_dict[cls]['cifar10_meta'][0] for cls in cls_mapping_list]
# meta_average = [cls_model_count_dict[cls]['cifar10_meta'][1] for cls in cls_mapping_list]
full_count = [cls_model_count_dict[cls]['cifar10_full'][0] for cls in cls_mapping_list]
full_average = [cls_model_count_dict[cls]['cifar10_full'][1] for cls in cls_mapping_list]
base_count.append(sum(base_count) / len(base_count)); base_average.append(sum(base_average) / len(base_average))
full_count.append(sum(full_count) / len(full_count)); full_average.append(sum(full_average) / len(full_average))


classes = ["Automobile", "Dog", "Cat", "Airplane", "Horse", "Bird", "Deer", "Frog", "Ship", "Truck", "Avg"]

# Define the width of each group
bar_width = 0.35

# Define the x positions for each group
x_base_1x = np.arange(len(classes))
# x_meta = x_base_1x + bar_width
x_full = x_base_1x + 1 * bar_width

# Plot
fig, axs = plt.subplots(1, 1, figsize=(35, 10), constrained_layout=False)
plt.subplots_adjust(bottom=0.32)

# Add RMD average plot
axs.bar(x_base_1x, base_average, width=bar_width, label="CIFAR-10 Base Prompt", color="lightsteelblue", edgecolor="k")
# axs.bar(x_meta, cifar_10_meta_rmd, width=bar_width, label="CIFAR-10 Meta Prompts", alpha=0.9, color="slateblue", edgecolor="k")
axs.bar(x_full, full_average, width=bar_width, label="CIFAR-10 Prompt Rewrites", alpha=0.9, color="indigo", edgecolor="k")
axs.set_xticks(x_full - 0.175)
axs.yaxis.set_tick_params(labelsize=35)
axs.set_xticklabels(classes, fontsize=30, rotation=45)
axs.set_title('Average RMD scores of top-30% samples', fontsize=45)
axs.set_ylim(4,10)

# Add RMD count plot
axs2 = axs.twinx()
axs2.yaxis.set_tick_params(labelsize=35)
axs.set_ylabel("Average RMD score", fontsize=30); axs2.set_ylabel("Number of samples", fontsize=30)
line1 = axs2.plot(x_full - 0.35, base_count, label="CIFAR-10 Base Prompt", color="#B0B3DF", marker='o', linewidth=3)
line2 = axs2.plot(x_full, full_count, label="CIFAR-10 Prompt Rewrites", color="#8000DF", marker='o', linewidth=3)

# # DISTS subplot
# axs[1].bar(x_base_1x, cifar_10_base_1x_rmd, width=bar_width, label="CIFAR-10 Base Prompt", color="lightsteelblue", edgecolor="k")
# axs[1].bar(x_meta, cifar_10_meta_dists, width=bar_width, label="CIFAR-10 Meta Prompts", alpha=0.9, color="slateblue", edgecolor="k")
# axs[1].bar(x_full, cifar_10_full_dists, width=bar_width, label="CIFAR-10 Prompt Rewrites", alpha=0.9, color="indigo", edgecolor="k")
# axs[1].set_xticks(x_meta)
# axs[1].set_xticklabels(classes, fontsize=30, rotation=45)
# axs[1].yaxis.set_tick_params(labelsize=35)
# axs[1].set_title('DISTS', fontsize=45)
# axs[1].set_ylim(0.325,0.46)

handles1, labels1 = axs.get_legend_handles_labels()
handles2, labels2 = axs2.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.51, -0.07), fontsize=40)
# lines_labels = [axs.get_legend_handles_labels()]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.51, -0.02), fontsize=40)
# Adding a horizontal line at y=0
# axs.axhline(y=0, color='black', linewidth=1.5)  # Adjust color and linewidth as needed
# plt.tight_layout()
plt.savefig('test.png')
plt.savefig('diversity_bar.pdf', bbox_inches='tight')
# plt.savefig('diversity_bar.pdf')
# plt.show()