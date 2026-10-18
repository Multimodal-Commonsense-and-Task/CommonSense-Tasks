import jsonlines
import json
import csv
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

mode_method_dict = {
    'no_sampling':'real_llava_sft_lora_scenario5_org_nocul',
    'random': 'real_llava_sft_lora_scenario5_random0_2',
    'dbp': 'final_llava_sft_lora_scenario5_dbp0_2'
}

# colors = ['#3A1730', '#C18A3D', '#588157', '#E63946', '#BCBD22', '#17BECF', '#457B9D']
colors = ['#457B9D', '#314832', '#D8CFC0', '#E63946', '#3A1730', '#C18A3D', '#588157', '#38322C', '#BCBD22', '#17BECF']

mode_color_dict = {
    'sft': colors[3],
    'fedavg': colors[1],
    'fedours': colors[0],
}

scenario_num = 2
method = "sft"
with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
    scenario = json.load(fp)

# iters = [0, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900]#, 995]
# iters = [0, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350]
# iters = [0, ,5, 10, 20, 30, 40, 50, 100]
num_rounds = 5
k_list = [0.03125,0.0625,0.125,0.25,0.5]
# plot_mode = 'seen_task'
from_task = 4
data_indices = [
    [from_task],[from_task],[from_task],[from_task],[from_task]
]

for client_data in scenario:

    k_final_dict = {}

    
    # Plotting the scores
    plt.figure(figsize=(8, 5))
    ax = plt.gca()  # Get the current axis

    ax.grid(axis='y', c='#e0e0e0')
    ax.grid(axis='x', c='#e0e0e0')

    y = list(range(from_task+1, num_rounds+1))

    # scen10
    k_final_dict["Random"] = [62.63, ]
    k_final_dict["GradNorm"] = [62.63, ]
    k_final_dict["DivBS"] = [62.63, ]
    k_final_dict["InfoBatch"] = [62.73, ]
    k_final_dict["Ours"] = [62.63, ]
    
    
    # data_name = "B"
    
    for mode, k_final_scores in k_final_dict.items():
        print("k_list", k_final_scores)
        plt.plot(k_list, k_final_scores, label=f'{mode}', linewidth=2.0)#, color=mode_color_dict[mode])#, marker='o')
        plt.xticks(ticks=k_list, labels=[str(x) for x in k_list])

        
    plt.title(f'Last_Score Average', fontsize=20)
    plt.xlabel('Batch Iterations', fontsize=18)
    plt.ylabel(r"$A_{last}$", fontsize=18)
    # plt.legend(fontsize=12, loc='upper right')
    plt.legend(
    loc='upper center',    # Position the legend at the top center of the plot
    bbox_to_anchor=(0.5, -0.17),  # Move the legend below the plot
    fancybox=True, shadow=True, ncol=3  # Display items in a single row
    )
    plt.subplots_adjust(bottom=0.25)
    plt.tick_params(axis='x', which='minor', bottom=False)
    # plt.grid(axis='y')
    # plt.grid(True)
    

    # Save the plot
    plt.savefig(f'plots/plot_5_train_sc{scenario_num}_K-means.png')