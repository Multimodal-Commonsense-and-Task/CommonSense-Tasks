import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

warnings.filterwarnings(action='ignore')
dir = 'NICO'
if 'PACS' in dir:
    in_dis = ['final_test_ma']
    ood_dis = ['final_cartoon', 'final_art_painting', 'final_sketch']
elif 'cct' in dir:
    in_dis = ['in_test_ma Test']
    ood_dis = ['out_test_ma Test']
elif 'OfficeHome' in dir:
    in_dis = ['test_ma']
    ood_dis = ['Clipart', 'Art', 'Product']
elif 'DomainNet' in dir:
    in_dis = ['test_ma']
    ood_dis = ['painting', 'quickdraw', 'sketch', 'clipart', 'infograph']
elif 'NICO' in dir:
    in_dis = []
    ood_dis = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']


target_exp_list = [
        '/home/user/mjlee/New_Nameonly_CL/results/NICO/rebuttal4_resnet18_er_NICO_iter2_mem200/glide', \
        '/home/user/mjlee/New_Nameonly_CL/results/NICO/rebuttal4_resnet18_er_NICO_iter2_mem200/sdxl', \
        '/home/user/mjlee/New_Nameonly_CL/results/DomainNet/resnet18_er_DomainNet_iter2_mem200/web2_fake',
        '/home/user/mjlee/New_Nameonly_CL/results/DomainNet/resnet18_er_DomainNet_iter2_mem200/RMD_classwise_temp_3',
        '/home/user/mjlee/New_Nameonly_CL/results/DomainNet/resnet18_er_DomainNet_iter2_mem200/RMD_classwise_temp_5'
        ]
label_list = ["glide", "sdxl", "web", "classwise_3", "classwise_5"]

target_exp_list = np.array(target_exp_list)[[0,1]]
label_list = np.array(label_list)[[0,1]]
seed = 2

if seed == 1:
    added_timing = []
elif seed == 2:
    added_timing = []
elif seed == 3:
    added_timing = []

added_timing = added_timing

def print_from_log(exp_name, in_dis, ood_dis):
    in_avg = []
    in_last = []
    ood_avg = []
    ood_last = []
    overall_avg = []
    overall_last = []
    f = open(f'{exp_name}/seed_{seed}.log', 'r')
    lines = f.readlines()
    in_dis_acc = []
    ood_dis_acc = []
    in_acc, ood_acc = [], []
    all_acc = []
    for line in lines[11:]:
        if 'Test' in line:
                domains = in_dis+ood_dis
                for domain in domains:
                    if domain in line:
                        if domain in in_dis:
                            in_acc.append(float(line.split(" ")[-4]))
                        else:
                            ood_acc.append(float(line.split(" ")[-4]))
        if 'Total Test' in line:
            in_dis_acc.append(np.mean(in_acc))
            ood_dis_acc.append(np.mean(ood_acc))
            in_acc, ood_acc = [], []
            all_acc.append(float(line.split(" ")[-4]))
    return in_dis_acc, ood_dis_acc

# exp_list = sorted([dir + '/' + exp for exp in os.listdir(dir)])

for exp, label in zip(target_exp_list, label_list):
    in_dis_acc,ood_dis_acc = print_from_log(exp, in_dis, ood_dis)
    plt.plot(range(len(in_dis_acc)), savgol_filter(in_dis_acc, 5, 3), label=f"{label}_in", linewidth=1.0)
    plt.plot(range(len(ood_dis_acc)), savgol_filter(ood_dis_acc, 5, 3), label=f"{label}_ood", linewidth=1.0)
for i in added_timing:
    plt.axvline(x=i/100, color='r', linestyle='--', linewidth=0.5)
plt.xlabel("# of iteration (X 100)", fontsize=15)
plt.ylabel("Out Accuracy", fontsize=15)
plt.legend()
plt.title(f"{dir}")
plt.savefig(f"{dir}.png")