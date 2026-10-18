import os
import numpy as np
import warnings
from scipy.stats import sem


warnings.filterwarnings(action='ignore')
dir = 'PACS_final'

if 'PACS' in dir:
    in_dis = ['final_test_ma']
    ood_dis = ['final_cartoon', 'final_art_painting', 'final_sketch']
elif 'cct' in dir:
    in_dis = ['in_test_ma']
    ood_dis = ['out_test_ma']
elif 'cifar10' in dir:
    in_dis = ['original']
    ood_dis = ['c Test', 'nc Test']
elif 'DomainNet' in dir:
    in_dis = ['test_ma']
    ood_dis = ['infograph', 'clipart', 'quickdraw', 'painting', 'sketch']

print("A_auc, A_last, IF_avg, KG_avg FLOPS")

exp_type_list = []
exp_list = sorted([dir + '/' + exp for exp in os.listdir(dir)])
for exp in exp_list:
    for exp_type in os.listdir(exp):
        if os.path.isdir(os.path.join(exp, exp_type)):
            exp_type_list.append(os.path.join(exp, exp_type))



def print_from_log(exp_name, in_dis, ood_dis, seeds=(1,2,3,4,5)):
    in_avg = []
    in_last = []
    ood_avg = []
    ood_last = []
    overall_avg = []
    overall_last = []
    for i in seeds:
        f = open(f'{exp_name}/seed_{i}.log', 'r')
        lines = f.readlines()
        in_dis_acc = []
        ood_dis_acc = []
        in_acc, ood_acc = [], []
        all_acc = []
        for line in lines:
            if 'Test' in line:
                domains = in_dis+ood_dis
                for domain in domains:
                    if domain in line and domain in in_dis:
                        # print(line)
                        in_acc.append(float(line.split(" ")[-4]))
                    if domain in line and domain in ood_dis:
                        ood_acc.append(float(line.split(" ")[-4]))
            if 'Total Test' in line:
                in_dis_acc.append(np.mean(in_acc))
                ood_dis_acc.append(np.mean(ood_acc))
                in_acc, ood_acc = [], []
                all_acc.append(float(line.split(" ")[-4]))
            
        in_avg.append(round(sum(in_dis_acc)/len(in_dis_acc)*100,2))
        in_last.append(round(in_dis_acc[-1]*100,2))
        ood_avg.append(round(sum(ood_dis_acc)/len(ood_dis_acc)*100,2))
        ood_last.append(round(ood_dis_acc[-1]*100,2))
        overall_avg.append(round(sum(all_acc)/len(all_acc)*100,2))
        overall_last.append(round(all_acc[-1]*100,2))
    
    # print(f'Exp:{exp_name} ')
    # print("In", in_avg)
    # print("In", in_last)
    # print("ood", ood_avg)
    # print("ood", ood_last)
        
    if np.isnan(np.mean(in_avg)):
        pass
    else:
        print(f'Exp:{exp_name} in-distribution \t\t\t {np.mean(in_avg):.2f}/{sem(in_avg):.2f} \t {np.mean(in_last):.2f}/{sem(in_last):.2f}')
        print(f'Exp:{exp_name} ood-distribution \t\t\t {np.mean(ood_avg):.2f}/{sem(ood_avg):.2f} \t {np.mean(ood_last):.2f}/{sem(ood_last):.2f}')
        print(f'Exp:{exp_name} overall \t\t\t {np.mean(overall_avg):.2f}/{sem(overall_avg):.2f} \t {np.mean(overall_last):.2f}/{sem(ood_last):.2f}')

for exp in exp_type_list:
    try:
        print_from_log(exp, in_dis, ood_dis)
    except Exception as e:
        pass