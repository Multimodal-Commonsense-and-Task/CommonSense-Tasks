import os
import numpy as np
import warnings
from scipy.stats import sem
from collections import defaultdict

warnings.filterwarnings(action='ignore')
dir = 'NICO'
if dir == 'NICO':
    n_samples = 12000
    ood_domain = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
    

domain_accs = defaultdict(list)
domain_accs_last = defaultdict(list)
def print_from_log(exp_name, seeds=(1, 2, 3, 4, 5)):
    A_auc = []
    A_last = []
    A_avg = []
    A_online = []
    F_last = []
    IF_avg = []
    KG_avg = []
    FLOPS = []
    for i in seeds:
        f = open(f'{exp_name}/seed_{i}.log', 'r')
        lines = f.readlines()

        for line in lines:
            if 'Test' in line:
                for domain in ood_domain:
                    if domain in line:
                        dom = line.split("|")[0].split(" ")[-3]
                        domain_accs[dom].append(float(line.split("|")[-2].split(" ")[-2]))
                        if "Sample # 12000" in line:
                            domain_accs_last[dom].append(float(line.split("|")[-2].split(" ")[-2]))
                
            # if 'gdumb' in exp_name:
            #     if 'Test' in line:
            #         list = line.split(' ')
            #         a_last = float(list[13])*100
            # if 'A_auc' in line:
            #     list = line.split(' ')
            #     A_auc.append(float(list[4])*100)
            #     A_avg.append(float(list[10])*100)
            #     if 'gdumb' in exp_name:
            #         A_last.append(a_last)
            #     else:
            #         A_last.append(float(list[7])*100)
            #     FLOPS.append(float(list[-1])/100)
            #     break
    all_domain_accs = []
    all_domain_accs_last = []
    for i, dom in enumerate(list(domain_accs.keys())):
        # print("here", domain_accs[dom])
        print(f'Exp:{exp_name} {dom} acc \t\t\t {np.mean(domain_accs[dom])*100:.2f}/{sem(domain_accs[dom])*100:.2f}')
        all_domain_accs.append(np.mean(domain_accs[dom]))
        all_domain_accs_last.append(np.mean(domain_accs_last[dom]))
    print(f'Exp:{exp_name} Overall AUC / Last acc \t\t\t {np.mean(all_domain_accs_last)*100:.2f}/{sem(all_domain_accs_last)*100:.2f} \t {np.mean(all_domain_accs)*100:.2f}/{sem(all_domain_accs)*100:.2f}')
    # print(f'Exp:{exp_name} Overall Last acc \t\t\t {np.mean(all_domain_accs)*100:.2f}/{sem(all_domain_accs)*100:.2f}')
    
    # if np.isnan(np.mean(A_auc)):
    #     pass
    # else:
    #     print(f'Exp:{exp_name} \t\t\t {np.mean(A_auc):.2f}/{sem(A_auc):.2f} \t {np.mean(A_avg):.2f}/{sem(A_avg):.2f} \t \t {np.mean(A_last):.2f}/{sem(A_last):.2f} \t  {np.mean(IF_avg):.2f}/{sem(IF_avg):.2f}  \t  {np.mean(KG_avg):.2f}/{sem(KG_avg):.2f}  \t  {np.mean(FLOPS):.2f}/{sem(FLOPS):.2f}|')

# print("A_auc, A_last, IF_avg, KG_avg FLOPS")

exp_type_list = []
exp_list = sorted([dir + '/' + exp for exp in os.listdir(dir)])
for exp in exp_list:
    for exp_type in os.listdir(exp):
        if os.path.isdir(os.path.join(exp, exp_type)):
            exp_type_list.append(os.path.join(exp, exp_type))
            
for exp in exp_type_list:
    try:
        print_from_log(exp)
    except Exception as e:
        pass

