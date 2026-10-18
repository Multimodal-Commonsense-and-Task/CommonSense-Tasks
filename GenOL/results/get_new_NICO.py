import os
import numpy as np
import warnings
from scipy.stats import sem
from collections import defaultdict

warnings.filterwarnings(action='ignore')
dir = 'DomainNet'

if 'PACS' in dir:
    in_dis = ['final_test_ma']
    ood_dis = ['final_cartoon', 'final_art_painting', 'final_sketch']
    n_samples, n_tasks = 1333, 3
    cls_per_task = [3,2,2]
elif 'cct' in dir:
    in_dis = ['in_test_ma']
    ood_dis = ['out_test_ma']
    n_samples, n_tasks = 2400, 4
    cls_per_task = [3]*n_tasks
elif 'cifar10' in dir:
    in_dis = ['original_test']
    ood_dis = ['> c', '> nc']
    n_samples, n_tasks = 10000, 5
    cls_per_task = [2]*n_tasks
elif 'DomainNet' in dir:
    in_dis = ['test_ma']
    ood_dis = ['infograph', 'clipart', 'quickdraw', 'painting', 'sketch']
    n_samples, n_tasks = 3459, 5
    cls_per_task = [69]*n_tasks

total_cls=0
cul_cls_per_task = []
for i in range(n_tasks):
    total_cls+=cls_per_task[i]
    cul_cls_per_task.append(total_cls)

print("A_auc, A_last, IF_avg, KG_avg FLOPS")

exp_type_list = []
exp_list = sorted([dir + '/' + exp for exp in os.listdir(dir)])
for exp in exp_list:
    for exp_type in os.listdir(exp):
        if os.path.isdir(os.path.join(exp, exp_type)) and "iclr" in os.path.join(exp, exp_type):
            exp_type_list.append(os.path.join(exp, exp_type))


print(exp_type_list)
def print_from_log(exp_name, in_dis, ood_dis, seeds=(1,2,3,4,5)):
    in_avg, imb_in_avg = [], []
    in_last, imb_in_last = [], []
    ood_avg, imb_ood_avg = [], []
    ood_last, imb_ood_last = [], []
    overall_avg, imb_overall_avg = [], []
    overall_last, imb_overall_last = [], []
    fast_adaptation = []
    aoa_auc = []
    aoa_last = []
    backward_transfer = defaultdict(list)
    fast_adapt_dict = defaultdict(list)
    imb_backward_transfer = defaultdict(list)
    imb_fast_adapt_dict = defaultdict(list)
    KLR = defaultdict(list)
    KGR = defaultdict(list)
    domain_list = in_dis+ood_dis
    for i in seeds:
        domain_task_accs = defaultdict(list)
        cur_task = 0
        f = open(f'{exp_name}/seed_{i}.log', 'r')
        lines = f.readlines()
        in_dis_acc = []
        ood_dis_acc = []
        in_acc, ood_acc = [], []
        all_acc = []
        fa_seed, aoa_seed, bt_seed = [],[],[]
        imb_in_dis_acc = []
        imb_ood_dis_acc = []
        imb_in_acc, imb_ood_acc = [], []
        imb_fa_seed, imb_aoa_seed, imb_bt_seed = [],[],[]
        for line in lines:
            if 'Test' in line:
                domains = in_dis+ood_dis
                for domain in domains:
                    if domain in line and domain in in_dis:
                        in_acc.append(float(line.split(" ")[-4]))
                        imb_in_acc.append(float(line.split(" ")[-1]))
                    if domain in line and domain in ood_dis:
                        ood_acc.append(float(line.split(" ")[-4]))
                        imb_ood_acc.append(float(line.split(" ")[-1]))
            elif 'ACC_PER_TASK' in line:
                acc_per_task = line.split("|")[-1].split(",")
                dom =  line.split("|")[0].split(" ")[-3]
                if cur_task+1 == n_tasks:
                    for n_t in range(n_tasks-1):
                        # backward_transfer[dom].append(float(acc_per_task[n_t].split(": ")[1])*cul_cls_per_task[n_t]/cls_per_task[n_t] - domain_task_accs[dom][n_t])
                        backward_transfer[dom].append(float(acc_per_task[n_t].split(": ")[1]) - domain_task_accs[dom][n_t])
                else:
                    # domain_task_accs[dom].append(float(acc_per_task[cur_task].split(": ")[1])*cul_cls_per_task[cur_task]/cls_per_task[cur_task])
                    domain_task_accs[dom].append(float(acc_per_task[cur_task].split(": ")[1]))
            # elif 'AOA' in line:
            #     aoa_seed.append(float(line.split(" ")[-1]))
            elif 'ADAPTATION' in line:
                adaptation = float(line.split("|")[-1][-5:])
                dom =  line.split("|")[0].split(" ")[-3]
                fast_adapt_dict[dom].append(adaptation)
            elif 'IMBALANCE_ADAPT' in line:
                adaptation = float(line.split("|")[-1][-5:])
                dom =  line.split("|")[0].split(" ")[-3]
                imb_fast_adapt_dict[dom].append(adaptation)
            # elif 'FORGETTING' in line:
            #     single_KLR = float(line.split("|")[-2].split(" ")[2])
            #     single_KGR = float(line.split("|")[-1].split(" ")[2])
            #     dom =  line.split("|")[0].split(" ")[-3]
            #     KLR[dom].append(single_KLR)
            #     KGR[dom].append(single_KGR)
            if 'Task Test' in line:
                cur_task += 1
            if 'Total Test' in line:
                in_dis_acc.append(np.mean(in_acc))
                ood_dis_acc.append(np.mean(ood_acc))
                in_acc, ood_acc = [], []
                all_acc.append(float(line.split(" ")[-4]))
                imb_in_dis_acc.append(np.mean(imb_in_acc))
                imb_ood_dis_acc.append(np.mean(imb_ood_acc))
                imb_in_acc, imb_ood_acc = [], []
                imb_all_acc = np.mean(imb_in_acc+imb_ood_acc)
                
        in_avg.append(round(sum(in_dis_acc)/len(in_dis_acc)*100,2))
        in_last.append(round(in_dis_acc[-1]*100,2))
        ood_avg.append(round(sum(ood_dis_acc)/len(ood_dis_acc)*100,2))
        ood_last.append(round(ood_dis_acc[-1]*100,2))
        overall_avg.append(round(sum(all_acc)/len(all_acc)*100,2))
        overall_last.append(round(all_acc[-1]*100,2))
        
        imb_in_avg.append(round(sum(imb_in_dis_acc)/len(imb_in_dis_acc)*100,2))
        imb_in_last.append(round(imb_in_dis_acc[-1]*100,2))
        imb_ood_avg.append(round(sum(imb_ood_dis_acc)/len(imb_ood_dis_acc)*100,2))
        imb_ood_last.append(round(imb_ood_dis_acc[-1]*100,2))
        imb_overall_avg.append(round(sum(imb_all_acc)/len(imb_all_acc)*100,2))
        imb_overall_last.append(round(imb_all_acc[-1]*100,2))
        
        # aoa_auc.append(round(sum(aoa_seed)/len(aoa_seed)*100,2))
        # aoa_last.append(round(aoa_seed[-1]*100,2))
        # fast_adaptation.append(round(sum(fa_seed)/len(fa_seed)*100,2))
        
    # print(f'Exp:{exp_name} ')
    # print("In", in_avg)
    # print("In", in_last)
    # print("ood", ood_avg)
    # print("ood", ood_last)
    if np.isnan(np.mean(ood_avg)):
        pass
    else:
        # print(f'Exp:{exp_name} in-distribution \t\t\t {np.mean(in_avg):.2f}/{sem(in_avg):.2f} \t {np.mean(in_last):.2f}/{sem(in_last):.2f}')
        print(f'Exp:{exp_name} ood-distribution \t\t\t {np.mean(ood_avg):.2f}/{sem(ood_avg):.2f} \t {np.mean(ood_last):.2f}/{sem(ood_last):.2f}')
        print(f'Exp:{exp_name} overall \t\t\t {np.mean(overall_avg):.2f}/{sem(overall_avg):.2f} \t {np.mean(overall_last):.2f}/{sem(overall_last):.2f}')
        
        # print(f'Exp:{exp_name} imb_in-distribution \t\t\t {np.mean(imb_in_avg):.2f}/{sem(imb_in_avg):.2f} \t {np.mean(imb_in_last):.2f}/{sem(imb_in_last):.2f}')
        print(f'Exp:{exp_name} imb_ood-distribution \t\t\t {np.mean(imb_ood_avg):.2f}/{sem(imb_ood_avg):.2f} \t {np.mean(imb_ood_last):.2f}/{sem(imb_ood_last):.2f}')
        print(f'Exp:{exp_name} imb_overall \t\t\t {np.mean(imb_overall_avg):.2f}/{sem(imb_overall_avg):.2f} \t {np.mean(imb_overall_last):.2f}/{sem(imb_overall_last):.2f}')
        # print(f'Exp:{exp_name} real_time_evaluation \t\t\t {np.mean(aoa_auc):.2f}/{sem(aoa_auc):.2f} \t {np.mean(aoa_last):.2f}/{sem(aoa_last):.2f}')
        # print(f'Exp:{exp_name} fast_adaptation \t\t\t {np.mean(fast_adaptation):.2f}/{sem(fast_adaptation):.2f}')
        ood_backward = []
        ood_adaptation = []
        imb_ood_adaptation = []
        ood_KGR, ood_KLR = [], []
        for i, dom in enumerate(list(backward_transfer.keys())):
            print(f'Exp:{exp_name} {dom} backward_transfer \t\t\t {np.mean(backward_transfer[dom])*100:.2f}/{sem(backward_transfer[dom])*100:.2f}')
            print(f'Exp:{exp_name} {dom} fast_adaptation \t\t\t {np.mean(fast_adapt_dict[dom])*100:.2f}/{sem(fast_adapt_dict[dom])*100:.2f}')
            print(f'Exp:{exp_name} {dom} imbalance_fast_adaptation \t\t\t {np.mean(imb_fast_adapt_dict[dom])*100:.2f}/{sem(imb_fast_adapt_dict[dom])*100:.2f}')
            # print(f'Exp:{exp_name} {dom} forgetting \t\t\t {np.mean(KGR[dom]):.2f}/{sem(KGR[dom]):.2f} / {np.mean(KLR[dom]):.2f}/{sem(KLR[dom]):.2f}')
            if dom not in in_dis:
                ood_backward.extend(backward_transfer[dom])
                ood_adaptation.extend(fast_adapt_dict[dom])
                imb_ood_adaptation.extend(imb_fast_adapt_dict[dom])
                # ood_KGR.extend(KGR[dom])
                # ood_KLR.extend(KLR[dom])
        print(f'Exp:{exp_name} OOD backward_transfer \t\t\t {np.mean(ood_backward)*100:.2f}/{sem(ood_backward)*100:.2f}')
        print(f'Exp:{exp_name} OOD adaptation \t\t\t {np.mean(ood_adaptation)*100:.2f}/{sem(ood_adaptation)*100:.2f}')
        print(f'Exp:{exp_name} OOD imbalance_adaptation \t\t\t {np.mean(imb_ood_adaptation)*100:.2f}/{sem(imb_ood_adaptation)*100:.2f}')
        # print(f'Exp:{exp_name} OOD forgetting \t\t\t {np.mean(ood_KGR):.2f}/{sem(ood_KGR):.2f} / {np.mean(ood_KLR):.2f}/{sem(ood_KLR):.2f}')
            
                
                

for exp in exp_type_list:
    try:
        print_from_log(exp, in_dis, ood_dis)
    except Exception as e:
        pass