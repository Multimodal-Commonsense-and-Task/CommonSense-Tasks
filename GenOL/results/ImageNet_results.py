import os
import numpy as np
import warnings
import pandas as pd
from scipy.stats import sem
from collections import defaultdict

warnings.filterwarnings(action='ignore')
dir = 'ImageNet_400'

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
elif 'NICO' in dir:
    in_dis = []
    ood_dis = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
    n_samples, n_tasks = 12000, 5
    cls_per_task = [12]*n_tasks
elif 'CUB_200' in dir:
    in_dis = ['test_ma']
    ood_dis = ['painting']
    n_samples, n_tasks = 5994, 5
    cls_per_task = [40]*n_tasks
elif 'ImageNet' in dir:
    in_dis = ['test_ma']
    # ood_dis = ['shot_noise_10', 'saturate_10', 'speckle_noise_10', 'brightness_10', 'zoom_blur_10', 'impulse_noise_10', 'contrast_10', 'jpeg_compression_10', 'splatter_10', 'pixelate_10', 'snow_10', 'elastic_transform_10', 'gaussian_noise_10', 'motion_blur_10', 'gaussian_blur_10', 'fog_10', 'glass_blur_10', 'defocus_blur_10', 'frost_10']
    ood_dis = ['r_50']
    nsamples, n_tasks = 5994, 5
    cls_per_task = [1000]*n_tasks
elif 'birds31' in dir:
    in_dis = ['test_ma']
    # ood_dis = ['shot_noise_10', 'saturate_10', 'speckle_noise_10', 'brightness_10', 'zoom_blur_10', 'impulse_noise_10', 'contrast_10', 'jpeg_compression_10', 'splatter_10', 'pixelate_10', 'snow_10', 'elastic_transform_10', 'gaussian_noise_10', 'motion_blur_10', 'gaussian_blur_10', 'fog_10', 'glass_blur_10', 'defocus_blur_10', 'frost_10']
    ood_dis = ['nabirds_test', 'inaturalist_test']
    nsamples, n_tasks = 1240, 5
    cls_per_task = [5,5,5,5,6]


total_cls=0
cul_cls_per_task = []
for i in range(n_tasks):
    total_cls+=cls_per_task[i]
    cul_cls_per_task.append(total_cls)

print("A_auc, A_last, IF_avg, KG_avg FLOPS")

exp_type_list = []
exp_list = sorted([dir + '/' + exp for exp in os.listdir(dir)])
exp_list = [exp for exp in exp_list if os.path.isdir(exp)] # Filter only dirs

for exp in exp_list:
    for exp_type in os.listdir(exp):
        if os.path.isdir(os.path.join(exp, exp_type)) and "eval_iclr_resnet18" in os.path.join(exp, exp_type) and "baseline" not in os.path.join(exp, exp_type):
            exp_type_list.append(os.path.join(exp, exp_type)) # iclr_resnet18_cifar10_xder/complete_ma


print(exp_type_list)
def print_from_log(exp_name, in_dis, ood_dis, seeds=([1])):
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
        if 'Summary' not in lines[-2]:
            return
        in_dis_acc = []
        ood_dis_acc = []
        in_acc, ood_acc = [], []
        all_acc = []
        fa_seed, aoa_seed, bt_seed = [],[],[]
        imb_in_dis_acc = []
        imb_ood_dis_acc = []
        imb_in_acc, imb_ood_acc = [], []
        imb_all_acc = []
        imb_fa_seed, imb_aoa_seed, imb_bt_seed = [],[],[]
        for line in lines:
            if 'Test' in line and "Total Test" not in line:
                # domains = in_dis+ood_dis
                # for domain in domains:
                #     if domain in line and domain in in_dis:
                #         in_acc.append(float(line.split(" ")[-4]))
                #         imb_in_acc.append(float(line.split(" ")[-1]))
                #     if domain in line and domain in ood_dis:
                ood_acc.append(float(line.split(" ")[-4]))
                
                # imb_ood_acc.append(float(line.split(" ")[-1]))
            # elif 'ACC_PER_TASK' in line:
            #     acc_per_task = line.split("|")[-1].split(",")
            #     dom =  line.split("|")[0].split(" ")[-3]
            #     if cur_task+1 == n_tasks:
            #         for n_t in range(n_tasks-1):
            #             # backward_transfer[dom].append(float(acc_per_task[n_t].split(": ")[1])*cul_cls_per_task[n_t]/cls_per_task[n_t] - domain_task_accs[dom][n_t])
            #             backward_transfer[dom].append(float(acc_per_task[n_t].split(": ")[1]) - domain_task_accs[dom][n_t])
            #     else:
            #         # domain_task_accs[dom].append(float(acc_per_task[cur_task].split(": ")[1])*cul_cls_per_task[cur_task]/cls_per_task[cur_task])
            #         domain_task_accs[dom].append(float(acc_per_task[cur_task].split(": ")[1]))
            # elif 'AOA' in line:
            #     aoa_seed.append(float(line.split(" ")[-1]))
            # elif 'ADAPTATION' in line:
            #     adaptation = float(line.split("|")[-1][-5:])
            #     dom =  line.split("|")[0].split(" ")[-3]
            #     fast_adapt_dict[dom].append(adaptation)
            # elif 'IMBALANCE_ADAPT' in line:
            #     adaptation = float(line.split("|")[-1][-5:])
            #     dom =  line.split("|")[0].split(" ")[-3]
            #     imb_fast_adapt_dict[dom].append(adaptation)
            # elif 'FORGETTING' in line:
            #     single_KLR = float(line.split("|")[-2].split(" ")[2])
            #     single_KGR = float(line.split("|")[-1].split(" ")[2])
            #     dom =  line.split("|")[0].split(" ")[-3]
            #     KLR[dom].append(single_KLR)
            #     KGR[dom].append(single_KGR)
            # if 'Task Test' in line:
            #     cur_task += 1
            if 'Total Test' in line:
                if len(in_acc)>0:
                    in_dis_acc.append(np.mean(in_acc))
                    imb_in_dis_acc.append(np.mean(imb_in_acc))
                # print("ood", ood_acc)
                if not np.isnan(np.mean(ood_acc)):
                    ood_dis_acc.append(np.mean(ood_acc))
                # print("ood_dis_acc", ood_dis_acc)
                # all_acc.append(float(line.split(" ")[-4]))
                if not np.isnan(np.mean(imb_ood_acc)):
                    imb_ood_dis_acc.append(np.mean(imb_ood_acc))
                # imb_all_acc.append(np.mean(imb_in_acc+imb_ood_acc))
                
                in_acc, ood_acc = [], []
                imb_in_acc, imb_ood_acc = [], []
        # print("ood_dis_acc", ood_dis_acc)
        if len(in_dis_acc)>0:
            in_avg.append(round(sum(in_dis_acc)/len(in_dis_acc)*100,2))
            in_last.append(round(in_dis_acc[-1]*100,2))
            imb_in_avg.append(round(sum(imb_in_dis_acc)/len(imb_in_dis_acc)*100,2))
            imb_in_last.append(round(imb_in_dis_acc[-1]*100,2))
            
        ood_avg.append(round(sum(ood_dis_acc)/len(ood_dis_acc)*100,2))
        ood_last.append(round(ood_dis_acc[-1]*100,2))
        # print("here2")
        # overall_avg.append(round(sum(all_acc)/len(all_acc)*100,2))
        # overall_last.append(round(all_acc[-1]*100,2))
        
        # imb_ood_avg.append(round(sum(imb_ood_dis_acc)/len(imb_ood_dis_acc)*100,2))
        # imb_ood_last.append(round(imb_ood_dis_acc[-1]*100,2))
        # imb_overall_avg.append(round(sum(imb_all_acc)/len(imb_all_acc)*100,2))
        # imb_overall_last.append(round(imb_all_acc[-1]*100,2))
        # aoa_auc.append(round(sum(aoa_seed)/len(aoa_seed)*100,2))
        # aoa_last.append(round(aoa_seed[-1]*100,2))
        # fast_adaptation.append(round(sum(fa_seed)/len(fa_seed)*100,2))
        

    # Create dataframe
    df_dict = {}
    # print("ood_avg", ood_avg)
    if np.isnan(np.mean(ood_avg)):
        pass
    else:
        if len(in_avg)>0:
            df_dict['id'] = f"{np.mean(in_avg):.2f}/{sem(in_avg):.2f} \t {np.mean(in_last):.2f}/{sem(in_last):.2f}"
            print(f'Exp:{exp_name} in-distribution \t\t\t {np.mean(in_avg):.2f}/{sem(in_avg):.2f} \t {np.mean(in_last):.2f}/{sem(in_last):.2f}')
            print(f'Exp:{exp_name} imb_in-distribution \t\t\t {np.mean(imb_in_avg):.2f}/{sem(imb_in_avg):.2f} \t {np.mean(imb_in_last):.2f}/{sem(imb_in_last):.2f}')
            
        df_dict['ood'] = f"{np.mean(ood_avg):.2f}/{sem(ood_avg):.2f} \t {np.mean(ood_last):.2f}/{sem(ood_last):.2f}"
        print(f'Exp:{exp_name} ood-distribution \t\t\t {np.mean(ood_avg):.2f}/{sem(ood_avg):.2f} \t {np.mean(ood_last):.2f}/{sem(ood_last):.2f}')
        # print(f'Exp:{exp_name} overall \t\t\t {np.mean(overall_avg):.2f}/{sem(overall_avg):.2f} \t {np.mean(overall_last):.2f}/{sem(overall_last):.2f}')

        print(f'Exp:{exp_name} imb_ood-distribution \t\t\t {np.mean(imb_ood_avg):.2f}/{sem(imb_ood_avg):.2f} \t {np.mean(imb_ood_last):.2f}/{sem(imb_ood_last):.2f}')
        # print(f'Exp:{exp_name} imb_overall \t\t\t {np.mean(imb_overall_avg):.2f}/{sem(imb_overall_avg):.2f} \t {np.mean(imb_overall_last):.2f}/{sem(imb_overall_last):.2f}')
        # print(f'Exp:{exp_name} real_time_evaluation \t\t\t {np.mean(aoa_auc):.2f}/{sem(aoa_auc):.2f} \t {np.mean(aoa_last):.2f}/{sem(aoa_last):.2f}')
        # print(f'Exp:{exp_name} fast_adaptation \t\t\t {np.mean(fast_adaptation):.2f}/{sem(fast_adaptation):.2f}')
        ood_backward = []
        ood_adaptation = []
        imb_ood_adaptation = []
        ood_KGR, ood_KLR = [], []
        # for i, dom in enumerate(list(backward_transfer.keys())):
        #     if dom in in_dis:
        #         df_dict['FA_ID'] = f"{np.mean(fast_adapt_dict[dom])*100:.2f}/{sem(fast_adapt_dict[dom])*100:.2f}"
        #         df_dict['BWT_ID'] = f"{np.mean(backward_transfer[dom])*100:.2f}/{sem(backward_transfer[dom])*100:.2f}"
        #     print(f'Exp:{exp_name} {dom} backward_transfer \t\t\t {np.mean(backward_transfer[dom])*100:.2f}/{sem(backward_transfer[dom])*100:.2f}')
        #     print(f'Exp:{exp_name} {dom} fast_adaptation \t\t\t {np.mean(fast_adapt_dict[dom])*100:.2f}/{sem(fast_adapt_dict[dom])*100:.2f}')
        #     print(f'Exp:{exp_name} {dom} imbalance_fast_adaptation \t\t\t {np.mean(imb_fast_adapt_dict[dom])*100:.2f}/{sem(imb_fast_adapt_dict[dom])*100:.2f}')
        #     # print(f'Exp:{exp_name} {dom} forgetting \t\t\t {np.mean(KGR[dom]):.2f}/{sem(KGR[dom]):.2f} / {np.mean(KLR[dom]):.2f}/{sem(KLR[dom]):.2f}')
        #     if dom not in in_dis:
        #         ood_backward.extend(backward_transfer[dom])
        #         ood_adaptation.extend(fast_adapt_dict[dom])
        #         imb_ood_adaptation.extend(imb_fast_adapt_dict[dom])
        #         # ood_KGR.extend(KGR[dom])
        #         # ood_KLR.extend(KLR[dom])
        # print(f'Exp:{exp_name} OOD backward_transfer \t\t\t {np.mean(ood_backward)*100:.2f}/{sem(ood_backward)*100:.2f}')
        # print(f'Exp:{exp_name} OOD adaptation \t\t\t {np.mean(ood_adaptation)*100:.2f}/{sem(ood_adaptation)*100:.2f}')
        # print(f'Exp:{exp_name} OOD imbalance_adaptation \t\t\t {np.mean(imb_ood_adaptation)*100:.2f}/{sem(imb_ood_adaptation)*100:.2f}')
        # df_dict["FA_OOD"] = f"{np.mean(ood_adaptation)*100:.2f}/{sem(ood_adaptation)*100:.2f}"
        # df_dict["BWT_OOD"] = f"{np.mean(ood_backward)*100:.2f}/{sem(ood_backward)*100:.2f}"
        # print(f'Exp:{exp_name} OOD forgetting \t\t\t {np.mean(ood_KGR):.2f}/{sem(ood_KGR):.2f} / {np.mean(ood_KLR):.2f}/{sem(ood_KLR):.2f}')
            
    # order = ('id', 'ood', 'FA_ID', 'FA_OOD', 'BWT_ID', 'BWT_OOD')
    # result_df_dict = {k: df_dict[k] for k in order}
    # result_df = pd.DataFrame({k: [v] for k, v in result_df_dict.items()})
    # output_csv_path = f"{exp_name}.csv"
    # result_df.to_csv(output_csv_path,index=False)
                

for exp in exp_type_list:
    try:
        print_from_log(exp, in_dis, ood_dis)
    except Exception as e:
        print(e, exp)
        pass
