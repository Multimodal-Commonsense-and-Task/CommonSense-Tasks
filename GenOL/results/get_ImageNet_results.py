import os
import numpy as np
import warnings
import pandas as pd
from scipy.stats import sem
from collections import defaultdict

warnings.filterwarnings(action='ignore')
dir = 'ImageNet'

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
    c_1_odd = ['contrast_1', 'defocus_blur_1', 'fog_1', 'gaussian_blur_1', 'gaussian_noise_1', 'glass_blur_1', 'impluse_noise_1', 'motion_blur_1', 'shot_noise_1', 'zoom_blur_1']
    c_2_odd = ['contrast_2', 'defocus_blur_2', 'fog_2', 'gaussian_blur_2', 'gaussian_noise_2', 'glass_blur_2', 'impluse_noise_2', 'motion_blur_2', 'shot_noise_2', 'zoom_blur_2']
    c_3_ood = ['contrast_3', 'defocus_blur_3', 'fog_3', 'gaussian_blur_3', 'gaussian_noise_3', 'glass_blur_3', 'impluse_noise_3', 'motion_blur_3', 'shot_noise_3', 'zoom_blur_3']
    r_ood = ['r_50']
    d_ood = ['background', 'texture', 'material']
    nsamples, n_tasks = 400000, 5
    cls_per_task = [200]*n_tasks
elif 'birds31' in dir:
    in_dis = ['test_ma']
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
        if os.path.isdir(os.path.join(exp, exp_type)) and "complete" in os.path.join(exp, exp_type):
            exp_type_list.append(os.path.join(exp, exp_type)) # iclr_resnet18_cifar10_xder/complete_ma


print(exp_type_list)
def print_from_log(exp_name, in_dis, ood_dis, seeds=([1])):
    print("here")
    in_avg, imb_in_avg = [], []
    r_ood_last, d_ood_last, c_1_ood_last, c_2_ood_last , c_3_ood_last = [], [], [], [], []
    r_ood_avg, d_ood_avg, c_1_ood_avg, c_2_ood_avg , c_3_ood_avg = [], [], [], [], []
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
        r_ood_dis_acc, d_ood_dis_acc, c_1_ood_dis_acc, c_2_ood_dis_acc, c_3_ood_dis_acc  = [], [], [], [], []
        in_acc = []
        r_ood_acc, d_ood_acc, c_1_ood_acc, c_2_ood_acc, c_3_ood_acc = [], [], [], [], []
        all_acc = []
        fa_seed, aoa_seed, bt_seed = [],[],[]
        imb_in_dis_acc = []
        imb_ood_dis_acc = []
        imb_all_acc = []
        imb_fa_seed, imb_aoa_seed, imb_bt_seed = [],[],[]
        for line in lines:
            if 'Test' in line:
                domains = in_dis+c_1_ood+c_2_ood+c_3_ood+d_ood+r_ood
                for domain in domains:
                    if domain in line:
                        if domain in in_dis:
                            in_acc.append(float(line.split(" ")[-4]))
                        elif domain in r_ood:
                            r_ood_acc.append(float(line.split(" ")[-4]))
                        elif domain in d_ood:
                            d_ood_acc.append(float(line.split(" ")[-4]))
                        elif domain in c_1_ood:
                            c_1_ood_acc.append(float(line.split(" ")[-4]))
                        elif domain in c_2_ood:
                            c_2_ood_acc.append(float(line.split(" ")[-4]))
                        elif domain in c_3_ood:
                            c_3_ood_acc.append(float(line.split(" ")[-4]))
            if 'Total Test' in line:
                if len(in_acc)>0:
                    in_dis_acc.append(np.mean(in_acc))
                r_ood_dis_acc.append(np.mean(r_ood_acc))
                d_ood_dis_acc.append(np.mean(d_ood_acc))
                c_1_ood_dis_acc.append(np.mean(c_1_ood_acc))
                c_2_ood_dis_acc.append(np.mean(c_2_ood_acc))
                c_3_ood_dis_acc.append(np.mean(c_3_ood_acc))
                all_acc.append(float(line.split(" ")[-4]))
                
                in_acc, r_ood_acc, d_ood_acc, c_1_ood_acc, c_2_ood_acc, c_3_ood_acc = [], [], [], [], [], []
        
        if len(in_dis_acc)>0:
            in_avg.append(round(sum(in_dis_acc)/len(in_dis_acc)*100,2))
            in_last.append(round(in_dis_acc[-1]*100,2))
            
        r_ood_avg.append(round(sum(r_ood_dis_acc)/len(r_ood_dis_acc)*100,2))
        d_ood_avg.append(round(sum(d_ood_dis_acc)/len(d_ood_dis_acc)*100,2))
        c_1_ood_avg.append(round(sum(c_1_ood_dis_acc)/len(c_1_ood_dis_acc)*100,2))
        c_2_ood_avg.append(round(sum(c_2_ood_dis_acc)/len(c_2_ood_dis_acc)*100,2))
        c_3_ood_avg.append(round(sum(c_3_ood_dis_acc)/len(c_3_ood_dis_acc)*100,2))
        r_ood_last.append(round(r_ood_dis_acc[-1]*100,2))
        d_ood_last.append(round(d_ood_dis_acc[-1]*100,2))
        c_1_ood_last.append(round(c_1_ood_dis_acc[-1]*100,2))
        c_2_ood_last.append(round(c_2_ood_dis_acc[-1]*100,2))
        c_3_ood_last.append(round(c_3_ood_dis_acc[-1]*100,2))
        overall_avg.append(round(sum(all_acc)/len(all_acc)*100,2))
        overall_last.append(round(all_acc[-1]*100,2))
        
        # aoa_auc.append(round(sum(aoa_seed)/len(aoa_seed)*100,2))
        # aoa_last.append(round(aoa_seed[-1]*100,2))
        # fast_adaptation.append(round(sum(fa_seed)/len(fa_seed)*100,2))
        

    # Create dataframe
    df_dict = {}
    
    if np.isnan(np.mean(ood_avg)):
        pass
    else:
        if len(in_avg)>0:
            # df_dict['id'] = f"{np.mean(in_avg):.2f}/{sem(in_avg):.2f} \t {np.mean(in_last):.2f}/{sem(in_last):.2f}"
            print(f'Exp:{exp_name} in-distribution \t\t\t {np.mean(in_avg):.2f}/{sem(in_avg):.2f} \t {np.mean(in_last):.2f}/{sem(in_last):.2f}')
            # print(f'Exp:{exp_name} imb_in-distribution \t\t\t {np.mean(imb_in_avg):.2f}/{sem(imb_in_avg):.2f} \t {np.mean(imb_in_last):.2f}/{sem(imb_in_last):.2f}')
            
        # df_dict['ood'] = f"{np.mean(ood_avg):.2f}/{sem(ood_avg):.2f} \t {np.mean(ood_last):.2f}/{sem(ood_last):.2f}"
        print(f'Exp:{exp_name} r_ood-distribution \t\t\t {np.mean(r_ood_avg):.2f}/{sem(r_ood_avg):.2f} \t {np.mean(r_ood_last):.2f}/{sem(r_ood_last):.2f}')
        print(f'Exp:{exp_name} d_ood-distribution \t\t\t {np.mean(d_ood_avg):.2f}/{sem(d_ood_avg):.2f} \t {np.mean(d_ood_last):.2f}/{sem(d_ood_last):.2f}')
        print(f'Exp:{exp_name} c_1_ood-distribution \t\t\t {np.mean(c_1_ood_avg):.2f}/{sem(c_1_ood_avg):.2f} \t {np.mean(c_1_ood_last):.2f}/{sem(c_1_ood_last):.2f}')
        print(f'Exp:{exp_name} c_2_ood-distribution \t\t\t {np.mean(c_2_ood_avg):.2f}/{sem(c_2_ood_avg):.2f} \t {np.mean(c_2_ood_last):.2f}/{sem(c_2_ood_last):.2f}')
        print(f'Exp:{exp_name} c_3_ood-distribution \t\t\t {np.mean(c_3_ood_avg):.2f}/{sem(c_3_ood_avg):.2f} \t {np.mean(c_3_ood_last):.2f}/{sem(c_3_ood_last):.2f}')
        print(f'Exp:{exp_name} c_4_overall \t\t\t {np.mean(overall_avg):.2f}/{sem(overall_avg):.2f} \t {np.mean(overall_last):.2f}/{sem(overall_last):.2f}')

        # print(f'Exp:{exp_name} imb_ood-distribution \t\t\t {np.mean(imb_ood_avg):.2f}/{sem(imb_ood_avg):.2f} \t {np.mean(imb_ood_last):.2f}/{sem(imb_ood_last):.2f}')
        # print(f'Exp:{exp_name} imb_overall \t\t\t {np.mean(imb_overall_avg):.2f}/{sem(imb_overall_avg):.2f} \t {np.mean(imb_overall_last):.2f}/{sem(imb_overall_last):.2f}')
        # print(f'Exp:{exp_name} real_time_evaluation \t\t\t {np.mean(aoa_auc):.2f}/{sem(aoa_auc):.2f} \t {np.mean(aoa_last):.2f}/{sem(aoa_last):.2f}')
        # print(f'Exp:{exp_name} fast_adaptation \t\t\t {np.mean(fast_adaptation):.2f}/{sem(fast_adaptation):.2f}')
        # ood_backward = []
        # ood_adaptation = []
        # imb_ood_adaptation = []
        # ood_KGR, ood_KLR = [], []
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
        # # print(f'Exp:{exp_name} OOD forgetting \t\t\t {np.mean(ood_KGR):.2f}/{sem(ood_KGR):.2f} / {np.mean(ood_KLR):.2f}/{sem(ood_KLR):.2f}')
            
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
