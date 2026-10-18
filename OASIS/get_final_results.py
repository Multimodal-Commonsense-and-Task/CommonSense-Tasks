import os
import json

scenario = 8
mode = "eval_results/infoBatch"

dir_bases = [
"2025_MLLM_llava_bs1_gradacc32_infoBatch_scenario20_random0.0625_seed1_iter0.125_fisher_MINZSCORE1.14_EMA0.9"
        ]
# dir_base = f"2025_1_23_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_125_{mode}_scenario{scenario}_10000_{method}"


for dir_base in dir_bases:
    print("exp:", dir_base)
    eval_rounds = [1,2,3,4]
    # k_list = [10,25,50,100,250,500,1000,2500]
    #0.4332
    averages = []
    print(dir_base)

    dir_path = os.path.join(mode,f'{dir_base}')
    all_round_result_files = os.listdir(dir_path)
    all_round_result_files = [file for file in all_round_result_files if "json" in file]

    for round_num in eval_rounds:
        k_avg = []
        round_result_files = [file for file in all_round_result_files if f"round{round_num}" in file]
        assert len(round_result_files) == round_num
        
        for file in round_result_files:
            with open(os.path.join(mode, dir_base, file), "r") as r_json:
                read_json = json.load(r_json)
            result = read_json[-1]["ROUGE_L"][0]*100
            k_avg.append(round(result,3))
        average = (round(sum(k_avg)/len(k_avg),3))
        averages.append(average)
    print("average", averages)

    print("A_avg:", round(sum(averages)/len(averages),3))
    print("A_last:", averages[-1])
