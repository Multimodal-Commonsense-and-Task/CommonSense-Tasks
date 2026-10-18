import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root_dir', type=str)
args = parser.parse_args()

eval_log_list = os.listdir(args.root_dir)
seeds = (1, 2, 3, 4, 5)
seed_results = [f"seed_{seed}.log" in eval_log_list for seed in seeds] # [T, T, T, T, T]
if all(seed_results):
    print(f"All seed logs are present")
    summary_present_in_all = True

    for seed in seeds:
        log_path = os.path.join(args.root_dir, f"seed_{seed}.log")
        with open(log_path, 'r') as file:
            if 'Summary' not in file.read():
                summary_present_in_all = False
                print(f"'Summary' not found in seed_{seed}.log")
                break
    
    if summary_present_in_all:
        print("All seed evaluations are complete and successful.")
        norm_path = os.path.normpath(args.root_dir)
        new_folder_name = f"complete_{os.path.basename(norm_path)}"
        new_path = os.path.join(os.path.dirname(norm_path), new_folder_name)
        os.rename(args.root_dir, new_path)
        print(f"Directory renamed to {new_folder_name}")