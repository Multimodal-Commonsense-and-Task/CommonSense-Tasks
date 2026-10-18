import shutils
import os

from_dir = "eval_results_from"
target_dir = "eval_results_target"

from_dir_list = os.listdir(from_dir)
target_dir_list = os.listdir(target_dir)

for from_dir_elem in from_dir_list:
    from_dir_elem_seeds = os.listdir(os.path.join(from_dir, from_dir_elem))
    for seed in from_dir_elem_seeds:
        print(os.path.join(from_dir, from_dir_elem, seed))
        if not os.path.exists(os.path.join(from_dir, from_dir_elem, seed)):
            os.makedirs(os.path.join(target_dir, from_dir_elem), exists_ok=True)
            shutils.move(os.path.join(from_dir, from_dir_elem, seed), os.path.join(target_dir, from_dir_elem, seed))
            print("move", os.path.join(from_dir, from_dir_elem, seed), "to", os.path.join(target_dir, from_dir_elem, seed))
