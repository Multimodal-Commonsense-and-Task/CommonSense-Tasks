from MTV.mtv_utils import *
from MTV.preprocess2 import *
from tqdm import tqdm
import torch
import argparse
torch.set_grad_enabled(False)
from transformers.utils import logging
import json
import os
import random
logging.set_verbosity_error() 


def eval_reinforce(args):

    train_dataset = open_data(args.data_name, args.train_path)
    val_dataset = open_data(args.data_name, args.val_path)
    reinforce_dataset = open_data(args.data_name, args.reinforce_path) if args.reinforce_path else train_dataset


    activation_data = train_dataset

    reinforce_sample_size = min(len(reinforce_dataset), 100)
    reinforce_data = random.sample(reinforce_dataset, reinforce_sample_size)
    
    eval_sample_size = min(len(val_dataset), 50)
    eval_data = val_dataset[:eval_sample_size]

    print(f"[MTV Info] Model: {args.model_name}, Data: {args.data_name}")
    print(f"[MTV Info] Train size: {len(train_dataset)}, Reinforce sample: {reinforce_sample_size}")
    if args.reinforce_path:
        print(f"[MTV Info] Reinforce pool size: {len(reinforce_dataset)}")
    print(f"[MTV Info] Validation size: {len(val_dataset)}, Eval sample: {eval_sample_size}")

    ##Load the model
    model_helper = load_model(args.model_name, args.data_name)
    model_helper.format_func = get_format_func(args.data_name)

    ##Mean activation of some in-context input
    if args.cur_mode != "clean":

        mean_activations = get_last_mean_head_activations(activation_data, model_helper, N_TRIALS = args.num_example, shot=args.num_shot)

        torch.save(mean_activations, args.activation_path)
        mean_activations = torch.load(args.activation_path)

        # ##Examples from the test set is used to visualize the validation loss
        bernoullis = reinforce(mean_activations, model_helper, reinforce_data, eval_data)
        # torch.save(bernoullis, args.bernoullis_path)
        # bernoullis = torch.load(args.bernoullis_path)

        best_heads = (999, None)
        ###Sample multiple times and pick the best set of heads.
        for i in range(10):
            ###Sample from the trained distribution and identify the intervention locations
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=0, max=1) for bernoulli in bernoullis])
            ###Thresholding heads with low probability from being sampled. Reduce the number of heads. Idefics2 empirically benefit from less heads.
            if args.model_name == "idefics2":
                sigmoid_tensor = torch.nn.functional.threshold(sigmoid_tensor, 0.8, 0)

            
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
            sampled = prob_dist.sample()
            intervention_locations = reinforce_intervention_location(sampled)
            # cur_heads_loss = validate_reinforce(model_helper, bernoullis, 1e-3, mean_activations, train_dataset[:50], 0, sampled=sampled)
            cur_heads_loss = validate_reinforce(model_helper, bernoullis, 1e-3, mean_activations, eval_data, 0, sampled=sampled)
            if cur_heads_loss < best_heads[0]:
                best_heads = (cur_heads_loss, intervention_locations)
                print(f"  [Iter {i}] New best loss: {cur_heads_loss:.4f} (Heads: {len(intervention_locations)})")
        torch.save(best_heads[1], args.bernoullis_path)
        intervention_locations = best_heads[1]

        print(f"Selected Heads Count: {len(intervention_locations)}")
    else:
        mean_activations = None
        intervention_locations = None

    # clean_answers = []
    # interv_answers = []
    # clean_count, interv_count = 0, 0

    # for item in tqdm(val_dataset):

    #     text, image_list, target_out, question_id = model_helper.format_func(train_dataset, item, num_shot=args.eval_num_shot)
    #     new_input = model_helper.insert_image(text, image_list)
    #     clean_out, interv_out = fv_intervention_natural_text(new_input, model_helper, max_new_tokens=args.max_token, return_item=args.cur_mode, intervention_locations=intervention_locations, avg_activations=mean_activations)


    #     clean_pred = clean_out.strip()
    #     interv_pred = interv_out.strip()
        
    #     result_item = {
    #         "question_id": question_id,
    #         "target": target_out,
    #         "prediction": "",
    #         "prompt": text
    #     }

    #     if args.cur_mode in ["clean", "both"]:
    #         clean_res = result_item.copy()
    #         clean_res["prediction"] = clean_pred
    #         clean_answers.append(clean_res)

    #     if args.cur_mode in ["interv", "both"]:
    #         interv_res = result_item.copy()
    #         interv_res["prediction"] = interv_pred
    #         interv_answers.append(interv_res)

    # save_dir = os.path.join("storage_json", args.model_name, args.experiment_name)
    # os.makedirs(save_dir, exist_ok=True)

    # if args.cur_mode in ["clean", "both"]:
    #     clean_path = os.path.join(save_dir, "clean.json")
    #     with open(clean_path, "w", encoding="utf-8") as f:
    #         json.dump(clean_answers, f, indent=4, ensure_ascii=False)
    #     print(f"Saved clean results to: {clean_path}")

    # if args.cur_mode in ["interv", "both"]:
    #     interv_path = os.path.join(save_dir, "interv.json")
    #     with open(interv_path, "w", encoding="utf-8") as f:
    #         json.dump(interv_answers, f, indent=4, ensure_ascii=False)
    #     print(f"Saved intervention results to: {interv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen")
    parser.add_argument("--data_name", type=str, default="vizwiz")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--reinforce_path", type=str, default=None)
    parser.add_argument("--num_example", type=int, default=100)
    parser.add_argument("--num_shot", type=int, default=4)
    parser.add_argument("--eval_num_shot", type=int, default=0)
    parser.add_argument("--max_token", type=int, default=10)
    parser.add_argument("--bernoullis_path", type=str, default=None)
    parser.add_argument("--is_eval", type=bool, default=False)
    parser.add_argument("--result_folder", type=str, default=None)
    parser.add_argument("--cur_mode", type=str, default="interv")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--activation_path", type=str, default=None)
    
    args = parser.parse_args()

    eval_reinforce(args)
