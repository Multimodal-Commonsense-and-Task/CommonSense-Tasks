import random
import os
import argparse
import json

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the method_name argument
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--client_num', type=int, help='Number of client')
    parser.add_argument('--max_task', type=int, default=4, help='Maximum number of tasks per client')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    with open('./all_tasks.json', 'r') as fp:
        task_list = json.load(fp)
        
    print('total available tasks:', len(task_list))
    
    max_task_per_client = min(args.max_task, len(task_list) // args.client_num)
    scenario_data = []
    
    random.shuffle(task_list)
    
    for client_id in range(args.client_num):
        scenario_data.append({
            'client_id':client_id,
            'datasets':task_list[client_id*max_task_per_client:(client_id+1)*max_task_per_client]
        })
    
    output_path = f'scenario_{args.client_num}clients_{max_task_per_client}tasks_seed{args.seed}.json'
    
    with open(output_path, 'w') as json_file:
        json.dump(scenario_data, json_file, indent=4)
    
if __name__ == "__main__":
    main()