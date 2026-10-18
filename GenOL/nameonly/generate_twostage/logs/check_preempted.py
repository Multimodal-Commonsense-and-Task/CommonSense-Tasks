import os
import argparse

def find_preemption_start_line(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    preemption_line_index = None
    for i, line in enumerate(lines):
        if "DUE TO PREEMPTION" in line:
            preemption_line_index = i
            break
        elif "No locks available" in line:
            preemption_line_index = i
            break

    if preemption_line_index is None:
        # print("No line contains 'DUE TO PREEMPTION'.")
        return None

    search_str = "Start generating images for class"
    for i in range(preemption_line_index, -1, -1):
        if search_str in lines[i]:
            return lines[i][lines[i].index(search_str) + len(search_str) + 1:].strip()

    print("No line with 'Start generating images for class' was found before the preemption line.")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--prefix", type=str, help="Prefix of the log file.")
    args = parser.parse_args()

    log_files = os.listdir("./")
    log_files = [f for f in log_files if args.prefix in f]

    preempted_classes = []
    preempted_logs = []
    for log_file in log_files:
        start_line = find_preemption_start_line(log_file)
        if start_line:
            preempted_classes.append(start_line)
            preempted_logs.append(log_file)
    
    print(f"Preempted classes: {preempted_classes}")
    remove_str = "rm " + " ".join(preempted_logs)

    print(f"Remove command: \n{remove_str}\n")
    
    

