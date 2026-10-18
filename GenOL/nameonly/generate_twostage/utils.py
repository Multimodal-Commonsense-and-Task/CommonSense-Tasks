import re
import os
import fcntl
import time
from pathlib import Path


def sanitize_filename(filename):
    forbidden_chars = r'[\/\\\?\%\*\:\|\"<>\.]'
    
    sanitized = re.sub(forbidden_chars, '', filename)
    
    sanitized = re.sub(r'\s+', '_', sanitized)
    
    max_length = 255
    sanitized = sanitized[:max_length]
    
    return sanitized

def read_and_lock_file(file_path):
    f = open(file_path, 'r+')
    fcntl.flock(f, fcntl.LOCK_EX)
    content = f.readlines()
    return content, f

def write_and_unlock_file(f, content):
    f.seek(0)
    f.writelines(content)
    f.truncate()
    fcntl.flock(f, fcntl.LOCK_UN)
    f.close()

def initialize_task_file(queue_name, start_idx, end_idx, cls_name):
    lock_file = "lock.lock"
    with open(lock_file, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        task_file = f"{queue_name}_task.txt"
        if not Path(task_file).exists():
            with open(task_file, 'w') as f:
                for i in range(start_idx, end_idx + 1):
                    f.write(f"{i} {cls_name[i - start_idx]} pending\n")
        fcntl.flock(lock_f, fcntl.LOCK_UN)

def get_next_task(queue_name):
    # This function reads the task file and returns the next task to be processed
    # NOTE: next task is the id from 0 to len(dataset_list), not the actual task id
    task_file = f"{queue_name}_task.txt"
    tasks, f = read_and_lock_file(task_file)
    next_task = None
    
    for i in range(len(tasks)):
        try:
            cls_idx, cls_id, status = tasks[i].strip().split()
            cls_idx = int(cls_idx)
        except ValueError:
            print(f"Invalid task line: {tasks[i]}")
            continue
        if status == 'pending':
            next_task = cls_idx
            tasks[i] = f"{cls_idx} {str(cls_id)} in_progress\n"
            break
    
    write_and_unlock_file(f, tasks)
    return next_task

def mark_task_done(queue_name, current_cls_id):
    task_file = f"{queue_name}_task.txt"
    tasks, f = read_and_lock_file(task_file)
    for i in range(len(tasks)):
        cls_idx, cls_id, status = tasks[i].strip().split()
        cls_idx = int(cls_idx)
        if cls_idx == current_cls_id:
            tasks[i] = f"{str(cls_idx)} {cls_id} done\n"
            break
    write_and_unlock_file(f, tasks)