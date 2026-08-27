import os
import sys
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from yacs.config import CfgNode as CN
from subprocess import call

import torch
import torch.distributed

from manager import DLManager

# torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

def get_cfg(cfg_path):
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    return cfg

def save_cfg_to_txt(cfg, save_dir):
    with open(f"{save_dir}/config.yaml", "w") as f:
        for key, value in cfg.items():
            f.write(f"{key}: {value}\n")
            
def set_random_seed(seed):
    # Python
    random.seed(seed)
    
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Numpy
    np.random.seed(seed)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default='/root/code/configs/config.yaml', help='Path to dataset details')
    parser.add_argument('--save_dir', default='/root/code/save', type=str, help='Path to save result')
    parser.add_argument('--checkpoint', action='store_true', help='Continue learning from saved checkpoint')
    # parser.add_argument('--local_rank', type=int, default=0)
    DEFAULT_LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    parser.add_argument('--local_rank', type=int, default=DEFAULT_LOCAL_RANK)
    args = parser.parse_args()
    
    # Set config
    cfg = get_cfg(Path(args.cfg_path))
    assert cfg.DATASET.NAME in ['MVSEC', 'DSEC', 'M3ED']
    args.data_dir = cfg.DATASET.DIR + cfg.DATASET.NAME
    assert Path(args.data_dir).is_dir()
    assert cfg.DATASET.PARAMS.representation in ['voxel', 'on_off', 'raw']
    args.cfg = cfg
    
    # Save data shape
    if cfg.DATASET.NAME == 'DSEC':
        args.height = 480
        args.width  = 640
    elif cfg.DATASET.NAME == 'MVSEC':
        args.height = 260
        args.width  = 346
    
    # Set seed
    # seed = np.random.randint(0, 2**32 - 1)
    seed = 930118773 
    set_random_seed(seed)
    
    try:
        assert int(os.environ['WORLD_SIZE']) >= 1
    except:
        print('\n####################################################')
        print('### Excute launch/main.sh file, not main.py file ###')
        print('###      Enter terminal command like below       ###')
        print('###          $ cd launch/ && . main.sh           ###')
        print('####################################################\n')
        sys.exit()
        
    if int(os.environ['WORLD_SIZE']) > 1:
        # Multi GPUs
        print('Use Multi GPUs')
        args.is_distributed = True
    else:
        # Or only one GPU
        print('Use Single GPU')
        args.is_distributed = False
        
    # Device config
    args.is_master   = args.local_rank == 0
    args.device      = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size  = torch.distributed.get_world_size()
    args.rank        = torch.distributed.get_rank()
    args.num_workers = 4 * args.world_size
    args.save_dir = args.save_dir + '_'+ str(seed)
    
    if args.is_master:
        # Save scripts
        source_folder = '/root/code/scripts'
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        shutil.copytree(source_folder, args.save_dir+'/scripts')
        
        # Save config
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        call(['chmod', '777', '-R', args.save_dir])
        call(['chmod', '777', '-R', '/root/code/save'])
        cfg.defrost()
        cfg.seed = seed
        cfg.freeze()
        save_cfg_to_txt(cfg, args.save_dir)
    
    m = DLManager(args, test_only=False)
    m.train()
    if args.is_master:
        # m.validation(args.cfg.EPOCHS)
        m.test(args.cfg.EPOCHS)
    
    call(['chmod', '777', '-R', args.save_dir])
    call(['chmod', '777', '-R', '/root/code/save'])
    print('# Save dir: ', args.save_dir, '\n')
