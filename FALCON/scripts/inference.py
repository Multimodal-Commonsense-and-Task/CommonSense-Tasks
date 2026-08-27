import torch
import argparse
from pathlib import Path
from yacs.config import CfgNode as CN

from manager import DLManager

from subprocess import call

def get_cfg(cfg_path):
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    return cfg


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default='/root/code/configs/config.yaml', help='Path to dataset details')
    parser.add_argument('--save_dir', default='/root/code/save', type=str, help='Path to save result')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    # Set config
    cfg = get_cfg(Path(args.cfg_path))
    assert cfg.DATASET.NAME in ['MVSEC', 'DSEC', 'M3ED']
    # args.data_dir = cfg.DATASET.DIR + cfg.DATASET.NAME
    args.data_dir = cfg.DATASET.DIR 
    assert Path(args.data_dir).is_dir(), args.data_dir
    assert cfg.DATASET.PARAMS.representation in ['voxel', 'on_off', 'raw']
    args.cfg = cfg
    
    # Save data shape
    if cfg.DATASET.NAME == 'DSEC':
        args.height = 480
        args.width  = 640
    elif cfg.DATASET.NAME == 'MVSEC':
        args.height = 260
        args.width  = 346
    
    # Only one GPU for inference
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
    
    m = DLManager(args, test_only=True)
    if args.is_master:
        m.test(model_path=args.checkpoint_path)
        print('Testing %s done.' % args.checkpoint_path)
        
        call(['chmod', '777', '-R', args.save_dir])
        call(['chmod', '777', '-R', '/root/code/results'])
        print('# Save dir: ', args.save_dir, '\n')