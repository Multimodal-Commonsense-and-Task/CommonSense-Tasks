import os
import shutil
import argparse

from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms.functional import to_pil_image


def get_time():
    fmt = '%Y-%m-%d %H:%M:%S %Z%z'
    kst_time = datetime.now().strftime(fmt)

    return kst_time


class Log:
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log_write(self, log, mode='a', is_print=True, add_time=True):
        if add_time:
            log = '%s: %s' %(get_time(), log)
        if is_print:
            print(log)
        with open(self.log_path, mode=mode) as f:
            f.write(log + '\n')
            
            
class ExpLogger(Log):
    _FILE_NAME = {'args'       : 'args.txt', 
                  'model'      : 'model.txt', 
                  'train'      : 'train_log.txt', 
                  'validation' : 'validation_log.txt', 
                  'test'       : 'test_log.txt'}
    
    _DIR_NAME  = {'scripts'    : 'scripts', 
                  'weight'     : 'weight', 
                  'spiking_img': 'spiking_img', 
                  'visualize'  : 'visualize'}
    
    def __init__(self, save_dir: str, mode='train'):
        assert mode in ['train', 'validation', 'test']
        self.save_dir      = save_dir
        self.mode           = mode
        self.summary_writer = SummaryWriter(self.save_dir)
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train(self):
        self.mode = 'train'
        
    def validation(self):
        self.mode = 'validation'
        
    def test(self):
        self.mode = 'test'
         
    def log_write(self, log, file_name=None, mode='a', is_print=True, add_time=True):
        if file_name is None:
            file_name = self._FILE_NAME[self.mode]
        super().__init__(os.path.join(self.save_dir, file_name))
        super().log_write(log=log, mode=mode, is_print=is_print, add_time=add_time)
        
    def add_scalar(self, tag, scalar_value, global_step):
        if self.summary_writer is None:
            self.summary_writer = SummaryWriter(self.save_root)
        self.summary_writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)
        
    def save_args(self, args: argparse):
        args_log = ''
        for arg in args.__dict__.keys():
            args_log += '--%s %s \\\n' % (arg, args.__dict__[arg])
        self.log_write(log=args_log, file_name=self._FILE_NAME['args'], mode='w', is_print=False, add_time=False)
        
    def save_model(self, model: torch.nn.Module):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        log = str(model) + '\n\n' + ('Total number of parameters: %d' % num_params)
        self.log_write(log=log, file_name=self._FILE_NAME['model'], mode='w', is_print=False, add_time=False)
        
    def save_code(self, code_dir: str):
        code_save_dir = os.path.join(self.save_dir, self._DIR_NAME['scripts'])
        if os.path.exists(code_save_dir):
            shutil.rmtree(code_save_dir)
        shutil.copytree(code_dir, code_save_dir)
        
    def save_file(self, file, file_name):
        torch.save(file, os.path.join(self.save_dir, file_name))

    def load_file(self, file_name):
        return torch.load(os.path.join(self.save_dir, file_name))
    
    def save_checkpoint(self, checkpoint, name):
        checkpoint_dir = os.path.join(self.save_dir, self._DIR_NAME['weight'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, name)
        torch.save(checkpoint, checkpoint_path)
        
        self.log_write(log='Checkpoint is saved to %s' % checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.log_write(log='Checkpoint is Loaded from %s' % checkpoint_path)
        
        return checkpoint
    
    def save_visualize(self, image, visual_type, sequence_name, image_name):
        visualize_dir = os.path.join(self.save_dir, self._DIR_NAME['visualize'], visual_type, sequence_name)
        os.makedirs(visualize_dir, exist_ok=True)
        visualize_path = os.path.join(visualize_dir, image_name)
        image.save(visualize_path)
        
    def save_spiking_img(self, spiking_img, epoch, local_rank):
        pil_img_dir = os.path.join(self.save_dir, self._DIR_NAME['spiking_img'])
        pil_img_dir = os.path.join(pil_img_dir, self.mode)
        pil_img_dir = os.path.join(pil_img_dir, str(epoch))
        os.makedirs(f'{pil_img_dir}/left', exist_ok=True)
        os.makedirs(f'{pil_img_dir}/right', exist_ok=True)
        
        for loc in ['left', 'right']:
            for i in range(len(spiking_img[loc])):
                pil_img = to_pil_image(spiking_img[loc][i])
                # print(f'{loc} spiking img {i}: {spiking_img[loc][i].shape}')
                pil_img.save(f'{pil_img_dir}/{loc}/{loc}_spiking_img_{i+(local_rank*len(spiking_img[loc]))}.jpg', 'JPEG')
                
    def save_ground_truth(self, disparity_gt, epoch):
        pil_img_dir = os.path.join(self.save_dir, self._DIR_NAME['spiking_img'])
        pil_img_dir = os.path.join(pil_img_dir, self.mode)
        pil_img_dir = os.path.join(pil_img_dir, str(epoch))
        os.makedirs(f'{pil_img_dir}/gt', exist_ok=True)
        
        for i in range(len(disparity_gt)):
            pil_img = to_pil_image(disparity_gt[i])
            # print(f'gt shape {i}: {disparity_gt[i].shape}')
            pil_img.save(f'{pil_img_dir}/gt/disparity_gt_{i}.jpg', 'JPEG')
            