import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import timedelta
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

import models
import utils.scheduler
import utils.visualizer
import dataset
from utils.metric import *

class DLManager:
    def __init__(self, args: argparse, test_only=False):
        self.args = args
        print(f'{args.local_rank}: {torch.cuda.get_device_name()} | {args.device}')
        
        # Prepare dataset
        self.dataset_provider = getattr(dataset, args.cfg.DATASET.NAME).dataset_provider.DatasetProvider(data_dir=args.data_dir, **args.cfg.DATASET.PARAMS)
        self.modality = self.args.cfg.DATASET.PARAMS.modality
        self.max_disp = self.args.cfg.MODEL.PARAMS.disparity_estimator.PARAMS.max_disp
        
        self.device = args.device
        self.EPOCHS = args.cfg.EPOCHS
        self.model  = getattr(models, args.cfg.MODEL.NAME)(**args.cfg.MODEL.PARAMS, 
                                                        )
        
        self._save_model_as_txt(self.model, args.save_dir+'/model.txt')
        
        if test_only:
            self.model = nn.DataParallel(self.model).cuda()
            self.optimizer = None
            self.scheduler = None
            self.save_dir = args.save_dir
            return
        
        if args.is_distributed:
            # Multi GPUs
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda()
            self.model = DistributedDataParallel(module=self.model, device_ids=[args.local_rank], find_unused_parameters=False)
        else:
            # Single GPU
            self.model = nn.DataParallel(self.model).cuda()
                
        if args.is_master:
            self.save_dir = args.save_dir
            # Save model
            if not args.checkpoint:
                self._save_model_as_txt(self.model, self.save_dir+'/model.txt')
            # Save tensorboard
            self.tb_logger = SummaryWriter(f'{self.save_dir}/')
            # For visualize the process
            plt.ion()
            self.fig = plt.figure()
            
        self.optimizer = getattr(torch.optim, args.cfg.OPTIMIZER.NAME)(self.model.module.get_params_group(args.cfg.OPTIMIZER.PARAMS.lr), **args.cfg.OPTIMIZER.PARAMS)
        self.scheduler = getattr(utils.scheduler, args.cfg.SCHEDULER.NAME)(self.optimizer, **args.cfg.SCHEDULER.PARAMS)
        
        # # Save initialized SMNet
        self.original_smnet_state_dict = deepcopy(self.model.module.stereo_matching_net1.state_dict())
        
    @torch.no_grad()
    def test(self, model_path:str):
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.train()
        vis_save_dir = self.save_dir + f'/visualization/'
        if not os.path.exists(vis_save_dir):
            os.makedirs(vis_save_dir)
            
        start_time = time.time()
        logfile = open(f'{self.save_dir}/test_log.txt', 'w+')
        # Test loss & metrics if test set has ground truth
        log_dict = OrderedDict([('Loss', AverageMeter(string_format='%6.3lf')), 
                                ('MDeE', MeanDepthError(average_by='image', string_format='%6.3lf')), 
                                ('MDisE', MeanDisparityError(average_by='image', string_format='%6.3lf')), 
                                ('1PA', NPixelAccuracy(n=1, average_by='image', string_format='%6.3lf'))
                                ])
            
        test_dataset = self.dataset_provider.get_test_dataset()
        test_loader  = DataLoader(dataset=test_dataset, 
                                **self.args.cfg.DATALOADER.TEST.PARAMS, 
                                num_workers=self.args.num_workers//self.args.world_size, 
                                pin_memory=True)
        
        for batch_data in tqdm(test_loader):
            # Data to CUDA
            batch_data = self._batch_to_cuda(batch_data)
            stereo_event = batch_data['event']              # ['left', 'right']
            stereo_image = None if self.modality == 'event' else batch_data['image']   # ['left', 'right'], [N, C, H, W]
            file_index  = batch_data['file_index']
            seq_name    = batch_data['seq_name']
            try:
                disparity_gt = batch_data['disparity_gt']   # [batch_size, H, W]
                gt_available = True
            except:
                disparity_gt = None
                gt_available = False
            
            # Test
            stereo_event_rep, pred, loss = self.model(stereo_event=stereo_event, 
                                                    stereo_image=stereo_image, 
                                                    disparity_gt=disparity_gt)            
            assert(stereo_event['left'].size(0) == 1)
            
            # Show process
            if self.args.cfg.SHOW.TEST:
                self._show_process('Test', stereo_image, stereo_event_rep, pred, disparity_gt)
            
            # Save prediction
            pred = pred.detach()
            self._pred_save(vis_save_dir, seq_name[0], file_index, pred)
            
            if gt_available:
                # Calculate loss
                loss = loss.mean()          # Loss for each pixel => loss.mean(): final loss
                # Calculate metrics
                pred = pred.detach()
                disparity_gt = disparity_gt.detach()
                mask = (disparity_gt > 0) & (disparity_gt < self.max_disp)
                if not mask.any():
                    continue
                log_dict['Loss'].update(loss.item(), pred.size(0))
                log_dict['MDeE'].update(pred, disparity_gt, mask)
                log_dict['MDisE'].update(pred, disparity_gt, mask)
                log_dict['1PA'].update(pred, disparity_gt, mask)
                
        if os.path.exists(f'{self.save_dir}/test_log.txt'):
            # Calculate test time
            test_time = time.time() - start_time
            time_for_test = self._time_checker(test_time)
            
            # Print test log
            test_summary = (f"Test | time for test: {time_for_test}\n"
                            f"Test | Loss: {log_dict['Loss']} | MDeE: {log_dict['MDeE']} | MDisE: {log_dict['MDisE']} | 1PA: {log_dict['1PA']}")
            print(test_summary)
            logfile.write(test_summary + '\n')
            logfile.flush()
        
    # ============================================== #
    # ============================================== #
    # ============================================== #
    
    def _save_model_as_txt(self, model, file_path):
        with open(file_path, "w") as f:
            # Save model structure
            f.write(str(model))
            
            # Save # of model parameters
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            f.write('\n\n' + ('Total number of parameters: %d' % num_params))
    
    def _batch_to_cuda(self, batch_data):
        def __batch_to_cuda(batch_data):
            if isinstance(batch_data, dict):
                for key in batch_data.keys():
                    batch_data[key] = __batch_to_cuda(batch_data[key])
            elif isinstance(batch_data, torch.Tensor):
                # batch_data = batch_data.to(self.device)
                batch_data = batch_data.cuda()
            else:
                raise NotImplementedError
            
            return batch_data
        
        for domain in ['image', 'event']:
            if domain not in batch_data.keys():
                batch_data[domain] = {}
            for location in ['left', 'right']:
                if location in batch_data[domain].keys():
                    batch_data[domain][location] = __batch_to_cuda(batch_data[domain][location])
                else:
                    batch_data[domain][location] = None
        if 'disparity_gt' in batch_data.keys() and batch_data['disparity_gt'] is not None:
            batch_data['disparity_gt'] = __batch_to_cuda(batch_data['disparity_gt'])
            
        return batch_data
    
    def _time_checker(self, t):
        t_delta = timedelta(seconds=t)
        total_seconds = int(t_delta.total_seconds())
        h, m, s = total_seconds // 3600, (total_seconds%3600) // 60, total_seconds % 60
        h_m_s = '{:02d}:{:02d}:{:02d}'.format(h, m, s)
        return h_m_s
    
    def _pred_save(self, vis_save_dir, seq_name, file_index, pred):
        if not os.path.exists(vis_save_dir + '/pred/' + seq_name):
            os.makedirs(vis_save_dir + '/pred/' + seq_name)
        if not os.path.exists(vis_save_dir + '/pred_magma/' + seq_name):
            os.makedirs(vis_save_dir + '/pred_magma/' + seq_name)
            
        for idx in range(len(file_index)):
            file_name = str(file_index[idx].item()).zfill(6)    # e.g., 000002, 000004, ...
            save_path = vis_save_dir + '/pred/' + seq_name + '/' + file_name + '.png'
            magma_save_path = vis_save_dir + '/pred_magma/' + seq_name + '/' + file_name + '.png'
            
            tensor_img = pred[idx, :self.args.height, :self.args.width].cpu()
            pred_img = utils.visualizer.tensor_to_disparity_image(tensor_img)
            pred_magma = utils.visualizer.tensor_to_disparity_magma_image(tensor_img, vmax=100)
            
            pred_img.save(save_path)
            pred_magma.save(magma_save_path)
            
    def _gather_log(self, log_dict):
        if log_dict is None:
            return None
        
        for key in log_dict.keys():
            if isinstance(log_dict[key], SummationMeter) or isinstance(log_dict[key], Metric):
                log_dict[key].all_gather(self.args.world_size)
                
        return log_dict
    
    def _show_process(self, title, stereo_image, stereo_event_rep, pred, disparity_gt=None):
        # 1. Intensity image
        if stereo_image is not None:
            left_img = stereo_image['left'][0].detach().cpu().numpy()       # [C, H, W]
            left_img = np.transpose(left_img, (1, 2, 0))
            left_img = (left_img * 255).astype(np.uint8)                    # [0.0 ~ 1.0] -> [0 ~ 255]
        else:
            left_img = np.zeros_like(pred[0].detach().cpu())
        
        # 2. Event Representation image
        left_rep = stereo_event_rep['left'][0][0].detach().cpu().numpy()    # [H, W]
        left_rep = (((left_rep - left_rep.min()) / (left_rep.max() - left_rep.min())) * 255).astype(np.uint8)
        
        # 3. Prediction
        pred = pred[0].detach().cpu().numpy()                           # [H, W]
        pred_frame = (((pred - pred.min()) / (pred.max() - pred.min())) * 255).astype(np.uint8)
        
        if disparity_gt is not None:
            # 4. Ground Truth
            disparity_gt = disparity_gt[0,:self.args.height,:self.args.width].detach().cpu().numpy() # [H, W]
            gt_frame = (((disparity_gt - disparity_gt.min()) / (disparity_gt.max() - disparity_gt.min())) * 255).astype(np.uint8)
            
            # 5. Error per pixel
            pred = pred[:self.args.height,:self.args.width]             # [H, W]
            err = np.abs(disparity_gt - pred)                           # [H, W]
            err_frame = (((err - err.min()) / (err.max() - err.min())) * 255).astype(np.uint8)
            err_frame[disparity_gt==0] = 0
        else:
            gt_frame = err_frame = np.zeros_like(pred_frame)
        
        # plt settings
        frames = [left_img, left_rep, pred_frame, gt_frame, err_frame]
        titles = ['Left Image', 'Left Representation', 
                  'Prediction', 'Disparity GT', 
                  'Error']
        fig, axs = plt.subplots(3, 2, figsize=(10, 12))  # Adjust the figure size
        
        for _ax, _img, _title in zip(axs.ravel(), frames, titles):
            if ((_title == 'Left Image') and (left_img.shape[-1] == 1)) or (_title == 'Left Representation'):
                cmap = 'gray'
            else:
                cmap = None
            _ax.imshow(_img, cmap=cmap)
            _ax.set_title(_title)
            _ax.axis('off')
        axs[2, 1].axis('off')  # Hide the last subplot
        
        plt.tight_layout()            
        plt.savefig(self.save_dir + f'/process_{title}.png')
        plt.clf()