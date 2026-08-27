from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..representations import *
from .. import transforms


class Sequence(Dataset):
    def __init__(self, seq_dir: str, mode: str='train', 
                 modality: str='event', representation: str='voxel', 
                 delta_t_ms: int=50, num_bins: int=15, 
                 edit_height=480, edit_width=640, cut_off_index: tuple=None):
        assert Path(seq_dir).is_dir()
        assert mode in ['train', 'validation', 'test']
        assert modality in ['event', 'image', 'EI']
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert num_bins >= 1
        # print(seq_dir, mode, modality, representation, delta_t_ms, num_bins, edit_height, edit_width)
        
        self.seq_dir = seq_dir
        self.seq_name = seq_dir.split('/')[-1]
        self.mode = mode
        
        self.modality = modality
        self.representation = representation
        self.delta_t_ms = delta_t_ms
        self.channel = num_bins
        
        self.edit_height = edit_height
        self.edit_width = edit_width
        self.strat_idx = cut_off_index[0]
        self.end_idx   = cut_off_index[1]
        self.locations = ['left', 'right']
        
        self.height = 260
        self.width  = 346
        
        # Set data transforms
        if self.mode in ['train']:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(crop_height=edit_height, crop_width=edit_width), 
                # transforms.RandomVerticalFlip(), 
            ])
        elif self.mode in ['validation', 'test']:
            self.transforms = transforms.Compose([
                transforms.Padding(pad_height=edit_height, pad_width=edit_width)
                # transforms.RandomCrop(crop_height=edit_height, crop_width=edit_width), 
            ])
        else:
            raise NotImplementedError
        
        # Load event(image) and gt data
        self.load_gt_path()
        if self.modality == 'event':
            self.load_event_path()
        elif self.modality == 'image':
            self.load_image_path()
        elif self.modality == 'EI':
            self.load_event_path()
            self.load_image_path()
        
        # assert len(self.gt_list) == len(self.event_list['left']) == len(self.event_list['right']) == len(self.image_list['left']) == len(self.image_list['right'])
        
        # Set event representation
        if self.representation == 'voxel':
            self.representation_method = VoxelGrid(self.channel, self.height, self.width, normalize=True)
        elif self.representation == 'on_off':
            self.representation_method = OnOffFrame(self.height, self.width)
        elif self.representation == 'raw':
            self.representation_method = RawEvent(self.channel, self.height, self.width)
            
    def load_gt_path(self):
        self.gt_list = list()
        gt_dir = Path(self.seq_dir) / 'disparity_gt'
        for entry in gt_dir.iterdir():
            assert str(entry.name).endswith('.png')
            self.gt_list.append(str(entry))
        self.gt_list.sort()
        self.gt_list = self.gt_list[self.strat_idx:self.end_idx]
        
    def load_event_path(self):
        self.event_list = dict()
        for loc in self.locations:
            event_list = list()
            ev_dir = Path(self.seq_dir) / f'events/{loc}'
            for entry in ev_dir.iterdir():
                assert str(entry.name).endswith('.npy')
                event_list.append(str(entry))
            event_list.sort()
            event_list = event_list[self.strat_idx:self.end_idx]
            self.event_list[loc] = event_list
            
    def load_image_path(self):
        self.image_list = dict()
        for loc in self.locations:
            image_list = list()
            img_dir = Path(self.seq_dir) / f'images/{loc}'
            for entry in img_dir.iterdir():
                assert str(entry.name).endswith('.png')
                image_list.append(str(entry))
            image_list.sort()
            image_list = image_list[self.strat_idx:self.end_idx]
            self.image_list[loc] = image_list
            
    def events_representation_method(self, event):
        t = event[:,0].astype('float64')
        t = t - t[0]
        t = (t/t[-1]).astype('float32')
        x = event[:,1].astype('float32')
        y = event[:,2].astype('float32')
        # pol = event[:,3].astype('float32')
        pol = event[:,3].astype('float16')
        
        return self.representation_method.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))
        
    def getHeightAndWidth(self):
        return self.edit_height, self.edit_width
    
    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32') / 256 / 7.0   # MVSEC Max disparity is 37
    
    @staticmethod
    def get_image_data(filepath: Path):
        assert filepath.is_file()
        image_16bit = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)   # Gray
        return image_16bit.astype('float32')/256
    
    def __len__(self):
        return self.end_idx - self.strat_idx
    
    def __getitem__(self, index):
        output = {}
        
        gt_path = Path(self.gt_list[index])
        gt = torch.from_numpy(self.get_disparity_map(gt_path))
        output['disparity_gt'] = gt
        output['file_index'] = int(gt_path.stem)
        output['seq_name'] = self.seq_name
        
        for loc in self.locations:
            # Load event data
            if self.modality == 'event' or self.modality == 'EI':
                if 'event' not in output:
                    output['event'] = dict()
                event_path = Path(self.event_list[loc][index])
                ev = np.load(event_path)
                
                event_representation = self.events_representation_method(ev)
                output['event'][loc] = event_representation
            
            # Load image data
            if self.modality == 'image' or self.modality == 'EI':
                if 'image' not in output:
                    output['image'] = dict()
                image_path = Path(self.image_list[loc][index])
                img = torch.from_numpy(np.copy(self.get_image_data(image_path)))
                img = img.unsqueeze(dim=0)
                
                output['image'][loc] = img
        
        # Apply transforms
        output = self.transforms(output)
        
        return output