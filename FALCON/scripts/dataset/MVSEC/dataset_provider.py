from pathlib import Path

import torch
import torch.utils.data

from .sequence import Sequence
from .constant import *


class DatasetProvider:
    def __init__(self, split: str, data_dir: str, modality: str, representation: str='voxel', delta_t_ms: int=50, num_bins: int=5, 
                 crop_height=228, crop_width=312, pad_height=264, pad_width=348):
        self.split = split
        if split == '1':
            self.train_sequences = ['indoor_flying2', 'indoor_flying3']
            self.valid_test_sequences = ['indoor_flying1']
            self.valid_indices = SPLIT1_VALID_INDICES
            self.test_indices  = SPLIT1_TEST_INDICES
        elif split == '2':
            self.train_sequences = ['indoor_flying1', 'indoor_flying2']
            self.valid_test_sequences = ['indoor_flying2']
            self.valid_indices = SPLIT2_VALID_INDICES
            self.test_indices  = SPLIT2_TEST_INDICES
        elif split == '3':
            self.train_sequences = ['indoor_flying1', 'indoor_flying2']
            self.valid_test_sequences = ['indoor_flying3']
            self.valid_indices = SPLIT3_VALID_INDICES
            self.test_indices  = SPLIT3_TEST_INDICES
        else:
            print('Not defined split')
            exit()
        self.data_dir = data_dir
        
        self.modality = modality
        self.representation = representation
        self.delta_t_ms = delta_t_ms
        self.num_bins = num_bins
        
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.pad_height = pad_height
        self.pad_width = pad_width
        
        self.valid_test_sequence = None
        
    def get_train_dataset(self):
        train_sequences = list()
        for child in self.train_sequences:
            cut_off_index = SEQUENCES_FRAMES['indoor_flying']['split'+self.split][child]
            child = self.data_dir + '/indoor_flying/' + child
            assert Path(child).is_dir()
            train_sequences.append(Sequence(seq_dir=child, mode='train', 
                                            modality=self.modality, representation=self.representation, 
                                            delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, 
                                            edit_height=self.crop_height, edit_width=self.crop_width, cut_off_index=cut_off_index))
            
        train_dataset = torch.utils.data.ConcatDataset(train_sequences)
        return train_dataset
    
    def get_valid_dataset(self):
        if self.valid_test_sequence is None:
            child = self.valid_test_sequences[0]
            cut_off_index = SEQUENCES_FRAMES['indoor_flying']['split'+self.split][child]
            child = self.data_dir + '/indoor_flying/' + child
            assert Path(child).is_dir()
            self.valid_test_sequence = (Sequence(seq_dir=child, mode='validation', 
                                                 modality=self.modality, representation=self.representation, 
                                                 delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, 
                                                 edit_height=self.pad_height, edit_width=self.pad_width, cut_off_index=cut_off_index))
            
        valid_dataset = torch.utils.data.Subset(self.valid_test_sequence, self.valid_indices)
        return valid_dataset
    
    def get_test_dataset(self):
        if self.valid_test_sequence is None:
            child = self.valid_test_sequences[0]
            cut_off_index = SEQUENCES_FRAMES['indoor_flying']['split'+self.split][child]
            child = self.data_dir + '/indoor_flying/' + child
            assert Path(child).is_dir()
            self.valid_test_sequence = (Sequence(seq_dir=child, mode='test', 
                                                 modality=self.modality, representation=self.representation, 
                                                 delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, 
                                                 edit_height=self.pad_height, edit_width=self.pad_width, cut_off_index=cut_off_index))
            
        test_dataset = torch.utils.data.Subset(self.valid_test_sequence, self.test_indices)
        return test_dataset