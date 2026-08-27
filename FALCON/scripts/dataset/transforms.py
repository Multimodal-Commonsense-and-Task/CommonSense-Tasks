# import numpy as np
# import torch


# class Compose:
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, sample):
#         for t in self.transforms:
#             sample = t(sample)
#         return sample
    

# class CenterCrop:
#     def __init__(self, crop_height, crop_width):
#         self.crop_height = crop_height
#         self.crop_width  = crop_width
        
#     def __call__(self, sample):
#         if 'event' in sample.keys():
#             ori_height, ori_width = sample['event']['left'].shape[1:]
#         elif 'image' in sample.keys():
#             ori_height, ori_width = sample['image']['left'].shape[1:]
#         else:
#             raise NotImplementedError
        
#         assert self.crop_height <= ori_height and self.crop_width <= ori_width
        
#         # Set offset: coordinates for crop
#         offset_x = (ori_width - self.crop_width) // 2
#         offset_y = (ori_height - self.crop_height) // 2
        
#         # Set crop coordinates
#         start_y, end_y = offset_y, offset_y + self.crop_height
#         start_x, end_x = offset_x, offset_x + self.crop_width
        
#         # Cropping
#         ## Event data
#         if 'event' in sample.keys():
#             # Crop event data
#             for loc in ['left', 'right']:
#                 # timestep, H, W
#                 sample['event'][loc] = sample['event'][loc][:, start_y:end_y, start_x:end_x]
#         ## Image data
#         if 'image' in sample.keys():
#             # Crop iamge data            
#             for loc in ['left', 'right']:
#                 # timestep, H, W
#                 sample['image'][loc] = sample['image'][loc][:, start_y:end_y, start_x:end_x]
#         ## Disparity_gt
#         if 'disparity_gt' in sample.keys():
#             # Crop disparity_gt
#             # H, W
#             if 'event' in sample.keys():
#                 sample['disparity_gt'] = sample['disparity_gt'][start_y:end_y, start_x:end_x]
        
#         return sample


# class RandomCrop:
#     def __init__(self, crop_height, crop_width):
#         self.crop_height = crop_height
#         self.crop_width  = crop_width
        
#     def __call__(self, sample):
#         if 'event' in sample.keys():
#             ori_height, ori_width = sample['event']['left'].shape[1:]
#         elif 'image' in sample.keys():
#             ori_height, ori_width = sample['image']['left'].shape[1:]
#         else:
#             raise NotImplementedError
        
#         assert self.crop_height <= ori_height and self.crop_width <= ori_width
        
#         # Set offset: coordinates for crop
#         offset_x = np.random.randint(ori_width - self.crop_width + 1)
#         offset_y = np.random.randint(ori_height - self.crop_height + 1)
        
#         # Set crop coordinates
#         start_y, end_y = offset_y, offset_y + self.crop_height
#         start_x, end_x = offset_x, offset_x + self.crop_width
        
#         # Cropping
#         ## Event data
#         if 'event' in sample.keys():
#             # Crop event data
#             for loc in ['left', 'right']:
#                 # # timestep, H, W
#                 sample['event'][loc] = sample['event'][loc][:, start_y:end_y, start_x:end_x]
#         ## Image data
#         if 'image' in sample.keys():
#             # Crop iamge data            
#             for loc in ['left', 'right']:
#                 # timestep, H, W
#                 sample['image'][loc] = sample['image'][loc][:, start_y:end_y, start_x:end_x]
#         ## Disparity_gt
#         if 'disparity_gt' in sample.keys():
#             # Crop disparity_gt
#             # H, W
#             sample['disparity_gt'] = sample['disparity_gt'][start_y:end_y, start_x:end_x]
        
#         return sample
    
    
# class RandomVerticalFlip:
#     def __init__(self):
#         pass
    
#     def __call__(self, sample):
#         if np.random.random() < 0.5:
#             if 'event' in sample.keys():
#                 for loc in ['left', 'right']:
#                     # # timestep, H, W
#                     sample['event'][loc] = torch.flip(sample['event'][loc], dims=(1,))
#             if 'image' in sample.keys():
#                 for loc in ['left', 'right']:
#                     # timestep, H, W
#                     sample['image'][loc] = torch.flip(sample['image'][loc], dims=(1,))
#             if 'disparity_gt' in sample.keys():
#                 # H, W
#                 sample['disparity_gt'] = torch.flip(sample['disparity_gt'], dims=(0,))
                
#         return sample
    
    
# # Horizontal flip should not be applied in stereo setup
# class RandomHorizontalFlip:
#     def __init__(self):
#         pass
    
#     def __call__(self, sample):
#         if np.random.random() < 0.5:
#             if 'event' in sample.keys():
#                 for loc in ['left', 'right']:
#                     # timestep, H, W
#                     sample['event'][loc] = torch.flip(sample['event'][loc], dims=(2,))
#             if 'image' in sample.keys():
#                 for loc in ['left', 'right']:
#                     # timestep, H, W
#                     sample['image'][loc] = torch.flip(sample['image'][loc], dims=(2,))
#             if 'disparity_gt' in sample.keys():
#                 # H, W
#                 sample['disparity_gt'] = torch.flip(sample['disparity_gt'], dims=(1,))
                
#         return sample
    
    
# class Padding:
#     def __init__(self, pad_height, pad_width):
#         self.pad_height = pad_height
#         self.pad_width  = pad_width
        
#     def __call__(self, sample):
#         if 'event' in sample.keys():
#             ori_height, ori_width = sample['event']['left'].shape[1:]
#         elif 'image' in sample.keys():
#             ori_height, ori_width = sample['image']['left'].shape[1:]
#         else:
#             raise NotImplementedError
        
#         top_pad   = self.pad_height - ori_height
#         right_pad = self.pad_width - ori_width

#         assert top_pad >= 0 and right_pad >= 0
        
#         if 'event' in sample.keys():
#             for loc in ['left', 'right']:
#                 # # timestep, H, W
#                 sample['event'][loc] = np.lib.pad(sample['event'][loc], 
#                                                 ((0,0), (top_pad,0), (0,right_pad)), 
#                                                 mode='constant', 
#                                                 constant_values=0)
#         if 'image' in sample.keys():
#             for loc in ['left', 'right']:
#                 # timestep, H, W
#                 sample['image'][loc] = np.lib.pad(sample['image'][loc], 
#                                                   ((0,0), (top_pad,0), (0,right_pad)), 
#                                                   mode='constant', 
#                                                   constant_values=0)
#         if 'disparity_gt' in sample.keys():
#             # H, W
#             sample['disparity_gt'] = np.lib.pad(sample['disparity_gt'], 
#                                                 ((top_pad,0), (0,right_pad)), 
#                                                 mode='constant', 
#                                                 constant_values=0)
            
#         return sample


import numpy as np
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    

class CenterCrop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width  = crop_width
        
    def __call__(self, sample):
        if 'event' in sample.keys():
            ori_height, ori_width = sample['event']['left'].shape[1:]
        elif 'image' in sample.keys():
            ori_height, ori_width = sample['image']['left'].shape[1:]
        else:
            raise NotImplementedError
        
        assert self.crop_height <= ori_height and self.crop_width <= ori_width
        
        # Set offset: coordinates for crop
        offset_x = (ori_width - self.crop_width) // 2
        offset_y = (ori_height - self.crop_height) // 2
        
        # Set crop coordinates
        start_y, end_y = offset_y, offset_y + self.crop_height
        start_x, end_x = offset_x, offset_x + self.crop_width
        
        # Cropping
        ## Event data
        if 'event' in sample.keys():
            # Crop event data
            for loc in ['left', 'right']:
                # timestep, H, W
                sample['event'][loc] = sample['event'][loc][:, start_y:end_y, start_x:end_x]
        ## Image data
        if 'image' in sample.keys():
            # Crop iamge data            
            for loc in ['left', 'right']:
                # timestep, H, W
                sample['image'][loc] = sample['image'][loc][:, start_y:end_y, start_x:end_x]
        ## Disparity_gt
        if 'disparity_gt' in sample.keys():
            # Crop disparity_gt
            # H, W
            if 'event' in sample.keys():
                sample['disparity_gt'] = sample['disparity_gt'][start_y:end_y, start_x:end_x]
        
        return sample


class RandomCrop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width  = crop_width
        
    def __call__(self, sample):
        if 'event' in sample.keys():
            ori_height, ori_width = sample['event']['left'].shape[1:]
        elif 'image' in sample.keys():
            ori_height, ori_width = sample['image']['left'].shape[1:]
        else:
            raise NotImplementedError
        
        assert self.crop_height <= ori_height and self.crop_width <= ori_width
        
        # Set offset: coordinates for crop
        offset_x = np.random.randint(ori_width - self.crop_width + 1)
        offset_y = np.random.randint(ori_height - self.crop_height + 1)
        
        # Set crop coordinates
        start_y, end_y = offset_y, offset_y + self.crop_height
        start_x, end_x = offset_x, offset_x + self.crop_width
        
        # Cropping
        ## Event data
        if 'event' in sample.keys():
            # Crop event data
            for loc in ['left', 'right']:
                # # timestep, H, W
                sample['event'][loc] = sample['event'][loc][:, start_y:end_y, start_x:end_x]
        ## Image data
        if 'image' in sample.keys():
            # Crop iamge data            
            for loc in ['left', 'right']:
                # C, H, W
                sample['image'][loc] = sample['image'][loc][:, start_y:end_y, start_x:end_x]
        ## Disparity_gt
        if 'disparity_gt' in sample.keys():
            # Crop disparity_gt
            # H, W
            sample['disparity_gt'] = sample['disparity_gt'][start_y:end_y, start_x:end_x]
        ## Semantic map
        # if 'semantic_map' in sample.keys():
        #     # Crop semantic_map
        #     # H, W
        #     sample['semantic_map'] = sample['semantic_map'][start_y:end_y, start_x:end_x]
        ## np_raw_image: 0~255
        if 'semantic_map' in sample.keys():
            # Crop semantic_map
            for loc in ['left', 'right']:
                # H, W, C
                sample['semantic_map'][loc] = sample['semantic_map'][loc][start_y:end_y, start_x:end_x]
        if 'longstack' in sample.keys():
            # Crop event data
            for loc in ['left', 'right']:
                # # timestep, H, W
                sample['longstack'][loc] = sample['longstack'][loc][:, start_y:end_y, start_x:end_x]
        
        return sample
    
    
class RandomVerticalFlip:
    def __init__(self):
        pass
    
    def __call__(self, sample):
        if np.random.random() < 0.5:
            if 'event' in sample.keys():
                for loc in ['left', 'right']:
                    # # timestep, H, W
                    sample['event'][loc] = torch.flip(sample['event'][loc], dims=(1,))
            if 'image' in sample.keys():
                for loc in ['left', 'right']:
                    # timestep, H, W
                    sample['image'][loc] = torch.flip(sample['image'][loc], dims=(1,))
            if 'disparity_gt' in sample.keys():
                # H, W
                sample['disparity_gt'] = torch.flip(sample['disparity_gt'], dims=(0,))
                
        return sample
    
    
# Horizontal flip should not be applied in stereo setup
class RandomHorizontalFlip:
    def __init__(self):
        pass
    
    def __call__(self, sample):
        if np.random.random() < 0.5:
            if 'event' in sample.keys():
                for loc in ['left', 'right']:
                    # timestep, H, W
                    sample['event'][loc] = torch.flip(sample['event'][loc], dims=(2,))
            if 'image' in sample.keys():
                for loc in ['left', 'right']:
                    # timestep, H, W
                    sample['image'][loc] = torch.flip(sample['image'][loc], dims=(2,))
            if 'disparity_gt' in sample.keys():
                # H, W
                sample['disparity_gt'] = torch.flip(sample['disparity_gt'], dims=(1,))
                
        return sample
    
    
class Padding:
    def __init__(self, pad_height, pad_width):
        self.pad_height = pad_height
        self.pad_width  = pad_width
        
    def __call__(self, sample):
        if 'event' in sample.keys():
            ori_height, ori_width = sample['event']['left'].shape[1:]            
        elif 'image' in sample.keys():
            ori_height, ori_width = sample['image']['left'].shape[1:]
        else:
            raise NotImplementedError
        
        top_pad   = self.pad_height - ori_height
        right_pad = self.pad_width - ori_width
        assert top_pad >= 0 and right_pad >= 0
        
        if 'event' in sample.keys():
            for loc in ['left', 'right']:
                # # timestep, H, W
                sample['event'][loc] = np.pad(sample['event'][loc], 
                                                ((0,0), (top_pad,0), (0,right_pad)), 
                                                mode='constant', 
                                                constant_values=0)
        if 'image' in sample.keys():
            for loc in ['left', 'right']:
                # C, H, W
                sample['image'][loc] = np.pad(sample['image'][loc], 
                                                  ((0,0), (top_pad,0), (0,right_pad)), 
                                                  mode='constant', 
                                                  constant_values=0)
        if 'disparity_gt' in sample.keys():
            # H, W
            sample['disparity_gt'] = np.pad(sample['disparity_gt'], 
                                                ((top_pad,0), (0,right_pad)), 
                                                mode='constant', 
                                                constant_values=0)
        # if 'semantic_map' in sample.keys():
        #     # H, W
        #     sample['semantic_map'] = np.lib.pad(sample['semantic_map'], 
        #                                           ((top_pad,0), (0,right_pad)), 
        #                                           mode='constant', 
        #                                           constant_values=0)
        if 'semantic_map' in sample.keys():
            for loc in ['left', 'right']:
                # H, W, C
                sample['semantic_map'][loc] = np.pad(sample['semantic_map'][loc], 
                                                         ((top_pad,0), (0,right_pad)), 
                                                         mode='constant', 
                                                         constant_values=0)
        
        if 'longstack' in sample.keys():
            for loc in ['left', 'right']:
                # # timestep, H, W
                sample['longstack'][loc] = np.pad(sample['longstack'][loc], 
                                                ((0,0), (top_pad,0), (0,right_pad)), 
                                                mode='constant', 
                                                constant_values=0)
                
        return sample