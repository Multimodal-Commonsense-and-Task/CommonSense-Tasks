import torch
import torch.nn as nn
import torch.nn.functional as F

from .repnet import RepNet
from .stereo_matching import StereoMatchingNet

class EventStereoMatchingNetwork(nn.Module):
    def __init__(self, event_processor=None, disparity_estimator=None):
        super(EventStereoMatchingNetwork, self).__init__()
        self.spiking_image_net1 = RepNet(**event_processor.PARAMS, act_func='snn')
        self.stereo_matching_net1 = StereoMatchingNet(**disparity_estimator.PARAMS)
        
        self.criterion = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, 
                stereo_event, 
                stereo_image=None, 
                disparity_gt=None,
                ):
        rep_img = {}
        ev_im_frame = {}
        
        for loc in ['left', 'right']:
            
            rep_img[loc] = self.spiking_image_net1(stereo_event[loc])    # [batch_size,C=T,H,W]
            
            if stereo_image is not None:
                # Stack repnet & image
                ev_im_frame[loc] = torch.cat([rep_img[loc], stereo_image[loc]], dim=1)
        
        if stereo_image is not None:
            disparity_pyramid_pred1 = self.stereo_matching_net1(ev_im_frame['left'], ev_im_frame['right'])
        else:
            disparity_pyramid_pred1 = self.stereo_matching_net1(rep_img['left'], rep_img['right'])
        
        loss_disp = None
        if disparity_gt is not None:
            loss_disp = self._cal_loss(disparity_pyramid_pred1, disparity_gt)

        return rep_img, disparity_pyramid_pred1[-1], loss_disp
    
    def get_params_group(self, learning_rate):
        def filter_specific_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return True
            return False

        def filter_base_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return False
            return True

        specific_params = list(filter(filter_specific_params,
                                      self.named_parameters()))
        base_params = list(filter(filter_base_params,
                                  self.named_parameters()))

        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (name_of_layer, params)
        base_params = [kv[1] for kv in base_params]

        specific_lr = learning_rate * 0.1
        params_group = [
            {'params': base_params, 'lr': learning_rate},
            {'params': specific_params, 'lr': specific_lr},
        ]

        return params_group
    
    def _cal_loss(self, disparity_pyramid_pred, disparity_gt):
        
        pyramid_weight = [1/3, 2/3, 1.0, 1.0, 1.0]
        loss = 0.0

        mask = disparity_gt > 0
        
        for idx in range(len(disparity_pyramid_pred)):
            pred_disp = disparity_pyramid_pred[idx]
            weight = pyramid_weight[idx]

            if pred_disp.size(-1) != disparity_gt.size(-1):
                pred_disp = pred_disp.unsqueeze(1)
                pred_disp = F.interpolate(pred_disp, size=(disparity_gt.size(-2), disparity_gt.size(-1)),
                                          mode='bilinear', align_corners=False) * (disparity_gt.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)

            cur_loss = self.criterion(pred_disp[mask], disparity_gt[mask])
            
            loss += weight * cur_loss
                
        return loss