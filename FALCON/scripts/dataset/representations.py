# import torch


# class EventRepresentation:
#     def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
#         raise NotImplementedError


# class VoxelGrid(EventRepresentation):
#     def __init__(self, channels: int, height: int, width: int, normalize: bool):
#         self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
#         self.nb_channels = channels
#         self.normalize = normalize

#     def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
#         # NOTE: The Discritized Event Volume
#         # t_norm   = (bins - 1)(t - t[0])/(t[-1]-t[0])
#         # V(x,y,t) = Sigma(p * k(x-x_i) * k(y-y_i) * k(t-t_i))
#         # k(a)     = max(0, 1 - |a|)
#         # 
#         # A. Z. Zhu, L. Yuan, K. Chaney, and K. Daniilidis, 
#         # “Unsupervised eventbased learning of optical flow, depth, and egomotion,” 
#         # in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2019.
        
#         assert x.shape == y.shape == pol.shape == time.shape
#         assert x.ndim == 1

#         C, H, W = self.voxel_grid.shape
#         with torch.no_grad():
#             self.voxel_grid = self.voxel_grid.to(pol.device)
#             voxel_grid = self.voxel_grid.clone()

#             t_norm = time
#             t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

#             x0 = x.int()
#             y0 = y.int()
#             t0 = t_norm.int()

#             value = 2*pol-1

#             for xlim in [x0,x0+1]:
#                 for ylim in [y0,y0+1]:
#                     for tlim in [t0,t0+1]:

#                         mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
#                         interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

#                         index = H * W * tlim.long() + \
#                                 W * ylim.long() + \
#                                 xlim.long()

#                         voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

#             if self.normalize:
#                 mask = torch.nonzero(voxel_grid, as_tuple=True)
#                 if mask[0].size()[0] > 0:
#                     mean = voxel_grid[mask].mean()
#                     std = voxel_grid[mask].std()
#                     if std > 0:
#                         voxel_grid[mask] = (voxel_grid[mask] - mean) / std
#                     else:
#                         voxel_grid[mask] = voxel_grid[mask] - mean

#         return voxel_grid


# class OnOffFrame(EventRepresentation):
#     def __init__(self, height: int, width: int):
#         self.on_off_frame = torch.zeros((2, height, width), dtype=torch.float, requires_grad=False)
        
#     def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
#         # NOTE: shape of (2, Height, Width) and contained positive integers, 
#         # corresponding to the number of spikes of each polarity that showed up 
#         # at each position of the scene during the time window
#         #
#         # U. Ranc¸on, J. Cuadrado-Anibarro, B. R. Cottereau, and T. Masquelier,
#         # “Stereospike: Depth learning with a spiking neural network,” 
#         # arXiv preprint arXiv:2109.13751, 2021.
        
#         assert x.shape == y.shape == pol.shape == time.shape
#         assert x.ndim == 1
        
#         C, H, W = self.on_off_frame.shape
#         with torch.no_grad():
#             self.on_off_frame = self.on_off_frame.to(pol.device)
#             on_off_frame = self.on_off_frame.clone()
            
#             x0 = x.int()
#             y0 = y.int()
#             for xlim in [x0, x0+1]:
#                 for ylim in [y0, y0+1]:
#                     mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0)
                    
#                     index = H * W * pol.long() + \
#                             W * ylim.long() + \
#                             xlim.long()
#                     spike_value = torch.ones_like(mask, dtype=torch.float)
#                     on_off_frame.put_(index[mask], spike_value[mask], accumulate=True)
                    
#         return on_off_frame
    
    
# class RawEvent(EventRepresentation):
#     def __init__(self, channels: int, height: int, width: int):
#         # E.g., 50ms, 1M events
#         #       channels = num_bins = 50
#         #       total 50-ch -> {50ms, 1M events} / 50 -> (1ms, 20K events) per 1-ch
#         self.raw_stream = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
#         self.dt = 1.0 / channels        # accumulate events within dt in one frame
        
#     def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
#         assert x.shape == y.shape == pol.shape == time.shape
#         assert x.ndim == 1
        
#         C, H, W = self.raw_stream.shape
#         # with torch.no_grad():
#         #     self.raw_stream = self.raw_stream.to(pol.device)
#         #     raw_stream = self.raw_stream.clone()
            
#         #     pix_x = x.int()
#         #     pix_y = y.int()
#         #     time /= self.dt
#         #     time  = time.int()
#         #     time[time==time[-1]] = time[-1] - 1
#         #     value = 2*pol-1
            
#         #     mask = (pix_x < W) & (pix_x >= 0) & (pix_y < H) & (pix_y >= 0) & (time < C) & (time >= 0)
            
#         #     index = H * W * time.long() + \
#         #             W * pix_y.long() + \
#         #             pix_x.long()
#         #     raw_stream.put_(index[mask], value[mask], accumulate=True)
#         # return raw_stream
        
#         with torch.no_grad():
#             self.raw_stream = self.raw_stream.to(pol.device)
#             raw_stream = self.raw_stream.clone()
            
#             pix_x = x.int()
#             pix_y = y.int()
#             time /= self.dt
#             time  = time.int()
#             time[time==time[-1]] = time[-1] - 1
#             value = (2*pol-1).to(torch.float32)
            
#             mask = (pix_x < W) & (pix_x >= 0) & (pix_y < H) & (pix_y >= 0) & (time < C) & (time >= 0)
            
#             index = H * W * time + \
#                     W * pix_y + \
#                     pix_x
#             index = index.long()
#             raw_stream.put_(index[mask], value[mask], accumulate=True)
#         return raw_stream

import torch


class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        # NOTE: The Discritized Event Volume
        # t_norm   = (bins - 1)(t - t[0])/(t[-1]-t[0])
        # V(x,y,t) = Sigma(p * k(x-x_i) * k(y-y_i) * k(t-t_i))
        # k(a)     = max(0, 1 - |a|)
        # 
        # A. Z. Zhu, L. Yuan, K. Chaney, and K. Daniilidis, 
        # “Unsupervised eventbased learning of optical flow, depth, and egomotion,” 
        # in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2019.
        
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(pol.device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = time
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            value = 2*pol-1

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid


class OnOffFrame(EventRepresentation):
    def __init__(self, height: int, width: int):
        self.on_off_frame = torch.zeros((2, height, width), dtype=torch.float, requires_grad=False)
        
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        # NOTE: shape of (2, Height, Width) and contained positive integers, 
        # corresponding to the number of spikes of each polarity that showed up 
        # at each position of the scene during the time window
        #
        # U. Ranc¸on, J. Cuadrado-Anibarro, B. R. Cottereau, and T. Masquelier,
        # “Stereospike: Depth learning with a spiking neural network,” 
        # arXiv preprint arXiv:2109.13751, 2021.
        
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1
        
        C, H, W = self.on_off_frame.shape
        with torch.no_grad():
            self.on_off_frame = self.on_off_frame.to(pol.device)
            on_off_frame = self.on_off_frame.clone()
            
            x0 = x.int()
            y0 = y.int()
            for xlim in [x0, x0+1]:
                for ylim in [y0, y0+1]:
                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0)
                    
                    index = H * W * pol.long() + \
                            W * ylim.long() + \
                            xlim.long()
                    spike_value = torch.ones_like(mask, dtype=torch.float)
                    on_off_frame.put_(index[mask], spike_value[mask], accumulate=True)
                    
        return on_off_frame
    
    
class RawEvent(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int):
        # E.g., 50ms, 1M events
        #       channels = num_bins = 50
        #       total 50-ch -> {50ms, 1M events} / 50 -> (1ms, 20K events) per 1-ch
        self.raw_stream = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        # self.ls_raw_stream = torch.zeros((1, height, width), dtype=torch.float, requires_grad=False)
        self.dt = 1.0 / channels        # accumulate events within dt in one frame
        
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1
        
        C, H, W = self.raw_stream.shape
        # with torch.no_grad():
        #     self.raw_stream = self.raw_stream.to(pol.device)
        #     raw_stream = self.raw_stream.clone()
            
        #     pix_x = x.int()
        #     pix_y = y.int()
        #     time /= self.dt
        #     time  = time.int()
        #     time[time==time[-1]] = time[-1] - 1
        #     value = 2*pol-1
            
        #     mask = (pix_x < W) & (pix_x >= 0) & (pix_y < H) & (pix_y >= 0) & (time < C) & (time >= 0)
            
        #     index = H * W * time.long() + \
        #             W * pix_y.long() + \
        #             pix_x.long()
        #     raw_stream.put_(index[mask], value[mask], accumulate=True)
        # return raw_stream
        
        with torch.no_grad():
            self.raw_stream = self.raw_stream.to(pol.device)
            raw_stream = self.raw_stream.clone()
            
            pix_x = x.int()
            pix_y = y.int()
            # time /= self.dt
            # time  = time.int()
            # time[time==time[-1]] = time[-1] - 1
            # value = (2*pol-1).to(torch.float32)
            
            # mask = (pix_x < W) & (pix_x >= 0) & (pix_y < H) & (pix_y >= 0) & (time < C) & (time >= 0)
            
            # index = H * W * time + \
            #         W * pix_y + \
            #         pix_x
            # index = index.long()
            # raw_stream.put_(index[mask], value[mask], accumulate=True)
        # return raw_stream
        
            # self.ls_raw_stream = self.ls_raw_stream.to(pol.device)
            # ls_raw_stream = self.ls_raw_stream.clone()
            
            time_org = time / self.dt
            # time /= self.dt
            time_org = time_org.int()
            # time  = time.int()
            # time[time==time[-1]] = time[-1] - 1
            time_org[time_org==time_org[-1]] = time_org[-1] - 1
            
            value = (2*pol-1).to(torch.float32)
            
            # mask = (pix_x < W) & (pix_x >= 0) & (pix_y < H) & (pix_y >= 0) & (time < C) & (time >= 0)
            mask_org = (pix_x < W) & (pix_x >= 0) & (pix_y < H) & (pix_y >= 0) & (time_org < C) & (time_org >= 0)
            
            # index = H * W * time.long() + \
            #         W * pix_y.long() + \
            #         pix_x.long()
            
            index_org = H * W * time_org.long() + \
                    W * pix_y.long() + \
                    pix_x.long()
                    
            # index = index.long()
            # index_org = index_org.long()
            raw_stream.put_(index_org[mask_org], value[mask_org], accumulate=True)
            # ls_raw_stream.put_(index[mask], value[mask], accumulate=True)
        
        # return raw_stream, ls_raw_stream
        return raw_stream # made on 8/3
    
class TwoRawEvent(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int):
        # E.g., 50ms, 1M events
        #       channels = num_bins = 50
        #       total 50-ch -> {50ms, 1M events} / 50 -> (1ms, 20K events) per 1-ch
        self.raw_stream = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.ls_raw_stream = torch.zeros((1, height, width), dtype=torch.float, requires_grad=False)
        self.dt = 1.0 / channels        # accumulate events within dt in one frame
        
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1
        
        C, H, W = self.raw_stream.shape
        # with torch.no_grad():
        #     self.raw_stream = self.raw_stream.to(pol.device)
        #     raw_stream = self.raw_stream.clone()
            
        #     pix_x = x.int()
        #     pix_y = y.int()
        #     time /= self.dt
        #     time  = time.int()
        #     time[time==time[-1]] = time[-1] - 1
        #     value = 2*pol-1
            
        #     mask = (pix_x < W) & (pix_x >= 0) & (pix_y < H) & (pix_y >= 0) & (time < C) & (time >= 0)
            
        #     index = H * W * time.long() + \
        #             W * pix_y.long() + \
        #             pix_x.long()
        #     raw_stream.put_(index[mask], value[mask], accumulate=True)
        # return raw_stream
        
        with torch.no_grad():
            self.raw_stream = self.raw_stream.to(pol.device)
            raw_stream = self.raw_stream.clone()
            
            pix_x = x.int()
            pix_y = y.int()
            # time /= self.dt
            # time  = time.int()
            # time[time==time[-1]] = time[-1] - 1
            # value = (2*pol-1).to(torch.float32)
            
            # mask = (pix_x < W) & (pix_x >= 0) & (pix_y < H) & (pix_y >= 0) & (time < C) & (time >= 0)
            
            # index = H * W * time + \
            #         W * pix_y + \
            #         pix_x
            # index = index.long()
            # raw_stream.put_(index[mask], value[mask], accumulate=True)
        # return raw_stream
        
            self.ls_raw_stream = self.ls_raw_stream.to(pol.device)
            ls_raw_stream = self.ls_raw_stream.clone()
            
            time_org = time / self.dt
            # time /= self.dt
            time_org = time_org.int()
            time  = time.int()
            time[time==time[-1]] = time[-1] - 1
            time_org[time_org==time_org[-1]] = time_org[-1] - 1
            
            value = (2*pol-1).to(torch.float32)
            
            mask = (pix_x < W) & (pix_x >= 0) & (pix_y < H) & (pix_y >= 0) & (time < C) & (time >= 0)
            mask_org = (pix_x < W) & (pix_x >= 0) & (pix_y < H) & (pix_y >= 0) & (time_org < C) & (time_org >= 0)
            
            index = H * W * time.long() + \
                    W * pix_y.long() + \
                    pix_x.long()
            
            index_org = H * W * time_org.long() + \
                    W * pix_y.long() + \
                    pix_x.long()
                    
            # index = index.long()
            # index_org = index_org.long()
            raw_stream.put_(index_org[mask_org], value[mask_org], accumulate=True)
            ls_raw_stream.put_(index[mask], value[mask], accumulate=True)
        
        # return raw_stream, ls_raw_stream
        return raw_stream # made on 8/3