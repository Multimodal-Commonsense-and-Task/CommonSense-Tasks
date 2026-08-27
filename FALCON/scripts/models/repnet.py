import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size, 
                      padding=padding, 
                      bias=bias), 
            nn.BatchNorm2d(out_channels)
        ])
        
    def forward(self, x):
        return self.block(x)
      
      
class RepNet(nn.Module):
    def __init__(self, in_channels=5, base_channels=32, attention_method='soft', act_func='snn'):
        super(RepNet, self).__init__()
        self.attention_method = attention_method
        self.act_func = act_func
        
        if act_func == 'snn':
            # Spiking Neuron and simulation params
            # Surrogate gradient function
            spike_grad = surrogate.atan(alpha=2.0)
            # Decay rate beta
            beta = 0.8
            
            # threshold1 = 0.1
            
            kernel = 1
            padding = 0
            
            self.conv1 = ConvBlock(in_channels=in_channels, out_channels=base_channels, kernel_size=kernel, padding=padding)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True, learn_threshold=True)
            self.conv2 = ConvBlock(in_channels=base_channels, out_channels=1, kernel_size=kernel, padding=padding)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True, learn_threshold=True)
        
        elif act_func == 'snn_synaptic':
            # Spiking Neuron and simulation params
            # Surrogate gradient function
            spike_grad = surrogate.atan(alpha=2.0)
            # Decay rate beta
            beta = 0.8
            alpha = 0.8
            
            kernel_size = 1
            padding = 0
            
            # [timestep, batch_size, in_channels, base_h, base_w]
            self.conv1 = ConvBlock(in_channels=in_channels, out_channels=base_channels, kernel_size=kernel_size, padding=padding)
            self.lif1 = snn.Synaptic(beta=beta, spike_grad=spike_grad, learn_beta=True, learn_threshold=True, 
                                     alpha=alpha, learn_alpha=True,
                                )
            self.conv2 = ConvBlock(in_channels=base_channels, out_channels=1, kernel_size=kernel_size, padding=padding)
            self.lif2 = snn.Synaptic(beta=beta, spike_grad=spike_grad, learn_beta=True, learn_threshold=True, 
                                   alpha=alpha, learn_alpha=True,
                                )
        else:
            raise ValueError("Improper argument given for activation function.")
        # self.bn = nn.BatchNorm2d(1)
        
        self.running_means = []
        self.count = 0
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor):
        
        if self.act_func == 'snn':

            # Initialize membrane potentials at each layer
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            
            # x: [N, C=num_bins, H, W]
            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            
            output = spk2+mem2
            
        elif self.act_func == 'snn_synaptic':
            # Initialize membrane potentials at each layer
            syn1, mem1 = self.lif1.init_synaptic()
            syn2, mem2 = self.lif2.init_synaptic()
            
            # x: [N, C=num_bins, H, W]
            spk1, syn1, mem1 = self.lif1(self.conv1(x), syn1, mem1)
            spk2, syn2, mem2 = self.lif2(self.conv2(spk1), syn2, mem2)
            
            output = spk2+syn2+mem2
        return output