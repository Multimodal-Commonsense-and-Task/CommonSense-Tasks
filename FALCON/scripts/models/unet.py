import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups, bias=False):
        super(UpSampleBlock, self).__init__()

        if groups is None:
            groups = in_channels
            
        self.block1 = ConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=groups, bias=bias)
        self.block2 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
        
        # self.block1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
    
    def forward(self, x):
        return self.block2(self.block1(x))
        # return self.block1(x)
    
class SpatialUNet(nn.Module):
    
    def __init__(self, in_channels: int = 25, concat_method: str = 'add'):
        super(SpatialUNet, self).__init__()
        allowed_concat_methods = ['concat', 'add']
        NOT_IMPLEMENTED_ERR_STR = f"'concat_method' argument passed needed to be one of {allowed_concat_methods}."
        
        if (concat_method not in allowed_concat_methods):
            raise NotImplementedError(NOT_IMPLEMENTED_ERR_STR)
        
        self.concat_method = concat_method
        
        first_layer_out_channels = 0
        
        if in_channels <= 10:
            first_layer_out_channels = 24
        elif in_channels <= 25:
            first_layer_out_channels = 32
        else:
            first_layer_out_channels = 64
        
        enc_in_ch = [first_layer_out_channels, 2 * first_layer_out_channels]
        enc_out_ch = [2 * first_layer_out_channels, 4 * first_layer_out_channels]
        
        dec_in_ch = [8 * first_layer_out_channels, 4 * first_layer_out_channels]
        dec_out_ch = [4 * first_layer_out_channels, 2 * first_layer_out_channels]
        
        self.zeroth_layer = ConvBlock(in_channels=in_channels, out_channels=first_layer_out_channels, kernel_size=3, 
                                      stride=1, padding=0, groups=1, bias=False)
        
        # Encoder Part
        self.conv1 = ConvBlock(in_channels=enc_in_ch[0], out_channels=enc_out_ch[0], kernel_size=3, stride=2, padding=0, bias=False)
        self.conv2 = ConvBlock(in_channels=enc_in_ch[1], out_channels=enc_out_ch[1], kernel_size=3, stride=2, padding=0, bias=False)
        
        # Bottleneck
        self.res1 = nn.Conv2d(in_channels=enc_out_ch[-1], out_channels=dec_in_ch[0], groups=1, kernel_size=3, stride=1, bias=False)
        
        # Decoder Part
        self.deconv2 = UpSampleBlock(in_channels=dec_in_ch[0], out_channels=dec_out_ch[0], kernel_size=3, groups=None, bias=False)
        
        if concat_method == 'add':
            self.deconv1 = UpSampleBlock(in_channels=dec_in_ch[1], out_channels=dec_out_ch[1], kernel_size=3, groups=None, bias=False)
            self.final_upsample = UpSampleBlock(in_channels=dec_out_ch[1], out_channels=1, kernel_size=1, groups=None, bias=False)
            
        elif concat_method == 'concat':
            self.deconv1 = UpSampleBlock(in_channels=dec_out_ch[0] + enc_out_ch[-1], out_channels=dec_out_ch[1], kernel_size=3, groups=None, bias=False)
            self.final_upsample = UpSampleBlock(in_channels=dec_out_ch[-1] + enc_out_ch[0], out_channels=1, kernel_size=1, groups=None, bias=False)
            
        
    def forward(self, x):
        # # Comment out for noforloop experiments:
        # x = rearrange(x, 't n h w -> n t h w')
        # print("x:", x.shape)
        
        zeroth_output = self.zeroth_layer(x)
        # print("zeroth_output:", zeroth_output.shape)
        
        enc1 = self.conv1(zeroth_output)
        # print("enc1:", enc1.shape)
        
        enc2 = self.conv2(enc1)
        # print("enc2:", enc2.shape)
        
        res = self.res1(enc2)
        # print("residual:", residual.shape)
        
        upsize2 = self.get_upsample_size_for_embedding((enc2.size(2), enc2.size(3)), kernel_size=3, padding=0, stride=1)
        dec2 = self.deconv2(F.interpolate(res, size=upsize2, mode='bilinear'))
        # print("dec2:", dec2.shape)

        if self.concat_method == 'add':
            dec2 = dec2 + enc2
        elif self.concat_method == 'concat':
            dec2 = torch.cat((dec2, enc2), dim=1)
            
        # print("dec2 after residual:", dec2.shape)
        
        upsize1 = self.get_upsample_size_for_embedding((enc1.size(2), enc1.size(3)), kernel_size=3, padding=0, stride=1)
        dec1 = self.deconv1(F.interpolate(dec2, size=upsize1, mode='bilinear'))
        # print("dec1:", dec1.shape)
        
        if self.concat_method == 'add':
            dec1 = dec1 + enc1
        elif self.concat_method == 'concat':
            dec1 = torch.cat((dec1, enc1), dim=1)
        
        # print("dec1 after residual:", dec1.shape)
        
        final_upsize = self.get_upsample_size_for_embedding((x.size(2), x.size(3)), kernel_size=1, padding=0, stride=1)
        output = self.final_upsample(F.interpolate(dec1, size=final_upsize, mode='bilinear'))        
        # print("output:", output.shape)
        # exit()
        return output
    
    def get_upsample_size_for_embedding(self, prev_size, kernel_size: int, padding: int, stride:int):
        prev_h, prev_w = prev_size
        
        new_h = ((prev_h - stride) * stride) + kernel_size - (2 * padding)
        new_w = ((prev_w - stride) * stride) + kernel_size - (2 * padding)
        
        return new_h, new_w
