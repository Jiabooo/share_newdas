import torch.nn as nn
import torch
from torchvision import models
from utils import save_net, load_net
import torch.nn.functional as F
import cv2

import json
    
    
class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.mask_feat = [512, 512, 512, 256, 128, 64]
        
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=2)
        self.mask = make_layers(self.mask_feat, in_channels=512, dilation=2)

        self.conv4 = nn.Conv2d(64, 1, kernel_size=1)
        self.act = nn.Sigmoid()
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            self._initialize_weights()
            with open("settings.json", 'r') as f:
                settings_dict = json.load(f)
            premodel = settings_dict['premodel_dir']
            # pre = torch.load(home+r"/csrnet_mask/new_mask.tar")
            pre = torch.load(premodel)
            pre = pre['state_dict']
            
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(pre.items())[i][1].data[:]
            for i in range(len(self.backend.state_dict().items())):
                j = i+len(self.frontend.state_dict().items())
                list(self.backend.state_dict().items())[i][1].data[:] = list(pre.items())[j][1].data[:]

                
            for i in range(len(self.mask.state_dict().items())):
                j = i+len(self.frontend.state_dict().items())+len(self.backend.state_dict().items())
                list(self.mask.state_dict().items())[i][1].data[:] = list(pre.items())[j][1].data[:]
            for i in range(len(self.conv4.state_dict().items())):
                j = i+len(self.frontend.state_dict().items())+len(self.backend.state_dict().items())+len(self.mask.state_dict().items())
                list(self.conv4.state_dict().items())[i][1].data[:] = list(pre.items())[j][1].data[:]

            for i in range(len(self.output_layer.state_dict().items())):
                j = i+len(self.frontend.state_dict().items())+len(self.backend.state_dict().items())+len(self.mask.state_dict().items())+len(self.conv4.state_dict().items())
                list(self.output_layer.state_dict().items())[i][1].data[:] = list(pre.items())[j][1].data[:]

            # for i in range(len(self.output_layer.state_dict().items())):
            #     j = i + len(self.frontend.state_dict().items()) + len(self.backend.state_dict().items())
            #     list(self.output_layer.state_dict().items())[i][1].data[:] = list(pre.items())[j][1].data[:]

                
    def forward(self, x,mask):
        x = self.frontend(x)
        x = x * mask
        x1 = x
        
        x1 = self.mask(x1)
        x1 = self.conv4(x1)
        x1 = self.act(x1)
        
        x = self.backend(x)
        x = self.output_layer(x)
        
        return x, x1
        
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=1, k_size=3):
    if dilation==1:
        d_rate = 1
    elif dilation==2:
        d_rate = 2
    else:
        d_rate = 3
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=k_size, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
