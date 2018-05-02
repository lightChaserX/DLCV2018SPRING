import os
import torch
from torch import nn
from torchvision import models
from utils import *

'''
TODO: Define the model of VGG16-FCN8s
'''
class FCN8s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        ## conv1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1   = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        ## conv2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2   = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        ## conv3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3   = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        ## conv4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4   = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        ## conv5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5   = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # fc6
        self.fc6   = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7   = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr    = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()
    
        self.features = [
                self.conv1_1, self.relu1_1,
                self.conv1_2, self.relu1_2,
                self.pool1,
                self.conv2_1, self.relu2_1,
                self.conv2_2, self.relu2_2,
                self.pool2,
                self.conv3_1, self.relu3_1,
                self.conv3_2, self.relu3_2,
                self.conv3_3, self.relu3_3,
                self.pool3,
                self.conv4_1, self.relu4_1,
                self.conv4_2, self.relu4_2,
                self.conv4_3, self.relu4_3,
                self.pool4,
                self.conv5_1, self.relu5_1,
                self.conv5_2, self.relu5_2,
                self.conv5_3, self.relu5_3,
                self.pool5,
            ]
    
    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                #layer.weight.data.normal_(0, 0.001)
                layer.weight.data.zero_()
                if layer.bias is not None:
                    layer.bias.data.zero_()
            if isinstance(layer, nn.ConvTranspose2d):
                assert layer.kernel_size[0] == layer.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    layer.in_channels, layer.out_channels, layer.kernel_size[0])
                layer.weight.data.copy_(initial_weight) 

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        
        ## 2x conv7
        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h
        
        ## 2x conv7 + pool4
        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h
        h = upscore2 + score_pool4c
        
        ## 2x (2x conv7 + pool4)
        h = self.upscore_pool4(h)
        upscore_pool4 = h
        
        ## 2x (2x conv7 + pool4) +pool3
        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h
        h = upscore_pool4 + score_pool3c
        
        ## 8x
        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self, vgg16):
        for layer1, layer2 in zip(vgg16.features, self.features):
            if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                assert layer1.weight.size() == layer2.weight.size()
                assert layer1.bias.size() == layer2.bias.size()
                layer2.weight.data = layer1.weight.data
                layer2.bias.data = layer1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
            
    def copy_params_from_FCN32s(self, FCN32s):
        for name, layer1 in FCN32s.named_children():
            try:
                layer2 = getattr(self, name)
                layer2.weight
            except Exception:
                continue
            layer2.weight.data.copy_(layer1.weight.data)
            if layer1.bias is not None:
                assert layer1.bias.size() == layer2.bias.size()
                layer2.bias.data.copy_(layer1.bias.data)