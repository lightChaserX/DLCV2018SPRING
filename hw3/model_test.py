import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_upsampling_weight

'''
TODO: Define the model of VGG16-FCN32s
'''
class FCN32s(nn.Module):
# VGG16-FCN32s model
    def __init__(self, n_class=7):
        super(FCN32s, self).__init__()
        self.block1_conv1 = nn.Conv2d(  3,  64, kernel_size=3, padding=100)
        self.block1_relu1 = nn.ReLU(inplace=True)
        self.block1_conv2 = nn.Conv2d( 64,  64, kernel_size=3, padding=1)
        self.block1_relu2 = nn.ReLU(inplace=True)
        self.block1_pool  = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.block2_conv1 = nn.Conv2d( 64, 128, kernel_size=3, padding=1)
        self.block2_relu1 = nn.ReLU(inplace=True)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block2_relu2 = nn.ReLU(inplace=True)
        self.block2_pool  = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block3_relu1 = nn.ReLU(inplace=True)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_relu2 = nn.ReLU(inplace=True)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_relu3 = nn.ReLU(inplace=True)
        self.block3_pool  = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.block4_relu1 = nn.ReLU(inplace=True)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_relu2 = nn.ReLU(inplace=True)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_relu3 = nn.ReLU(inplace=True)
        self.block4_pool  = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_relu1 = nn.ReLU(inplace=True)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_relu2 = nn.ReLU(inplace=True)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_relu3 = nn.ReLU(inplace=True)
        self.block5_pool  = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv6 = nn.Conv2d( 512, 4096, kernel_size=7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        
        self.score = nn.Conv2d(4096, n_class, 1)
        
        self.upsampling = nn.ConvTranspose2d(n_class, n_class, kernel_size=64, stride=32, bias=False)
        
        self._initialize_weights()
        
        # features
        self.features = [
        self.block1_conv1, self.block1_relu1,
        self.block1_conv2, self.block1_relu2,
        self.block1_pool,
        self.block2_conv1, self.block2_relu1,
        self.block2_conv2, self.block2_relu2,
        self.block2_pool,
        self.block3_conv1, self.block3_relu1,
        self.block3_conv2, self.block3_relu2,
        self.block3_conv3, self.block3_relu3,
        self.block3_pool,
        self.block4_conv1, self.block4_relu1,
        self.block4_conv2, self.block4_relu2,
        self.block4_conv3, self.block4_relu3,
        self.block4_pool,
        self.block5_relu1, self.block5_relu1,
        self.block5_relu2, self.block5_relu2,
        self.block5_relu3, self.block5_relu3,
        self.block5_pool,
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
        y = self.block1_relu1(self.block1_conv1(x))
        y = self.block1_relu2(self.block1_conv2(y))
        y = self.block1_pool(y)
        
        y = self.block2_relu1(self.block2_conv1(y))
        y = self.block2_relu2(self.block2_conv2(y))
        y = self.block2_pool(y)
        
        y = self.block3_relu1(self.block3_conv1(y))
        y = self.block3_relu2(self.block3_conv2(y))
        y = self.block3_relu3(self.block3_conv3(y))
        y = self.block3_pool(y)
        
        y = self.block4_relu1(self.block4_conv1(y))
        y = self.block4_relu2(self.block4_conv2(y))
        y = self.block4_relu3(self.block4_conv3(y))
        y = self.block4_pool(y)
        
        y = self.block5_relu1(self.block5_conv1(y))
        y = self.block5_relu2(self.block5_conv2(y))
        y = self.block5_relu3(self.block5_conv3(y))
        y = self.block5_pool(y)
        
        y = self.relu6(self.conv6(y))
        y = self.drop6(y)
        y = self.relu7(self.conv7(y))
        y = self.drop7(y)
        score = self.score(y)
        upsampling = self.upsampling(score)
        out = upsampling[:, :, 19: (19 + x.size(2)), 19: (19 + x.size(3))].contiguous()
        #out = F.upsample_bilinear(score, x.size()[2:])    
        return out
    
    def copy_params_from_vgg16(self, vgg16):
        for layer1, layer2 in zip(vgg16.features, self.features):
            if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                assert layer1.weight.size() == layer2.weight.size()
                assert layer1.bias.size() == layer2.bias.size()
                layer2.weight.data = layer1.weight.data
                layer2.bias.data = layer1.bias.data
        for i, name in zip([0, 3], ['conv6', 'conv7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())