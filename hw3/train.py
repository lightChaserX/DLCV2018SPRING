import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as UtiData
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import cv2
import numpy as np

import os, csv, argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from fcn32s import VGG16_FCN32s
from utils import *
from mean_iou_evaluate import *
from dataset import *
from criterion import *


## Hyper Parameters 
parser = argparse.ArgumentParser(description='DLCV HW3')
parser.add_argument('--batch-size', type=int, default=5, metavar='B',
                help='input batch size for training (default: 5)')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                help='number of epochs to training (default: 20)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                help='learning rate (default: 1e-3)')
args = parser.parse_args()

lr         = args.lr
batch_size = args.batch_size
MAX_EPOCHS = args.epochs
n_class    = 7
model_save_name  = 'VGG16FCNs_model.pkl'

trainfilepath = 'dataset/train'
validfilepath = 'dataset/validation'
train_loader = get_data_loader(trainfilepath, batch_size)
valid_loader = get_data_loader(validfilepath, batch_size)

VGG16_net = torchvision.models.vgg16(pretrained=True).cuda()
net = VGG16_FCN32s(n_class).cuda()
net.copy_vgg16(VGG16_net)
del VGG16_net
#net = torch.nn.DataParallel(net).cuda()

criterion = CrossEntropyLoss2d().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

################################################################
for ep in range(0, MAX_EPOCHS):
    epoch_loss = 0
    for it, (images, labels) in enumerate(train_loader):
        assert images.size(0) == labels.size(0)
        images = Variable(images.cuda()).cuda()
        labels = Variable(labels.cuda()).cuda()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        print('epoch: {}, it: {}, loss: {}'.format(ep, it, loss.data[0]))
        epoch_loss += loss.data[0]
    print('epoch:{} avr loss: {}'.format(ep, epoch_loss/it))
    torch.save(net.state_dict(), model_save_name)