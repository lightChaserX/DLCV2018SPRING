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

from fcn32s import *
from fcn8s import *
from utils import *
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
parser.add_argument('--down-scale', type=int, default=2, metavar='D',
                help='downscale (default: 2)')
parser.add_argument('--model-id', type=int, default=0, metavar='M',
                help='downscale (default: 0)')
parser.add_argument('--parallel', type=bool, default=False, metavar='P',
                help='parallel (default: False)')
parser.add_argument('--resume', type=bool, default=0, metavar='R',
                help='resume (default: 0)')
parser.add_argument('--model-name', type=str, default='FCN32s', metavar='N',
                help='model-name (default: FCN32s)')
args = parser.parse_args()

lr         = args.lr
batch_size = args.batch_size
MAX_EPOCHS = args.epochs
n_class    = 7
down_scale = args.down_scale
model_id   = args.model_id
model_name = args.model_name
parallel   = args.parallel
resume     = args.resume

trainfilepath = 'dataset/train'
validfilepath = 'dataset/validation'
train_loader = get_data_loader(trainfilepath, batch_size, down_scale)
valid_loader = get_data_loader(validfilepath, batch_size=4)

loss_dir = '{}{:02d}_Loss'.format(model_name, model_id)
model_dir = '{}{:02d}_Model'.format(model_name, model_id)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

exec('net = %s(n_class)' % (model_name))

if resume:
    pkl_list = [file for file in os.listdir(model_dir) if file.endswith('.pkl')]
    pkl_list.sort()
    net.load_state_dict(torch.load(os.path.join(model_dir, pkl_list[-1])))
    curr_ep = int(pkl_list[-1][2:6])
    print('resume from epoch {:04d}'.format(curr_ep))
else:
    if model_name == 'FCN8s':
        FCN32s_model = FCN32s(n_class)
        FCN32s_model.load_state_dict(torch.load(os.path.join('FCN32s04_Model', 'ep0079_model.pkl')))
        net.copy_params_from_FCN32s(FCN32s_model)
    else:
        VGG16_net = torchvision.models.vgg16(pretrained=False)
        VGG16_net.load_state_dict(torch.load('vgg16-new.pth'))
        net.copy_params_from_vgg16(VGG16_net)
    curr_ep = 0

net.cuda()
if parallel:
    print('Train parallel')
    net = torch.nn.DataParallel(net).cuda()

#lbl_weight = Variable(torch.FloatTensor([0.29873678, 0.0560519 , 0.38079318, 0.2786428 , 1.,
#       0.40306586, 0.2]))
lbl_weight = Variable(torch.FloatTensor([1, 0.75, 1.25, 1, 1, 1, 1]))
#criterion = CrossEntropyLoss2d(weight=lbl_weight).cuda()
criterion = CrossEntropyLoss2d().cuda()
#optimizer = optim.Adam(net.parameters(), lr=lr)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.95, weight_decay=5e-4)

################################################################
train_loss_total = []
train_accu_total = []
test_loss_total = []
test_accu_total = []
for ep in range(curr_ep, MAX_EPOCHS):
    ## training
    train_loss = 0
    train_acc = 0
    for it_1, (images, labels) in enumerate(train_loader):
        assert images.size(0) == labels.size(0)
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        lbl_pred = output.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = labels.data.cpu().numpy()
        acc = np.sum(lbl_pred.flatten()==lbl_true.flatten())/len(lbl_true.flatten())
        #acc = torch_mean_iou(lbl_pred, lbl_true)
        
        per_loss = loss.data[0]
        train_loss += per_loss
        train_acc += acc
        print('epoch: {:3}, it: {:4}, train loss: {:10.4f}, train acc: {:10.2f}'.format(ep, it_1, per_loss, acc))
        train_loss_total.append(per_loss)
        train_accu_total.append(acc)
     
    ## validation
    test_loss = 0
    test_acc = 0
    for it_2, (images, labels) in enumerate(valid_loader):
        assert images.size(0) == labels.size(0)
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        
        lbl_pred = output.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = labels.data.cpu().numpy()
        acc = np.sum(lbl_pred.flatten()==lbl_true.flatten())/len(lbl_true.flatten())
        
        per_loss = loss.data[0]
        test_loss += per_loss
        test_acc += acc
        print('epoch: {:3}, it: {:4}, test loss: {:10.4f}, test acc: {:10.2f}'.format(ep, it_2, per_loss, acc))
        test_loss_total.append(per_loss)
        test_accu_total.append(acc)
        
    print('epoch:%03d, train loss: %2.2f, train acc: %2.2f, test loss: %2.2f, test acc: %2.2f' % (ep, train_loss/it_1, train_acc/it_1, test_loss/it_2, test_acc/it_2))
    # write files
    loss_file_name = os.path.join(loss_dir, 'ep{:04d}_loss.data'.format(ep))
    loss_df = pd.DataFrame({"loss" : train_loss_total, "acc" : train_accu_total})
    loss_df.to_csv(loss_file_name, index=False)
    
    val_loss_file_name = os.path.join(loss_dir, 'ep{:04d}_valid_loss.data'.format(ep))
    val_loss_df = pd.DataFrame({"test_loss" : test_loss_total, "test_acc" : test_accu_total})
    val_loss_df.to_csv(val_loss_file_name, index=False)
    
    with open(os.path.join(loss_dir,'log.txt'), 'a') as fwirte:
        fwirte.write('epoch:%03d, train loss: %2.2f, train acc: %2.2f, test loss: %2.2f, test acc: %2.2f\n' % (ep, train_loss/it_1, train_acc/it_1, test_loss/it_2, test_acc/it_2))
                
    if (ep+1) % 5 == 0 or ep ==0:
        model_file_name = os.path.join(model_dir, 'ep{:04d}_model.pkl'.format(ep))
        torch.save(net.state_dict(), model_file_name)