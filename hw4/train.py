import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as UtiData
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from tensorboardX import SummaryWriter

import cv2
import numpy as np
import os, csv, argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from utils import *
from dataset import *
from model import *
from loss import *

#####################################################################
## Hyper Parameters 
#####################################################################
parser = argparse.ArgumentParser(description='DLCV HW4')
parser.add_argument('--batch-size', type=int, default=5, metavar='B',
                help='input batch size for training (default: 5)')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                help='number of epochs to training (default: 20)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                help='learning rate (default: 1e-3)')
parser.add_argument('--model-id', type=int, default=0, metavar='M',
                help='downscale (default: 0)')
parser.add_argument('--parallel', type=bool, default=False, metavar='P',
                help='parallel (default: False)')
parser.add_argument('--resume', type=bool, default=0, metavar='R',
                help='resume (default: 0)')
parser.add_argument('--model-name', type=str, default='VAE', metavar='N',
                help='model-name (default: VAE)')
parser.add_argument('--kl', type=float, default=1e-5, metavar='KL',
                help='kl loss (default: 1e-5)')
parser.add_argument('--comment', type=str, default='train', metavar='C',
                help='comment')
args = parser.parse_args()


lr               = args.lr
batch_size       = args.batch_size
MAX_EPOCHS       = args.epochs
model_id         = args.model_id
model_name       = args.model_name
parallel         = args.parallel
resume           = args.resume
momentum         = 0.95
weight_decay     = 5e-4
save_iterations  = 1000
trainfilepath    = 'hw4_data/train'
validfilepath    = 'hw4_data/test'
lambda_kl        = args.kl
comment          = args.comment

# CPU seed
torch.manual_seed(42)
# GPU seed
torch.cuda.manual_seed_all(42)

train_loader = get_data_loader(trainfilepath, batch_size)
valid_loader = get_data_loader(validfilepath, batch_size)

loss_dir  = '{}{:02d}_Loss'.format(model_name, model_id)
model_dir = '{}{:02d}_Model'.format(model_name, model_id)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

exec('net = %s()' % (model_name))

iterations = 0
curr_ep = 0
if resume:
    iterations = net.resume(model_dir)
    curr_ep = len(os.listdir(loss_dir))- 1

if torch.cuda.is_available(): 
    print(args.parallel)
    if parallel:
        print('Train parallel')
        net = torch.nn.DataParallel(net)
    net.cuda()

net.train()
writer = SummaryWriter(comment=comment)

#criterion = nn.MSELoss()
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer = optim.Adam(net.parameters(), lr=lr)

#####################################################################
## trainning process 
#####################################################################
train_loss_total = []
MSE_loss_total = []
KLD_total = []
for ep in range(curr_ep, MAX_EPOCHS):
    ## training
    train_loss    = 0
    train_MSE_loss = 0
    train_KLD      = 0
    for it_1, images in enumerate(train_loader):
        images = Variable(images.cuda())
        
        optimizer.zero_grad()
        output, mu, logvar = net(images)
        
        #loss = criterion(output, images)
        loss, MSE_loss_cu, KLD_cu = loss_function(output, images, mu, logvar, lambda_kl)
        loss.backward()
        optimizer.step()
        
        per_loss = loss.data[0]
        MSE_loss = MSE_loss_cu.data[0]
        KLD      = KLD_cu.data[0]
        train_loss     += per_loss
        train_MSE_loss += MSE_loss
        train_KLD      += KLD
        if it_1 % 1 == 0:
            print('epoch: {:3}, it: {:4}, train loss: {:10.4f}, MSE loss: {:10.4f}'.format(ep, it_1, per_loss, MSE_loss))
        train_loss_total.append(per_loss)
        MSE_loss_total.append(MSE_loss)
        KLD_total.append(KLD)
        
        writer.add_scalar('loss/train_loss', per_loss,  iterations+1)
        writer.add_scalar('loss/MSE',        MSE_loss,  iterations+1)
        writer.add_scalar('loss/KLD',        KLD,       iterations+1)
        
        if (iterations+1) % save_iterations == 0:
            if parallel:
                net.module.save(model_dir, iterations)
            else:
                net.save(model_dir, iterations)
            ## save images
            x = vutils.make_grid(output.data, normalize=True, scale_each=True)
            writer.add_image('Image', x, iterations)
            x = vutils.make_grid(images.data, normalize=True, scale_each=True)
            writer.add_image('Origin', x, iterations)

        iterations += 1
        
    print('epoch:%03d, train loss: %2.3f, MSE loss: %2.3f\n' % (ep, train_loss/it_1, train_MSE_loss/it_1))
    
    # write log files
    loss_file_name = os.path.join(loss_dir, 'ep{:04d}_loss.data'.format(ep))
    loss_df = pd.DataFrame({"loss" : train_loss_total, "MSE_loss" : MSE_loss_total})
    loss_df.to_csv(loss_file_name, index=False)
    with open(os.path.join(loss_dir,'log.txt'), 'a') as fwirte:
        fwirte.write('epoch:%03d, train loss: %2.3f, MSE loss: %2.3f\n' % (ep, train_loss/it_1, train_MSE_loss/it_1))
                
    #if (ep+1) % 5 == 0 or ep ==0:
    #    model_file_name = os.path.join(model_dir, 'ep{:04d}_model.pkl'.format(ep))
    #    torch.save(net.state_dict(), model_file_name)
    
# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()