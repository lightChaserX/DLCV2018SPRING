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
from sklearn.manifold import TSNE

from utils import *
from dataset import *
from model import *
from loss import *

#####################################################################
## Hyper Parameters 
#####################################################################
parser = argparse.ArgumentParser(description='DLCV HW4')
parser.add_argument('--batch-size', type=int, default=5, metavar='B', help='input batch size for training (default: 5)')
parser.add_argument('--model-name', type=str, default='VAE', metavar='N', help='model-name (default: VAE)')
parser.add_argument('--dir', type=str, default='Image', metavar='D', help='dir')
parser.add_argument('--data-path', type=str, default='hw4_data', metavar='P', help='data path')
args = parser.parse_args()


batch_size       = args.batch_size
model_name       = args.model_name
validfilepath    = os.path.join(args.data_path, 'test')

# np seed
np.random.seed(10000)
# CPU seed
torch.manual_seed(10000)
# GPU seed
torch.cuda.manual_seed_all(10000)

if not os.path.exists(args.dir):
    os.makedirs(args.dir)

exec('net = %s()' % (model_name))
pkl_name = model_name+'_model.pkl'

if not (os.path.exists(pkl_name)):
    file_id = '1w9e_wcejeCKfOkIam0VP-RIsO4Cyfr_0'
    destination = pkl_name
    print('start downloading model')
    download_file_from_google_drive(file_id, destination)
    print('finished downloading model')
        
                                          
net.load_state_dict(torch.load(pkl_name))

if torch.cuda.is_available(): 
    net.cuda()

net.eval()

## 1
images = torch.FloatTensor(10,3,64,64)
for i in range(10):
    images[i,:,:,:] = dataset_image(validfilepath).__getitem__(i)
    
images = Variable(images.cuda())
output, mu, logvar = net(images, test=True)

## save images

## 0
KLD_loss = pd.read_csv('VAE-loss_KLD.csv')
plt.figure(num=1, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1,2,1)
plt.plot(KLD_loss['Step'], KLD_loss['Value'], 'b', alpha=0.6)
plt.grid(True) 
plt.xlabel('Steps')
plt.ylabel('KLD')

plt.subplot(1,2,2)
MSE_loss = pd.read_csv('VAE-loss_MSE.csv')
plt.plot(KLD_loss['Step'], MSE_loss['Value'], 'r')
plt.grid(True) 
plt.xlabel('Steps')
plt.ylabel('MSE')
plt.savefig(os.path.join(args.dir,'fig1_2.jpg'), dpi=600, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='jpg',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
print('save fig1_2.jpg to {}'.format(args.dir))

## 1
img_now = torch.cat([images.data, output.data]) 
vutils.save_image(img_now, os.path.join(args.dir,'fig1_3.jpg'), nrow=10, normalize=True, scale_each=True)
print('save fig1_3.jpg to {}'.format(args.dir))

## 2
noise = Variable(torch.cuda.FloatTensor(32, 512).normal_())
output,_,_ = net(noise, test=True, generate=True)
vutils.save_image(output.data, os.path.join(args.dir,'fig1_4.jpg'), nrow=8, normalize=True, scale_each=True)
print('save fig1_4.jpg to {}'.format(args.dir))

## 3
valid_loader = get_data_loader(validfilepath, batch_size, lbl=True)
for it_1, (images, lbl) in enumerate(valid_loader, 0):
    images = Variable(images.cuda())
    n_lbl = 1 - lbl.numpy()
    z = net(images, test=True, generate=False, latent=True)
    Z_embedded = TSNE(n_components=2).fit_transform(z.data.cpu().numpy())
    
    plt.figure(4)
    plt.plot(Z_embedded[lbl.numpy().astype(np.bool),0], Z_embedded[lbl.numpy().astype(np.bool),1],'b.', label='Bangs')
    plt.plot(Z_embedded[n_lbl.astype(np.bool),0], Z_embedded[n_lbl.astype(np.bool),1],'r.', label='no Bangs')
    plt.legend(loc=0)
    plt.savefig(os.path.join(args.dir,'fig1_5.jpg'), dpi=600, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='jpg',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
    print('save fig1_5.jpg to {}'.format(args.dir))
    break

mse_all = 0
reconstruction_function = nn.MSELoss()
valid_loader = get_data_loader(validfilepath, 1)
for it_1, images in enumerate(valid_loader):
    images = Variable(images.cuda())
    out, _, _ = net(images, test=True)
    MSE = reconstruction_function(out, images)
    mse_all += MSE.data.cpu().numpy()
    
print('MSE of test images are {}'.format(mse_all/it_1))