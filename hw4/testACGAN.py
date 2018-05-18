from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as UtiData
import torch.optim as optim
from   torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from   tensorboardX import SummaryWriter
import cv2
import numpy as np
import os, csv, argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from   utils import *
from   dataset import *
from   model import *
from   loss import *

#####################################################################
## Hyper Parameters 
#####################################################################
parser = argparse.ArgumentParser(description='DLCV HW4')
parser.add_argument('--batch-size', type=int,   default=10,       help='input batch size for training (default: 5)')
parser.add_argument('--model-name', type=str,   default='ACGAN',   help='model-name (default: GAN)')
parser.add_argument('--manualSeed', type=int,   default=1000,    help='manualSeed')
parser.add_argument('--dir', type=str, default='Image', metavar='D', help='dir')
args = parser.parse_args()


batch_size       = args.batch_size
model_name       = args.model_name
num_z            = 100

if not os.path.exists(args.dir):
    os.makedirs(args.dir)

# numpy seed
np.random.seed(args.manualSeed)
# CPU seed
torch.manual_seed(args.manualSeed)
# GPU seed
torch.cuda.manual_seed_all(args.manualSeed)


## 3-2
g_loss = pd.read_csv('ACGAN_gloss.csv')
d_loss = pd.read_csv('ACGAN_dloss.csv')
plt.figure(num=1, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1,2,1)
plt.plot(g_loss['Step'], g_loss['Value'], 'b', alpha=0.6, label='g_loss')
plt.plot(d_loss['Step'], d_loss['Value'], 'r', alpha=0.6, label='d_loss')
plt.grid(True) 
plt.xlabel('Steps')
plt.ylabel('discriminator loss')
plt.legend(loc=0)

real_acc = pd.read_csv('ACGAN_real_acc.csv')
fake_acc = pd.read_csv('ACGAN_fake_acc.csv')
plt.subplot(1,2,2)
plt.plot(real_acc['Step'], real_acc['Value'], 'b', alpha=0.6, label='real images')
plt.plot(fake_acc['Step'], fake_acc['Value'], 'r', alpha=0.6, label='fake images')
plt.grid(True) 
plt.xlabel('Steps')
plt.ylabel('classify accuracy')
plt.legend(loc=0)

plt.savefig(os.path.join(args.dir,'fig3_2.jpg'), dpi=600, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='jpg',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
print('save fig3_2.jpg to {}'.format(args.dir))

## 3-3
G = netG_ACGAN(num_z).cuda()
pkl_name = model_name+'_model.pkl'
if not (os.path.exists(pkl_name)):
    file_id = '1cES0xPttAo-djuSbuI5J4OFOG69RnBAe'
    destination = pkl_name
    print('start downloading model')
    download_file_from_google_drive(file_id, destination)
    print('finished downloading model')
    
G.eval()
G.load_state_dict(torch.load(pkl_name))

z = Variable(torch.FloatTensor(batch_size, num_z, 1, 1).normal_()).cuda()
yes_lbl = Variable(torch.FloatTensor(np.ones(batch_size))).cuda()
no_lbl = Variable(torch.FloatTensor(np.zeros(batch_size))).cuda()
yes_fake_img = G(z, yes_lbl)
no_fake_img = G(z, no_lbl)
output = torch.cat([yes_fake_img.data, no_fake_img.data])
vutils.save_image(output, os.path.join(args.dir,'fig3_3.jpg'), nrow=10, normalize=True, scale_each=True)
print('save fig3_3.jpg to {}'.format(args.dir))
