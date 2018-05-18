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
parser.add_argument('--batch-size', type=int,   default=32,       help='input batch size for training (default: 5)')
parser.add_argument('--model-name', type=str,   default='GAN',   help='model-name (default: GAN)')
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
        
## 2-2
g_loss = pd.read_csv('CSV/GAN_gloss.csv')
d_loss = pd.read_csv('CSV/GAN_dloss.csv')
plt.figure(num=1, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1,2,1)
plt.plot(g_loss['Step'], g_loss['Value'], 'b', alpha=0.6, label='g_loss')
plt.plot(d_loss['Step'], d_loss['Value'], 'r', alpha=0.6, label='d_loss')
plt.grid(True) 
plt.xlabel('Steps')
plt.ylabel('discriminator loss')
plt.legend(loc=0)

real_acc = pd.read_csv('CSV/GAN_real_im_scores.csv')
fake_acc = pd.read_csv('CSV/GAN_fake_im_scores.csv')
gene_acc = pd.read_csv('CSV/GAN_generator_scores.csv')
plt.subplot(1,2,2)
plt.plot(real_acc['Step'], real_acc['Value'], 'b', alpha=0.6, label='real image')
plt.plot(fake_acc['Step'], 1-fake_acc['Value'], 'r', alpha=0.6, label='fake image')
#plt.plot(gene_acc['Step'], gene_acc['Value'], 'g', alpha=0.6, label='generator')
plt.grid(True) 
plt.xlabel('Steps')
plt.ylabel('discriminator accuracy')
plt.legend(loc=0)

plt.savefig(os.path.join(args.dir,'fig2_2.jpg'), dpi=600, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='jpg',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
print('save fig2_2.jpg {}'.format(args.dir))

## 2-3
pkl_name = model_name+'_model.pkl'
if not (os.path.exists(pkl_name)):
    file_id = '1-HsY_K8xf0eevok1mmepw8DfXFd6F4zr'
    destination = pkl_name
    print('start downloading model')
    download_file_from_google_drive(file_id, destination)
    print('finished downloading model')
        
G = generator(num_z).cuda()
G.load_state_dict(torch.load(pkl_name))
        
z = Variable(torch.FloatTensor(batch_size, num_z, 1, 1).normal_()).cuda()
fake_img = G(z)
vutils.save_image(fake_img.data, os.path.join(args.dir,'fig2_3.jpg'), nrow=8, normalize=True, scale_each=True)
print('save fig2_3.jpg to {}'.format(args.dir))
