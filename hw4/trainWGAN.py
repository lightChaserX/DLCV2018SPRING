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
parser.add_argument('--batch-size', type=int,   default=5,       help='input batch size for training (default: 5)')
parser.add_argument('--epochs',     type=int,   default=20,      help='number of epochs to training (default: 20)')
parser.add_argument('--g_lr',       type=float, default=1e-3,    help='g_lr (default: 1e-3)')
parser.add_argument('--d_lr',       type=float, default=1e-3,    help='d_lr (default: 1e-3)')
parser.add_argument('--beta',       type=float, default=1e-3,    help='beta (default: 1e-3)')
parser.add_argument('--model-id',   type=int,   default=0,       help='downscale (default: 0)')
parser.add_argument('--parallel',   type=bool,  default=False,   help='parallel (default: False)')
parser.add_argument('--resume',     type=bool,  default=0,       help='resume (default: 0)')
parser.add_argument('--model-name', type=str,   default='GAN',   help='model-name (default: GAN)')
parser.add_argument('--save-itr',   type=int,   default=100,     help='save model per x iterations (default: 100)')
parser.add_argument('--comment',    type=str,   default='train', help='comment')
args = parser.parse_args()


g_lr             = args.g_lr
d_lr             = args.d_lr
beta1            = args.beta
batch_size       = args.batch_size
MAX_EPOCHS       = args.epochs
model_id         = args.model_id
model_name       = args.model_name
parallel         = args.parallel
resume           = args.resume
save_iterations  = args.save_itr
comment          = args.comment
trainfilepath    = 'hw4_data/train'
validfilepath    = 'hw4_data/test'
num_z            = 100

# CPU seed
#torch.manual_seed(42)
# GPU seed
#torch.cuda.manual_seed_all(42)

train_loader = get_data_loader(trainfilepath, batch_size)
valid_loader = get_data_loader(validfilepath, batch_size)

loss_dir  = '{}{:02d}_Loss'.format(model_name, model_id)
model_dir = '{}{:02d}_Model'.format(model_name, model_id)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

D = discriminator().cuda()
G = generator(num_z).cuda()
d_optimizer = optim.Adam(D.parameters(), lr=d_lr, betas=(beta1, 0.999))
g_optimizer = optim.Adam(G.parameters(), lr=g_lr, betas=(beta1, 0.999))
criterion = nn.BCELoss()

iterations = 0
curr_ep = 0
writer = SummaryWriter(comment=comment)

one = torch.FloatTensor([1])
mone = one * -1
one = one.cuda()
mone = mone.cuda()
#####################################################################
## trainning process 
#####################################################################
train_loss_total = []
gen_iterations = 0
for ep in range(curr_ep, MAX_EPOCHS):
    ## training
    train_loss = 0
    data_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in D.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update
        
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = 100
        j = 0
        while j < Diters and i < len(train_loader):
            j += 1
        
        for p in D.parameters():
            p.data.clamp_(-0.1, 0.1)
        
        images = data_iter.next()
        i += 1
        
        num_imgs = images.size(0)
        real_img = Variable(images).cuda()
        
        #real_label = Variable(torch.ones(num_imgs)).cuda()
        fake_label = Variable(torch.zeros(num_imgs)).cuda()
        real_label = Variable(torch.FloatTensor(np.random.rand(num_imgs).clip(0.7, 1.0))).cuda()
        
        # ===============train D
        D.zero_grad()
        
        # compute loss of real_img
        real_out = D(real_img)
        real_scores = real_out
        real_out.backward(one)
        d_loss_real = real_out
        
        # compute loss of fake_img
        z = Variable(torch.randn(num_imgs, num_z, 1, 1)).cuda()
        fake_img = G(z)
        fake_out = D(fake_img)
        fake_scores = fake_out
        fake_out.backward(mone)
        d_loss_fake = fake_out
        
        # bp and optimize
        d_loss = d_loss_real - d_loss_fake
        d_optimizer.step()
        
        # ===============train generator
        # compute loss of fake_img
        for p in D.parameters():
            p.requires_grad = False # to avoid computation
        z = Variable(torch.randn(num_imgs, num_z, 1, 1)).cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_scores = output
        g_loss = output
        output.backward(one)
        optimizerG.step()
        gen_iterations += 1
        
        g_loss_vis = g_loss.data[0]
        d_loss_vis = d_loss.data[0]
        #real_scores_vis = real_scores.data.cpu().numpy().mean()
        #fake_scores_vis = fake_scores.data.cpu().numpy().mean()
        #g_scores_vis    = g_scores.data.cpu().numpy().mean()
        
        if it_1 % 1 == 0:
            print('epoch: {:3}, it: {:4}, g_loss: {:10.4f}, d_loss: {:10.4f}, true: {:10.4f}, fake: {:10.4f}'.format(ep, it_1, g_loss_vis, d_loss_vis))
        writer.add_scalars('loss', {'g_loss':g_loss_vis,
                                    'd_loss':d_loss_vis},  iterations+1)
        #writer.add_scalars('scores', {'real_im':real_scores_vis,
        #                              'fake_im':fake_scores_vis,
        #                              'generator':g_scores_vis}, iterations+1)
        
        if (iterations+1) % save_iterations == 0:
            #if parallel:
                #net.module.save(model_dir, iterations)
            #else:
                #net.save(model_dir, iterations)
            ## save images
            x = vutils.make_grid(fake_img.data, normalize=True, scale_each=True)
            writer.add_image('Image', x, iterations)
            x = vutils.make_grid(real_img.data, normalize=True, scale_each=True)
            writer.add_image('Origin', x, iterations)

        iterations += 1
    
# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()