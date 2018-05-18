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
parser.add_argument('--model-name', type=str,   default='ACGAN', help='model-name (default: ACGAN)')
parser.add_argument('--save-itr',   type=int,   default=100,     help='save model per x iterations (default: 100)')
parser.add_argument('--comment',    type=str,   default='train', help='comment (default: train)')
parser.add_argument('--manualSeed', type=int,   default=1000,    help='manualSeed (default: 1000)')
parser.add_argument('--num-z',      type=int,   default=100,     help='num-z (default: 100)')
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
model_save_itr   = 1000
comment          = args.comment
trainfilepath    = 'hw4_data/train'
validfilepath    = 'hw4_data/test'
num_z            = args.num_z

# numpy seed
np.random.seed(args.manualSeed)
# CPU seed
torch.manual_seed(args.manualSeed)
# GPU seed
torch.cuda.manual_seed_all(args.manualSeed)

# Training data
train_loader = get_data_loader(trainfilepath, batch_size, lbl=True, attr='Smiling')
#valid_loader = get_data_loader(validfilepath, batch_size, lbl=True)

# Save dir
model_dir = '{}{:02d}_Model'.format(model_name, model_id)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

D = netD_ACGAN().cuda()
G = netG_ACGAN(num_z).cuda()
d_optimizer = optim.Adam(D.parameters(), lr=d_lr, betas=(beta1, 0.999))
g_optimizer = optim.Adam(G.parameters(), lr=g_lr, betas=(beta1, 0.999))
# loss functions
dis_criterion = nn.BCELoss()
aux_criterion = nn.CrossEntropyLoss()#nn.NLLLoss()

iterations = 0
curr_ep = 0
if resume:
    iterations = D.resume(model_dir)
    iterations = G.resume(model_dir)
    
writer = SummaryWriter(comment=model_name+comment)

#####################################################################
## trainning process 
#####################################################################
train_loss_total = []
for ep in range(curr_ep, MAX_EPOCHS):
    ## training
    train_loss = 0
    for it_1, (images, labels) in enumerate(train_loader, 0):
        num_imgs = images.size(0)
        real_img = Variable(images).cuda()
        real_aux_label = Variable(torch.LongTensor(labels)).cuda()
        random_lbl = np.random.randint(0, 2, num_imgs)
        fake_aux_label = Variable(torch.LongTensor(random_lbl)).cuda()
        fake_aux_label_em = Variable(torch.FloatTensor(random_lbl)).cuda()
        #real_dis_label = Variable(torch.FloatTensor(np.random.normal(0.85,0.04,num_imgs).clip(0.7,1))).cuda()
        fake_dis_label = Variable(torch.ones(num_imgs)).cuda()
        real_dis_label = Variable(torch.zeros(num_imgs)).cuda()
        
        # ===============train D
        D.zero_grad()
        # compute loss of real_img
        real_dis_out, real_aux_out = D(real_img)
        dis_errD_real = dis_criterion(real_dis_out, real_dis_label)
        aux_errD_real = aux_criterion(real_aux_out, real_aux_label)
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        
        real_scores = real_dis_out
        real_acc = compute_acc(real_aux_out, real_aux_label)

        # compute loss of fake_img
        z = Variable(torch.randn(num_imgs, num_z, 1, 1)).cuda()
        fake_img = G(z, fake_aux_label_em)
        fake_dis_out, fake_aux_out = D(fake_img.detach())
        dis_errD_fake = dis_criterion(fake_dis_out, fake_dis_label)
        aux_errD_fake = aux_criterion(fake_aux_out, fake_aux_label)
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        fake_scores = fake_dis_out
        fake_acc = compute_acc(fake_aux_out, fake_aux_label)
        errD = errD_real + errD_fake
        d_optimizer.step()
        
        # ===============train generator
        # compute loss of fake_img
        G.zero_grad()
        fake_dis_out, fake_aux_out = D(fake_img)
        dis_errG = dis_criterion(fake_dis_out, real_dis_label)
        aux_errG = aux_criterion(fake_aux_out, fake_aux_label)
        errG = dis_errG + aux_errG
        #errG = dis_errG
        errG.backward()
        g_scores = fake_dis_out
        g_optimizer.step()

        g_loss_vis = errG.data[0]
        d_loss_vis = errD.data[0]
        real_scores_vis = real_scores.data.cpu().numpy().mean()
        fake_scores_vis = fake_scores.data.cpu().numpy().mean()
        g_scores_vis    = g_scores.data.cpu().numpy().mean()
        
        if it_1 % 1 == 0:
            print('epoch: {:3}, it: {:4}, g_loss: {:10.4f}, d_loss: {:10.4f}, true: {:10.4f}, fake: {:10.4f}'.format(ep, it_1, g_loss_vis, d_loss_vis, real_scores_vis, fake_scores_vis))
        writer.add_scalars('loss', {'g_loss':g_loss_vis,
                                    'd_loss':d_loss_vis},  iterations+1)
        writer.add_scalars('scores', {'real_im':real_scores_vis,
                                      'fake_im':fake_scores_vis,
                                      'generator':g_scores_vis}, iterations+1)
        writer.add_scalars('acc', {'fake_acc':fake_acc,
                                      'real_acc':real_acc}, iterations+1)
        
        if (iterations+1) % save_iterations == 0:
            ## save images
            x = vutils.make_grid(fake_img.data, normalize=True, scale_each=True)
            writer.add_image('Image', x, iterations)
            x = vutils.make_grid(real_img.data, normalize=True, scale_each=True)
            writer.add_image('Origin', x, iterations)
        if (iterations+1) % model_save_itr == 0:
            if parallel:
                D.module.save(model_dir, iterations)
                G.module.save(model_dir, iterations)
            else:
                D.save(model_dir, iterations)
                G.save(model_dir, iterations)

        iterations += 1
    
# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()