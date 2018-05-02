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

parser = argparse.ArgumentParser(description='DLCV HW3')
parser.add_argument('--model-name', type=str, default='FCN32s', metavar='N',
                help='model-name (default: FCN32s)')
parser.add_argument('--g', type=str, metavar='g', help='ground truth')
parser.add_argument('--p', type=str, metavar='p', help='predict')
args = parser.parse_args()

n_class     = 7
down_scale  = 1
model_id    = 5
ep          = 0
model_name  = args.model_name
input_filepath    = args.g
output_filepath   = args.p

if not (os.path.exists('baseline_model.pkl') and os.path.exists('improved_model.pkl')):
    file_id = '1f26NDZJ7qRgf0FK8rdt4cgH-KRdFV79_'
    destination = 'model.zip'
    download_file_from_google_drive(file_id, destination)
    
    zip_ref = zipfile.ZipFile(destination, 'r')
    zip_ref.extractall('.')
    zip_ref.close()
    
exec('net = %s(n_class)' % (model_name))
if model_name == 'FCN32s':    
    net.load_state_dict(torch.load('baseline_model.pkl'))
elif model_name == 'FCN8s':
    net.load_state_dict(torch.load('improved_model.pkl'))
net.cuda()


im_list = [os.path.join(input_filepath, file) 
                   for file in os.listdir(input_filepath) if file.endswith('.jpg')]
im_list.sort()

if not os.path.exists(output_filepath):
    os.makedirs(output_filepath)

totol_acc = 0
for im_name in im_list:
        img_test, lbl_test = dataset_sat_image(input_filepath)._load_one_image(im_name, 
                                                                         test=True)
        img_test = np.float64(img_test)
        img_test -= VGG_mean
        img_CxHxW = img_test.transpose((2, 0, 1))  # convert to CHW
        img_test = ((torch.FloatTensor(img_CxHxW)))
        lbl_test = ((torch.LongTensor(lbl_test)))
        img_test_final = img_test.contiguous()
        lbl_test_final = lbl_test.contiguous()
        img_test_final = Variable(img_test_final.view(1,
                                                      img_test_final.size(0),
                                                      img_test_final.size(1),
                                                      img_test_final.size(2))).cuda()
        lbl_test_final = Variable(lbl_test_final.view(1,
                                                      lbl_test_final.size(0),
                                                      lbl_test_final.size(1))).cuda()
        output = net(img_test_final)
        lbl_pred = output.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = lbl_test_final.data.cpu().numpy()
        acc = np.sum(lbl_pred.flatten()==lbl_true.flatten())/len(lbl_true.flatten())
        print(acc)
        totol_acc += acc
        label_im = labels2im(output.data.max(1)[1].cpu().numpy().squeeze())
        im_save_name = os.path.join(output_filepath, im_name[-12:-8]+'_mask.png')
        cv2.imwrite(im_save_name, label_im[:,:,::-1])
        print('saving {} finished'.format(im_save_name))
print('avr loss{}'.format(totol_acc/len(im_list)))