from __future__ import print_function
import os
import numpy as np
import cv2
import pandas as pd
import torch
import torch.utils.data as data

'''
TODO: Get Data Loader
'''
def get_data_loader(filepath, batch_size, lbl=False):
  dataset = []
  if not lbl:
      dataset = dataset_image(filepath)
  else:
      dataset = dataset_image_lbl(filepath)
  return torch.utils.data.DataLoader(dataset=dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True, 
                                     num_workers=10,
                                     pin_memory=True)

'''
TODO: Image Dataset
'''
class dataset_image(data.Dataset):
  def __init__(self, filepath):
    self.filepath = filepath
    #with open(filepath + '.csv') as f:
    #    content = f.readlines()
    #    f.close()
    #self.img_list = [x[:-1].split(',')[0] for x in content[1:]]
    #self.attributions = np.array([list(map(float, x[:-1].split(',')[1:])) for x in content[1:]], dtype=np.int)
    full_list = pd.read_csv(filepath + '.csv')
    self.img_list = full_list['image_name']
    self.dataset_size = len(self.img_list)

  def __getitem__(self, index):
    img_raw = self._load_one_image(os.path.join(self.filepath, self.img_list[index]))
    img = img_raw.transpose((2, 0, 1))  # convert to CHW
    img = torch.FloatTensor((img / 255.0 - 0.5) * 2.0)
    return img

  def _load_one_image(self, img_name, test=False):
    img = cv2.imread(img_name)
    img = img[:,:,::-1]
    #if test == False:
    #  if np.random.rand(1) > 0.8:
    #    axis = int(np.round(np.random.rand(1)))
    #    img = np.flip(img, axis=1)      
    return img

  def __len__(self):
    return self.dataset_size

'''
TODO: Image Dataset
'''
class dataset_image_lbl(data.Dataset):
  def __init__(self, filepath):
    self.filepath = filepath
    full_list = pd.read_csv(filepath + '.csv')
    self.img_list = full_list['image_name']
    self.Bangs = full_list['Bangs']
    self.dataset_size = len(self.img_list)

  def __getitem__(self, index):
    img_raw = self._load_one_image(os.path.join(self.filepath, self.img_list[index]))
    img = img_raw.transpose((2, 0, 1))  # convert to CHW
    img = torch.FloatTensor((img / 255.0 - 0.5) * 2.0)
    lbl = np.int(self.Bangs[index])
    return img, lbl

  def _load_one_image(self, img_name, test=False):
    img = cv2.imread(img_name)
    img = img[:,:,::-1]    
    return img

  def __len__(self):
    return self.dataset_size