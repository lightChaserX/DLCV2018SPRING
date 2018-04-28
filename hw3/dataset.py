from __future__ import print_function
import os
import numpy as np
import cv2
import torch
import torch.utils.data as data

class dataset_sta_image(data.Dataset):
  def __init__(self, filepath):
    self.filepath = filepath
    self.images = [os.path.join(filepath, file) 
                   for file in os.listdir(filepath) if file.endswith('.jpg')]
    self.labels = [file[0:-len('sat.jpg')]+'mask.png' for file in self.images]
    self.dataset_size = len(self.images)
    self.data = dict(zip(self.images, self.labels))

  def __getitem__(self, index):
    crop_img, labels = self._load_one_image(self.images[index])
    raw_data = crop_img.transpose((2, 0, 1))  # convert to HWC
    #data = ((torch.FloatTensor(raw_data)/255.0)-0.5)*2
    data = (torch.FloatTensor(raw_data)/255.0)
    #labels = torch.FloatTensor(labels)
    return data, labels

  def _load_one_image(self, img_name, test=False):
    if test == True:
      img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    else:
      img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
      img = np.float32(img)
    lbl = im2labels(self.data[img_name])
    
    #img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    #lbl = cv2.resize(lbl, (0,0), fx=0.5, fy=0.5) 
    return img, lbl

  def __len__(self):
    return self.dataset_size

'''
TODO: 
'''
def get_data_loader(filepath, batch_size):
  dataset = []
  dataset = dataset_sta_image(filepath)
  return torch.utils.data.DataLoader(dataset=dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True, 
                                     num_workers=10,
                                     pin_memory=True)
'''
TODO: 
'''
def im2labels(mask_name):
    mask = cv2.cvtColor(cv2.imread(mask_name), cv2.COLOR_BGR2RGB)
    mask = (mask >= 128).astype(int)
    labels = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    labels[labels == 3] = 0  # (Cyan: 011) Urban land 
    labels[labels == 6] = 1  # (Yellow: 110) Agriculture land 
    labels[labels == 5] = 2  # (Purple: 101) Rangeland 
    labels[labels == 2] = 3  # (Green: 010) Forest land 
    labels[labels == 1] = 4  # (Blue: 001) Water 
    labels[labels == 7] = 5  # (White: 111) Barren land 
    labels[labels == 0] = 6  # (Black: 000) Unknown
    return labels