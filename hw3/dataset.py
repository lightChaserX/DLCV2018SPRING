from __future__ import print_function
import os
import numpy as np
import cv2
import torch
import torch.utils.data as data

VGG_mean = np.array([103.939, 116.779, 123.68])

'''
TODO: Sat Image Dataset
'''
class dataset_sat_image(data.Dataset):
  def __init__(self, filepath, down_scale=1):
    self.filepath = filepath
    self.down_scale = down_scale
    self.images = [os.path.join(filepath, file) 
                   for file in os.listdir(filepath) if file.endswith('.jpg')]
    self.labels = [file[0:-len('sat.jpg')]+'mask.png' for file in self.images]
    self.dataset_size = len(self.images)
    self.data = dict(zip(self.images, self.labels))

  def __getitem__(self, index):
    img_raw, lbl_raw = self._load_one_image(self.images[index])
    img_raw = np.float64(img_raw)
    img_raw -= VGG_mean
    img_CxHxW = img_raw.transpose((2, 0, 1))  # convert to CHW
    img = torch.from_numpy(img_CxHxW.copy()).float()
    lbl = torch.from_numpy(lbl_raw.copy()).long()
    return img, lbl

  def _load_one_image(self, img_name, test=False):
    img = cv2.imread(img_name)
    if test == False:
      lbl = im2labels(self.data[img_name], down_scale=self.down_scale)
      #img = cv2.resize(img, None, fx=1.0/self.down_scale, fy=1.0/self.down_scale)
      if np.random.rand(1) > 0.5:
        axis = int(np.round(np.random.rand(1)))
        img = np.flip(img, axis=axis)
        lbl = np.flip(lbl, axis=axis)
        
      h, w, c = img.shape
      w_post = w//self.down_scale
      h_post = h//self.down_scale
      x_offset = np.int32(np.random.randint(0, w - w_post + 1, 1))[0]
      y_offset = np.int32(np.random.randint(0, h - h_post + 1, 1))[0]
      img = img[y_offset:(y_offset+h_post), x_offset:(x_offset+w_post), :]
      lbl = lbl[y_offset:(y_offset+h_post), x_offset:(x_offset+w_post)]
    else:
      lbl = im2labels(self.data[img_name], test=test)
      #print(self.data[img_name], img_name)
    return img, lbl

  def __len__(self):
    return self.dataset_size

'''
TODO: Transfer im 'xxxx_mask.png' to Labels
'''
def im2labels(mask_name, test=False, down_scale=1, ref_name=True):
    if ref_name:
        mask = cv2.imread(mask_name)
    else:
        mask = mask_name
    #if test == False:
    #    mask = cv2.resize(mask, None, fx=1.0/down_scale, fy=1.0/down_scale)
    mask = mask[:,:,::-1] # BGR->RGB
    mask = (mask >= 128).astype(int) # threshold
    labels = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    lbl = np.zeros(labels.shape)
    lbl[labels == 3] = 0  # (Cyan:   011) Urban land 
    lbl[labels == 6] = 1  # (Yellow: 110) Agriculture land 
    lbl[labels == 5] = 2  # (Purple: 101) Rangeland 
    lbl[labels == 2] = 3  # (Green:  010) Forest land 
    lbl[labels == 1] = 4  # (Blue:   001) Water 
    lbl[labels == 7] = 5  # (White:  111) Barren land 
    lbl[labels == 0] = 6  # (Black:  000) Unknown
    lbl[labels == 4] = 6
    return lbl

'''
TODO: Transfer Labels to im
'''
def labels2im(lbl):
    h, w = lbl.shape
    label = np.zeros((h, w, 3), dtype=np.uint8)
    label[lbl == 0, :] = [  0, 255, 255]
    label[lbl == 1, :] = [255, 255,   0]
    label[lbl == 2, :] = [255,   0, 255]
    label[lbl == 3, :] = [  0, 255,   0]
    label[lbl == 4, :] = [  0,   0, 255]
    label[lbl == 5, :] = [255, 255, 255]
    label[lbl == 6, :] = [  0,   0,   0]
    return label

'''
TODO: Get Data Loader
'''
def get_data_loader(filepath, batch_size, down_scale=1):
  dataset = []
  dataset = dataset_sat_image(filepath, down_scale)
  return torch.utils.data.DataLoader(dataset=dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True, 
                                     num_workers=10,
                                     pin_memory=True)