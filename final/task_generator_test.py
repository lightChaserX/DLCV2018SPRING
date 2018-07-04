# code is based on https://github.com/katerakelly/pytorch-maml
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
import glob


def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x


def omniglot_character_folders(data_folder):
    class_folders = glob.glob(os.path.join(data_folder, 'novel/class*/*'), recursive=True)
    class_folders.sort()

    random.seed(1)
    random.shuffle(class_folders)

    metatrain_character_folders = [pngs_item for pngs_item in class_folders if 'test' not in pngs_item]
    metaval_character_folders = '../../test'

    return metatrain_character_folders, metaval_character_folders


class OmniglotTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, character_folders, num_classes, train_num, test_num, iter, test_dir='../../test'):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders, self.num_classes)

        # tmp_class_folders = self.character_folders + self.character_folders[:self.num_classes]
        # class_folders = tmp_class_folders[iter:iter + self.num_classes]

        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        self.lbl_map = labels
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            # samples[c] = random.sample(temp, len(temp))
            samples[c] = temp

            self.train_roots += samples[c][:train_num]

        self.test_roots = [os.path.join(test_dir, '{}.png'.format(i)) for i in range(2000)]  # query

        self.train_labels = [labels[self.get_class(x)] for x in
                             self.train_roots]  # C*K, [0, ..., 0, ..., K-1, ..., K-1]
        self.test_labels = []
        #print(self.train_labels, self.train_roots)
        #print(self.test_roots)

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

    def remapping_lbl(self, lbl):
        sample = list(self.lbl_map.keys())[list(self.lbl_map.values()).index(lbl)]
        return int(sample.split('/')[-2][-2:])


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else []

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        image = image.resize((28, 28), resample=Image.LANCZOS)  # per Chelsea's implementation
        # image = np.array(image, dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        if len(self.labels) == 0:
            #print(image_root)
            return image
        else:
            label = self.labels[idx]
            if self.target_transform is not None:
                label = self.target_transform(label)
            #print(image_root, label)
            return image, label


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train', shuffle=True, rotation=0):
    # NOTE: batch size here is # instances PER CLASS
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    dataset = Omniglot(task, split=split,
                       transform=transforms.Compose([Rotate(rotation), transforms.ToTensor(), normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, shuffle=shuffle)

    return loader

