#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils import data
from torchvision import transforms
from model_utils import *


# In[21]:


class FemnistLoader(data.Dataset):
    def __init__(
        self,
        path,
        clientID = 0,
        img_size = 28,
        niid_degree = 0,
        split = 'fed', #central, fed, test
        aug = False,
        is_transform = True,
    ):
        self.aug = aug
        self.is_transform = is_transform
        self.img_size = img_size
        self.path = path
        self.niid_degree = niid_degree
        self.clientID = clientID
        self.split = split
        
        if self.split == 'central':
            self.loc = os.path.join(self.path, 'trainset', 'data.txt')
        elif self.split == 'fed':
            self.loc = os.path.join(self.path, 'trainingset_for_clients', 'non-iid_level_'+str(self.niid_degree), 'client_'+str(self.clientID), 'data.txt')
        elif self.split == 'test':
            self.loc = os.path.join(self.path, 'testset', 'data.txt')
            
        self.file = tuple(open(self.loc, 'r'))
        self.n = 0
        self.datalist = []
        self.tf = transforms.Resize(self.img_size, interpolation = Image.NEAREST)
        for x in self.file:
            temp1 = list(x.rstrip().rsplit('   '))
            temp2 = []
            temp2.append(temp1[0])
            img = torch.load(temp1[1])
            img = torch.unsqueeze(img, dim = 0)
            if self.is_transform == True:
                img = self.tf(img)
            img = torch.squeeze(img, dim = 0)
            temp2.append(img)
            self.datalist.append(temp2)
            self.n += 1
            
    def __len__(self):
        return self.n
        
    def __getitem__(self, index):
        if self.n == 0:
            return (None, None)
        temp = self.datalist[index]
        img = temp[1]
        lbl = int(temp[0])
            
        return img, lbl

