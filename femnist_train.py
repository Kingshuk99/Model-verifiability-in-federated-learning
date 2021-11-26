#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import time
import copy
import random
import argparse
from PIL import Image
from tqdm import tqdm as tq
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable, Function

from model_utils import *
from FemnistDataloader import *
from inference import *


# In[2]:


def train(model_state_dict, user_dataset, round_, args, mal = 0):
    torch.autograd.set_detect_anomaly(True)
    start = time.time()
    
    # get dataloader for training
    if args.dataset == 'femnist':
        trainloader = data.DataLoader(user_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False)
    else:
        trainloader = None
        
    Model = get_model(args)
    Model.load_state_dict(model_state_dict)
    Model = Model.to(args.device)
    
    if args.algo =='FedProx':
        init_model = copy.deepcopy(Model)
    
    # define optimizers
    if mal == 0:
        optimizer = optim.Adam(Model.parameters(), lr=args.lr)
    else :
        optimizer = optim.Adam(Model.parameters(), lr=1)
    
    # define loss functions
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    for epoch in range(args.loc_epoch):
        
        # variable initialization to store losses and count batches
        trainLoss = 0.0
        trainBatches = 0.0
        
        Model.train(True)
        
        for data_ in trainloader:
            Prox_loss = 0.0
            
            Model.zero_grad()
            optimizer.zero_grad()
            
            input_, label = data_
            if args.model == 'MLP':
                input_ = input_.view(input_.size(0),-1).unsqueeze(1)
            else:
                input_ = input_.unsqueeze(1)
            
            input_ =input_.to(args.device)
            label = label.to(args.device)
            
            output = Model(input_)
            
            Loss = criterion(output, label)
            if args.algo =='FedProx':
                for paramA, paramB in zip(init_model.parameters(), Model.parameters()):
                    Prox_loss += torch.square(torch.norm((paramA.detach() - paramB)))
                Prox_loss = (args.mu/2)*Prox_loss
            
            (Loss + Prox_loss).backward()
            optimizer.step()
            
            if epoch == args.loc_epoch -1:
               trainLoss += Loss.item()
               trainBatches+=1
    
    trainEndLoss = trainLoss/trainBatches
    
    # print performance
    if args.verbose == 1:
        print("Client with UserId '{}' has completed {} epochs in {:.2f} second.\n\t Train-end loss: {:.4f}\n".format(user+1, args.loc_epoch, time.time() - start, trainEndLoss))
    
    return Model.state_dict()

