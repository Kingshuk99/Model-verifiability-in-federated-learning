import os
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from model_utils import *
from FemnistDataloader import *
from inference import calc_acc

def verify(model_state_dict, user, round_, args):
    # get dataloader for training
    if args.dataset == 'mnist':
        loader = data.DataLoader(mnistLoader(split='fed', clientID=user, niid_degree=args.niid_degree), batch_size=args.batch_size, shuffle=True, pin_memory=False)
    else:
        loader = None
    model = get_model(args)
    model.load_state_dict(model_state_dict)
    model = model.to(args.device)
    correct = 0
    total = 0
    for data_ in loader:
        input_, label = data_
        if args.model == 'MLP':
            input_ = input_.view(input_.size(0),-1).unsqueeze(1)
        else:
            input_ = input_.unsqueeze(1)
        input_ =input_.to(args.device)
        label = label.to(args.device)
        output = model(input_)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    return (correct/total)

def train(model_state_dict, user, round_, args):

    torch.autograd.set_detect_anomaly(True)
    start = time.time()

    # get dataloader for training
    if args.dataset == 'mnist':
        trainloader = data.DataLoader(mnistLoader(split='fed', clientID=user, niid_degree=args.niid_degree), batch_size=args.batch_size, shuffle=True, pin_memory=False)
    else:
        trainloader = None

    Model = get_model(args)
    Model.load_state_dict(model_state_dict)
    Model = Model.to(args.device)

    if args.algo =='FedProx':
        init_model = copy.deepcopy(Model)

    #malicious_users = [-1]
    malicious_users = np.random.choice(list(range(0,150)),15)
    #malicious_users = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    if user in malicious_users:
        print(user, "yes")
        optimizer = optim.Adam(Model.parameters(), lr=0.01)
		 # define optimizers
    else:
        optimizer = optim.Adam(Model.parameters(), lr=args.lr)
    # define optimizers
    #optimizer = optim.Adam(Model.parameters(), lr=args.lr)

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



