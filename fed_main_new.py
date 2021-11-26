#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import copy
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm as tq
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from model_utils import *
from FemnistDataloader import *
from femnist_train import train
from inference import *
from model_selection_utils import model_selection


# In[ ]:


def avg_weights(w, args):
    
    running_mean_sqr, running_sqr_mean = [], []
    wts = torch.ones(len(w)).to(args.device)
    
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key].to(args.device)
        layer = key.split('.')[-1]
        if layer == 'running_mean':
            for i in range(len(w)):
                running_mean_sqr.append(torch.square(w[i][key].to(args.device)))
                if i != 0:
                    w_avg[key] += torch.mul(w[i][key].to(args.device),wts[i].to(float))
            key_prev = key
            w_avg[key] = torch.true_divide(w_avg[key].to(args.device), sum(wts))
            
        elif layer =='running_var':
            for i in range(len(w)):
                running_sqr_mean.append(torch.mul(torch.add(w[i][key].to(args.device),running_mean_sqr[i]),wts[i].to(float)))
            running_sqr_mean_avg = torch.true_divide(sum(running_sqr_mean), sum(wts))
            w_avg[key] = torch.sub(running_sqr_mean_avg, torch.square(w_avg[key_prev].to(args.device)))
            running_mean_sqr, running_sqr_mean = [], []
        elif layer == 'num_batches_tracked':
            batches = 0
            for i in range(1,len(w)):
                w_avg[key] += w[i][key].to(args.device)
            w_avg[key] = torch.true_divide(w_avg[key].to(args.device), len(w)).to(torch.int64)
            
        else:
            for i in range(1,len(w)):
                w_avg[key] += torch.mul(w[i][key].to(args.device), wts[i].to(float))
            w_avg[key] = torch.true_divide(w_avg[key].to(args.device), sum(wts))
            
    return w_avg


# In[ ]:


def server_coordination(args,userset,testload):
    
    Model = get_model(args)
    global_model = Model.state_dict()
    
    if args.dataset == 'femnist':
        user_datasets=userset
        testloader = testload
    else:
        uder_datasets = None
        testloader = None
    
    test_loss = []
    test_acc = []
    test_dice = []
    test_prec = []
    test_rec = []
    
    for round_ in range(args.round):
        print("\nRound {} is started.\n".format(round_+1))
        received_models = []
        sz = max(int(args.num_users*args.frac), 1)
        selected_users = np.random.choice(np.arange(args.num_users), size=sz, replace=False)
        mal_cl = np.random.choice([0,1], size = sz, p = [1-args.malicious, args.malicious])
        for user in range(len(selected_users)):
            local_model = train(global_model, user_datasets[selected_users[user]], round_+1, args, mal_cl[user])
            received_models.append(local_model)
        global_model = avg_weights(received_models, args)
        Model.load_state_dict(global_model)
        loss, acc, dice, prec, rec = calc_loss_metrices(testloader, Model, args)
        acc = acc.cpu()
        test_loss.append(loss)
        test_acc.append(acc)
        test_dice.append(dice)
        test_prec.append(prec)
        test_rec.append(rec)
        print("'Global Model' on test dataset:\n\t Loss : {:.2f} | Accuracy : {:.2f}%".format(test_loss[-1], test_acc[-1]))
    
    test_loss_metrices = {'loss':test_loss, 'Accuracy':test_acc, 'Dice':test_dice, 'Precision':test_prec, 'Recall':test_rec}
    torch.save(global_model, args.modelpath+'model_niid-degree_'+str(args.niid_degree)+'.pt')
    torch.save(test_loss_metrices, args.resultpath+'test_loss_metrices_'+str(args.niid_degree)+'.pt')

    
    # test accuracy plot
    
    plt.figure()
    plt.plot(range(1,len(test_acc)+1), test_acc, '-b')
    plt.xlabel('Communication rounds')
    plt.ylabel('Accuracy')
    plt.title('Percentage Accuracy, evaluated on Test set')
    plt.xlim(1,len(test_acc))
    plt.ylim(0, 100)
    plt.savefig(args.resultpath+'test_acc_'+str(args.niid_degree)+'.pdf')


# In[ ]:


def args_parser():
    parser = argparse.ArgumentParser()
    
    # federated arguments (Notation for the arguments followed from paper)
    
    parser.add_argument('--algo', choices=['FedAvg', 'FedProx'], type=str, default='FedAvg', help="Name of algorithm. Allowable values: FedAvg and FedProx")
    parser.add_argument('--loc_epoch', '-le', type=int, default=10, help="Number of local epochs: E")
    parser.add_argument('--frac', '-c', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--malicious', '-m', type=float, default=0.1, help='the fraction of malicious clients')
    parser.add_argument('--drop_percent', '-dp', type=float, default=0.0, help='percentage of slow devices')
    parser.add_argument('--mu', type=float, default=0, help="The proximal loss for the FedProx algo")
    parser.add_argument('--batch_size', '-b', type=int, default=128, help="Local batch size: B")
    parser.add_argument('--aug', type=int, default=0, help="1 inplies augmentation enabled")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for generator networks")
    parser.add_argument('--k', default=1, type=int, help='Top k accuracy')
    parser.add_argument('--round', '-r', type=int, default=10, help="number of communication round")
    
    # other arguments
    
    parser.add_argument('--dataset', type=str, default='femnist', help="name of dataset")
    parser.add_argument('--model', choices=['MLP', 'CNN'], type=str, default='MLP', help='model name')
    parser.add_argument('--hidden_nodes', '-hn', default=1000, type=int, help='Number of hidden nodes in MLP')
    parser.add_argument('--gpu', default=0, type=int, help='GPU number')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--niid_degree', '-nd', type=int, default=0, help='Default set to IID. Set integers from 1 to 4 for incresing degree of non-IID.')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--run', type=str, default=-999, help="Run number")
    
    args = parser.parse_args()
    
    
    return args


# In[ ]:




