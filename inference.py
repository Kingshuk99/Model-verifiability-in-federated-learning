#!/usr/bin/env python
# coding: utf-8

# In[29]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tq
import torch.optim as optim
from torch.utils import data
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix


# In[35]:


def calc_loss_metrices(loader, model, args):
    losses = []
    correct = 0
    total = 0
    tp = np.zeros(args.num_classes)
    fp = np.zeros(args.num_classes)
    fn = np.zeros(args.num_classes)
    model = model.to(args.device)
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    model.eval()
    for data_ in tq(loader):
        input_, label = data_
        if args.model == 'MLP':
            input_ = input_.view(input_.size(0),-1).unsqueeze(1)
        else:
            input_ = input_.unsqueeze(1)
        input_ =input_.to(args.device)
        label = label.to(args.device)
        output = model(input_)
        Loss = criterion(output, label)
        losses.append(Loss.item())
        total += label.size(0)
        correct += topk_correct(label, output, args.k)
        cf = conf_matrix(label, output, args)
        tp += cf[0]
        fp = cf[1]
        fn += cf[2]
        
    final_loss = (torch.tensor(losses).mean()).item()
    dice = np.divide(2*tp, (2*tp+fp+fn))
    prec = np.divide(tp, (tp+fp), out = np.zeros_like(tp), where = (tp+fp)!=0)
    rec = np.divide(tp, (tp+fn), out = np.zeros_like(tp), where = (tp+fn)!=0)
    return final_loss, ((100*correct)/total), dice, prec, rec


# In[36]:


def topk_correct(label, output, k):
    _, predicted = torch.topk(output, k, dim = 1)
    label = label.unsqueeze(dim = 1)
    tot_corr = torch.eq(predicted, label).sum()
    return tot_corr


# In[37]:


def conf_matrix(label, output, args):
    _, predicted = torch.topk(output, args.k, dim = 1)
    label2 = label.unsqueeze(dim = 1)
    corr = torch.eq(predicted, label2).sum(0)
    label = label.cpu().numpy()
    class_labels = [i for i in range(args.num_classes)]
    pred = np.zeros_like(label)
    for i in range(len(corr)):
        if corr[i] == 1:
            pred[i] = label[i]
        else :
            pred[i] = predicted[i][0]
    c = confusion_matrix(label, pred, labels = class_labels)
    TP = np.diag(c)
    FP = np.sum(c, axis = 0)-TP
    FN = np.sum(c, axis = 1)-TP
    return TP, FP, FN

