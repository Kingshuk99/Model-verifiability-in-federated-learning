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
from train import train
from inference import calc_acc
from model_selection_utils import model_selection


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

def server_coordination(args):
    
    Model = get_model(args)
    global_model = Model.state_dict()
    
    if args.dataset == 'mnist':
        testloader = data.DataLoader(mnistLoader(split='test'), batch_size=args.batch_size, shuffle=False, pin_memory=False)
    else:
        testloader = None
    
    test_acc = []
    
    for round_ in range(args.round):
        print("\nRound {} is started.\n".format(round_+1))
        received_models = []
        selected_users = np.random.choice(np.arange(args.num_users), size=max(int(args.num_users*args.frac), 1), replace=False)
        for user in selected_users:
            local_model = train(global_model, user, round_+1, args)
            received_models.append(local_model)
        selected_models = model_selection(received_models, global_model, args)
        global_model = avg_weights(selected_models, args)
        Model.load_state_dict(global_model)
        test_acc.append(calc_acc(testloader, Model, args))
        print("\t Accuracy of the 'Global Model' on test dataset: {:.2f}%".format(test_acc[-1]))
    
    torch.save(global_model, args.modelpath+'model_niid-degree_'+str(args.niid_degree)+'.pt')
    torch.save(test_acc, args.resultpath+'test_acc_'+str(args.niid_degree)+'.pt')
    
    # test accuracy plot
    plt.figure()
    plt.plot(range(1,len(test_acc)+1), test_acc, '-b')
    plt.xlabel('Communication rounds')
    plt.ylabel('Accuracy')
    plt.title('Percentage Accuracy, evaluated on Test set')
    plt.xlim(1,len(test_acc))
    plt.ylim(0, 100)
    plt.savefig(args.resultpath+'test_acc_'+str(args.niid_degree)+'.pdf')


def args_parser():
    parser = argparse.ArgumentParser()
    
    # federated arguments (Notation for the arguments followed from paper)
    
    parser.add_argument('--algo', choices=['FedAvg', 'FedProx'], type=str, default='FedAvg', help="Name of algorithm. Allowable values: FedAvg and FedProx")
    parser.add_argument('--loc_epoch', '-le', type=int, default=10, help="Number of local epochs: E")
    parser.add_argument('--frac', '-c', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--drop_percent', '-dp', type=float, default=0.4, help='percentage of slow devices')
    parser.add_argument('--mu', type=float, default=0, help="The proximal loss for the FedProx algo")
    parser.add_argument('--batch_size', '-b', type=int, default=10, help="Local batch size: B")
    parser.add_argument('--aug', type=int, default=0, help="1 inplies augmentation enabled")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for generator networks")
    parser.add_argument('--round', '-r', type=int, default=10, help="number of communication round")
    
    # other arguments
    
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--model', choices=['MLP', 'CNN'], type=str, default='MLP', help='model name')
    parser.add_argument('--hidden_nodes', '-hn', default=100, type=int, help='Number of hidden nodes in MLP')
    parser.add_argument('--gpu', default=0, type=int, help='GPU number')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--niid_degree', '-nd', type=int, default=0, help='Default set to IID. Set integers from 1 to 4 for incresing degree of non-IID.')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--run', type=str, default=-999, help="Run number")
    
    args = parser.parse_args()
    
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    # directory to save results
    args.root = os.path.dirname(os.path.abspath('./.'))
    savepath = args.root+'/save/'
    args.resultpath = savepath +'results/run_'+str(args.run)+'/'
    args.modelpath = savepath+'/models/run_'+str(args.run)+'/'
    if args.dataset == 'mnist':
        args.num_users = 150
        args.num_classes = 10
    else:
        args.num_users = 0
        args.num_classes = 0
    
    if not os.path.isdir(args.resultpath):
        os.makedirs(args.resultpath)
    if not os.path.isdir(args.modelpath): 
        os.makedirs(args.modelpath)
    
    # saving args as dict
    torch.save(vars(args), args.resultpath+'args_'+str(args.niid_degree)+'.pt')
    print("Algo: {}  | No. local epochs: {} | No. of communication round: {} | Dataset: {} | Model: {}\nDegree of Non-IID-ness: {} | No. users: {} | Client selection fraction: {}".format(args.algo, args.loc_epoch, args.round, args.dataset, args.model, args.niid_degree, args.num_users, args.frac))
    
    return args
    
if __name__ == "__main__":
    
    args = args_parser()
    server_coordination(args)
    time.sleep(20)
