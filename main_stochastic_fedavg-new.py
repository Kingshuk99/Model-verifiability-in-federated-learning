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
import math
from torch.utils import data
from model_utils import *
from FemnistDataloader import *
from train_stochastic import train, verify
from inference import *
import pickle
#from scipy import spatial
import networkx as nx

def select_clients(client_list, num_clients_train, num_clients_verify):

    length = len(client_list)
    temp_client_list = client_list.copy()
    client_order = []
    while(length > 0):
         if length <= num_clients_train :
             sample = random.sample(range(0,length), length)
         else:
             sample = random.sample(range(0, length), num_clients_train)
         temp_train_list = []
         temp_verify_list = []

         for idx in sample: #append clients for training
             temp_train_list.append(temp_client_list[idx])

         for client in client_list: #append clients for verification
             if client not in temp_train_list:
                 temp_verify_list.append(client)
         if len(temp_verify_list) > num_clients_verify:
             temp_verify_list = random.sample(temp_verify_list, num_clients_verify) #make sure every clients get equal number of verification opportunities

         client_order.append((temp_train_list, temp_verify_list))
		 for client in temp_train_list: #remove training clients
             temp_client_list.remove(client)
         length = len(temp_client_list)

    return client_order
	
	
def calculate_intra_cluster_cohesion(models):
    weight_list = []
    for i in models:
        w1 = i['layer_hidden.weight'].cpu().numpy()
        w1 = w1.flatten()
        weight_list.append(w1)

    weight_array = np.array(weight_list)
    centroid = np.mean(weight_array, axis=0)
    sum_dist = 0
    for i in range(0,len(weight_list)):
        cur_point = weight_array[i,:]
        dist = np.linalg.norm([centroid,cur_point],'fro')
        sum_dist = sum_dist + dist

    return sum_dist

def avg_weights_stochastic(w, args):
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

def initiate_stochastic_fedavg(args, iters,  received_models):
       testloader = data.DataLoader(mnistLoader(split='test'), batch_size=args.batch_size, shuffle=False, pin_memory=False)
       iter_dict = {}
       iter_max = {}
       score_max = -1
       iter_choose = -1
       grp_idx = -1
       iter_groups = {}
       for i in range(0,iters):
            print("iteration",i)
            frac = np.random.choice(list(range(4,16)),1)
            num_clients_train = math.ceil(frac)
            num_clients_verify = args.num_users - num_clients_train
            users = list(range(0,args.num_users))
            clients_order = select_clients(list(range(0,args.num_users)), num_clients_train,
                                                                          num_clients_verify)#
            group_avg = {}
            group_acc = {}
            group_idx = 0
            group_clients = {}
            group_dia = {}
            for clients_train, clients_verify in clients_order:
                    #print("Begin averaging")
                    #write code for averaging
                    group_clients[group_idx] = clients_train
                    stored_models = []
                    for client in clients_train:
                          stored_models.append(received_models[client])
                    cc = calculate_intra_cluster_cohesion(stored_models)

                    global_model = avg_weights_stochastic(stored_models,  args)
                    group_avg[group_idx] = global_model
                    #print("Finished averaging")
					
					#print("check acc of group global model")
                    Model = get_model(args)
                    Model.load_state_dict(global_model)
                    group_acc[group_idx] = calc_acc(testloader, Model, args)
                    group_dia[group_idx] = cc

                    print(group_acc[group_idx], cc)

                    if group_acc[group_idx] > score_max:
                        print(group_acc[group_idx])
                        score_max = group_acc[group_idx]
                        iter_choose = i
                        grp_idx = group_idx

                    group_idx = group_idx + 1

            iter_groups[i] = group_clients
            iter_dict[i] = (group_avg, group_acc, group_dia)

       with open(r"stoch_fed_avg_cohesion_test_dict.pkl", "wb") as output_file:
            pickle.dump(iter_dict, output_file)

	   with open(r"stoch_fed_avg_cohesion_test_groups.pkl", "wb") as output_file:
            pickle.dump(iter_groups, output_file)

       
       acc_threshold = 0.50
       if score_max > acc_threshold:
            group_avg, group_acc = iter_dict[iter_choose]
            good_groups = []
            for k in group_acc.keys():
                v = group_acc[k]
                if v >= acc_threshold:
                     good_groups.append(k)
                good_groups.append(group_avg[k])
            group_clients = iter_groups[iter_choose]


            return good_groups
			
	   else:

            print("couldn't find a good starting point")
            import sys
            sys.exit(1)


def train_clients(clients_train, global_model, round_, received_models, args):
    for client in clients_train: #train
                local_model = train(global_model, client, round_, args)
                received_models[client] = local_model #received models contains model state_dict() of each stored model

def verify_clients(clients_train, clients_verify, received_models, received_scores, round_, args, client_log_dict):

    for c_train in clients_train: #verification score for each model
         for c_verify in clients_verify:
                score = verify(received_models[c_train], c_verify, round_, args)
                received_scores[c_train]['scores'].append(score)
                client_log_dict[c_train]['individual_scores'][round_][c_verify] = score

    print('verification done!!!')
    for client in clients_train: #calculate mean score for each model
         scores = received_scores[client]['scores']
         mean_score = np.mean(scores)
         received_scores[client]['mean_score'] = mean_score
         client_log_dict[client]['mean_score'][round_] = mean_score


    #print(received_scores)
    score_list = []
    for key in received_scores:
          val = received_scores[key]['mean_score']
          if val:
                score_list.append(val)
    score_array = np.array(score_list)
    temperature = 0.2
    score_array = score_array/temperature
    score_softmax = F.softmax(torch.from_numpy(score_array),dim=0) #torch.from_numpy(score_array) #convert mean score to probability distribution
    count = 0
    temp_score_softmax = score_softmax.numpy()
    for client in clients_train:
        client_log_dict[client]['softmax_score'][round_] = temp_score_softmax[count]
        count = count + 1

    print('softmax weights calculated')
    print(score_softmax)
    return score_softmax


def create_log_dictionary(args):
    log_dict = {}
    for i in range(0, args.num_users):
        temp_l1 = [0]*args.num_users
        temp_l2 = [temp_l1]*args.round
        temp_l3 = [0]*args.round
        log_dict[i] = {'individual_scores': temp_l2,'mean_score': temp_l3, 'softmax_score': temp_l3}

    return log_dict

def write_logs(log_dict, round_, log_path, name):
    try:
         with open(log_path+"/round_"+str(round_)+name+".pkl","ab") as f:
            pickle.dump(log_dict, f)
    except FileNotFoundError:
         with open(log_path+"/round_"+str(round_)+name+".pkl","wb") as f:
            pickle.dump(log_dict, f)


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
    
#     print(global_model.keys())
#     w = global_model['layer_hidden.weight'].numpy()
#     sz = w.shape
#     print(w.reshape((1,sz[0] * sz[1])).shape)
#     import sys
#     sys.exit(1)

    '''if args.dataset == 'femnist':
        testloader = data.DataLoader(mnistLoader(split='test'), batch_size=args.batch_size, shuffle=False, pin_memory=False)
    else:
        testloader = None

    test_acc = []'''
    log_path = "logs_stoch_fedavg_new"
    client_log_dict = create_log_dictionary(args)
    global_acc = []
	
	for round_ in range(args.round):
        print("\nRound {} is started.\n".format(round_+1))

        received_models = {}
        received_scores = {}

        #train in all clients
        for user in range(0,args.num_users):
            train_clients([user], global_model, round_, received_models, args)

        print(round_, "round")
        #global_model, clients_order = initiate_stochastic_fedavg(args, 15, received_models)
        global_models = initiate_stochastic_fedavg(args, 15, received_models)


#         good_received_models = {}
#         for clients_train in clients_order:
#             #train_clients(clients_train, global_model, round_, good_received_models, args)
#             good_stored_models = []
#             for client in clients_train:
#                 good_stored_models.append(good_received_models[client])

        #good_stored_models.append(global_model)
        temp_global_model = avg_weights_stochastic(global_models,  args)

        Model.load_state_dict(temp_global_model)
        tacc = 100*(calc_acc(testloader, Model, args))
        test_acc.append(100*(calc_acc(testloader, Model, args)))
        global_acc.append(100*(calc_acc(testloader, Model, args)))
		if tacc > 0.50:
            global_model = temp_global_model

        print("\t Accuracy of the 'Global Model' on  test dataset: {:.2f}%".format(test_acc[-1]))

    torch.save(global_model, args.modelpath+'model_no_sftmax_niid-degree_'+str(args.niid_degree)+'.pt')
    torch.save(test_acc, args.resultpath+'test_acc_no_sftmax_'+str(args.niid_degree)+'.pt')

    # test accuracy plot
    plt.figure()
    plt.plot(range(1,len(test_acc)+1), test_acc, '-b')
    plt.xlabel('Communication rounds')
    plt.ylabel('Accuracy')
    plt.title('Percentage Accuracy, evaluated on Test set')
    plt.xlim(1,len(test_acc))
    plt.ylim(0, 100)
    plt.savefig(args.resultpath+'test_acc_no_sftmax_'+str(args.niid_degree)+'.pdf')

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
    print("Algo: {}  | No. local epochs: {} | No. of communication round: {} | Dataset: {} | Model: {}\nDegree of Non-IID-ness: {} | No. users: {} | Client selection fraction: {}".format(a$

    return args

if __name__ == "__main__":

    args = args_parser()
    server_coordination(args)
    time.sleep(20)





	
