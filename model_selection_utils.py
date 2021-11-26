import torch
import numpy as np
import matplotlib.pyplot as plt

def model_selection(all_models, global_model, args):
    gparams = []
    for param in global_model:
        gparams += list(global_model[param].view(-1).detach().cpu().numpy())
    gparams = torch.Tensor(gparams)
    
    params = [[] for i in range(len(all_models))]
    for idx, model in enumerate(all_models):
        for param in model:
            params[idx] += list(model[param].view(-1).detach().cpu().numpy())
        params[idx] = torch.Tensor(params[idx]) - gparams
    
    pca_angle_selection(params) 
    """if args.select = 'pca-angle':
       selected_idxs = pca_angle_selection(params)
    else:
       selected_idxs = np.arange(len(all_models))
    
    selected_models = []
    for idx in selected_idxs:
        selected_models.append(all_models[idx])"""
    
    return all_models

def pca_angle_selection(vectors):
    angles = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            angles[i,j] = 1 - (np.matmul(vectors[i].T, vectors[j])/(np.linalg.norm(vectors[i])*np.linalg.norm(vectors[j])))
    a = []
    for i in range(len(vectors)):
        a.append(np.sum(angles[i,:]) + np.sum(angles[:,i]))
    
    print(max(a)/min(a))
    
