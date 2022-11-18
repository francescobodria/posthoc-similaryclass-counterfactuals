from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
pd.set_option('display.max_columns', None)
import time
import warnings
warnings.filterwarnings("ignore")

import torch
from sklearn import datasets, svm
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, LocalOutlierFactor
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist, pdist
from numpy.random import default_rng
from collections import Counter
from sklearn.cluster import KMeans
from numpy.linalg import norm

def knn_clf(nbr_vec, y):
    '''
    Helper function to generate knn classification result.
    '''
    y_vec = y[nbr_vec]
    c = Counter(y_vec)
    return c.most_common(1)[0][0]

def knn_eval_series(X, y, n_neighbors_list=[1, 2, 3, 4, 5, 10, 15, 20], n_jobs=-1):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by an k-nearest neighbor classifier.
    A series of accuracy will be calculated for the given n_neighbors.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        n_neighbors_list: A list of int.
        kwargs: Any keyword argument that is send into the knn clf.
    Output:
        accs: The avg accuracy generated by the clf, using leave one out cross val.
    '''
    avg_accs = []
    max_acc = X.shape[0]
    # Train once, reuse multiple times
    nbrs = NearestNeighbors(n_neighbors=n_neighbors_list[-1]+1, n_jobs=n_jobs).fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices [:, 1:]
    distances = distances[:, 1:]
    for n_neighbors in n_neighbors_list:
        sum_acc = 0
        for i in range(X.shape[0]):
            indices_temp = indices[:, :n_neighbors]
            result = knn_clf(indices_temp[i], y)
            if result == y[i]:
                sum_acc += 1
        avg_acc = sum_acc / max_acc
        avg_accs.append(avg_acc)
    return 1-np.array(avg_accs)

def random_triplet_eval(X, X_new, y):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An triplet satisfaction score is calculated by evaluating how many randomly
    selected triplets have been violated. Each point will generate 5 triplets.
    Input:
        X: A numpy array with the shape [N, p]. The higher dimension embedding
           of some dataset. Expected to have some clusters.
        X_new: A numpy array with the shape [N, k]. The lower dimension embedding
               of some dataset. Expected to have some clusters as well.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset. Used to identify clusters
    Output:
        acc: The score generated by the algorithm.
    '''    
    # Sampling Triplets
    # Five triplet per point
    anchors = np.arange(X.shape[0])
    rng = default_rng()
    triplets = rng.choice(anchors, (X.shape[0], 5, 2))
    triplet_labels = np.zeros((X.shape[0], 5))
    anchors = anchors.reshape((-1, 1, 1))
    # Calculate the distances and generate labels
    b = np.broadcast(anchors, triplets)
    distances = np.empty(b.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u,v) in b]
    labels = distances[:, :, 0] < distances[: , :, 1]
    # Calculate distances for LD
    b = np.broadcast(anchors, triplets)
    distances_l = np.empty(b.shape)
    distances_l.flat = [np.linalg.norm(X_new[u] - X_new[v]) for (u,v) in b]
    pred_vals = distances_l[:, :, 0] < distances_l[:, :, 1]
    correct = np.sum(pred_vals == labels)
    acc = correct/X.shape[0]/5
    return acc

def lof_eval(X, Z, n_jobs=-1):
    clf = LocalOutlierFactor(n_jobs=n_jobs)
    clf.fit(X)
    outlier_factor_input_space = clf.negative_outlier_factor_
    clf = LocalOutlierFactor(n_jobs=n_jobs)
    clf.fit(Z)
    outlier_factor_latent_space = clf.negative_outlier_factor_
    lof_score = np.mean((outlier_factor_input_space-outlier_factor_latent_space)**2)
    return lof_score
 
def isf_eval(X, Z, n_jobs=-1):
    clf = IsolationForest(n_jobs=n_jobs)
    clf.fit(X)
    outlier_factor_input_space = clf.score_samples(X)
    clf = IsolationForest(n_jobs=n_jobs)
    clf.fit(Z)
    outlier_factor_latent_space = clf.score_samples(Z)
    isf_score = np.mean((outlier_factor_input_space-outlier_factor_latent_space)**2)
    return isf_score

def sse_eval(Z):
    kmeans = KMeans(n_clusters=2).fit(Z)
    sse_score = kmeans.inertia_
    return sse_score

def spars_eval(model, Z):
    y_contrib = model.fc1.weight.detach().numpy()[:,-1]
    thetas = np.arccos(np.round(np.dot(Z/norm(Z,axis=1).reshape(-1,1),y_contrib/norm(y_contrib)),5))
    spars_score = np.std(np.linalg.norm(Z)*np.cos(thetas))
    return spars_score

def compute_metrics(model, X, Z, Y, n_jobs=-1):    
    knn_score = np.mean(knn_eval_series(Z, Y, n_jobs=n_jobs))
    triplet_score = random_triplet_eval(X, Z, Y)
    lof_score = lof_eval(X, Z, n_jobs=n_jobs)
    isf_score = isf_eval(X, Z, n_jobs=n_jobs)
    sse_score = sse_eval(Z)
    spars_score = spars_eval(model, Z)
    return {'KNN':knn_score,
            'Triplet':triplet_score,
            'LOF':lof_score,
            'IsF':isf_score,
            'sse':sse_score,
            'spars':spars_score}

d = {}

for dataset_name in ['adult','german', 'compas','diva']:
    print(dataset_name)
    d[dataset_name]={}
    for bb_name in ['xgb','rf','svc','nn']:
        print(bb_name)
        d[dataset_name]={bb_name:{}}
        from exp.data_loader import load_tabular_data
        from exp.bb_loader import load_bb
        X_train, X_test, y_train, y_test = load_tabular_data(dataset_name)
        y_train_pred, y_test_pred, clf = load_bb(X_train, X_test, dataset_name, bb_name)
        X_train = np.hstack((X_train,y_train_pred.reshape(-1,1)))
        y_train = y_train.values
        X_test = np.hstack((X_test,y_test_pred.reshape(-1,1)))
        y_test = y_test.values

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        class LinearModel(nn.Module):
            def __init__(self, input_shape, latent_dim=2):
                super(LinearModel, self).__init__()
                # encoding components
                self.fc1 = nn.Linear(input_shape, latent_dim)
            def encode(self, x):
                x = self.fc1(x)
                return x
            def forward(self, x):
                z = self.encode(x)
                return z

        if dataset_name == 'adult':
            latent_dims = [2,3,4,5,6,7,8]
        elif dataset_name == 'fico':
            latent_dims = [2,3,4,6,9,12,15,18,21,24]
        elif dataset_name == 'german':
            latent_dims = [2,3,5,10,20,30,40,50,60,71]
        elif dataset_name == 'diva':
            latent_dims = [2,3,4,5,7,10,15,20,25,33]

        for latent_dim in latent_dims:
            model = LinearModel(X_train.shape[1], latent_dim=latent_dim)
            model.load_state_dict(torch.load(f'./models/{dataset_name}_latent_{bb_name}_{latent_dim}.pt'))
            with torch.no_grad():
                model.eval()
                Z_train = model(torch.tensor(X_train).float()).cpu().detach().numpy()
                Z_test = model(torch.tensor(X_test).float()).cpu().detach().numpy()

            d[dataset_name][bb_name][str(latent_dim)] = compute_metrics(model, X_train, Z_train, y_train)

        fig, ax = plt.subplots(3,2,figsize=(30,15))
        ax.ravel()[0].plot(latent_dims,[r['KNN'] for r in d[dataset_name][bb_name].values()], '-o')
        ax.ravel()[0].grid()
        ax.ravel()[0].set_xticks(latent_dims)
        ax.ravel()[0].set_title('KNN')
        ax.ravel()[1].plot(latent_dims,[r['Triplet'] for r in d[dataset_name][bb_name].values()], '-o')
        ax.ravel()[1].grid()
        ax.ravel()[1].set_xticks(latent_dims)
        ax.ravel()[1].set_title('Triplet')
        ax.ravel()[2].plot(latent_dims,[r['LOF'] for r in d[dataset_name][bb_name].values()], '-o')
        ax.ravel()[2].grid()
        ax.ravel()[2].set_xticks(latent_dims)
        ax.ravel()[2].set_title('LOF')
        ax.ravel()[3].plot(latent_dims,[r['IsF'] for r in d[dataset_name][bb_name].values()], '-o')
        ax.ravel()[3].grid()
        ax.ravel()[3].set_xticks(latent_dims)
        ax.ravel()[3].set_title('IsF')
        ax.ravel()[4].plot(latent_dims,[r['sse'] for r in d[dataset_name][bb_name].values()], '-o')
        ax.ravel()[4].grid()
        ax.ravel()[4].set_xticks(latent_dims)
        ax.ravel()[4].set_title('sse')
        ax.ravel()[4].set_xlabel('latent_dim')
        ax.ravel()[5].plot(latent_dims,[r['spars'] for r in d[dataset_name][bb_name].values()], '-o')
        ax.ravel()[5].grid()
        ax.ravel()[5].set_xticks(latent_dims)
        ax.ravel()[5].set_title('spars')
        ax.ravel()[5].set_xlabel('latent_dim')
        fig.savefig(f'./plots/{dataset_name}_{bb_name}_space_metrics.jpeg', bbox_inches='tight')
        plt.close()


