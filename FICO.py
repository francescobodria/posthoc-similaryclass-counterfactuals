import tensorflow as tf
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
random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/heloc_dataset.csv')
df['RiskPerformance']=(df.RiskPerformance=='Bad')+0

scaler = MinMaxScaler((-1,1))
X = scaler.fit_transform(df.values[:,1:])
y = df['RiskPerformance'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# BlackBoxes

### XGBOOST

from xgboost import XGBClassifier

clf_xgb = XGBClassifier(n_estimators=60, reg_lambda=3, use_label_encoder=False, eval_metric='logloss')

clf_xgb.fit(X_train, Y_train)
pickle.dump(clf_xgb,open('./BlackBoxes/fico_xgboost.p','wb'))

clf_xgb = pickle.load(open('./BlackBoxes/fico_xgboost.p','rb'))
y_train_pred = clf_xgb.predict(X_train)
y_test_pred = clf_xgb.predict(X_test)
print('XGBOOST')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

### Random Forest

from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(random_state=random_seed)
clf_rf.fit(X_train, Y_train)

pickle.dump(clf_rf,open('./BlackBoxes/fico_rf.p','wb'))

clf_rf = pickle.load(open('./BlackBoxes/fico_rf.p','rb'))
y_train_pred = clf_rf.predict(X_train)
y_test_pred = clf_rf.predict(X_test)
print('RF')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

### SVC

from sklearn.svm import SVC
clf_svc = SVC(gamma='auto', probability=True)
clf_svc.fit(X_train, Y_train)

pickle.dump(clf_svc,open('./BlackBoxes/fico_svc.p','wb'))

clf_svc = pickle.load(open('./BlackBoxes/fico_svc.p','rb'))
y_train_pred = clf_svc.predict(X_train)
y_test_pred = clf_svc.predict(X_test)
print('SVC')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

### NN tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

BATCH_SIZE = 1024
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(2048).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(BATCH_SIZE)

clf_nn = keras.Sequential([
    keras.layers.Dense(units=10, activation='relu'),
    keras.layers.Dense(units=5, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid'),
])

early_stopping = EarlyStopping(patience=5)

clf_nn.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = clf_nn.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=500,
    callbacks=[early_stopping],
    verbose=0
)

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.grid()
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

#plot_metric(history, 'loss')
clf_nn.save_weights('./blackboxes/fico_tf_nn')

from sklearn.metrics import accuracy_score
clf_nn.load_weights('./blackboxes/fico_tf_nn')
clf_nn.trainable = False
print('NN')
print(accuracy_score(np.round(clf_nn.predict(X_train)),Y_train))
print(accuracy_score(np.round(clf_nn.predict(X_test)),Y_test))
print('---------------')

# ### NN pt

#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import TensorDataset, DataLoader
#import pytorch_lightning as pl
#
#class DataModule(pl.LightningDataModule):
#    def __init__(self, batch_size, X_train, Y_train, X_test, Y_test):
#        self.batch_size = batch_size
#        self.X_train = X_train
#        self.Y_train = Y_train
#        self.X_test = X_test
#        self.Y_test = Y_test
#
#    def train_dataloader(self):
#        train_dataset = TensorDataset(torch.tensor(self.X_train).float(),torch.tensor(self.Y_train,dtype=torch.long).float())
#        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) 
#
#    def val_dataloader(self):
#        val_dataset = TensorDataset(torch.tensor(self.X_test).float(),torch.tensor(self.Y_test,dtype=torch.long).float())
#        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) 
#
#class Lit_Dense_Network(pl.LightningModule):
#    def __init__(self):
#        super().__init__()
#        self.fc1 = nn.Linear(23, 10)
#        self.b1 = nn.BatchNorm1d(10)
#        self.fc2 = nn.Linear(10, 5)
#        self.b2 = nn.BatchNorm1d(5)
#        self.fc3 = nn.Linear(5,1)
#
#    def forward(self, x):
#        if len(x.shape)==1:
#            x = torch.reshape(x,[1,x.shape[0]])
#        x = F.relu(self.fc1(x))
#        x = self.b1(x)
#        x = F.relu(self.fc2(x))
#        x = self.b2(x)
#        x = F.sigmoid(self.fc3(x))
#        return x
#        
#    def training_step(self, batch, batch_idx):
#        x, y = batch
#        y_pred = self.forward(x)
#        loss = F.binary_cross_entropy(y_pred, y)
#        self.log('train_loss', loss)
#        return loss
#
#    def validation_step(self, batch, batch_idx):
#        x, y = batch
#        y_pred = self.forward(x)
#        loss = F.binary_cross_entropy(y_pred, y)
#        self.log('val_loss', loss)
#        return loss
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#        return optimizer

#from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from pytorch_lightning.callbacks.progress import TQDMProgressBar
#from pytorch_lightning.loggers import TensorBoardLogger
#
#net = Lit_Dense_Network()
#BATCH_SIZE = 1024
#dm = DataModule(BATCH_SIZE, X_train, Y_train.reshape(-1,1), X_test, Y_test.reshape(-1,1))
#bar = TQDMProgressBar(refresh_rate=20)
#es = EarlyStopping(monitor="val_loss", mode="min", patience=5)
#logger = TensorBoardLogger("lightning_logs", name="FICO_model", version=1)
#trainer = pl.Trainer(callbacks=[bar, es], 
#                     max_epochs=100, 
#                     log_every_n_steps=10, 
#                     logger=logger, 
#                     enable_checkpointing=False)
#
#trainer.fit(net, dm)
##trainer.save_checkpoint("fico_black_box.ckpt")

#clf_pt = Lit_Dense_Network()
#clf_pt.eval()
#clf_pt = clf_pt.load_from_checkpoint(checkpoint_path="./BlackBoxes/fico_pt_nn.ckpt")

#from sklearn.metrics import accuracy_score
#
#print(accuracy_score(np.round(clf_pt(torch.tensor(X_train).float()).detach().numpy()),Y_train))
#print(accuracy_score(np.round(clf_pt(torch.tensor(X_test).float()).detach().numpy()),Y_test))

# # Latent Space

for black_box in ['xgb', 'svc', 'nn']:
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if black_box=='xgb':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_xgb.predict_proba(x)[:,1].ravel()
            else: return clf_xgb.predict(x).ravel().ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
    elif black_box=='rf':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_rf.predict_proba(x)[:,1].ravel()
            else: return clf_rf.predict(x).ravel().ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
    elif black_box=='svc':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_svc.predict_proba(x)[:,1].ravel()
            else: return clf_svc.predict(x).ravel().ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
    elif black_box=='nn':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_nn.predict(x).ravel()
            else: return np.round(clf_nn.predict(x).ravel()).astype(int).ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)

    #Latent 

    X_train = np.hstack((X_train,y_train_pred.reshape(-1,1)))
    X_test = np.hstack((X_test,y_test_pred.reshape(-1,1)))

    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    latent_dim = 7
    alpha = 1
    batch_size = 512
    sigma = 1
    max_epochs = 1000
    early_stopping = 3
    learning_rate = 1e-3

    similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')

    def compute_similarity_X(X, sigma, idx_cat=None, alpha=1):
        D_features = torch.cdist(X[:,:-1],X[:,:-1])
        D_class = torch.cdist(X[:,-1].reshape(-1,1),X[:,-1].reshape(-1,1))
        D = D_features + alpha * D_class
        M = torch.exp((-D**2)/(2*sigma**2))
        return M / (torch.ones([M.shape[0],M.shape[1]])*(torch.sum(M, axis = 0)-1)).transpose(0,1)

    def compute_similarity_Z(Z, sigma):
        D = torch.cdist(Z,Z)
        M = torch.exp((-D**2)/(2*sigma**2))
        return M / (torch.ones([M.shape[0],M.shape[1]])*(torch.sum(M, axis = 0)-1)).transpose(0,1)

    def loss_function(X, Z, sigma=1, idx_cat=None ):
        Sx = compute_similarity_X(X, sigma, idx_cat, alpha=alpha)
        Sz = compute_similarity_Z(Z, sigma=1)
        loss = similarity_KLD(torch.log(Sx), Sz)
        return loss

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

    # Create Model
    model = LinearModel(X_train.shape[1], latent_dim=latent_dim)

    train_dataset = TensorDataset(torch.tensor(X_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_dataset = TensorDataset(torch.tensor(X_test).float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    def check_and_clear(dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        else:
            os.system('rm -r ' + dir_name)
            os.mkdir(dir_name)

    check_and_clear('./models/weights')

    model_params = list(model.parameters())
    optimizer = torch.optim.Adam(model_params, lr=learning_rate)

    # record training process
    epoch_train_losses = []
    epoch_test_losses = []

    #validation parameters
    epoch = 1
    best = np.inf

    # progress bar
    pbar = tqdm(bar_format="{postfix[0]} {postfix[1][value]:03d} {postfix[2]} {postfix[3][value]:.5f} {postfix[4]} {postfix[5][value]:.5f} {postfix[6]} {postfix[7][value]:d}",
                postfix=["Epoch:", {'value':0}, "Train Sim Loss", {'value':0}, "Test Sim Loss", {'value':0}, "Early Stopping", {"value":0}])

    # start training
    while epoch <= max_epochs:

        # ------- TRAIN ------- #
        # set model as training mode
        model.train()
        batch_loss = []

        for batch, (X_batch,) in enumerate(train_loader):
            optimizer.zero_grad()
            Z_batch = model(X_batch)  #
            loss  = loss_function(X_batch, Z_batch, sigma) 
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        # save result
        epoch_train_losses.append(np.mean(batch_loss))
        pbar.postfix[3]["value"] = np.mean(batch_loss)

        # -------- VALIDATION --------

        # set model as testing mode
        model.eval()
        batch_loss = []

        with torch.no_grad():
            for batch, (X_batch,) in enumerate(test_loader):
                Z_batch = model(X_batch)
                loss = loss_function(X_batch, Z_batch, sigma)
                batch_loss.append(loss.item())

    # save information
        epoch_test_losses.append(np.mean(batch_loss))
        pbar.postfix[5]["value"] = np.mean(batch_loss)
        pbar.postfix[1]["value"] = epoch

        if epoch_test_losses[-1] < best:
            wait = 0
            best = epoch_test_losses[-1]
            best_epoch = epoch
            torch.save(model.state_dict(), f'./models/weights/LinearTransparent_fico.pt')
        else:
            wait += 1
        pbar.postfix[7]["value"] = wait
        if wait == early_stopping:
            break    
        epoch += 1
        pbar.update()

    model.load_state_dict(torch.load(f'./models/weights/LinearTransparent_fico.pt'))
    with torch.no_grad():
        model.eval()
        Z_train = model(torch.tensor(X_train).float()).cpu().detach().numpy()
        Z_test = model(torch.tensor(X_test).float()).cpu().detach().numpy()

    torch.save(model.state_dict(), f'./models/fico_latent_{black_box}_{latent_dim}_{str(alpha).replace(".", "")}.pt')

    model.load_state_dict(torch.load(f'./models/fico_latent_{black_box}_{latent_dim}_{str(alpha).replace(".", "")}.pt'))
    with torch.no_grad():
        model.eval()
        Z_train = model(torch.tensor(X_train).float()).cpu().detach().numpy()
        Z_test = model(torch.tensor(X_test).float()).cpu().detach().numpy()

    w = model.fc1.weight.detach().numpy()
    b = model.fc1.bias.detach().numpy()
    y_contrib = model.fc1.weight.detach().numpy()[:,-1]

    def compute_cf(q, indexes):
        q_pred = predict(q[:-1].reshape(1,-1),return_proba=True)
        q_cf = q.copy()
        q_cf_preds = []
        q_cf_preds.append(float(predict(q_cf[:-1].reshape(1,-1),return_proba=True)))
        if q_pred > 0.5:
            m = -0.1
        else:
            m = +0.1
        while np.round(q_pred) == np.round(q_cf_preds[-1]):
            v = np.array(model(torch.tensor(q_cf).float()).detach().numpy()+m*y_contrib)
            c_l = [v[l] - np.sum(q_cf*w[l,:]) - b[l] for l in range(latent_dim)]
            M = []
            for l in range(latent_dim):
                M.append([np.sum(w[k,indexes]*w[l,indexes]) for k in range(latent_dim)])
            M = np.vstack(M)
            lambda_k = np.linalg.solve(M, c_l)
            delta_i = [np.sum(lambda_k*w[:,i]) for i in indexes]
            #print(delta_i)
            q_cf[indexes] += delta_i
            q_cf = np.clip(q_cf,-1,1)
            if float(predict(q_cf[:-1].reshape(1,-1),return_proba=True)) in q_cf_preds:
                return q_cf
            q_cf_preds.append(float(predict(q_cf[:-1].reshape(1,-1),return_proba=True)))
            q_cf[-1] = q_cf_preds[-1]
        return q_cf

    from itertools import combinations
    from scipy.spatial.distance import cdist

    d_dist = []
    d_impl = []
    d_count = []
    d_adv = []
    num = []
    div_dist = []
    div_count = []

    for idx in tqdm(range(100)):
        q = X_test[idx,:].copy()
        q_pred = predict(q[:-1].reshape(1,-1),return_proba=False)
        q_cfs = []
        l_i = []
        l_f = []

        for indexes in list(combinations(list(range(24)),1)):    
            q_cf = compute_cf(q, list(indexes))
            q_cf_pred = predict(q_cf[:-1].reshape(1,-1),return_proba=True)
            if q_pred:
                if q_cf_pred<0.5:
                    q_cfs.append(q_cf)
            else:
                if q_cf_pred>0.5:
                    q_cfs.append(q_cf) 

        for indexes in list(combinations(list(range(24)),2)):    
            q_cf = compute_cf(q, list(indexes))
            q_cf_pred = predict(q_cf[:-1].reshape(1,-1),return_proba=True)
            if q_pred:
                if q_cf_pred<0.5:
                    q_cfs.append(q_cf)
            else:
                if q_cf_pred>0.5:
                   q_cfs.append(q_cf) 
            l_i.append([list(indexes),q_cf_pred])
        r = np.argsort(np.stack(np.array(l_i,dtype=object)[:,1]).ravel())[-10:]
        l_i = np.array(l_i,dtype=object)[r,0]

        while len(l_i[0])<4:
            for e in l_i:
                for i in list(np.delete(range(24),e)):
                    q_cf = compute_cf(q, e+[i])
                    q_cf_pred = predict(q_cf[:-1].reshape(1,-1),return_proba=True)
                    if q_pred:
                        if q_cf_pred<0.5:
                            q_cfs.append(q_cf)
                    else:
                        if q_cf_pred>0.5:
                            q_cfs.append(q_cf) 
                    l_f.append([e+[i],q_cf_pred])
            r = np.argsort(np.stack(np.array(l_f,dtype=object)[:,1]).ravel())[-10:]
            l_f = np.array(l_f,dtype=object)[r,0]
            l_i = l_f.copy()
            l_f = []
        
        if len(q_cfs)<1:
            continue
        else:
            q_cfs = np.vstack(q_cfs)
            d_dist.append(np.min(cdist(q_cfs[:,:-1],q[:-1].reshape(1,-1))))
            d_impl.append(np.min(cdist(q_cfs[:,:-1],X_train[:,:-1])))
            d_count.append(np.min(np.sum(q_cfs[:,:-1]!=q[:-1],axis=1)))
            r = np.argsort(cdist(q_cfs[:,:-1],X_train[:,:-1]),axis=1)[:,:10]
            d_adv.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==q_pred) for i in range(q_cfs.shape[0])])))
            num.append(len(q_cfs))
            div_dist.append(1/(q_cfs.shape[0]**2)*np.sum(cdist(q_cfs[:,:-1],q_cfs[:,:-1])))
            div_count.append(23/(q_cfs.shape[0]**2)*np.sum(cdist(q_cfs[:,:-1], q_cfs[:,:-1],metric='hamming')))

    with open('./results/fico_results.txt','a') as f:
        f.write('latent '+black_box+'\n')
        f.write(str(np.round(np.mean(d_dist),5))+','+str(np.round(np.std(d_dist),5))+'\n')
        f.write(str(np.round(np.mean(d_count),5))+','+str(np.round(np.std(d_count),5))+'\n')
        f.write(str(np.round(np.mean(d_impl),5))+','+str(np.round(np.std(d_impl),5))+'\n')
        f.write(str(np.round(np.mean(d_adv),5))+','+str(np.round(np.std(d_adv),5))+'\n')
        f.write(str(np.round(np.mean(num),5))+','+str(np.round(np.std(num),5))+'\n')
        f.write(str(np.round(np.mean(div_dist),5))+','+str(np.round(np.std(div_dist),5))+'\n')
        f.write(str(np.round(np.mean(div_count),5))+','+str(np.round(np.std(div_count),5))+'\n')
        f.write('success_rate: '+str(len(d_dist)/100)+'\n')

    
    #print('d_dist: \t',    np.round(np.mean(d_dist),5),   np.round(np.std(d_dist),5))
    #print('d_count: \t',   np.round(np.mean(d_count),5),  np.round(np.std(d_count),5))
    #print('implicity: \t', np.round(np.mean(d_impl),5),   np.round(np.std(d_impl),5))
    #print('number: \t',    np.round(np.mean(num),5),'\t', np.round(np.std(num),5))
    #print('div_dist: \t',  np.round(np.mean(div_dist),5), np.round(np.std(div_dist),5))
    #print('div_count: \t', np.round(np.mean(div_count),5),np.round(np.std(div_count),5))
    #print('success_rate: \t', len(d_dist)/10)

    # Growing Spheres

    from growingspheres import counterfactuals as cf
    from scipy.spatial.distance import hamming, euclidean

    d_dist_GS = []
    d_count_GS = []
    d_impl_GS = []
    d_adv_GS = []

    for idx in tqdm(range(100)):
        q = X_test[idx,:-1].reshape(1,-1).copy()
        pred = int(predict(q))
        CF = cf.CounterfactualExplanation(q, predict, method='GS')
        CF.fit(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False)
        q_cf_GS = CF.enemy
        d_dist_GS.append(euclidean(q_cf_GS,q))
        d_count_GS.append(np.sum(q_cf_GS!=q))
        d_impl_GS.append(np.min(cdist(q_cf_GS.reshape(1,-1),X_train[:,:-1])))
        r = np.argsort(cdist(q_cf_GS.reshape(1,-1),X_train[:,:-1]),axis=1)[:,:10]
        d_adv_GS.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==pred) for i in range(q_cf_GS.reshape(1,-1).shape[0])])))

    with open('./results/fico_results.txt','a') as f:
        f.write('GSG '+black_box+'\n')
        f.write(str(np.round(np.mean(d_dist_GS),5))+','+str(np.round(np.std(d_dist_GS),5))+'\n')
        f.write(str(np.round(np.mean(d_count_GS),5))+','+str(np.round(np.std(d_count_GS),5))+'\n')
        f.write(str(np.round(np.mean(d_impl_GS),5))+','+str(np.round(np.std(d_impl_GS),5))+'\n')
        f.write(str(np.round(np.mean(d_adv_GS),5))+','+str(np.round(np.std(d_adv_GS),5))+'\n')

    #print('d_dist: \t',    np.round(np.mean(d_dist_GS),5),   np.round(np.std(d_dist_GS),5))
    #print('d_count: \t',   np.round(np.mean(d_count_GS),5),  np.round(np.std(d_count_GS),5))
    #print('implicity: \t', np.round(np.mean(d_impl_GS),5),   np.round(np.std(d_impl_GS),5))

    ## # FAT

    #import fatf.utils.data.datasets as fatf_datasets
    #import fatf.utils.models as fatf_models
    #import fatf.transparency.predictions.counterfactuals as fatf_cf

    ## Create a Counterfactual Explainer
    #if black_box =='xgb':
    #    cf_explainer = fatf_cf.CounterfactualExplainer(
    #        model=clf_xgb,
    #        dataset=X_train[:,:-1],
    #        categorical_indices=[],
    #        default_numerical_step_size=0.1)
    #elif black_box == 'rf':
    #    cf_explainer = fatf_cf.CounterfactualExplainer(
    #        model=clf_rf,
    #        dataset=X_train[:,:-1],
    #        categorical_indices=[],
    #        default_numerical_step_size=0.1)
    #elif black_box == 'svc':
    #    cf_explainer = fatf_cf.CounterfactualExplainer(
    #        model=clf_svc,
    #        dataset=X_train[:,:-1],
    #        categorical_indices=[],
    #        default_numerical_step_size=0.1)
    #elif black_box == 'nn':
    #    class FAT_wrapper():
    #        def __init__(self,model):
    #            self.model = model

    #        def predict(self,x):
    #            return np.round(self.model.predict(x).ravel()).astype(int).ravel()

    #        def fit(self, x, y):
    #            pass

    #        def predict_proba(self,x):
    #            return self.model.predict(x).ravel()
    #    fat_model = FAT_wrapper(clf_nn)
    #    cf_explainer = fatf_cf.CounterfactualExplainer(
    #        model=fat_model,
    #        dataset=X_train[:,:-1],
    #        categorical_indices=[],
    #        default_numerical_step_size=0.1)

    #d_dist_fat = []
    #d_count_fat = []
    #d_impl_fat = []
    #num_fat = []
    #div_dist_fat = []
    #div_count_fat = []

    #for i in tqdm(range(100)):
    #    q = X_test[i,:-1].copy()
    #    dp_1_cf_tuple = cf_explainer.explain_instance(q)
    #    q_cfs_fat, _, _ = dp_1_cf_tuple
    #    if len(q_cfs_fat) > 0:
    #        d_dist_fat.append(np.min(cdist(q_cfs_fat,q.reshape(1,-1))))
    #        #d_count_fat.append(1/(q_cfs_fat.shape[1]*q_cfs_fat.shape[0])*np.sum(q_cfs_fat!=q.reshape(1,-1)))
    #        d_count_fat.append(np.min(np.sum(q_cfs_fat!=q.reshape(1,-1),axis=1)))
    #        d_impl_fat.append(np.min(cdist(q_cfs_fat,X_train)))
    #        num_fat.append(len(q_cfs_fat))
    #        div_dist_fat.append(1/(q_cfs_fat.shape[0]**2)*np.sum(cdist(q_cfs_fat, q_cfs_fat)))
    #        div_count_fat.append(23/(q_cfs_fat.shape[0]**2)*np.sum(cdist(q_cfs_fat, q_cfs_fat, metric='hamming')))

    #with open('./results/fico_results.txt','a') as f:
    #    f.write('fat '+black_box+'\n')
    #    #f.write(str(np.round(np.mean(d_dist_fat),5))+','+str(np.round(np.std(d_dist_fat),5))+'\n')
    #    f.write(str(np.round(np.mean(d_count_fat),5))+','+str(np.round(np.std(d_count_fat),5))+'\n')
    #    #f.write(str(np.round(np.mean(d_impl_fat),5))+','+str(np.round(np.std(d_impl_fat),5))+'\n')
    #    #f.write(str(np.round(np.mean(num_fat),5))+','+str(np.round(np.std(num_fat),5))+'\n')
    #    #f.write(str(np.round(np.mean(div_dist_fat),5))+','+str(np.round(np.std(div_dist_fat),5))+'\n')
    #    #f.write(str(np.round(np.mean(div_count_fat),5))+','+str(np.round(np.std(div_count_fat),5))+'\n')
    #    #f.write('success_rate: '+str(len(d_dist_fat)/100)+'\n')

    ##print('d_dist: \t',    np.round(np.mean(d_dist_fat),5),   np.round(np.std(d_dist_fat),5))
    ##print('d_count: \t',   np.round(np.mean(d_count_fat),5),  np.round(np.std(d_count_fat),5))
    ##print('implicity: \t', np.round(np.mean(d_impl_fat),5),   np.round(np.std(d_impl_fat),5))
    ##print('number: \t',    np.round(np.mean(num_fat),5),'\t', np.round(np.std(num_fat),5))
    ##print('div_dist: \t',  np.round(np.mean(div_dist_fat),5), np.round(np.std(div_dist_fat),5))
    ##print('div_count: \t', np.round(np.mean(div_count_fat),5),np.round(np.std(div_count_fat),5))

    ##print('success_rate: \t', len(d_dist_fat)/10)

    # Watcher
    
    from scipy.spatial.distance import cdist, euclidean
    from scipy.optimize import minimize
    from scipy import stats

    d_dist_watch = []
    d_count_watch = []
    d_impl_watch = []
    d_adv_watch = []

    for i in tqdm(range(100)):
        # initial conditions
        lamda = 0.1 
        x0 = np.zeros([1,X_train.shape[1]-1]) # initial guess for cf
        q = X_test[i:i+1,:-1].copy()
        pred = predict(q,return_proba=False)

        def dist_mad(cf, eg):
            manhat = [cdist(eg.T, cf.reshape(1,-1).T ,metric='cityblock')[i][i] for i in range(len(eg.T))]
            #mad = stats.median_absolute_deviation(X_train)
            return sum(manhat)

        def loss_function_mad(x_dash):
            target = 1-pred
            if target == 0:
                L = lamda*(predict(x_dash.reshape(1,-1),return_proba=True)-1)**2 + dist_mad(x_dash.reshape(1,-1), q)
            else:
                L = lamda*(1-predict(x_dash.reshape(1,-1),return_proba=True)-1)**2 + dist_mad(x_dash.reshape(1,-1), q) 
            return L

        res = minimize(loss_function_mad, x0, method='nelder-mead', options={'maxiter':1000, 'xatol': 1e-8})
        cf = res.x.reshape(1, -1)

        i = 0
        r = 1
        while pred == predict(cf):
            lamda += 0.1
            x0 = cf 
            res = minimize(loss_function_mad, x0, method='nelder-mead', options={'maxiter':1000, 'xatol': 1e-8})
            cf = res.x.reshape(1, -1)
            i += 1
            if i == 100:
                r = 0
                break

        if r == 1:
            d_dist_watch.append(euclidean(cf,q))
            d_count_watch.append(1/(cf.shape[0])*np.sum(cf!=q))
            d_impl_watch.append(np.min(cdist(cf.reshape(1,-1),X_train[:,:-1])))
            r = np.argsort(cdist(cf,X_train[:,:-1]),axis=1)[:,:10]
            d_adv_watch.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==pred) for i in range(cf.shape[0])])))

    with open('./results/fico_results.txt','a') as f:
        f.write('Watcher '+black_box+'\n')
        f.write(str(np.round(np.mean(d_dist_watch),5))+','+str(np.round(np.std(d_dist_watch),5))+'\n')
        f.write(str(np.round(np.mean(d_count_watch),5))+','+str(np.round(np.std(d_count_watch),5))+'\n')
        f.write(str(np.round(np.mean(d_impl_watch),5))+','+str(np.round(np.std(d_impl_watch),5))+'\n')
        f.write(str(np.round(np.mean(d_adv_watch),5))+','+str(np.round(np.std(d_adv_watch),5))+'\n')
        f.write('success_rate: '+str(len(d_dist_watch)/100)+'\n')







