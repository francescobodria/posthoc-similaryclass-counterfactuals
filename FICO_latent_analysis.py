import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os

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

# # BlackBoxes

# ### XGBOOST

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

# ### Random Forest

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

# ### SVC

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

# ### NN tf

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

pbar_model = tqdm(total=4, desc='model')
pbar_latent = tqdm(total=7, desc='latent')
pbar_alpha = tqdm(total=10, desc='alpha')

for black_box in ['xgb','rf','svc','nn']:

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

    # Latent Space

    X_train = np.hstack((X_train,y_train_pred.reshape(-1,1)))
    X_test = np.hstack((X_test,y_test_pred.reshape(-1,1)))

    for latent_dim in [2, 3, 4, 6, 9, 12, 15, 18, 21, 24]:
        for alpha in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5]:
            batch_size = 1024
            sigma = 1
            max_epochs = 1000
            early_stopping = 3
            learning_rate = 1e-3

            similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')

            def compute_similarity_Z(Z, sigma):
                D = torch.cdist(Z,Z)
                M = torch.exp((-D**2)/(2*sigma**2))
                return M / (torch.ones([M.shape[0],M.shape[1]])*(torch.sum(M, axis = 0)-1)).transpose(0,1)

            def compute_similarity_X(X, sigma, alpha, idx_cat=None):
                D_class = torch.cdist(X[:,-1].reshape(-1,1),X[:,-1].reshape(-1,1))
                X = X[:, :-1]
                if idx_cat:
                    X_cat = X[:, idx_cat]
                    X_cont = X[:, np.delete(range(X.shape[1]),idx_cat)]
                    h = X_cat.shape[1]
                    m = X.shape[1]
                    D_cont = torch.cdist(X_cont,X_cont)
                    D_cat = torch.cdist(X_cat, X_cat, p=0)/h
                    D = h/m * D_cat + ((m-h)/m) * D_cont + alpha * D_class
                else:
                    D = torch.cdist(X,X) + alpha * D_class
                M = torch.exp((-D**2)/(2*sigma**2))
                return M / (torch.ones([M.shape[0],M.shape[1]])*(torch.sum(M, axis = 0)-1)).transpose(0,1)

            def loss_function(X, Z, idx_cat, sigma=1):
                Sx = compute_similarity_X(X, sigma, alpha)
                Sz = compute_similarity_Z(Z, 1)
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
            #pbar = tqdm(bar_format="{postfix[0]} {postfix[1][value]:03d} {postfix[2]} {postfix[3][value]:.5f} {postfix[4]} {postfix[5][value]:.5f} {postfix[6]} {postfix[7][value]:d}",
            #            postfix=["Epoch:", {'value':0}, "Train Sim Loss", {'value':0}, "Test Sim Loss", {'value':0}, "Early Stopping", {"value":0}])

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
                #pbar.postfix[3]["value"] = np.mean(batch_loss)

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
                #pbar.postfix[5]["value"] = np.mean(batch_loss)
                #pbar.postfix[1]["value"] = epoch

                if epoch_test_losses[-1] < best:
                    wait = 0
                    best = epoch_test_losses[-1]
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'./models/weights/LinearTransparent_fico.pt')
                else:
                    wait += 1
                #pbar.postfix[7]["value"] = wait
                if wait == early_stopping:
                    break    
                epoch += 1
                #pbar.update()

            model.load_state_dict(torch.load(f'./models/weights/LinearTransparent_fico.pt'))
            with torch.no_grad():
                model.eval()
                Z_train = model(torch.tensor(X_train).float()).cpu().detach().numpy()
                Z_test = model(torch.tensor(X_test).float()).cpu().detach().numpy()

            torch.save(model.state_dict(), f'./models/fico_latent_{black_box}_{latent_dim}_{str(alpha).replace(".", "")}.pt')
            pbar_alpha.update(1)
        pbar_latent.update(1)
    pbar_model.update(1)       

pbar_alpha.close()
pbar_latent.close()
pbar_model.close()


