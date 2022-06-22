#!/usr/bin/env python
# coding: utf-8

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

rnd = 42
np.random.seed(rnd)
torch.manual_seed(rnd)

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict

def get_features_map(feature_names, real_feature_names):
    features_map = defaultdict(dict)
    i = 0
    j = 0

    while i < len(feature_names) and j < len(real_feature_names):
        if feature_names[i] == real_feature_names[j]:
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
            j += 1
        elif feature_names[i].startswith(real_feature_names[j]):
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
        else:
            j += 1
    return features_map

def get_real_feature_names(rdf, numeric_columns, class_name):
    if isinstance(class_name, list):
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c not in class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c not in class_name]
    else:
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c != class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c != class_name]
    return real_feature_names

def one_hot_encoding(df, class_name):
    if not isinstance(class_name, list):
        dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep='=')
        class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
        dfY = df[class_name].map(class_name_map)
        df = pd.concat([dfX, dfY], axis=1)
        df =df.reindex(dfX.index)
        feature_names = list(dfX.columns)
        class_values = sorted(class_name_map)
    else: # isinstance(class_name, list)
        dfX = pd.get_dummies(df[[c for c in df.columns if c not in class_name]], prefix_sep='=')
        # class_name_map = {v: k for k, v in enumerate(sorted(class_name))}
        class_values = sorted(class_name)
        dfY = df[class_values]
        df = pd.concat([dfX, dfY], axis=1)
        df = df.reindex(dfX.index)
        feature_names = list(dfX.columns)
    return df, feature_names, class_values

def remove_missing_values(df):
    for column_name, nbr_missing in df.isna().sum().to_dict().items():
        if nbr_missing > 0:
            if column_name in df._get_numeric_data().columns:
                mean = df[column_name].mean()
                df[column_name].fillna(mean, inplace=True)
            else:
                mode = df[column_name].mode().values[0]
                df[column_name].fillna(mode, inplace=True)
    return df

def get_numeric_columns(df):
    numeric_columns = list(df._get_numeric_data().columns)
    return numeric_columns

class_name = 'default'
# Load and transform dataset 
df = pd.read_csv('./data/german_credit.csv', skipinitialspace=True, na_values='?', keep_default_na=True)
df.columns = [c.replace('=', '') for c in df.columns]

df = remove_missing_values(df)
numeric_columns = get_numeric_columns(df)
rdf = df
df, feature_names, class_values = one_hot_encoding(df, class_name)
real_feature_names = get_real_feature_names(rdf, numeric_columns, class_name)
rdf = rdf[real_feature_names + (class_values if isinstance(class_name, list) else [class_name])]
features_map = get_features_map(feature_names, real_feature_names)
std = MinMaxScaler(feature_range=(-1,1))
df.iloc[:,[0,1,4]] = std.fit_transform(df.values[:,[0,1,4]])
hot_enc = OneHotEncoder(handle_unknown='ignore')
for i in [2,3,5,6]:
    df.iloc[:,i] = [df.columns[i]+'='+df.iloc[:,i].values.astype(str)[j] for j in range(len(df))] 
hot_enc.fit(df.iloc[:,[2,3,5,6]])
df[np.hstack(hot_enc.categories_).tolist()]=hot_enc.transform(df.iloc[:,[2,3,5,6]]).toarray().astype(int)
df = df.drop(list(df.columns[[2,3,5,6]]), axis=1, inplace=False)
y = df['default'].astype(int)
df.drop(['default'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.values, y, test_size=0.2, stratify=y, random_state=rnd)

import boto3
import smtplib, ssl

AWS_REGION = "eu-west-1"
S3_BUCKET_NAME = "transparentfeaturereduction"

s3_client = boto3.client("s3", region_name=AWS_REGION)

def upload_files(file_name, bucket, object_name=None, args=None):
    if object_name is None:
        object_name = file_name
    s3_client.upload_file(file_name, bucket, object_name, ExtraArgs=args)
    #print(f"'{file_name}' has been uploaded to '{S3_BUCKET_NAME}'")

class Mail:
    def __init__(self):
        self.port = 465
        self.smtp_server_domain_name = "smtp.gmail.com"
        self.sender_mail = "francesco.bodria@sns.it"
        self.password = "ysjrwdfhehikunui"
    def send(self, emails, subject, content):
        ssl_context = ssl.create_default_context()
        service = smtplib.SMTP_SSL(self.smtp_server_domain_name, self.port, context=ssl_context)
        service.login(self.sender_mail, self.password)
        for email in emails:
            result = service.sendmail(self.sender_mail, email, f"Subject: {subject}\n{content}")
        service.quit()

### XGBOOST

from xgboost import XGBClassifier
clf_xgb = XGBClassifier(n_estimators=10, reg_lambda=1, use_label_encoder=False, eval_metric='logloss')
clf_xgb.fit(X_train, y_train)
pickle.dump(clf_xgb,open('./blackboxes/german_xgboost.p','wb'))

clf_xgb = pickle.load(open('./blackboxes/german_xgboost.p','rb'))
def predict(x, return_proba=False):
    if return_proba:
        return clf_xgb.predict_proba(x)[:,1].ravel()
    else: return clf_xgb.predict(x).ravel().ravel()
y_train_pred = predict(X_train)
y_test_pred = predict(X_test)
print('XGBOOST')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

### RF
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(max_depth=7,random_state=rnd)
clf_rf.fit(X_train, y_train)

pickle.dump(clf_rf,open('./blackboxes/german_rf.p','wb'))

clf_rf = pickle.load(open('./blackboxes/german_rf.p','rb'))

def predict(x, return_proba=False):
    if return_proba:
        return clf_rf.predict_proba(x)[:,1].ravel()
    else: return clf_rf.predict(x).ravel().ravel()

y_test_pred = predict(X_test, return_proba=True)
y_train_pred = predict(X_train, return_proba=True)
print('RF')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

### SVC

from sklearn.svm import SVC
clf_svc = SVC(gamma='auto', probability=True)
clf_svc.fit(X_train, y_train)
pickle.dump(clf_svc,open('./blackboxes/german_svc.p','wb'))

clf_svc = pickle.load(open('./blackboxes/german_svc.p','rb'))

def predict(x, return_proba=False):
    if return_proba:
        return clf_svc.predict_proba(x)[:,1].ravel()
    else: return clf_svc.predict(x).ravel().ravel()

y_train_pred = predict(X_train, return_proba=True)
y_test_pred = predict(X_test, return_proba=True)
print('SVC')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

# ### NN tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

BATCH_SIZE = 1024
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(2048).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

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
    epochs=10000,
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
clf_nn.save_weights('./blackboxes/german_tf_nn')

from sklearn.metrics import accuracy_score
clf_nn.load_weights('./blackboxes/german_tf_nn')
clf_nn.trainable = False

def predict(x, return_proba=False):
    if return_proba:
        return clf_nn.predict(x).ravel()
    else: return np.round(clf_nn.predict(x).ravel()).astype(int).ravel()

print('NN')
print(accuracy_score(np.round(predict(X_train, return_proba = True)),y_train))
print(accuracy_score(np.round(predict(X_test, return_proba = True)),y_test))
print('---------------')

pbar_model = tqdm(total=4, desc='model')
pbar_latent = tqdm(total=10, desc='latent')

for black_box in ['xgb','rf','svc','nn']:

    X_train, X_test, y_train, y_test = train_test_split(df.values, y, test_size=0.2, stratify=y, random_state=rnd)

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

    # # Latent Space

    X_train = np.hstack((X_train,y_train_pred.reshape(-1,1)))
    X_test = np.hstack((X_test,y_test_pred.reshape(-1,1)))

    for latent_dim in [2, 3, 5, 10, 20, 30, 40, 50, 60, 71]:

        batch_size = 1024
        sigma = 1
        max_epochs = 10000
        early_stopping = 3
        learning_rate = 1e-3
        idx_cat = np.arange(3,71,1).tolist()

        similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')

        def compute_similarity_Z(Z, sigma):
            D = 1 - F.cosine_similarity(Z[:, None, :], Z[None, :, :], dim=-1)
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
                D_cont = 1 - F.cosine_similarity(X[:, None, :], X[None, :, :], dim=-1)
                D_cat = torch.cdist(X_cat, X_cat, p=0)/h
                D = h/m * D_cat + ((m-h)/m) * D_cont + D_class
            else:
                D_features = 1 - F.cosine_similarity(X[:, None, :], X[None, :, :], dim=-1) 
                D = D_features + D_class
            M = torch.exp((-D**2)/(2*sigma**2))
            return M / (torch.ones([M.shape[0],M.shape[1]])*(torch.sum(M, axis = 0)-1)).transpose(0,1)

        def loss_function(X, Z, idx_cat, sigma=1):
            Sx = compute_similarity_X(X, sigma, idx_cat)
            Sz = compute_similarity_Z(Z, sigma)
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
                loss  = loss_function(X_batch, Z_batch, idx_cat, sigma) 
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
                    loss = loss_function(X_batch, Z_batch, idx_cat, sigma)
                    batch_loss.append(loss.item())

            # save information
            epoch_test_losses.append(np.mean(batch_loss))
            #pbar.postfix[5]["value"] = np.mean(batch_loss)
            #pbar.postfix[1]["value"] = epoch

            if epoch_test_losses[-1] < best:
                wait = 0
                best = epoch_test_losses[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), f'./models/weights/LinearTransparent_german.pt')
            else:
                wait += 1
            #pbar.postfix[7]["value"] = wait
            if wait == early_stopping:
                break    
            epoch += 1
            #pbar.update()

        model.load_state_dict(torch.load(f'./models/weights/LinearTransparent_german.pt'))

        with torch.no_grad():
            model.eval()
            Z_train = model(torch.tensor(X_train).float()).cpu().detach().numpy()
            Z_test = model(torch.tensor(X_test).float()).cpu().detach().numpy()

        torch.save(model.state_dict(), f'./models/german_latent_{black_box}_{latent_dim}.pt')
        upload_files(f'/home/ubuntu/posthoc-similaryclass-counterfactuals/models/german_latent_{black_box}_{latent_dim}.pt', S3_BUCKET_NAME)

        pbar_latent.update(1)
    mails = ['francesco.bodria@sns.it']
    subject = 'risultati esperimento'
    content = f'{black_box} completato'
    mail = Mail()
    mail.send(mails, subject, content)
    pbar_model.update(1)       

pbar_latent.close()
pbar_model.close()
