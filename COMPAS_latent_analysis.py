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

rnd = 384
np.random.seed(rnd)
torch.manual_seed(rnd)

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

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/compas-scores-two-years.csv')
df = df[(df.days_b_screening_arrest>=-30)*(df.days_b_screening_arrest<=30)]
df = df.drop(['name','id','dob','age_cat','c_case_number','compas_screening_date','c_jail_in','c_jail_out','c_offense_date',
              'c_arrest_date','r_case_number','r_charge_degree','r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in',
              'r_jail_out','violent_recid','vr_case_number','vr_charge_degree','vr_offense_date','vr_charge_desc','v_screening_date',
              'in_custody','out_custody','v_type_of_assessment','type_of_assessment','screening_date','is_recid','first','last','c_charge_desc'],axis=1,inplace=False)
cat_columns = ['sex','race','c_charge_degree','score_text','v_score_text','event','is_violent_recid']
idx = np.ravel([np.where(np.array(list(df.columns)) == i) for i in cat_columns])
df = df.loc[:,cat_columns+list(np.delete(df.columns, idx))]
df.loc[:,'event'] = df.event.replace({0:'event_0',1:'event_1'})
df.loc[:,'is_violent_recid'] = df.is_violent_recid.replace({0:'not_violent_recid',1:'violent_recid'})
df.loc[:,'c_charge_degree'] = df.c_charge_degree.replace({'M':'charge_degree_M','F':'charge_degree_F'})
df.loc[:,'score_text'] = df.score_text.replace({'Low':'score_text_low','Medium':'score_text_medium','High':'score_text_high'})
df.loc[:,'v_score_text'] = df.v_score_text.replace({'Low':'v_score_text_low','Medium':'v_score_text_medium','High':'v_score_text_high'})
df = df[~df.duplicated()]

X = df.copy()
std = MinMaxScaler(feature_range=(-1,1))
X.iloc[:,7:] = std.fit_transform(X.values[:,7:])
hot_enc = OneHotEncoder(handle_unknown='ignore')
hot_enc.fit(X.iloc[:,:7])
X[np.hstack(hot_enc.categories_).tolist()]=hot_enc.transform(X.iloc[:,:7]).toarray().astype(int)
X.drop(['sex','race','c_charge_degree','score_text','v_score_text','event','is_violent_recid'], axis=1, inplace=True)
y = df["two_year_recid"].astype(int)
X = X.drop(['two_year_recid'],axis=1,inplace=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rnd)

# ### XGBOOST

from xgboost import XGBClassifier
clf_xgb = XGBClassifier(n_estimators=60, use_label_encoder=False, eval_metric='logloss', random_state=rnd)
clf_xgb.fit(X_train, y_train)
pickle.dump(clf_xgb,open('./BlackBoxes/compas_xgboost.p','wb'))

clf_xgb = pickle.load(open('./BlackBoxes/compas_xgboost.p','rb'))
def predict(x, return_proba=False):
    if return_proba:
        return clf_xgb.predict_proba(x)[:,1].ravel()
    else: return clf_xgb.predict(x).ravel().ravel()
y_train_pred = predict(X_train)
y_test_pred = predict(X_test)
print('XGBOOST')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

# ### RF

from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(max_depth=12,random_state=rnd)
clf_rf.fit(X_train, y_train)

pickle.dump(clf_rf,open('./BlackBoxes/compas_rf.p','wb'))
clf_rf

clf_rf = pickle.load(open('./BlackBoxes/compas_rf.p','rb'))

def predict(x, return_proba=False):
    if return_proba:
        return clf_rf.predict_proba(x)[:,1].ravel()
    else: return clf_rf.predict(x).ravel().ravel()

y_test_pred = predict(X_test, return_proba=True)
y_train_pred = predict(X_train, return_proba=True)
print('RF')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

# ### SVC

from sklearn.svm import SVC
clf_svc = SVC(gamma='auto', probability=True)
clf_svc.fit(X_train, y_train)
pickle.dump(clf_svc,open('./BlackBoxes/compas_svc.p','wb'))

clf_svc = pickle.load(open('./BlackBoxes/compas_svc.p','rb'))

def predict(x, return_proba=False):
    if return_proba:
        return clf_svc.predict_proba(x)[:,1].ravel()
    else: return clf_svc.predict(x).ravel().ravel()

y_train_pred = predict(X_train, return_proba=True)
y_test_pred = predict(X_test, return_proba=True)
print('SVC')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

### NN

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
    epochs=1000,
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

plot_metric(history, 'loss')
clf_nn.save_weights('./blackboxes/compas_tf_nn')

from sklearn.metrics import accuracy_score
clf_nn.load_weights('./blackboxes/compas_tf_nn')
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
pbar_alpha = tqdm(total=10, desc='alpha')

for black_box in ['xgb', 'rf', 'svc', 'nn']:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rnd)

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

    for latent_dim in [2, 3, 4, 5, 7, 10, 15, 20, 25, 33]:
        for alpha in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5]:
            batch_size = 2048
            sigma = 1
            max_epochs = 1000
            early_stopping = 3
            learning_rate = 1e-3
            idx_cat = list(range(13,33,1))

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
                Sx = compute_similarity_X(X, sigma, alpha, idx_cat)
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
                    torch.save(model.state_dict(), f'./models/weights/LinearTransparent_compas.pt')
                else:
                    wait += 1
                #pbar.postfix[7]["value"] = wait
                if wait == early_stopping:
                    break    
                epoch += 1
                #pbar.update()

            model.load_state_dict(torch.load(f'./models/weights/LinearTransparent_compas.pt'))
            with torch.no_grad():
                model.eval()
                Z_train = model(torch.tensor(X_train).float()).cpu().detach().numpy()
                Z_test = model(torch.tensor(X_test).float()).cpu().detach().numpy()

            torch.save(model.state_dict(), f'./models/compas_latent_{black_box}_{latent_dim}_{str(alpha).replace(".", "")}.pt')
            upload_files(f'/home/ubuntu/posthoc-similaryclass-counterfactuals/models/compas_latent_{black_box}_{latent_dim}_{str(alpha).replace(".", "")}.pt', S3_BUCKET_NAME)

            pbar_alpha.update(1)
        pbar_latent.update(1)
    mails = ['francesco.bodria@sns.it']
    subject = 'risultati esperimento'
    content = f'{black_box} completato'
    mail = Mail()
    mail.send(mails, subject, content)
    pbar_model.update(1)       

pbar_alpha.close()
pbar_latent.close()
pbar_model.close()
  