from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torchvision
import torch
import numpy as np
import pandas as pd

def load_tabular_dataset(dataset_name, dataset_path):
    def get_features(filename):
        data = open(filename, 'r')
        features = list()
        feature_names = list()
        usecols = list()
        col_id = 0
        for row in data:
            field = row.strip().split(',')
            feature_names.append(field[0])
            if field[2] != 'ignore':
                usecols.append(col_id)
                if field[2] != 'class':
                    features.append((field[0], field[1], field[2]))
            col_id += 1
        return feature_names, features, usecols

    target = 'class'
    # df = pd.read_csv(dataset_path + dataset_name + '.csv.gz', delimiter=',')

    feature_names, features, col_indexes = get_features(dataset_path + dataset_name + '.names')
    df = pd.read_csv(dataset_path + dataset_name + '.data.gz', delimiter=',', names=feature_names, usecols=col_indexes)

    # features = yadt.metadata(df, ovr_types={})

    features2binarize = list()
    class_observed = False
    classes_names = []
    for idx, col in enumerate(df.columns):
        dtype = df[col].dtype
        if dtype != np.float64:
            if dtype.kind == 'O':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                if len(le.classes_) > 2:
                    classes_names.append(le.classes_)
                    if col != target:
                        # print(col, idx if not class_observed else idx - 1)
                        features2binarize.append(idx if not class_observed else idx - 1)
        if col == target:
            class_observed = True
    
    y = df[target].values
    df = df.drop(columns='class')
    
    categorical_columns = list(df.columns[features2binarize])
    scalar_columns = list(df.columns[np.where([np.array(range(len(features)))[i] not in features2binarize for i in range(len(features))])[0]])

    feature_names = []
    for i in range(len(categorical_columns)):
        feature_names += [f'{categorical_columns[i]}_{j}' for j in classes_names[i]]
    enc = OneHotEncoder(handle_unknown='ignore')

    df[feature_names] = enc.fit_transform(df.loc[:,categorical_columns]).astype(int).toarray()
    #df[enc.get_feature_names(categorical_columns)] = enc.fit_transform(df.loc[:,categorical_columns]).astype(int).toarray()

    std = MinMaxScaler(feature_range=(-1,1))
    df.loc[:, scalar_columns] = std.fit_transform(df.loc[:,scalar_columns])

    df = df.drop(columns=categorical_columns)
    df_old = df.copy()
    X = df.values
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    return X_train, X_test, Y_train, Y_test, df_old, scalar_columns


def load_image_dataset(dataset_name, path, batch_size=2048):

    class ReshapeTransform:
        def __init__(self, new_size):
            self.new_size = new_size

        def __call__(self, img):
            return torch.reshape(img, self.new_size)

    mnist_transforms = [torchvision.transforms.ToTensor(),
                        #torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                        ReshapeTransform((28*28,))]
    
    if dataset_name=='MNIST':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(f'{path}', 
                                       train=True, 
                                       download=True,
                                       transform=torchvision.transforms.Compose(mnist_transforms)),
            batch_size=batch_size, 
            shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(f'{path}', 
                                       train=False, 
                                       download=True,
                                       transform=torchvision.transforms.Compose(mnist_transforms)),
            batch_size=batch_size, 
            shuffle=False)
    elif dataset_name=='FashionMNIST':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(f'{path}', 
                                       train=True, 
                                       download=True,
                                       transform=torchvision.transforms.Compose(mnist_transforms)),
            batch_size=batch_size, 
            shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(f'{path}', 
                                       train=False, 
                                       download=True,
                                       transform=torchvision.transforms.Compose(mnist_transforms)),
            batch_size=batch_size, 
            shuffle=False)

    with torch.no_grad():
        x_train=[]
        y_train = []
        for i, data in enumerate(train_loader):
            batch_X, batch_y = data[0], data[1]
            x_train.append(batch_X)
            y_train.append(batch_y)
        x_train = np.vstack(x_train)
        y_train = np.hstack(y_train)
    with torch.no_grad():
        y_test = []
        x_test = []
        for i, data in enumerate(test_loader):
            batch_X, batch_y = data[0], data[1]
            x_test.append(batch_X)
            y_test.append(batch_y)
        x_test = np.vstack(x_test)
        y_test = np.hstack(y_test)
        
    if dataset_name=='MNIST':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(f'{path}', 
                                       train=True, 
                                       download=True,
                                       transform=torchvision.transforms.Compose(mnist_transforms)),
            batch_size=batch_size, 
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(f'{path}', 
                                       train=False, 
                                       download=True,
                                       transform=torchvision.transforms.Compose(mnist_transforms)),
            batch_size=batch_size, 
            shuffle=False)
    elif dataset_name=='FashionMNIST':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(f'{path}', 
                                       train=True, 
                                       download=True,
                                       transform=torchvision.transforms.Compose(mnist_transforms)),
            batch_size=batch_size, 
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(f'{path}', 
                                       train=False, 
                                       download=True,
                                       transform=torchvision.transforms.Compose(mnist_transforms)),
            batch_size=batch_size, 
            shuffle=False)
    
    return x_train, x_test, y_train, y_test, train_loader, test_loader

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
