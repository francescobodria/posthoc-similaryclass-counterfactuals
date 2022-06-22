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

def load_tabular_data(name):
    if name == 'adult':
        df = pd.read_csv('./data/adult_clear.csv')
        df = df[df["workclass"] != "?"]
        df = df[df["occupation"] != "?"]
        df = df[df["native-country"] != "?"]
        df.replace(['Divorced', 'Married-AF-spouse',
                    'Married-civ-spouse', 'Married-spouse-absent',
                    'Never-married', 'Separated', 'Widowed'],
                ['notmarried', 'married', 'married', 'married',
                    'notmarried', 'notmarried', 'notmarried'], inplace=True)
        df['education'].replace(['Preschool', '10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th'], 'dropout',
                                inplace=True)
        df['education'].replace(['HS-Grad', 'HS-grad'], 'HighGrad', inplace=True)
        df['education'].replace(['Some-college', 'Assoc-acdm', 'Assoc-voc'], 'CommunityCollege', inplace=True)
        df = df[df.race == 'White']
        # excludes 10 observations
        df = df[df['workclass'] != 'Never-worked']
        # excludes 14 observations
        df = df[df['occupation'] != 'Armed-Forces']
        # excludes 21 observations
        df = df[df['workclass'] != 'Without-pay']
        df.drop(['fnlwgt', 'educational-num', 'relationship', 'race', 'capital-gain', 'capital-loss'],
                    axis=1, inplace=True)
        df['workclass'].replace(['Local-gov', 'State-gov', 'Federal-gov'], 'Gov', inplace=True)
        df['workclass'].replace(['Private', 'Self-emp-not-inc', 'Self-emp-inc'], 'Private', inplace=True)
        df['occupation'].replace(
            ['Craft-repair', 'Machine-op-inspct', 'Handlers-cleaners', 'Transport-moving', 'Adm-clerical',
            'Farming-fishing'], 'BlueCollar', inplace=True)
        df['occupation'].replace(['Other-service', 'Protective-serv', 'Tech-support', 'Priv-house-serv'], 'Services',
                                inplace=True)
        df['occupation'].replace(['Exec-managerial'], 'ExecManagerial', inplace=True)
        df['occupation'].replace(['Prof-specialty'], 'ProfSpecialty', inplace=True)
        df['education'].replace(['Prof-school'], 'ProfSchool', inplace=True)
        df['native-country'].replace(['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', \
                                    'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti',\
                                    'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland',\
                                    'Italy', 'Jamaica', 'Japan', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)',\
                                    'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South',\
                                    'Taiwan','Thailand', 'Trinadad&Tobago', 'Vietnam','Yugoslavia'], 'Non_US', inplace=True)
        df.rename(columns={'hours-per-week': 'hoursPerWeek'}, inplace=True)
        df.rename(columns={'marital-status': 'marital_status'}, inplace=True)
        df.rename(columns={'native-country': 'native_country'}, inplace=True)
        columns_titles = ["age","hoursPerWeek","education","marital_status","occupation","gender","native_country","income"]
        df=df.reindex(columns=columns_titles)
        df = df[~df.duplicated()]
        X = df.copy()
        ord_enc = OrdinalEncoder()
        X.iloc[:,2:] = ord_enc.fit_transform(X.values[:,2:]).astype(int)
        print('ord_enc')
        print(ord_enc.categories_)
        std = MinMaxScaler(feature_range=(-1,1))
        print('std')
        X.iloc[:,:2] = std.fit_transform(X.values[:,:2])
        print(std.inverse_transform(np.array([[-0.15068493,0],[-0.383117,0],[-0.315358, -0.263194],[-0.150685, -0.091183],[-0.337704,-0.243003],[-0.15068493,0.08495728]])))
        X.drop(['income'], axis=1, inplace=True)
        y = df["income"].apply(lambda x: ">50K" in x).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rnd)
        return X_train, X_test, y_train, y_test, df

    elif name == 'fico':
        df = pd.read_csv('./data/heloc_dataset.csv')
        df['RiskPerformance']=(df.RiskPerformance=='Bad')+0

        scaler = MinMaxScaler((-1,1))
        df.iloc[:,1:] = scaler.fit_transform(df.values[:,1:])
        y = df['RiskPerformance']
        df.drop(['RiskPerformance'], axis=1, inplace=True)

        X_train, X_test, Y_train, Y_test = train_test_split(df, y, test_size=0.2, random_state=rnd)
        return X_train, X_test, Y_train, Y_test, df
    
    elif name == 'german':
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

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=rnd)
        return X_train, X_test, y_train, y_test, df

    elif name == 'compas':
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
        return X_train, X_test, y_train, y_test, df

