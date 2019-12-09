import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import pickle
import external
from data_preparation import smiles_to_fps_labels

def apply_normalization(df, normalization):

    df_out = df.copy()
    for col in normalization:
        a = normalization[col][1]
        b = normalization[col][2]

        if normalization[col][0] is 'minmax':

            df_out[col] = (df_out[col] - a)/(b - a) # using broadcasting
            df_out[col].clip(0, 1, inplace = True)
        elif normalization[col][0] is "zscore":

            df_out[col] = df_out[col].apply(lambda x: (x - a)/b)

    return df_out.fillna(df_out.median())

def create_normalization(df, normalizationtype = 'minmax'):

    _df = df.select_dtypes(include = ['float', 'int'])

    if normalizationtype is 'minmax':

        normalization = {col: (normalizationtype, _df[col].min(), _df[col].max()) for col in _df.columns}
    elif normalizationtype is 'zscore':

        normalization = {col: (normalizationtype, _df[col].mean(), _df[col].std()) for col in _df.columns}

    df_out = apply_normalization(_df, normalization)
    return df_out, normalization

df_train = pd.read_csv('assignment4/data/train_fingerprints.csv')
t_target = df_train['ACTIVE'].to_numpy()
df_train = df_train.drop(columns = ['INDEX', 'SMILES', 'ACTIVE'], errors = 'ignore')
df_train, normalization = create_normalization(df_train)
train = df_train.to_numpy()

# Load datasets
df_est = pd.read_csv('assignment4/data/test_fingerprints.csv')
df_test = pd.read_csv('assignment4/data/test_smiles.csv')   # This is the one without labels
X_est = smiles_to_fps_labels(df_est, fp_length = 1024).drop(columns = ['INDEX', 'SMILES', 'ACTIVE'], errors = 'ignore')
X_est = apply_normalization(X_est, normalization)
X_test = smiles_to_fps_labels(df_test, fp_length = 1024).drop(columns = ['INDEX', 'SMILES', 'ACTIVE'], errors = 'ignore')
X_test = apply_normalization(X_test, normalization)

nn = pickle.load(open('assignment4/old/fingerprints/stratified/best_model.sav', 'rb'))
pred_prob = nn.predict_proba(X_est)
AUC_est = metrics.roc_auc_score(df_est['ACTIVE'], pred_prob[:, 1])

def results_txt(model, X_test, AUC_est):

    prob = model.predict_proba(X_test)
    file = open('assignment4/6.txt', "w")
    file.write(str(AUC_est))
    for p in prob[:, 1]:
        file.write("\n")
        file.write(str(p))
    file.close()
    print(sum(pred_prob[:, 1] <= 0.5), sum(pred_prob[:, 1] > 0.5))


results_txt(nn, X_test, AUC_est)
