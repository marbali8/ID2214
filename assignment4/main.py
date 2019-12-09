import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import pickle
from progress.bar import Bar
import external
import matplotlib.pyplot as plt
import time

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

    return df_out.fillna(0)

def create_normalization(df, normalizationtype = 'minmax'):

    _df = df.select_dtypes(include = ['float', 'int'])

    if normalizationtype is 'minmax':

        normalization = {col: (normalizationtype, _df[col].min(), _df[col].max()) for col in _df.columns}
    elif normalizationtype is 'zscore':

        normalization = {col: (normalizationtype, _df[col].mean(), _df[col].std()) for col in _df.columns}

    df_out = apply_normalization(_df, normalization)
    return df_out, normalization

dataset = 'fingerprints'

df_train = pd.read_csv('assignment4/data/train_' + dataset + '.csv')
active = df_train.loc[df_train['ACTIVE'] == 1]
df_train = pd.concat([active, df_train.loc[df_train['ACTIVE'] == 0][0: round(len(active)*1.2)]])
t_target = df_train['ACTIVE'].to_numpy()
df_train = df_train.drop(columns = ['INDEX', 'SMILES', 'ACTIVE'])
df_train, normalization = create_normalization(df_train)
train = df_train.to_numpy()

df_val = pd.read_csv('assignment4/data/validation_' + dataset + '.csv')
v_target = df_val['ACTIVE'].to_numpy()
df_val = df_val.drop(columns = ['INDEX', 'SMILES', 'ACTIVE'])
df_val = apply_normalization(df_val, normalization)
val = df_val.to_numpy()

# pix = [(v_target == i).sum() for i in range(len(np.unique(v_target)))]
# w = sum(pix)/pix
# v_weights = [w[i] for i in v_target]

hidden_layer_sizes = [(50,), (100,), (20, 2), (50, 2)]
activation = ['relu', 'logistic', 'tanh'] #, 'identity']
solver = ['adam', 'lbfgs', 'sgd']
alpha = [0.0001, 0.0005]
batch_size = ['auto', 100, 500, 1000] #, 5000, 10000]
learning_rate = ['constant', 'invscaling', 'adaptive']
#learning_rate_init = [0.001]
#power_t = [0.5] # for sgd
#max_iter = [200]
#shuffle = [True]
#random_state = None
#tol = [0.0001]
#warm_start = False
momentum = [0.9, 0.8, 0.7] # for sgd
#nesterovs_momentum = True
#early_stopping = False
#validation_fraction = 0.1
#beta_1 = [0.9] # for adam
#beta_2 = [0.999] # for adam
#epsilon = [1e-08] # for adam
#n_iter_no_change = [10]
#max_fun = [15000] # for lbfgs

parameters = [(h, a, s, al, b, l, m)
              for h in hidden_layer_sizes
              for a in activation
              for s in solver
              for al in alpha
              for b in batch_size
              for l in learning_rate
              for m in momentum]

best_AUC = 0
best_AUC_parameters = None
best_AUC_model = None
best_AUC_idx = None

cols = ['hidden_layer_sizes', 'activation', 'solver', 'alpha', 'batch_size', 'learning_rate', 'momentum']
cols += ['AUC_validation', 'AUC_test', 'time']
results_df = pd.DataFrame(index = range(len(parameters)), columns = cols)
for i in Bar('Processing').iter(range(len(parameters))):

    start = time.time()
    nn = MLPClassifier(hidden_layer_sizes = parameters[i][0],
                        activation = parameters[i][1],
                        solver = parameters[i][2],
                        alpha = parameters[i][3],
                        batch_size = parameters[i][4],
                        learning_rate = parameters[i][5],
                        momentum = parameters[i][6])
    nn.fit(train, t_target)
    pred_prob = nn.predict_proba(val)
    AUC = metrics.roc_auc_score(v_target, pred_prob[:, 1]) #, sample_weight = v_weights)

    results_df.loc[i] = [parameters[i][0],
                            parameters[i][1],
                            parameters[i][2],
                            parameters[i][3],
                            parameters[i][4],
                            parameters[i][5],
                            parameters[i][6],
                            AUC, None, time.time() - start]

    if AUC > best_AUC:

        print(' ' + str(AUC) + ' > ' + str(best_AUC))
        best_AUC = AUC
        best_AUC_parameters = parameters[i]
        best_AUC_model = nn
        best_AUC_idx = i

    if i % 100 == 0: # just in case

        results_df.to_csv('assignment4/results.csv', index = False)
        pickle.dump(best_AUC_model, open('assignment4/best_model.sav', 'wb'))

# test evaluation
df_test = pd.read_csv('assignment4/data/test_' + dataset + '.csv')
te_target = df_test['ACTIVE'].to_numpy().reshape((len(df_test['ACTIVE']), 1))
df_test = df_test.drop(columns = ['INDEX', 'SMILES', 'ACTIVE'])
df_test = apply_normalization(df_test, normalization)
test = df_test.to_numpy()

# pix = [(te_target == i).sum() for i in range(len(np.unique(te_target)))]
# w = sum(pix)/pix
# te_weights = [w[i] for i in te_target]

pred_prob_test = best_AUC_model.predict_proba(test)
AUC_test = metrics.roc_auc_score(te_target, pred_prob_test[:, 1]) #, sample_weight = te_weights)
# aux = np.ndarray((len(te_weights),))
# aux[:, ] = te_weights
results_df.loc[best_AUC_idx]['AUC_test'] = AUC_test

y_pred = best_AUC_model.predict(test)
external.plot_confusion_matrix(te_target, y_pred, classes = np.array([0, 1]), normalize = True, title = 'Normalized confusion matrix (AUC val ' + str(AUC_test) + ')')
plt.savefig('assignment4/confusion_matrix_test.png')

y_pred = best_AUC_model.predict(val)
external.plot_confusion_matrix(v_target, y_pred, classes = np.array([0, 1]), normalize = True, title = 'Normalized confusion matrix (AUC val ' + str(best_AUC) + ')')
plt.savefig('assignment4/confusion_matrix_val.png')

results_df.to_csv('assignment4/results.csv', index = False)
pickle.dump(best_AUC_model, open('assignment4/best_model.sav', 'wb'))

print('best validation result on index ' + str(best_AUC_idx))
print('results witht AUC_test >= 0.8:')
print(results_df.loc[results_df['AUC_test'] >= 0.8])
