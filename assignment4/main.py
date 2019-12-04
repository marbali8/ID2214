import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import pickle
from progress.bar import Bar
import external
import matplotlib.pyplot as plt

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
    return df_out

def create_normalization(df, normalizationtype = 'minmax'):

    _df = df.select_dtypes(include = ['float', 'int'])

    if normalizationtype is 'minmax':

        normalization = {col: (normalizationtype, _df[col].min(), _df[col].max()) for col in _df.columns}
    elif normalizationtype is 'zscore':

        normalization = {col: (normalizationtype, _df[col].mean(), _df[col].std()) for col in _df.columns}

    df_out = apply_normalization(_df, normalization)
    return df_out, normalization


df_train = pd.read_csv('assignment4/data/train_lipinski.csv')
t_target = df_train['ACTIVE'].to_numpy()
df_train = df_train.drop(columns = [df_train.columns[0], 'SMILES', 'ACTIVE'])
df_train, normalization = create_normalization(df_train)
train = df_train.to_numpy()

df_val = pd.read_csv('assignment4/data/validation_lipinski.csv')
v_target = df_val['ACTIVE'].to_numpy()
df_val = df_val.drop(columns = [df_val.columns[0], 'SMILES', 'ACTIVE'])
df_val = apply_normalization(df_val, normalization)
val = df_val.to_numpy()

pix = [(v_target == i).sum() for i in range(len(np.unique(v_target)))]
w = sum(pix)/pix
v_weights = [w[i] for i in v_target]

hidden_layer_sizes = [(50,), (100,), (500,), (50, 2), (100, 2), (500, 2)]
activation = ['relu', 'logistic', 'tanh'] #, 'identity']
solver = ['adam', 'lbfgs', 'sgd']
alpha = [0.0001, 0.0005]
batch_size = ['auto', 500, 1000, 5000, 10000]
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
cols += ['acc_validation', 'AUC_validation', 'acc_test', 'AUC_test']
results_df = pd.DataFrame(index = range(len(parameters)), columns = cols)
for i in Bar('Processing').iter(range(len(parameters))):

    nn = MLPClassifier(hidden_layer_sizes = parameters[i][0],
                        activation = parameters[i][1],
                        solver = parameters[i][2],
                        alpha = parameters[i][3],
                        batch_size = parameters[i][4],
                        learning_rate = parameters[i][5],
                        momentum = parameters[i][6],
                        max_iter = 1000)
    nn.fit(train, t_target)
    pred_prob = nn.predict_proba(val)
    fpr, tpr, thresholds = metrics.roc_curve(v_target, pred_prob[:, 1], sample_weight = v_weights)
    AUC = metrics.auc(fpr, tpr)
    aux = np.ndarray((len(v_weights),))
    aux[:, ] = v_weights
    acc = nn.score(val, v_target, sample_weight = aux)

    results_df.loc[i] = [parameters[i][0],
                            parameters[i][1],
                            parameters[i][2],
                            parameters[i][3],
                            parameters[i][4],
                            parameters[i][5],
                            parameters[i][6],
                            acc, AUC, None, None]

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
df_test = pd.read_csv('assignment4/data/test_lipinski.csv')
te_target = df_test['ACTIVE'].to_numpy().reshape((len(df_test['ACTIVE']), 1))
df_test = df_test.drop(columns = [df_test.columns[0], 'SMILES', 'ACTIVE'])
df_test = apply_normalization(df_test, normalization)
test = df_test.to_numpy()

pix = [(te_target == i).sum() for i in range(len(np.unique(te_target)))]
w = sum(pix)/pix
te_weights = [w[i] for i in te_target]

pred_prob_test = nn.predict_proba(test)
fpr, tpr, thresholds = metrics.roc_curve(te_target, pred_prob_test[:, 1], sample_weight = te_weights)
AUC_test = metrics.auc(fpr, tpr)
aux = np.ndarray((len(te_weights),))
aux[:, ] = te_weights
acc_test = nn.score(test, te_target, sample_weight = aux)
results_df.loc[best_AUC_idx]['AUC_test'] = AUC_test
results_df.loc[best_AUC_idx]['acc_test'] = acc_test

y_pred = nn.predict(val)
external.plot_confusion_matrix(v_target, y_pred, classes = np.array([0, 1]))
plt.savefig('assignment4/confusion_matrix.png')

results_df.to_csv('assignment4/results.csv', index = False)
pickle.dump(best_AUC_model, open('assignment4/best_model.sav', 'wb'))

print('best validation result on index' + str(best_AUC_idx))
print('results witht AUC_test >= 0.8:')
print(results_df.loc[results_df['AUC_test'] >= 0.8])

# for i in range(1000): # trains the NN 1.000 times
#
#     if i % 100 == 0:
#
#         print('# ' + str(i))
#         fpr, tpr, thresholds = metrics.roc_curve(v_target, NN.forward(val))
#         AUC = metrics.auc(fpr, tpr)
#         print('AUC of validation data: ' + str(AUC))
#         print('\n')
#     NN.train(train, t_target)
#
# df_test = pd.read_csv('assignment4/data/test_lipinski.csv')
# test = df_test.drop(columns = [df_test.columns[0], 'SMILES', 'ACTIVE']).to_numpy()
# NN.predict(test)
