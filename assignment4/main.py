import numpy as np
import pandas as pd
from NN import OneLayerNN

df_train = pd.read_csv('assignment4/data/train_lipinski.csv')
t_target = df_train['ACTIVE'].to_numpy().reshape((len(df_train['ACTIVE']), 1))
train = df_train.drop(columns = ['INDEX', 'SMILES', 'ACTIVE']).to_numpy()

df_val = pd.read_csv('assignment4/data/validation_lipinski.csv')
v_target = df_val['ACTIVE'].to_numpy().reshape((len(df_val['ACTIVE']), 1))
val = df_val.drop(columns = ['INDEX', 'SMILES', 'ACTIVE']).to_numpy()

NN = OneLayerNN(train.shape[1])
for i in range(1000): # trains the NN 1.000 times

    if i % 100 == 0:
        print("# " + str(i))
        # print("Input (scaled): \n" + str(X))
        # print("Actual Output: \n" + str(y))
        # print("Predicted Output: \n" + str(NN.forward(X)))
        print("Loss of training data: " + str(np.mean(np.square(t_target - NN.forward(train))))) # mean sum squared loss
        print("Loss of validation data: " + str(np.mean(np.square(v_target - NN.forward(val))))) # mean sum squared loss
        print("\n")
    NN.train(train, t_target)

df_test = pd.read_csv('assignment4/data/test_lipinski.csv').drop(columns = ['INDEX', 'SMILES', 'ACTIVE'])
NN.predict(df_test.to_numpy())
