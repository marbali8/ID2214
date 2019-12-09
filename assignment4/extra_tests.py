# rdkit.Chem.Descriptors.NumValenceElectrons(m) https://oneclass.com/homework-help/chemistry/18168-how-do-valence-electrons-determ.en.html
# valence = [chemd.NumValenceElectrons(m) for m in moles]
# df['VAL'] = valence
# print(np.corrcoef(df['ACTIVE'], df['VAL'])[1, 0])
# th = 100
# print('< TH')
# print(df.loc[df['VAL'] < th]['ACTIVE'].value_counts())
# print('>= TH')
# print(df.loc[df['VAL'] >= th]['ACTIVE'].value_counts())

# rdkit.Chem.Descriptors.NumRadicalElectrons(m) no he trobat correlacio
# radical = [chemd.NumRadicalElectrons(m) for m in moles]
# df['RAD'] = radical
# print(np.corrcoef(df['ACTIVE'], df['RAD'])[1, 0])

# wt = [d.CalcExactMolWt(m) for m in moles]
# df['WT'] = wt
# print(np.corrcoef(df['ACTIVE'], df['WT'])[1, 0])

# rdkit.Chem.Lipinski.HeavyAtomCount(m) no he trobat relacio ni que es lol
# heavy = [lip.HeavyAtomCount(m) for m in moles]
# df['HEAVY'] = heavy
# print(np.corrcoef(df['ACTIVE'], df['HEAVY'])[1, 0])


# df = df.drop(columns = ['INDEX', 'SMILES'])
# print(df.head())
# df.to_csv('assignment4/training_smiles_lipinski.csv', index = False)
# print(np.unique(np.array(valence)))


# rdkit.Chem.EState.AtomTypes.TypeAtoms algo amb SMARTS que es similar a SMILES (https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html)
# rdkit.Chem.QED.properties(m) i don't think they apply
# rdkit.Chem.SATIS.SATISTypes(m) tf

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
df_train = df_train.drop(columns = ['INDEX', 'SMILES', 'ACTIVE'])
df_train, normalization = create_normalization(df_train)
train = df_train.to_numpy()

nn = pickle.load(open('assignment4/old/fingerprints/stratified/best_model.sav', 'rb'))

# val evaluation
df_val = pd.read_csv('assignment4/data/test_fingerprints.csv')
v_target = df_val['ACTIVE'].to_numpy().reshape((len(df_val['ACTIVE']), 1))
df_val = df_val.drop(columns = ['INDEX', 'SMILES', 'ACTIVE'])
df_val = apply_normalization(df_val, normalization)
val = df_val.to_numpy()

pred_prob = nn.predict_proba(val)
AUC = metrics.roc_auc_score(v_target, pred_prob[:, 1])
y_pred = nn.predict(val)
external.plot_confusion_matrix(v_target, y_pred, classes = np.array([0, 1]), normalize = True, title = 'Normalized confusion matrix (AUC val ' + str(AUC) + ')')
plt.savefig('assignment4/confusion_matrix_fingerprints_test.png')
