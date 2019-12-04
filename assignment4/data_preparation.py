import numpy as np
import pandas as pd
from rdkit import Chem as chem
from rdkit.Chem import Lipinski as l
from sklearn.model_selection import train_test_split

def get_lipinski(df, verbose = True):

    _df = df.copy()

    for i, r in enumerate(df['SMILES']):

        m = chem.MolFromSmiles(r)

        _df.loc[i, 'FRAC'] = l.FractionCSP3(m)
        _df.loc[i, 'HEAVY'] = l.HeavyAtomCount(m)
        _df.loc[i, 'NHOH'] = l.NHOHCount(m)
        _df.loc[i, 'NO'] = l.NOCount(m)
        _df.loc[i, 'ALIPHC'] = l.NumAliphaticCarbocycles(m)
        _df.loc[i, 'ALIPHH'] = l.NumAliphaticHeterocycles(m)
        _df.loc[i, 'ALIPHR'] = l.NumAliphaticRings(m)
        _df.loc[i, 'AROMC'] = l.NumAromaticCarbocycles(m)
        _df.loc[i, 'AROMH'] = l.NumAromaticHeterocycles(m)
        _df.loc[i, 'AROMR'] = l.NumAromaticRings(m)
        _df.loc[i, 'HACC'] = l.NumHAcceptors(m)
        _df.loc[i, 'HDON'] = l.NumHDonors(m)
        _df.loc[i, 'HETERAT'] = l.NumHeteroatoms(m)
        _df.loc[i, 'SATC'] = l.NumSaturatedCarbocycles(m)
        _df.loc[i, 'SATH'] = l.NumSaturatedHeterocycles(m)
        _df.loc[i, 'SATR'] = l.NumSaturatedRings(m)
        _df.loc[i, 'R'] = l.RingCount(m)

        if verbose and i % 1000 == 0:
            print(str(i) + "/" + str(len(_df)))

    return _df

# train
train_df = pd.read_csv('assignment4/out/training.csv')
df_l = get_lipinski(train_df)

df_l.to_csv('assignment4/data/train_lipinski.csv', index = False)

# test
test_df = pd.read_csv('assignment4/out/test.csv')
df_l = get_lipinski(test_df)
df_l.to_csv('assignment4/data/test_lipinski.csv', index = False)

# validation
val_df = pd.read_csv('assignment4/out/validation.csv')
df_l = get_lipinski(val_df)
df_l.to_csv('assignment4/data/validation_lipinski.csv', index = False)
