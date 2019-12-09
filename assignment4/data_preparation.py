import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import inspect

from rdkit import Chem as chem
from rdkit.Chem import Lipinski as l
from rdkit.Chem import AllChem
import rdkit.Chem.Fragments as f
import rdkit.Chem.rdMolDescriptors as molDesc

# def get_lipinski(df, verbose = True):
#
#     _df = df.copy()
#
#     for i in range(len(df)):
#
#         m = chem.MolFromSmiles(df['SMILES'].loc[i])
#
#         _df.loc[i, 'FRAC'] = l.FractionCSP3(m)
#         _df.loc[i, 'HEAVY'] = l.HeavyAtomCount(m)
#         _df.loc[i, 'NHOH'] = l.NHOHCount(m)
#         _df.loc[i, 'NO'] = l.NOCount(m)
#         _df.loc[i, 'ALIPHC'] = l.NumAliphaticCarbocycles(m)
#         _df.loc[i, 'ALIPHH'] = l.NumAliphaticHeterocycles(m)
#         _df.loc[i, 'ALIPHR'] = l.NumAliphaticRings(m)
#         _df.loc[i, 'AROMC'] = l.NumAromaticCarbocycles(m)
#         _df.loc[i, 'AROMH'] = l.NumAromaticHeterocycles(m)
#         _df.loc[i, 'AROMR'] = l.NumAromaticRings(m)
#         _df.loc[i, 'HACC'] = l.NumHAcceptors(m)
#         _df.loc[i, 'HDON'] = l.NumHDonors(m)
#         _df.loc[i, 'HETERAT'] = l.NumHeteroatoms(m)
#         _df.loc[i, 'SATC'] = l.NumSaturatedCarbocycles(m)
#         _df.loc[i, 'SATH'] = l.NumSaturatedHeterocycles(m)
#         _df.loc[i, 'SATR'] = l.NumSaturatedRings(m)
#         _df.loc[i, 'R'] = l.RingCount(m)
#
#         if verbose and i % 1000 == 0:
#             print(str(i) + "/" + str(len(df)))
#
#     return _df

# generate fingerprints: Morgan fingerprint with radius 2
def smiles_to_fps_labels(df, fp_radius = 2, fp_length = 256):

    df_tmp = df.copy()
    cols = ['F' + str(i) for i in range(fp_length)]
    fps = np.empty((len(df_tmp), len(cols)))
    for i, smiles in enumerate(df['SMILES']):

        molecule = chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(
                 molecule, fp_radius, nBits = fp_length)
        fps[i] = np.array(list(fp.ToBitString()))

        if i % 1000 == 0:

            print(str(i) + "/" + str(len(df)))

    for i, c in enumerate(cols):

        df_tmp.loc[:, c] = fps[:, i]

    return df_tmp

def smiles_to_all_labels(df):

    smilesList = df['SMILES']
    feature_df = df.copy()

    # get all functions of modules
    all_lipinski = inspect.getmembers(l, inspect.isfunction)
    all_fragments = inspect.getmembers(f, inspect.isfunction)


    # bad features have the same value for all our compounds
    bad_features = []
    for (columnName, columnData) in df.iteritems():
        if(len(set(columnData.values)) == 1):
            bad_features.append(columnName)

    # add fragment features
    for i in range(len(all_fragments)):
        new_col = []

        # exclude attributes which start with _ and exclude bad features
        if all_fragments[i][0].startswith('_') == False and all_fragments[i][0] not in bad_features:

            for smiles in smilesList:
                molecule = chem.MolFromSmiles(smiles)
                mol_method = all_fragments[i][1](molecule)
                new_col.append(mol_method)

            # add new col with feature name to our df
            feature_df[all_fragments[i][0]] = new_col

    print('fragments over')

    # add lipinski features
    for i in range(len(all_lipinski)):

        new_col = []
        if all_lipinski[i][0].startswith('_') == False and all_fragments[i][0] not in bad_features:

            for smiles in smilesList:

                molecule = chem.MolFromSmiles(smiles)
                mol_method = all_lipinski[i][1](molecule)
                new_col.append(mol_method)

            feature_df[all_lipinski[i][0]] = new_col

    print('lipinski over')

    new_col = []
    for smiles in smilesList:

        molecule = chem.MolFromSmiles(smiles)
        new_col.append(f.fr_Al_COO(molecule))

    feature_df["fr_Al_COO"] = new_col

    # new_col = []
    for smiles in smilesList:

        molecule = chem.MolFromSmiles(smiles)
        new_col.append(l.HeavyAtomCount(molecule))

    feature_df["HeavyAtomCount"] = new_col

    # add getnumatoms as feature
    new_col = []
    for smiles in smilesList:

        molecule = chem.MolFromSmiles(smiles)
        new_col.append(molecule.GetNumAtoms())

    feature_df["GetNumAtoms"] = new_col

    # add CalcExactMolWt as feature
    new_col = []
    for smiles in smilesList:

        molecule = chem.MolFromSmiles(smiles)
        new_col.append(molDesc.CalcExactMolWt(molecule))

    feature_df["CalcExactMolWt"] = new_col

    # print('other over')

    return feature_df

# import initial data
X = pd.read_csv('assignment4/data/training_smiles.csv')
y = X['ACTIVE']

# split into training validation and test
train_df, test_df, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 8)
val_df, test_df, y_validation, y_test = train_test_split(test_df, y_test, test_size = 0.5, stratify = y_test, random_state = 8)

# train
# train_df = pd.read_csv('assignment4/out/training.csv')
df_l = get_lipinski(train_df)
df_l.to_csv('assignment4/data/train_lipinski.csv', index = False)
df_l = smiles_to_fps_labels(train_df, fp_length = 1024)
df_l.to_csv('assignment4/data/train_fingerprints.csv', index = False)
df_l = smiles_to_all_labels(train_df)
df_l.to_csv('assignment4/data/train_all.csv', index = False)

# test
# test_df = pd.read_csv('assignment4/out/test.csv')
df_l = get_lipinski(test_df)
df_l.to_csv('assignment4/data/test_lipinski.csv', index = False)
df_l = smiles_to_fps_labels(test_df, fp_length = 1024)
df_l.to_csv('assignment4/data/test_fingerprints.csv', index = False)
df_l = smiles_to_all_labels(test_df)
df_l.to_csv('assignment4/data/test_all.csv', index = False)

# validation
val_df = pd.read_csv('assignment4/out/validation.csv')
df_l = get_lipinski(val_df)
df_l.to_csv('assignment4/data/validation_lipinski.csv', index = False)
df_l = smiles_to_fps_labels(val_df, fp_length = 1024)
df_l.to_csv('assignment4/data/validation_fingerprints.csv', index = False)
df_l = smiles_to_all_labels(val_df)
df_l.to_csv('assignment4/data/validation_all.csv', index = False)
