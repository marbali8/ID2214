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
