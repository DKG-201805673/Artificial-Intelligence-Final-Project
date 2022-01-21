import pandas as pd

# replace filepath to unmodified dataset
df = pd.read_csv([your path here])

# convert Sex column to numerical values
df['is_male'] = [1 if sex == 'M' else 0 for sex in df['Sex']]
df.drop('Sex', axis=1, inplace=True)

#convert ExerciseAngina column to numerical values
df['ExerciseAngina'] = [1 if EA == 'Y' else 0 for EA in df['ExerciseAngina']]

# one-hot-encode ChestPainType column
df['CP_TA'] = [1 if CP == 'TA' else 0 for CP in df['ChestPainType']]
df['CP_ATA'] = [1 if CP == 'ATA' else 0 for CP in df['ChestPainType']]
df['CP_ASY'] = [1 if CP == 'ASY' else 0 for CP in df['ChestPainType']]
df['CP_NAP'] = [1 if CP == 'NAP' else 0 for CP in df['ChestPainType']]
df.drop('ChestPainType', axis=1, inplace=True)

# one-hot-encode RestingECG column
df['ECG_normal'] = [1 if ECG == 'Normal' else 0 for ECG in df['RestingECG']]
df['ECG_ST'] = [1 if ECG == 'ST' else 0 for ECG in df['RestingECG']]
df['ECG_LVH'] = [1 if ECG == 'LVH' else 0 for ECG in df['RestingECG']]
df.drop('RestingECG', axis=1, inplace=True)

# one-hot-encode ST_Slope column
df['ST_Up'] = [1 if ST == 'Up' else 0 for ST in df['ST_Slope']]
df['ST_Down'] = [1 if ST == 'Down' else 0 for ST in df['ST_Slope']]
df['ST_Flat'] = [1 if ST == 'Flat' else 0 for ST in df['ST_Slope']]
df.drop('ST_Slope', axis=1, inplace=True)

# save dataframe to csv (put your filepath here)
df.to_csv([your path here], index=False)
