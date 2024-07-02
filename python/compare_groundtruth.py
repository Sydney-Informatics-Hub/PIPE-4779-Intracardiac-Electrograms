# Simple script to compare ground-truth with predictions

import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    precision_score,
    matthews_corrcoef)
from inference_pipe import test_inference, test_injest_and_inference


fname_csv = 'NestedDataAll_rawsignal_unipolar.parquet'
inpath = '../results'

df = pd.read_parquet(os.path.join(inpath, fname_csv))
df = df[df['sheep'] == 'S18']
# remove 'S18_' from the names in Point_Number column
df['Point_Number'] = df['Point_Number'].str.replace('S18_', '')
df['endocardium_scar'] = df['endocardium_scar'].astype(int)
df['intramural_scar'] = df['intramural_scar'].astype(int)
df['epicardial_scar'] =df['epicardial_scar'].astype(int)
df['scar'] = df[['endocardium_scar', 'intramural_scar', 'epicardial_scar']].max(axis=1)
df['NoScar'] = 1 - df['scar']
df['AtLeastEndo'] = df['endocardium_scar']
df['AtLeastIntra'] = df['intramural_scar'] & ~df['endocardium_scar']
df['epiOnly'] = df['epicardial_scar'] & ~df['endocardium_scar'] & ~df['intramural_scar']

# save to parquet as groundtruth
path = '../../../data/deploy/test_output'
df.to_parquet(os.path.join(path, 'S18_groundtruth.parquet'))


### run prediction with 
test_inference() #sebs version says results are accurate

### now check the predictions for accuracy
target = 'epiOnly'

fname_pred = f'predictions_clf_{target}_RVp_120epochs.parquet'
fname_true = 'S18_groundtruth.parquet'

df_pred = pd.read_parquet(os.path.join(path, fname_pred))
df_true = pd.read_parquet(os.path.join(path, fname_true))
# rename column prediction to truth

df_true = df_true[df_true['WaveFront'] == 'RVp']
df_true['truth'] = df_true[target]
df_true = df_true[['Point_Number', 'truth']]
# remove duplicates
df_true.drop_duplicates(subset='Point_Number', keep='first', inplace=True)
df_true.loc[df_true['truth']==0, 'truth'] = -1

df_true['Point_Number'] = df_true['Point_Number'].astype(int)
df_pred['Point_Number'] = df_pred['Point_Number'].astype(int)


# merge df_prediction on df_true using Point_Number column
df = pd.merge(df_true, df_pred, on='Point_Number', how = 'left')

df.dropna(inplace=True)
df['truth'] = df['truth'].astype(int)
df['prediction'] = df['prediction'].astype(int)

accuracy = accuracy_score(df['truth'].values, df['prediction'].values)
print(accuracy) 
# 98.3% for NoScar RVp
# 97.8% for AtLeastEndo RVp
# 96.8% for AtLeastIntra RVp
# 99.1% for epiOnly RVp
