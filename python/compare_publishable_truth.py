# Simple script to compare ground-truth publishable with predictions
# After test_injest_and_inference() is run that does data injest and inference

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

# settings to choose
target = 'NoScar' # 'AtLeastIntra' #'AtLeastEndo' # 'NoScar' #'epiOnly' ### check the predictions for accuracy
fname_csv = 'publishable_model_data_TSAI.parquet'
inpath = '../data/generated/'
inpath_predictions = '../deploy/output'
select_wavefront = 'RVp' #'RVp' 'LVp' 'SR'
select_sheep = 'S18'
path = '../deploy/test_output'


#settings done by choices
fname_pred = f'predictions_clf_{target}_{select_wavefront}_120epochs.parquet'
fname_true = 'S18_groundtruth.parquet'

def prepare_ground_truth():
    #read publishable ground truth data and save as parquet truth
    df = pd.read_parquet(os.path.join(inpath, fname_csv))
    df = df[df['sheep'] == select_sheep]
    df['endocardium_scar'] = df['endocardium_scar'].astype(int)
    df['intramural_scar'] = df['intramural_scar'].astype(int)
    df['epicardial_scar'] =df['epicardial_scar'].astype(int)
    df['scar'] = df[['endocardium_scar', 'intramural_scar', 'epicardial_scar']].max(axis=1)
    df['NoScar'] = 1 - df['scar']
    df['AtLeastEndo'] = df['endocardium_scar']
    df['AtLeastIntra'] = df['intramural_scar'] & ~df['endocardium_scar']
    df['epiOnly'] = df['epicardial_scar'] & ~df['endocardium_scar'] & ~df['intramural_scar']
    df.dropna(inplace=True)
    # save to parquet as groundtruth
    df.to_parquet(os.path.join(path, 'S18_groundtruth.parquet'))

# test inference --------------------------------

def compare_truth_to_inference():
        
    #read truth that was saved and compare to inference
    df_pred = pd.read_parquet(os.path.join(inpath_predictions, fname_pred))
    df_true = pd.read_parquet(os.path.join(path, fname_true))
    df_true = df_true[df_true['WaveFront'] == select_wavefront]
    df_true['truth'] = df_true[target]
    df_true = df_true[['Point_Number', 'truth']]
    df_true.drop_duplicates(subset='Point_Number', keep='first', inplace=True)
    df_true.loc[df_true['truth']==0, 'truth'] = -1
    df_true['Point_Number'] = df_true['Point_Number'].astype(int)
    df_pred['Point_Number'] = df_pred['Point_Number'].astype(int)
    # merge df_prediction on df_true using Point_Number column
    df = pd.merge(df_true, df_pred, on='Point_Number', how = 'left')

    df['truth'] = df['truth'].astype(int)
    df['prediction'] = df['prediction'].astype(int)

    accuracy = accuracy_score(df['truth'].values, df['prediction'].values)
    print(f'{select_sheep} - {select_wavefront} - {target}: {accuracy}')
    return(accuracy)

if __name__ == '__main__':
    #vary the target option to get the different accuracy scores
    prepare_ground_truth()
    accuracy = compare_truth_to_inference()
#compare sebs run (from compare_groundtruth.py) to this file
# 98.3% for NoScar RVp - This file has 0.7778342653787493
# 97.8% for AtLeastEndo RVp - This file has  0.8291814946619217
# 96.8% for AtLeastIntra RVp - This file has 0.9791560752414845
# 99.1% for epiOnly RVp  - This file has 0.961870869344179

# TM - compare sebs run after dropping NaNs in the input ground truth
# S18 - RVp - NoScar: 0.9836233367451381
# S18 - RVp - AtLeastEndo: 0.977482088024565
# S18 - RVp - AtLeastIntra: 0.9682702149437052
# S18 - RVp - epiOnly: 0.9907881269191402

# KM - S18 using 'publishable_model_data_TSAI.parquet' rather than imputed confirms TM results
#S18 - RVp - NoScar: 0.9836233367451381
# S18 - RVp - AtLeastEndo: 0.977482088024565
# S18 - RVp - AtLeastIntra: 0.9682702149437052
# S18 - RVp - epiOnly: 0.9907881269191402
# S18 - LVp - NoScar: 0.9335083114610674
# S18 - LVp - AtLeastEndo: 0.8985126859142607
# S18 - LVp - AtLeastIntra: 0.9142607174103237
# S18 - LVp - epiOnly: 0.9711286089238845
# S18 - SR - NoScar: 0.9826937547027841
# S18 - SR - AtLeastEndo: 0.9661399548532731
# S18 - SR - AtLeastIntra: 0.9721595184349134
# S18 - SR - epiOnly: 0.9781790820165538