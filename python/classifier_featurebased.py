"""
Timeseries classification for ECG data using feature-selection and Random Forest/XGboost.

Feature extraction and selection is performed using the FeatureExtraction class from the features.py module.

How to use:
    - preprocess: Run the preprocess function to prepare the ECG data for classification
    - classify: Run the classify function to classify the data using the selected features and the target label

Python example:
from classifier_featurebased import run_all
run_all()

If all targets and wavefronts are to be classified, use the run_all function.
The results are saved to a txt and csf file in the specified output folder.

The combine_txt function can be used to combine all txt files into one (for easier comparison).

For installation of required packages and dependencies, see environment.yaml.

Author: Sebastian Haan
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    precision_score,
    matthews_corrcoef)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from features import FeatureExtraction

inpath = '../../../data/generated_w_unipolar'
fname_csv_core = 'NestedData'
animal_labels = ['S9', 'S12', 'S15', 'S17', 'S18', 'S20']
outpath = '../results'
target = 'scar'

def preprocess(outpath = '../results', 
               fname_csv_core = 'NestedData', 
               animal_labels = ['S9', 'S12', 'S15', 'S17', 'S18', 'S20'], 
               inpath = '../../../data/generated',
               fname_out = 'NestedDataAll_clean.csv'):
    """
    The preprocess function reads in all data of list_fnames_csv as dataframe and concatenates them.
    The data is then saved to a csv file.

    Parameters
    ----------
    outpath : str
        Path to the output folder
    fname_csv_core : str
        Core name of the csv files
    animal_labels : list
        List of animal labels
    inpath : str
        Path to the data
    fname_out : str
        Name of the output file
    """
    usecols = [
                'Point_Number',
                'WaveFront',
                'sheep',
                'signal_data',
                'endocardium_scar',
                'intramural_scar',
                'epicardial_scar',
            ]
    df = None
    for label in animal_labels:
            file = os.path.join(inpath, fname_csv_core + label + '.csv')
            # check if file exists
            if not os.path.isfile(file):
                raise FileNotFoundError(f'File {file} not found')
            dfnew = pd.read_csv(file, usecols=usecols)
            # remove nan values
            dfnew = dfnew.dropna()
            # modify point number to avoid duplication
            dfnew['Point_Number'] = dfnew['sheep'] + '_' + dfnew['Point_Number'].astype(str)
            # concatenate dataframes
            # check if 
            if df is None:
                df = dfnew
            else:
                df = pd.concat([df, dfnew], ignore_index=True)
    os.makedirs(outpath, exist_ok=True)
    df.to_csv(os.path.join(outpath, fname_out), index=False)


def classify(path, 
             infile_features = 'NestedDataAll_clean.csv', 
             target = 'scar', 
             wavefront = 'SR',
             method = 'xgboost'):
    """
    Classify the data using the selected features and the target label.
    Generates a txt file with the results.

    Parameters
    ----------
    path : str
        Path to the data
    target : str
        Target label for classification, either 'scar','endocardium_scar','intramural_scar','epicardial_scar
    infile_features : str
        Name of the file containing the features
    wavefront : str
        Wavefront label for classification, either 'SR', 'LVp', or 'RVp'
    method : str
        Classification method, either 'xgboost' or 'rf'

    Returns
    -------
    tuple
        Tuple containing the accuracy, precision, AUC, and MCC of the classifier
    """
    
    outpath = os.path.join(path, target+'_'+wavefront)
    os.makedirs(outpath, exist_ok=True)

    print('Extracting and selecting features...')
    fe = FeatureExtraction(path, infile_features, outpath)
    fe.run_wavefront_target(wavefront, target)

    X = fe.selected_features
    y = fe.y
    print('Class balance:', y.value_counts())
    # save y to file
    y.to_csv(os.path.join(outpath, f'selected_features_label_{target}_{wavefront}.csv'))

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if method == 'xgboost':
        print('Training XGBoost classifier...')
        classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        # Fit the classifier to the training set
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        y_probs = classifier.predict_proba(X_test)[:,1]

    elif method == 'rf':
        print('Training Random Forest classifier...')
        # train random forest classifier
        rf = RandomForestClassifier(n_estimators=1000, 
                                    max_depth=20, 
                                    min_samples_leaf=2,
                                    random_state=42, 
                                    warm_start=False,
                                    n_jobs=-1, 
                                    class_weight='balanced',
                                    criterion = 'log_loss',                         
                                    )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_probs = rf.predict_proba(X_test)[:,1]
    else:
        raise ValueError('Method not supported')
    
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['no scar', 'scar'])

     # save results to txt file
    with open(os.path.join(outpath, f'results_{target}_{method}.txt'), 'w') as f:
        f.write(f"Accuracy: {round(accuracy,4)}\n")
        f.write(f"Precision: {round(precision,4)}\n")
        f.write(f"AUC: {round(auc,4)}\n")
        f.write(f"MCC: {round(mcc,4)}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\nClassification Report:\n")
        f.write(class_report)

    print(f"Accuracy: {round(accuracy,4)}")
    print(f"Precision: {round(precision,4)}")
    print(f"AUC: {round(auc,4)}")
    print(f"MCC: {round(mcc,4)}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    return (accuracy, precision, auc, mcc)


def run_all(path='../results', infile_features = 'NestedDataAll_clean.csv'):
    """
    Run the preprocess and classify functions for all targets and wavefronts.
    Saves the results to a csv file.
    """
    preprocess()
    results = pd.DataFrame(columns=['target', 'wavefront', 'method', 'accuracy', 'precision', 'auc', 'mcc'])
    for target in ['scar','endocardium_scar','intramural_scar','epicardial_scar']:
        for wavefront in ['SR', 'LVp', 'RVp']:
            for method in ['xgboost', 'rf']:
                accuracy, precision, auc, mcc = classify(path, infile_features, target, wavefront, method)
                results = results.append({'target': target, 'wavefront': wavefront, 'method': method, 'accuracy': accuracy, 'precision': precision, 'auc': auc, 'mcc': mcc}, ignore_index=True)
    results.to_csv(os.path.join(path, 'results_stats_all.csv'), index=False)


def combine_txt():
    """
    Combine all txt files into one incl file name
    """
    outpath = '../results'
    all_txt = ''
    for target in ['scar','endocardium_scar','intramural_scar','epicardial_scar']:
        for wavefront in ['SR', 'LVp', 'RVp']:
            for method in ['xgboost', 'rf']:
                file = os.path.join(outpath, target+'_'+wavefront, f'results_{target}_{method}.txt')
                with open(file, 'r') as f:
                    data = f.read()
                all_txt += f'{file}:\n'
                all_txt += data
                all_txt += '\n\n'
    with open(os.path.join(outpath, 'all_results.txt'), 'w') as f:
        f.write(all_txt)
