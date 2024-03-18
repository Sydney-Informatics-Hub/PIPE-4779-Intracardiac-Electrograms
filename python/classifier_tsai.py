# Train and test unsupervised classifier using tsai package
# use conda environment tsai (Python 3.10)


from tsai.basics import (
    Categorize, 
    TSDatasets, 
    Learner, accuracy, 
    TSStandardize, 
    TSClassification,
    ClassificationInterpretation,
    combine_split_data,
    TSClassifier,
    accuracy,
    F1Score,
    Recall,
    Precision,
    ShowGraph
)
from tsai.inference import load_learner
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score
from scipy.signal import resample
import os
import numpy as np
import pandas as pd

inpath = '../../../data/generated'
fname_csv = 'NestedDataS18.csv'

wavefront = 'SR'
target = 'scar'

def load_data(inpath, fname_csv):
    usecols = [
        'Point_Number',
        'WaveFront',
        'sheep',
        'signal_data',
        'endocardium_scar',
        'intramural_scar',
        'epicardial_scar',
    ]
    # check if file exists
    if not os.path.isfile(os.path.join(inpath, fname_csv)):
        raise FileNotFoundError(f'File {fname_csv} not found in {inpath}')
    df = pd.read_csv(os.path.join(inpath, fname_csv), usecols=usecols)
    # remove nan values
    df = df.dropna()
    # add column "time" to df which starts at 0 and has the same length as "signal_data" for each Point_Number and WaveFront
    df['time'] =df.groupby(['Point_Number', 'WaveFront']).cumcount()
    # Generate a column 'scar' that is 1 if either of the scar columns is 1, otherwise 0
    df['endocardium_scar'] = df['endocardium_scar'].astype(int)
    df['intramural_scar'] = df['intramural_scar'].astype(int)
    df['epicardial_scar'] =df['epicardial_scar'].astype(int)
    df['scar'] = df[['endocardium_scar', 'intramural_scar', 'epicardial_scar']].max(axis=1)
    return df

def df_to_ts(df, wavefront, target='scar'):
    """
    Converts the dataframe to tsai format for a given wavefront and target tissue

    Args:
        wavefront (str): 'LVp', 'RVp', or 'SR'
        target (str): 'scar' (Default) or 'endocardium_scar', 'intramural_scar', 'epicardial_scar'
    """
    dfsel = df[df['WaveFront'] == wavefront][['Point_Number', 'time', 'signal_data', target]]
    npoints_unique = dfsel['Point_Number'].nunique()
    signal = [] #np.zeros((npoints_unique, timeseries['signal_data'].apply(len).max()))
    y = dfsel[['Point_Number', target]].drop_duplicates()
    # get length of signal_data for each point
    signal_length = dfsel.groupby('Point_Number')['signal_data'].apply(len)
    X = np.zeros((len(y), signal_length.max()))
    for i in range(len(y)):
        point = y.iloc[i]['Point_Number']
        # get signal_data
        data = dfsel[dfsel['Point_Number'] == point]['signal_data']
        X[i, :len(data)] = data

    return X.reshape((len(y), 1, -1)), y[target].values

def resample_signals(signals, sample_length):
    """
    Resample the signals to a given sample length using Fourier method.

    Args:
        signals (np.array): Array of signals
        sample_length (int): Length of the resampled signal

    Returns:
        np.array: Array of resampled signals    
    """
    resampled_signals = []
    for signal in signals:
        resampled_signal = []
        for si in signal:
            resampled_signal.append(resample(si, sample_length))
        resampled_signal = np.array(resampled_signal)
        resampled_signals.append(resampled_signal)
    return np.array(resampled_signals)

def train_model(X, y):
    """
    Run the model using the tsai package.

    Pipeline:
    - Split data
    - Standardize data
    - Build model
    - Train model
    - Save model
    - Evaluate model
    - Print classification report

    Args:
        X (np.array): Array of signals
        y (np.array): Array of labels
    
    Returns:
        TSClassifier: The trained classifier
    """

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])

    # calculate weights for y
    weights = len(y_train) / (2 * np.bincount(y_train))
    sample_weight = np.zeros(len(y))
    sample_weight[y==0] = weights[0]
    sample_weight[y==1] = weights[1]

    y[y==0] = -1
    
    tfms  = [None, [TSClassification()]]
    batch_tfms = TSStandardize()
    # see https://timeseriesai.github.io/tsai/tslearner.html
    clf = TSClassifier(X, y, 
                    splits=splits, 
                    path='models', 
                    arch="InceptionTimePlus", 
                    tfms=tfms, 
                    batch_tfms=batch_tfms, 
                    #metrics=[accuracy, Precision, Recall],
                    metrics = accuracy,
                    weights = sample_weight,
                    cbs=ShowGraph()
                    )
    clf.fit_one_cycle(200)#, 3e-4)

    # save the model
    clf.export("clf_balanced.pkl")
    # load the model
    #clf = load_learner("models/clf.pkl")
    return clf

def eval_model(clf, X_test, y_test, outpath=None):
    """
    Evaluate the trained classifier using the test data.
    Writes the results to a txt file if outpath is given.

    Args:
        clf (TSClassifier): The trained classifier
        X_test (np.array): Array of test signals
        y_test (np.array): Array of test labels
        outpath (str): Path to save the results to a txt file
    
    """
    y_probs, _, y_pred = clf.get_X_preds(X_test)
    y_pred = np.array([int(t) for t in y_pred])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred) 
    auc = roc_auc_score(y_test, y_probs[:,1])
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['no scar', 'scar'])

    print(f"Accuracy: {round(accuracy,4)}")
    print(f"Precision: {round(precision,4)}")
    print(f"AUC: {round(auc,4)}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

     # save results to txt file
    if outpath:
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, f'results_{target}_{wavefront}.txt'), 'w') as f:
            f.write(f"Accuracy: {round(accuracy,4)}\n")
            f.write(f"Precision: {round(precision,4)}\n")
            f.write(f"AUC: {round(auc,4)}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(conf_matrix))
            f.write("\nClassification Report:\n")
            f.write(class_report)

def predict(X, clf):
    """
    Predict the labels for the given signals using the trained classifier.

    Args:
        X (np.array): Array of signals
        clf (TSClassifier): The trained classifier
    
    Returns:
        np.array: Array of predicted labels
    """
    probas, _, preds = clf.get_X_preds(X)
    return preds, probas


def main():
    # load data
    df = load_data(inpath, fname_csv)
    X_orig, y_orig = df_to_ts(df, wavefront=wavefront, target=target)
    # train and evalutae model
    clf = train_model(X_orig, y_orig)
    eval_model(clf, X_orig, y_orig, outpath='results')

