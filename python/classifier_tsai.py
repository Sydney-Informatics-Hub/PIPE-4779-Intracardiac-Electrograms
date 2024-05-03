""" Software for training and evaluation of CNN models for ECG Time Series Classification (TSC).

This classifier uses InceptionTime, which is an ensemble of five deep Convolutional Neural Network (CNN) models for TSC.
The implementation leverages the python package tsai: https://timeseriesai.github.io/tsai

TSAI assumes that the input data is pre-processed (see preprocessing functions in combinedata.py) and contains ECG signals for different wavefronts and target labels.

The input data is expected to be in csv or parquet format and should contain the following columns:
- Point_Number: Unique identifier for each data point
- WaveFront: 'LVp', 'RVp', or 'SR'
- signal_data: ECG signal data as float values
and the columns for the target labels, e.g.:
- scar: 1 if there is a scar, 0 otherwise
- endocardium_scar: 1 if there is an endocardial scar, 0 otherwise
- intramural_scar: 1 if there is an intramural scar, 0 otherwise
- epicardial_scar: 1 if there is an epicardial scar, 0 otherwise

How to use for training and evaluation:
    inpath = 'DATA_PATH'
    fname_csv = 'DATA_FILE.csv or DATA_FILE.parquet'
    outpath = 'OUTPUT_PATH'
    run_all(inpath, 
            fname_csv, 
            outpath,
            target_list = ['NoScar', 'AtLeastEndo', 'AtLeastIntra', 'epiOnly'],
            method = 'CNN')

or just run the script via command line:
    python classifier_tsai.py


The results are saved to a csv file.

References: 
Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., … & Petitjean, F. (2020). 
Inceptiontime: Finding alexnet for time series classification. Data Mining and Knowledge Discovery, 34(6), 1936-1962.
Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime

Functionality:
    - load ECG data
    - convert data to tsai format for univariate TSC
    - train InceptionTime
    - Evaluation of model
    - Inference

Installation:
use conda environment tsai (Python 3.10), see environment_tsai.yml

How to use:
    from classifier_tsai import TSai
    tsai = TSai(inpath, fname_csv)
    df = tsai.load_data(inpath, fname_csv)
    X, y = tsai.df_to_ts(df, wavefront, target)
    tsai.train_model(X, y)
    accuracy, precision, auc, mcc = tsai.eval_model()

Author: Sebastian Haan
"""

from tsai.basics import (
    Learner,
    TSStandardize, 
    TSClassification,
    combine_split_data,
    TSClassifier,
    ShowGraph
)
from tsai.basics import accuracy as tsai_accuracy
from tsai.inference import load_learner
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    precision_score,
    matthews_corrcoef)
from scipy.signal import resample
import os
import numpy as np
import pandas as pd
import time



class TSai:
    """
    This classifier uses InceptionTime, which is an ensemble of five deep Convolutional Neural Network (CNN) models for TSC.
    The implementation leverages the python package tsai: https://timeseriesai.github.io/tsai

    This class includes functionality to:
      - load ECG data
      - convert data to tsai format for univariate TSC
      - train InceptionTime
      - Evaluation of model
      - Inference

    Args:
        inpath (str): Path to the input data
        fname_csv (str): Filename of the csv file containing the data

    Example:
        tsai = TSai(inpath, fname_csv)
        df = tsai.load_data(inpath, fname_csv)
        X, y = tsai.df_to_ts(df, wavefront, target)
        tsai.train_model(X, y)
        accuracy, precision, auc, mcc = tsai.eval_model()
    """

    def __init__(self, inpath, fname_csv):
        self.inpath = inpath
        self.fname_csv = fname_csv


    def load_data(self, inpath, fname_csv):
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
        # check if file exists and csv
        if fname_csv.endswith('.csv'):
            df = pd.read_csv(os.path.join(inpath, fname_csv), usecols=usecols)
        elif fname_csv.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(inpath, fname_csv), columns=usecols)
        else:
            raise ValueError(f'File {fname_csv} is not a csv or parquet file')
        # remove nan values
        df = df.dropna()
        # add column "time" to df which starts at 0 and has the same length as "signal_data" for each Point_Number and WaveFront
        df['time'] =df.groupby(['Point_Number', 'WaveFront']).cumcount()
        # Generate a column 'scar' that is 1 if either of the scar columns is 1, otherwise 0
        df['endocardium_scar'] = df['endocardium_scar'].astype(int)
        df['intramural_scar'] = df['intramural_scar'].astype(int)
        df['epicardial_scar'] =df['epicardial_scar'].astype(int)
        df['scar'] = df[['endocardium_scar', 'intramural_scar', 'epicardial_scar']].max(axis=1)
        df['NoScar'] = 1 - df['scar']
        df['AtLeastEndo'] = df['endocardium_scar']
        df['AtLeastIntra'] = df['intramural_scar'] & ~df['endocardium_scar']
        df['epiOnly'] = df['epicardial_scar'] & ~df['endocardium_scar'] & ~df['intramural_scar']
        return df

    def df_to_ts(self, df, wavefront, target='scar'):
        """
        Converts the dataframe to tsai format for a given wavefront and target tissue

        Args:
            wavefront (str): 'LVp', 'RVp', or 'SR'
            target (str): 'scar' (Default) or 'endocardium_scar', 'intramural_scar', 'epicardial_scar'
        """
        self.wavefront = wavefront
        self.target = target
        dfsel = df[df['WaveFront'] == wavefront][['Point_Number', 'time', 'signal_data', target]]
        npoints_unique = dfsel['Point_Number'].nunique()
        signal = [] #np.zeros((npoints_unique, timeseries['signal_data'].apply(len).max()))
        y = dfsel[['Point_Number', target]].drop_duplicates()
        # get length of signal_data for each point
        signal_length = dfsel.groupby('Point_Number')['signal_data'].apply(len)
        signal_length_max = signal_length.max()
        print(f"Max signal length: {signal_length_max}")
        X = np.zeros((len(y), signal_length_max))
        #aggregate 'signal_data' directly 
        aggregated_data = dfsel.groupby('Point_Number')['signal_data'].agg(list)
        for i, point in enumerate(y['Point_Number']):
            data = np.array(aggregated_data[point])
            X[i, :len(data)] = data

        return X.reshape((len(y), 1, -1)), y[target].values

    def resample_signals(self, signals, sample_length):
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

    def train_model(self, X, y, epochs = 100, balance_classes = True):
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
            epochs (int): Number of epochs to train the model (Default: 100)
            balance_classes (bool): Whether to balance the classes (Default: True)
        
        Returns:
            TSClassifier: The trained classifier
        """

        # split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.X, self.y, self.splits = combine_split_data([self.X_train, self.X_test], [self.y_train, self.y_test])

        # calculate weights for y
        if balance_classes:
            weights = len(self.y_train) / (2 * np.bincount(self.y_train))
            self.sample_weight = np.zeros(len(y))
            self.sample_weight[y==0] = weights[0]
            self.sample_weight[y==1] = weights[1]
        else:
            self.sample_weight = None

        self.y[self.y==0] = -1
        
        tfms  = [None, [TSClassification()]]
        batch_tfms = TSStandardize()
        # see https://timeseriesai.github.io/tsai/tslearner.html
        self.clf = TSClassifier(self.X, self.y, 
                        splits=self.splits, 
                        path='models', 
                        arch="InceptionTimePlus", 
                        tfms=tfms, 
                        batch_tfms=batch_tfms, 
                        metrics = tsai_accuracy,
                        weights = self.sample_weight,
                        #cbs=ShowGraph()
                        )
        self.clf.fit_one_cycle(epochs)#, 3e-4)

        # save the model
        outname = f"clf_{self.target}_{self.wavefront}_{epochs}epochs.pkl"
        self.clf.export(outname)
        # load the model
        #clf = load_learner("models/clf.pkl")

    def eval_model(self, outpath=None):
        """
        Evaluate the trained classifier using the test data.
        Writes the results to a txt file if outpath is given.

        Args:
            outpath (str): Path to save the results to a txt file

        Returns:
            tuple: Tuple of accuracy, precision, AUC, and MCC (Matthews correlation coefficient)
        
        """
        y_probs, _, y_pred = self.clf.get_X_preds(self.X_test)
        y_pred = np.array([int(t) for t in y_pred])
        y_pred[y_pred==-1] = 0
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred) 
        auc = roc_auc_score(self.y_test, y_probs[:,1])
        mcc = matthews_corrcoef(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred, target_names=['no scar', 'scar'])

        print(f"Accuracy: {round(accuracy,4)}")
        print(f"Precision: {round(precision,4)}")
        print(f"AUC: {round(auc,4)}")
        print(f"MCC: {round(mcc,4)}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        # save results to txt file
        if outpath:
            os.makedirs(outpath, exist_ok=True)
            with open(os.path.join(outpath, f'results_{self.target}_{self.wavefront}.txt'), 'w') as f:
                f.write(f"Accuracy: {round(accuracy,4)}\n")
                f.write(f"Precision: {round(precision,4)}\n")
                f.write(f"AUC: {round(auc,4)}\n")
                f.write(f"MCC: {round(mcc,4)}\n")
                f.write("Confusion Matrix:\n")
                f.write(str(conf_matrix))
                f.write("\nClassification Report:\n")
                f.write(class_report)
        return (accuracy, precision, auc, mcc)

    def predict(self, X, path_model):
        """
        Predict the labels for the given signals using the trained classifier.
        The filename of the model needs to include the target name and wavefront name.

        Args:
            X (np.array): Array of signals
            path_model (str): Path to save the trained model 
        
        Returns:
            np.array: Array of predicted labels
            np.array: Array of predicted probabilities

        """
        clf = load_learner(path_model)
        probas, _, preds = clf.get_X_preds(X)
        return preds, probas
    
    def predict_from_file(self, path_data, path_model):
        """
        Predict the labels for the given dataframe file using the trained classifier.
        The filename of the model needs to include the  wavefront name.

        Args:
            path_data (str): Path/filename to the input data, including columns:
                'Point_Number', 
                'WaveFront', 
                'signal_data', 
            path_model (str): Path/filename to the trained model

        Returns:
            np.array: Array of predicted labels
            np.array: Array of predicted probabilities

        """
        usecols = [
            'Point_Number',
            'WaveFront',
            'signal_data'
        ]
        # check if file exists
        if not os.path.isfile(\path_data):
            raise FileNotFoundError(f'File {path_data} not found in')
        # check if file exists and csv
        if path_data.endswith('.csv'):
            df = pd.read_csv(path_data, usecols=usecols)
        elif path_data.endswith('.parquet'):
            df = pd.read_parquet(path_data, columns=usecols)
        else:
            raise ValueError(f'File {path_data} is not a csv or parquet file')
        
        column_names = list(df)
        fname_model = os.path.basename(path_model)

        # Check if model filename is valid, need to include at least 3 underscores and end with .pkl
        if fname_model.count('_') < 2 or not fname_model.endswith('.pkl'):
            raise ValueError(f'Model file name {fname_model} not valid. Must include at least 2 underscores and end with .pkl')
        # Try to extract wavefront and target name from filename (e.g.: clf_AtLeastIntra_RVp_120epochs.pkl)
        target = fname_model.split('_')[1]
        wavefront = fname_model.split('_')[2]
        # check that wavefront is either LVp, RVp or SR
        if wavefront not in ['LVp', 'RVp', 'SR']:
            raise ValueError(f'Wavefront name {wavefront} in model file not recognized. Must be LVp, RVp, or SR') 
        #print(f'Loaded {len(df)} datapoints from file.')
        print(f'Extracting signal data for wavefront {wavefront}')
        X, y = self.df_to_ts(df, wavefront, target)
        print(f'Predicting labels and probabilities for {len(X)} signals...')
        preds, probas = self.predict(X, path_model)  
        # Adding coordinates
        print('Predictions done.')  
        return preds, probas
    
    

def run_all(inpath,
            fname_csv, 
            outpath, 
            target_list = ['NoScar', 'AtLeastEndo', 'AtLeastIntra', 'epiOnly'],
            method = 'CNN'):
    """
    Train and evaluate the model for all targets and wavefronts.

    Args:
        inpath (str): Path to the input data
        fname_csv (str): Filename of the csv file containing the data
        target_list (list): List of target labels (Default: ['scar','endocardium_scar','intramural_scar','epicardial_scar'])
            example options: ['NoScar', 'AtLeastEndo', 'AtLeastIntra', 'epiOnly'] 
            or ['scar','endocardium_scar','intramural_scar','epicardial_scar']
        method (str): Method to use (Default: 'CNN') 
        rawsignal (bool): Whether to use raw signal data (Default: True) or window_of_interest data
    """
    tsai = TSai(inpath, fname_csv)
    df = tsai.load_data(inpath, fname_csv)
    results = pd.DataFrame(columns=['target', 'wavefront', 'method', 'accuracy', 'precision', 'auc', 'mcc'])
    # date and time in string format
    os.makedirs(outpath, exist_ok=True)
    for target in target_list:
        for wavefront in ['SR', 'LVp', 'RVp']:
            X, y = tsai.df_to_ts(df, wavefront, target)
            tsai.train_model(X, y, epochs = 120, balance_classes = True)
            path_name = outpath + f'_{target}_{wavefront}' 
            accuracy, precision, auc, mcc = tsai.eval_model(outpath=path_name)
            new_row = [{'target': target, 
                        'wavefront': wavefront, 
                        'method': method, 
                        'accuracy': accuracy, 
                        'precision': precision, 
                        'auc': auc,
                        'mcc': mcc}]
            results = pd.concat([results, pd.DataFrame(new_row)], ignore_index=True)
    results.to_csv(os.path.join(outpath, 'results_stats_all.csv'), index=False)


def test_tsai(wavefront, target, inpath, fname_csv):
    """
    Test the TSai classifier for a given wavefront and target.

    Args:
        wavefront (str): 'LVp', 'RVp', or 'SR'
        target (str): 'scar' (Default) or 'endocardium_scar', 'intramural_scar', 'epicardial_scar'
        inpath (str): Path to the input data
        fname_csv (str): Filename of the csv file containing the data
    """
    tsai = TSai(inpath, fname_csv)
    df = tsai.load_data(inpath, fname_csv)
    X, y = tsai.df_to_ts(df, wavefront, target)
    tsai.train_model(X, y, epochs = 120, balance_classes = True)
    path_name = '../results/tsai' + f'_{target}_{wavefront}' 
    accuracy, precision, auc, mcc = tsai.eval_model(outpath=path_name)


def test_all():
    inpath = '../results'
    #fname_csv = 'NestedDataAll_clean.csv'
    #fname_csv = 'NestedDataAll_rawsignal_clean.parquet'
    fname_csv = 'NestedDataAll_rawsignal_unipolar.parquet'
    outpath = '../results/tsai_test_raw_unipolar'
    run_all(inpath, 
        fname_csv, 
        outpath,
        target_list = ['NoScar', 'AtLeastEndo', 'AtLeastIntra', 'epiOnly'],
        method = 'CNN')
    

if __name__ == '__main__':
    inpath = input("Enter the path to the input data: ")
    fname_csv = input("Enter the filename of the csv/parquet file containing the pre-processed data: ")
    outpath = input("Enter the path to the output folder: ")
    run_all(inpath, 
            fname_csv, 
            outpath,
            target_list = ['NoScar', 'AtLeastEndo', 'AtLeastIntra', 'epiOnly'],
            method = 'CNN')
