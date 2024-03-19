# Class for traininig and evaluation of unsupervised classifier using tsai package
# use conda environment tsai (Python 3.10)


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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score
from scipy.signal import resample
import os
import numpy as np
import pandas as pd

inpath = '../results'
fname_csv = 'NestedDataAll_clean.csv'

wavefront = 'SR'
target = 'scar'



class TSai:
    """
    Class to train and evaluate a classifier using the tsai package.

    This class allows to load the data, convert it to the tsai format, train the model, and evaluate the model.

    Args:
        inpath (str): Path to the input data
        fname_csv (str): Filename of the csv file containing the data

    Example:
        tsai = TSai(inpath, fname_csv)
        df = tsai.load_data(inpath, fname_csv)
        X, y = tsai.df_to_ts(df, wavefront, target)
        tsai.train_model(X, y)
        tsai.eval_model()
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

    def df_to_ts(self, df, wavefront, target='scar'):
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
        signal_length_max = signal_length.max()
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
        outname = f"clf_{target}_{wavefront}_{epochs}epochs.pkl"
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
            tuple: Tuple of accuracy, precision, and AUC
        
        """
        y_probs, _, y_pred = self.clf.get_X_preds(self.X_test)
        y_pred = np.array([int(t) for t in y_pred])
        y_pred[y_pred==-1] = 0
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred) 
        auc = roc_auc_score(self.y_test, y_probs[:,1])
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred, target_names=['no scar', 'scar'])

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
        return (accuracy, precision, auc)

    def predict(self, X, reload_model_from_path=None):
        """
        Predict the labels for the given signals using the trained classifier.

        Args:
            X (np.array): Array of signals
            load_model_from_path (str): Path to the trained model to reload (optional, default: None)
        
        Returns:
            np.array: Array of predicted labels
            np.array: Array of predicted probabilities

        """
        if reload_model_from_path:
            self.clf = load_learner(path_model)
        probas, _, preds = self.clf.get_X_preds(X)
        return preds, probas
    
    

def run_all(inpath, fname_csv):
    tsai = TSai(inpath, fname_csv)
    df = tsai.load_data(inpath, fname_csv)
    results = pd.DataFrame(columns=['target', 'wavefront', 'method', 'accuracy', 'precision', 'auc'])
    method = 'CNN'
    path = '../results/tsai'
    for target in ['scar','endocardium_scar','intramural_scar','epicardial_scar']:
        for wavefront in ['SR', 'LVp', 'RVp']:
            X, y = tsai.df_to_ts(df, wavefront, target)
            tsai.train_model(X, y, epochs = 120, balance_classes = True)
            path_name = path + f'_{target}_{wavefront}' 
            accuracy, precision, auc = tsai.eval_model(outpath=path_name)
            new_row = [{'target': target, 'wavefront': wavefront, 'method': method, 'accuracy': accuracy, 'precision': precision, 'auc': auc}]
            results = pd.concat([results, pd.DataFrame(new_row)], ignore_index=True)
    results.to_csv(os.path.join(path, 'results_stats_all.csv'), index=False)


def test_tsai(wavefront, target, inpath, fname_csv):
    tsai = TSai(inpath, fname_csv)
    df = tsai.load_data(inpath, fname_csv)
    X, y = tsai.df_to_ts(df, wavefront, target)
    tsai.train_model(X, y, epochs = 100, balance_classes = True)
    path_name = '../results/tsai' + f'_{target}_{wavefront}' 
    accuracy, precision, auc = tsai.eval_model(outpath=path_name)