# Inference Pipeline for ECG Classification

import os
import pandas as pd
import numpy as np
import logging

# for preprocessing:
from aggregating_data import retrieve_signal
from combinedata import preprocess_rawsignal_singlefile
# for inference with tsai model:
from classifier_tsai import TSai

# Settings
data_dir = 'data'
fname_data = 'data.csv'
path_model = 'model'
output_dir = '../../results/inference/'



# Set up logging
logging.basicConfig(level=logging.INFO)


def preprocess_data(data_dir):
    """
    Preprocess data for inference, including:
        - loading data
        - cleaning and filtering  data
        - converting to ML-ready format (TSai)
    """
    pass


def classify_ecg(model, path_data, fname_data, path_model, save_results=False):
    """
    Classify ECG data with TSai model

    Returns:
        - y_pred: predicted class labels
        - y_proba: predicted probabilities
    """
    # 
    tsai = TSai(path_data, fname_data)
    y_pred, y_proba = tsai.predict_from_file(os.path.join(path_data, fname_data), path_model)
    if save_results:
        # save y_pred and y_proba to csv file
        df = pd.DataFrame({'labels_pred': y_pred, 'labels_proba': y_proba})
    


def postprocess_data(labels_points, proba_points, meshfile_ref):
    """
    Postprocess data for inference, including:
        - mapping predictions to mesh
        - saving predictions to file

    Returns:
        projected labels and probabilities on mesh
    """
    pass

def main():
    # Preprocess data
    preprocess_data(data_dir)

    # Classify ECG data
    labels_points, proba_points = classify_ecg(model, datafile)

    # Postprocess data
    postprocess_data(labels_points, proba_points, meshfile_ref)


if __name__ == '__main__':
    main()

