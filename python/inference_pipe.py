# Inference Pipeline for ECG Classification (WIP

import os
import pandas as pd
import numpy as np
import logging

# for preprocessing:
#from aggregating_data import retrieve_signal
from data_injest import DataIngest
from combinedata import preprocess_rawsignal_singlefile
from points2mesh import MeshDataMapper
# for inference with tsai model:
from classifier_tsai import TSai

# Settings
raw_data_path = "../../../data/deploy/data/Export_Analysis"
export_analysis_path =  "../../../data/deploy/data"
fname_data = 'data.csv'
path_model = 'models'

catheter_type = "Penta"



# Set up logging
logging.basicConfig(level=logging.INFO)


def preprocess_data(data_dir, output_dir, catheter_type):
    """
    Preprocess data for inference, including:
        - loading data
        - cleaning and filtering  data
        - converting to ML-ready format (TSai)

    Input:
        - data_dir: path to raw data
        - output_dir: path to save preprocessed data
        - catheter_type: type of catheter used for data collection

    Returns:
        - path to preprocessed data
    """
    data_ingest = DataIngest(data_dir, output_dir, catheter_type)
    data_ingest.collect_data()
    filename_output = data_ingest.filename_output 
    return os.path.join(output_dir, filename_output)


def classify_ecg(model, path_data, path_model):
    """
    Classify ECG data with TSai model

    Input:
        - model: TSai model
        - path_data: path to preprocessed data
        - path_model: list of to model
        - save_results: flag to save results to file

    Returns:
        - y_pred: predicted class labels
        - y_proba: predicted probabilities
    """
    # 
    #path_data =  "../../../data/deploy/data/preprocessed_rawsignal_unipolar_penta.parquet"
    #path_model = "./models/clf_NoScar_SR_120epochs.pkl"

    tsai = TSai('','')
    dfres = tsai.predict_from_file(path_data, path_model)
    # get coordinates of points
    dfcoord = pd.read_parquet(path_data, columns=['Point_Number','WaveFront','X','Y','Z'])
    # merge predictions with coordinates on Point_Number and Wavefront
    dfres = dfres.merge(dfcoord, on=['Point_Number','WaveFront'], how='left')
    dfres.to_parquet(os.path.join(export_analysis_path, 'predictions.parquet'), index=False)
    


def postprocess_data(labels_points, proba_points, meshfile_ref):
    """
    Postprocess data for inference, including:
        - mapping predictions to mesh
        - saving predictions to file

    Returns:
        projected labels and probabilities on mesh
    """
    mapper = MeshDataMapper(meshfile_ref, labels_points, proba_points, 'predictions.vtk', fname_out_vtk, meta_text)
    mapper.run()

def main():
    # Preprocess data
    preprocess_data(data_dir)

    # Classify ECG data
    labels_points, proba_points = classify_ecg(model, datafile)

    # Postprocess data
    postprocess_data(labels_points, proba_points, meshfile_ref)


if __name__ == '__main__':
    main()

