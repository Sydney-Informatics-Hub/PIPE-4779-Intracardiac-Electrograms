# Inference Pipeline for ECG Classification (WIP

import os
import pandas as pd
import numpy as np
import logging
import concurrent.futures

# for preprocessing:
#from aggregating_data import retrieve_signal
from data_injest import DataIngest
from combinedata import preprocess_rawsignal_singlefile
from point2mesh import MeshDataMapper
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


def classify_ecg_parallel(path_data, path_models, outname_parquet):
    """
    Classify ECG data with TSai model in parallel (using concurrent.futures.ProcessPoolExecutor)
    Parallel processing for CPU only, not for GPU.
    Not debugged yet! Might not work as expected.

    Input:
        - path_data: path to preprocessed data
        - path_models: list of models
        - outname_parquet: name of parquet file to save predictions

    Returns:
        - dataframe with predictions, probabilities, and coordinates
    
    Example: 
        
    path_data =  "../../../data/deploy/data/preprocessed_rawsignal_unipolar_penta.parquet"
    path_models = [
        "./models/clf_NoScar_RVp_120epochs.pkl", 
        "./models/clf_NoScar_LVp_120epochs.pkl", 
        "./models/clf_NoScar_SR_120epochs.pkl"
    ]
    outname_parquet = "../../../data/deploy/data/predictions_NoScar.parquet"
    """

    # check if path_models is a list
    if not isinstance(path_models, list):
        path_models = [path_models]

     # get coordinates of points
    dfcoord = pd.read_parquet(path_data, columns=['Point_Number','WaveFront','X','Y','Z'])

    def predict_model(path_model):
        """
        Function to predict using a model file path.
        """
        tsai = TSai('','') 
        df = tsai.predict_from_file(path_data, path_model)
        return df

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(predict_model, path_models)
        dfres = pd.concat(results, axis=0)
    # merge predictions with coordinates on Point_Number and Wavefront
    dfres = dfres.merge(dfcoord, on=['Point_Number','WaveFront'], how='left')
    dfres.to_parquet(outname_parquet, index=False)


def classify_ecg(path_data, path_models, outname_parquet):
    """
    Classify ECG data with TSai model

    Input:
        - path_data: path to preprocessed data
        - path_models: list of models
        - outname_parquet: name of parquet file to save predictions

    Returns:
        - dataframe with predictions, probabilities, and coordinates
    
    Example: 
        
    path_data =  "../../../data/deploy/data/preprocessed_rawsignal_unipolar_penta.parquet"
    path_models = [
        "./models/clf_NoScar_RVp_120epochs.pkl", 
        "./models/clf_NoScar_LVp_120epochs.pkl", 
        "./models/clf_NoScar_SR_120epochs.pkl"
    ]
    outname_parquet = "../../../data/deploy/data/predictions_NoScar.parquet"
    """

    # check if path_models is a list
    if not isinstance(path_models, list):
        path_models = [path_models]

     # get coordinates of points
    dfcoord = pd.read_parquet(path_data, columns=['Point_Number','WaveFront','X','Y','Z'])

    tsai = TSai('','') 
    dfres = pd.DataFrame()
    for path_model in path_models:
        df = tsai.predict_from_file(path_data, path_model)
        dfres = pd.concat([dfres, df], axis=0)

    # merge predictions with coordinates on Point_Number and Wavefront
    dfres = dfres.merge(dfcoord, on=['Point_Number','WaveFront'], how='left')
    dfres.to_parquet(outname_parquet, index=False)


def postprocess_data(path_data_export, point_data_file, meshfile, fname_out_vtk, meta_text):
    """
    Postprocess data for inference, including:
        - mapping predictions to mesh
        - saving predictions to file

    Returns:
        projected labels and probabilities on mesh
    """
    mapper = MeshDataMapper(path_data_export, point_data_file, meshfile, fname_out_vtk, meta_text)
    mapper.run()

def main():
    # Preprocess data
    fname_preprocessed = preprocess_data(data_dir)

    # Classify ECG data
    for path_model in path_models:
        # get path name  from path_models
        path = os.path.dirname(path_model)
        # get basename w/o .pkl
        basename = os.path.basename(path_model).split('.')[0]
        outname_parquet = os.path.join(path, f"predictions_{basename}.parquet")
        classify_ecg(fname_preprocessed, path_model, outname_parquet)
        # Postprocess data
        postprocess_data(data_dir, outname_parquet, meshfile, fname_out_vtk, meta_text)
    


if __name__ == '__main__':
    main()

