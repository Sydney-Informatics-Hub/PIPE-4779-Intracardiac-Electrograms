"""
Inference Pipeline for ECG Classification.

How to use:
    - preprocess_data: Preprocess data for inference
    - classify_ecg: Classify ECG data with TSai model
    - postprocess_data: Postprocess data for inference

Python example:
from inference_pipe import run
run(data_dir, path_model, models, meshfile, path_out, meta_text, fname_preprocessed, combine_models)
"""

import os
import pandas as pd
import numpy as np
import logging

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
mesh_file= '../../../data/deploy/data/Export_Analysis/9-LV SR Penta.mesh'
path_model = './models'
combine_models = False
vtk_meta_text = 'PatientData S18 S18 4290_S18'
fname_out = '../../../data/deploy/data/predictions_mapped_NoScar.vtk'
path_data_parquet =  "../../../data/deploy/data/preprocessed_rawsignal_unipolar_penta.parquet"

catheter_type = "Penta"

models = [
        "clf_NoScar_RVp_120epochs.pkl", 
        "clf_NoScar_LVp_120epochs.pkl", 
        "clf_NoScar_SR_120epochs.pkl",
        "clf_AtLeastEndo_RVp_120epochs.pkl",
        "clf_AtLeastEndo_LVp_120epochs.pkl",
        "clf_AtLeastEndo_SR_120epochs.pkl",
        "clf_AtLeastIntra_RVp_120epochs.pkl",
        "clf_AtLeastIntra_LVp_120epochs.pkl",
        "clf_AtLeastIntra_SR_120epochs.pkl",
        "clf_epiOnly_RVp_120epochs.pkl",
        "clf_epiOnly_LVp_120epochs.pkl",
        "clf_epiOnly_SR_120epochs.pkl"
    ]

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
    import concurrent.futures

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
        
    path_data =  "../../../data/deploy/data/S18_RVp_NoScar_groundtruth.parquet"
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


def run(data_dir, 
        path_model, 
        models, 
        meshfile, 
        path_out, 
        meta_text, 
        fname_preprocessed = None, 
        combine_models = False):
    """
    Run inference pipeline for ECG classification

    Input:
        - data_dir: path to raw data
        - path_model: path to models
        - models: list of models
        - meshfile: path to mesh file for reference geometry that is mapped to
        - path_out: path to save output
        - meta_text: metadata text. This text will be written to the second line of the VTK file.
        - fname_preprocessed: path to preprocessed data. If None, preprocess data. Default None
        - combine_models: boolean to combine models, default False
    
    """
    # check that data_dir, path_model, and meshfile exist
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found.")
    if not os.path.exists(path_model):
        raise FileNotFoundError(f"Model directory {path_model} not found.")
    if not os.path.exists(meshfile):
        raise FileNotFoundError(f"Mesh file {meshfile} not found.")
    # check that models exist
    path_models = [os.path.join(path_model, m) for m in models]
    for path_model in path_models:
        if not os.path.exists(path_model):
            raise FileNotFoundError(f"Model file {path_model} not found.")

    if not os.path.exists(path_out):
        os.makedirs(path_out, exist_ok=True)

    # preprocess data if needed
    if not fname_preprocessed:
        fname_preprocessed = preprocess_data(data_dir)

    # classify ECG data for each model
    if combine_models:
        print("Generating combined model ...")
        outname_parquet = os.path.join(path, f"predictions_combined.parquet")
        classify_ecg(fname_preprocessed, path_model, outname_parquet)
        fname_out_vtk = os.path.join(path_out, f"predictions_combined.vtk")
        postprocess_data(data_dir, outname_parquet, meshfile, fname_out_vtk, meta_text)
    else:
        for path_model in path_models:
            print(f"Processing model {os.path.basename(path_model)} ...")
            # get path name  from path_models
            path = os.path.dirname(path_model)
            # get basename w/o .pkl
            basename = os.path.basename(path_model).split('.')[0]
            outname_parquet = os.path.join(path, f"predictions_{basename}.parquet")
            classify_ecg(fname_preprocessed, path_model, outname_parquet)
            # Postprocess data
            fname_out_vtk = os.path.join(path_out, f"predictions_{basename}.vtk")
            postprocess_data(data_dir, outname_parquet, meshfile, fname_out_vtk, meta_text)
    print("Inference completed.")


def test_inference():
    data_dir = "../../../data/deploy/data/Export_Analysis"
    path_model = './models'
    models = [
        "clf_NoScar_RVp_120epochs.pkl", 
        "clf_NoScar_LVp_120epochs.pkl", 
        "clf_NoScar_SR_120epochs.pkl",
        "clf_AtLeastEndo_RVp_120epochs.pkl",
        "clf_AtLeastEndo_LVp_120epochs.pkl",
        "clf_AtLeastEndo_SR_120epochs.pkl",
        "clf_AtLeastIntra_RVp_120epochs.pkl",
        "clf_AtLeastIntra_LVp_120epochs.pkl",
        "clf_AtLeastIntra_SR_120epochs.pkl",
        "clf_epiOnly_RVp_120epochs.pkl",
        "clf_epiOnly_LVp_120epochs.pkl",
        "clf_epiOnly_SR_120epochs.pkl"
    ]
    #models = [
    #    "clf_NoScar_SR_120epochs.pkl",
    #    "clf_AtLeastEndo_SR_120epochs.pkl",
    #    "clf_AtLeastIntra_SR_120epochs.pkl",
    #    "clf_epiOnly_SR_120epochs.pkl"
    #]
    meshfile= '../../../data/deploy/data/Export_Analysis/9-LV SR Penta.mesh'
    path_out = '../../../data/deploy/data'
    meta_text = 'PatientData S18 S18 4290_S18'
    combine_models = False
    fname_preprocessed = preprocess_data(data_dir, path_out, catheter_type)
    
    #fname_preprocessed = "../../../data/deploy/data/preprocessed_rawsignal_unipolar_penta.parquet"
    run(data_dir, path_model, models, meshfile, path_out, meta_text, fname_preprocessed, combine_models)


    
def main():
    test_inference()

if __name__ == '__main__':
    main()

