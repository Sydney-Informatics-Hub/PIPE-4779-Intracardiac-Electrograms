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

import glob
import os
import pandas as pd
import numpy as np
import logging
import argparse

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


def preprocess_data(data_dir, output_dir, catheter_type = "Penta"):
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


def postprocess_data(path_data_export, meshfile, point_data_file, fname_out_vtk, meta_text):
    """
    Postprocess data for inference, including:
        - mapping predictions to mesh
        - saving predictions to file

    Returns:
        projected labels and probabilities on mesh
    """
    mapper = MeshDataMapper(path_data_export, meshfile, point_data_file, fname_out_vtk, meta_text)
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
    print(" Performing data aggregation from path",data_dir,"...")
    if not fname_preprocessed:
        fname_preprocessed = preprocess_data(data_dir)

    print("Starting Inference ...")
    # classify ECG data for each model
    if combine_models:
        print("Generating combined model ...")
        outname_parquet = os.path.join(path_out, f"predictions_combined.parquet")
        classify_ecg(fname_preprocessed, path_model, outname_parquet)
        fname_out_vtk = os.path.join(path_out, f"predictions_combined.vtk")
        postprocess_data(data_dir, meshfile, outname_parquet, fname_out_vtk, meta_text)
    else:
        for path_model in path_models:
            print(f"Processing model {os.path.basename(path_model)} ...")
            # get path name  from path_models
            path = os.path.dirname(path_model)
            # get basename w/o .pkl
            basename = os.path.basename(path_model).split('.')[0]
            outname_parquet = os.path.join(path_out, f"predictions_{basename}.parquet")
            classify_ecg(fname_preprocessed, path_model, outname_parquet)
            # Postprocess data
            fname_out_vtk = os.path.join(path_out, f"predictions_{basename}.vtk")
            postprocess_data(data_dir, meshfile, outname_parquet, fname_out_vtk, meta_text)
    print("Inference completed.")


def test_inference(fname_preprocessed = "../../../data/deploy/data/preprocessed_rawsignal_unipolar_penta.parquet",
                   data_dir = "../../../data/deploy/data/Export_Analysis",
                   path_model = './models',
                   meshfile= '../../../data/deploy/data/Export_Analysis/9-LV SR Penta.mesh',
                   path_out = '../../../data/deploy/test_output',
                   wavefront_selected = None,
                   meta_text = 'Inference'):
    """
    Inference test script.

    This test will run the inference pipeline for ECG classification across all models.
    Then it will save each prediction to a VTK file.

    Input:
        - fname_preprocessed: path to preprocessed data (see function preprocess_data)
    """

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
    #only use a selected wavefront
    if wavefront_selected:
        filtered_models = [model for model in models if wavefront_selected in model]
        models = filtered_models
    print("using models...",models)
    print("checking parameters fname_preprocessed:  ", fname_preprocessed,
          "data_dir: ", data_dir,
          "path_model",path_model,
          "meshfile: ", meshfile,
          "path_out: ", path_out)
    
    print("meta_text set as: ",meta_text)
    combine_models = False
    # check if fname_preprocessed exists
    if fname_preprocessed is not None:
        if not os.path.exists(fname_preprocessed):
            raise FileNotFoundError(f"Preprocessed data file {fname_preprocessed} not found.")
    if fname_preprocessed is None:
        fname_preprocessed = preprocess_data(data_dir, path_out, catheter_type)
    run(data_dir, path_model, models, meshfile, path_out, meta_text, fname_preprocessed, combine_models)

def find_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Mesh file not found matching the pattern: {pattern}")
    return files[0]

def test_injest_and_inference(wavefront_selected = None, catheter_type = "Penta",meta_text = 'Inference: '):
    data_dir = "../deploy/data/Export_Analysis"    
    path_model = './models'
    find_mesh_file = '*RVp*' + catheter_type+ '.mesh'
    pattern_meshfile = os.path.join(data_dir, find_mesh_file) #this name can vary
    try:
        meshfile = find_file(pattern_meshfile)
        print(f"Using mesh file: {meshfile}")
    except FileNotFoundError as e:
        raise(e)

    path_out = '../deploy/output'
    print("Running Data Injest using relative folder ",data_dir)
    fname_preprocessed = preprocess_data(data_dir,path_out,catheter_type) #this will run data injest
    #for testing only
    #fname_preprocessed = "../deploy/output/preprocessed_rawsignal_unipolar_penta.parquet" #testing only
    print("Running Inference ... ")
    test_inference(fname_preprocessed,data_dir,path_model,meshfile,path_out,wavefront_selected,meta_text)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Inference Pipeline for ECG Classification')
    parser.add_argument('--data_dir', type=str, help='Path to raw data')
    parser.add_argument('--path_model', type=str, help='Path to models')
    parser.add_argument('--models', nargs='+', help='List of models')
    parser.add_argument('--meshfile', type=str, help='Path to mesh file')
    parser.add_argument('--path_out', type=str, help='Path to save output')
    parser.add_argument('--meta_text', type=str, help='Metadata text')
    parser.add_argument('--fname_preprocessed', type=str, help='Path to preprocessed data', default=None)
    args = parser.parse_args()

    # check if args are provided, if not ask for user input
    if not args.data_dir:
        args.data_dir = input("Enter path to raw data: ")
    if not args.path_model:
        args.path_model = input("Enter path to models: ")
    if not args.models:
        args.models = input("Enter list of models: ").split()
    if not args.meshfile:
        args.meshfile = input("Enter path to mesh file: ")
    if not args.path_out:
        args.path_out = input("Enter path to save output: ")
    if not args.meta_text:
        args.meta_text = input("Enter metadata text for adding to second line of vtk file: ")

    run(args.data_dir, args.path_model, args.models, args.meshfile, args.path_out, args.meta_text, args.fname_preprocessed)

if __name__ == '__main__':
    #main()
    parser = argparse.ArgumentParser(description='Inference Pipeline for ECG Classification')
    parser.add_argument('--wavefront', type=str, help='Path to raw data')
    parser.add_argument('--catheter', type=str, help='Path to raw data')
    parser.add_argument('--meta', type=str, help='meta text for carto. Default is Inference')
    args = parser.parse_args()
    if not args.wavefront:
        args.wavefront = None
    else:
        if args.wavefront not in ["RVp","SR","LVp"]:
            raise ValueError('wavefront must be either RVp or SR or LVp')
    if not args.catheter:
        args.catheter = "Penta"
    else:
        if args.catheter not in ["Penta","DecaNav"]:
            raise ValueError('wavefront must be either "Penta" or "DecaNav"')

    test_injest_and_inference(args.wavefront,args.catheter,args.meta)