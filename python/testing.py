import pandas as pd
import os
from data_injest import DataIngest
from inference_pipe import preprocess_data, classify_ecg, run


# Generates the prediction files - all of them
def test_python_deploy():
    data_dir = "../deploy/data/Export_Analysis"
    output_dir = "../deploy/data"
    catheter_type = "Penta"
    path_models = [
        "./models/clf_NoScar_RVp_120epochs.pkl", 
        "./models/clf_NoScar_LVp_120epochs.pkl", 
        "./models/clf_NoScar_SR_120epochs.pkl",
        "./models/clf_AtLeastEndo_RVp_120epochs.pkl",
        "./models/clf_AtLeastEndo_LVp_120epochs.pkl",
        "./models/clf_AtLeastEndo_SR_120epochs.pkl",
        "./models/clf_AtLeastIntra_RVp_120epochs.pkl",
        "./models/clf_AtLeastIntra_LVp_120epochs.pkl",
        "./models/clf_AtLeastIntra_SR_120epochs.pkl",
        "./models/clf_epiOnly_RVp_120epochs.pkl",
        "./models/clf_epiOnly_LVp_120epochs.pkl",
        "./models/clf_epiOnly_SR_120epochs.pkl"
    ]

    
    #once run the path can be substituted to save time for testing
    #fname_preprocessed = preprocess_data(data_dir,output_dir,catheter_type)
    fname_preprocessed = "../deploy/data/preprocessed_rawsignal_unipolar_penta.parquet"

    for path_model in path_models:
        # get path name  from path_models
        path = os.path.dirname(path_model)
        # get basename w/o .pkl
        basename = os.path.basename(path_model).split('.')[0]
        outname_parquet = os.path.join(path, f"predictions_{basename}.parquet")
        print("Running prediction basename: ", basename, "outname: ", outname_parquet)
        df = classify_ecg(fname_preprocessed, path_model, outname_parquet)
        # inspecting these is done in r_scripts/testing_predictions.R
    
# Given prediction files create a speicifc mesh    
def generate_mesh_for_specific_models():
    path_models = [
        "./models/clf_NoScar_RVp_120epochs.pkl", 
        "./models/clf_AtLeastEndo_RVp_120epochs.pkl",
        "./models/clf_AtLeastIntra_RVp_120epochs.pkl",
        "./models/clf_epiOnly_RVp_120epochs.pkl",
    ]
    data_dir = "../deploy/data/Export_Analysis"
    output_dir = "../deploy/data"
    catheter_type = "Penta"
    fname_preprocessed = "../deploy/data/preprocessed_rawsignal_unipolar_penta.parquet"

    for path_model in path_models:
        path = os.path.dirname(path_model)
        basename = os.path.basename(path_model).split('.')[0]
        outname_parquet = os.path.join(path, f"predictions_{basename}.parquet")
        postprocess_data(data_dir, outname_parquet, meshfile, fname_out_vtk, meta_text)


def test_inference_RvP():
    """
    Run only RVP
    """
    data_dir = "../deploy/data/Export_Analysis"
    path_model = './models'
    models = [
        "clf_NoScar_RVp_120epochs.pkl", 
        "clf_AtLeastEndo_RVp_120epochs.pkl",
        "clf_AtLeastIntra_RVp_120epochs.pkl",
        "clf_epiOnly_RVp_120epochs.pkl",
    ]
    meshfile= '../deploy/data/Export_Analysis/9-1-ReLV RVp Penta.mesh'
    path_out = '../deploy/data'
    meta_text = 'PatientData S18 S18 4290_S18'
    catheter_type = "Penta"
    fname_preprocessed = "../deploy/data/preprocessed_rawsignal_unipolar_penta.parquet"
    combine_models = False
    # check if fname_preprocessed exists
    if fname_preprocessed is not None:
        if not os.path.exists(fname_preprocessed):
            raise FileNotFoundError(f"Preprocessed data file {fname_preprocessed} not found.")
    if fname_preprocessed is None:
        fname_preprocessed = preprocess_data(data_dir, path_out, catheter_type)
    run(data_dir, path_model, models, meshfile, path_out, meta_text, fname_preprocessed, combine_models)


def test_inference():
    """
    Run all models

    This test will run the inference pipeline for ECG classification across all models.
    Then it will save each prediction to a VTK file.

    Input:
        - fname_preprocessed: path to preprocessed data (see function preprocess_data)
    """
    data_dir = "../deploy/data/Export_Analysis"
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
    meshfile= '../deploy/data/Export_Analysis/9-1-ReLV RVp Penta.mesh'
    path_out = '../deploy/data'
    meta_text = 'PatientData S18 S18 4290_S18'
    catheter_type = "Penta"
    fname_preprocessed = "../deploy/data/preprocessed_rawsignal_unipolar_penta.parquet"
    combine_models = False
    # check if fname_preprocessed exists
    if fname_preprocessed is not None:
        if not os.path.exists(fname_preprocessed):
            raise FileNotFoundError(f"Preprocessed data file {fname_preprocessed} not found.")
    if fname_preprocessed is None:
        fname_preprocessed = preprocess_data(data_dir, path_out, catheter_type)
    run(data_dir, path_model, models, meshfile, path_out, meta_text, fname_preprocessed, combine_models)


#test_python_deploy()
test_inference()