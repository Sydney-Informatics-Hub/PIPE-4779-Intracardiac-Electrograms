# Python Software Kit for ECG Timeseries Classification

## Overview

This software kit provides Python tools for the classification of ECG timeseries data using various methods including feature extraction, feature-based classification, and deep learning. The components of the software are designed to handle preprocessing, classification, and post-processing tasks, ensuring a comprehensive workflow for ECG analysis.

## List of Software Components

### Data Exploration and Model Training
- `features.py`: Feature extraction and selection for ECG data using the `tsfresh` library.
- `classifier_featurebased.py`: Timeseries classification for ECG data using feature-selection and Random Forest/XGBoost.
- `classifier_tsai.py`: Software for training and evaluation of deep Convolutional Neural Network (CNN) models for ECG timeseries classification. Includes also inference 
- `plot_statsresults.py`: Plot and compare statistical results for all classification methods, wavefronts, and target types (Endo, Intra, Epi).
- `test_features.py`: Test functions for the `FeatureExtraction` class in `features.py`.

### Classification Inference and 3D Mapping
- `inference_pipe.py`: Inference pipeline for ECG Classification, including the functions:
    - preprocess_data: Preprocess data for inference
    - classify_ecg: Classify ECG data with TSai model
    - postprocess_data: Postprocess data for inference
- `data_injest.py`: custom pre-processing of raw intracardiac electrograms for ML inference
- `predict_from_file()` in `classifier_tsai.py`: Predict the labels for the given dataframe file using the trained classifier.
- `point2mesh.py`: Interpolate point data onto mesh geometry and convert to Carto vtk format


A detailed description for each module is provided in the file headers. For example:
```python
import classifier_tsai
print(classifier_tsai.__doc__)
```

## Installation

### Feature-Based Classification

To run feature selection and feature-based classification (`classifier_featurebased.py`), install via conda/mamba:
```shell
conda env create -f environment.yaml
conda activate ecg
```

### CNN-Based Classification

To run classification based on CNNs, install via conda/mamba:
```shell
conda env create -f environment_tsai.yml
conda activate tsai
```

## Inference Pipeline for ECG Classification

### Overview

The inference pipeline consists of several stages including preprocessing data, classifying ECG data with TSai models, and post-processing the results. This pipeline ensures a streamlined process for obtaining and interpreting ECG classification results.

### How to Use

1. **Preprocess Data**
    ```python
    from inference_pipe import preprocess_data
    preprocess_data(data_dir, output_dir, catheter_type)
    ```

2. **Classify ECG**
    ```python
    from inference_pipe import classify_ecg
    classify_ecg(path_data, path_models, outname_parquet)
    ```

3. **Postprocess and Map Data**
    ```python
    from inference_pipe import postprocess_data
    postprocess_data(path_data_export, meshfile, point_data_file, fname_out_vtk, meta_text)
    ```

### Complete Example for Inference

```python
from inference_pipe import run

data_dir = "/path/to/raw/data"
path_model = "/path/to/models"
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
meshfile = "/path/to/mesh/file"
path_out = "/path/to/save/output"
meta_text = "Metadata text for VTK file, e.g. 'PatientData S18 S18 4290_S18'"

run(data_dir, path_model, models, meshfile, path_out, meta_text)
```

### Inference via Command Line Arguments

Use the python command to run the inference_pipe script:

```python
python inference_pipe.py \
    --data_dir /path/to/raw/data \
    --path_model /path/to/models \
    --models clf_NoScar_RVp_120epochs.pkl clf_NoScar_LVp_120epochs.pkl clf_NoScar_SR_120epochs.pkl \
    --meshfile /path/to/mesh/file \
    --path_out /path/to/save/output \
    --meta_text "PatientData S18 S18 4290_S18" \
    --fname_preprocessed /path/to/preprocessed/data.parquet
```


Here is a list of command line arguments for inference_pipe:

- `--data_dir`:
  - **Description**: Path to the directory containing raw ECG data.
  - **Type**: `str`
  - **Example**: `--data_dir /path/to/raw/data`

- `--path_model`:
  - **Description**: Path to the directory containing trained models.
  - **Type**: `str`
  - **Example**: `--path_model /path/to/models`

- `--models`:
  - **Description**: List of model filenames to be used for classification.
  - **Type**: `list of str`
  - **Example**: `--models clf_NoScar_RVp_120epochs.pkl clf_NoScar_LVp_120epochs.pkl`

- `--meshfile`:
  - **Description**: Path to the mesh file used for reference geometry.
  - **Type**: `str`
  - **Example**: `--meshfile /path/to/mesh/file`

- `--path_out`:
  - **Description**: Path to the directory where output files will be saved.
  - **Type**: `str`
  - **Example**: `--path_out /path/to/save/output`

- `--meta_text`:
  - **Description**: Metadata text to be added to the second line of the VTK output file.
  - **Type**: `str`
  - **Example**: `--meta_text "Experiment XYZ, Date: 2024-05-15"`

- `--fname_preprocessed`:
  - **Description**: Path to the preprocessed data file. If not provided, data will be preprocessed.
  - **Type**: `str`
  - **Default**: `None`
  - **Example**: `--fname_preprocessed /path/to/preprocessed/data.parquet`


## Attribution and Acknowledgement
Acknowledgments are an important way for us to demonstrate the value we bring to your research. Your research outcomes are vital for ongoing funding of the Sydney Informatics Hub.

If you make use of this software for your research project, please include the following acknowledgment:

“This research was supported by the Sydney Informatics Hub, a Core Research Facility of the University of Sydney."

Author: Sebastian Haan