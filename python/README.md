# Python Software Kit for ECG Timeseries Classification 

## List of software components

- `features.py`: Feature extraction and selection for ECG data using the tsfresh library.
- `classifier_featurebased.py`: Timeseries classification for ECG data using feature-selection and Random Forest/XGboost.
- `classifier_tsai.py`: Software for training and evaluation of deep Convolutional Neural Network (CNN) models for ECG timeseries classification 
- `plot_statsresults.py`: Plot and compare statsresults for all classifiaction methods, wavefronts, and target types (Endo, Intra, Epi)
- `test_features.py`: test functions for class FeatureExtraction in features.py

A detailed description for each module is provided in the file headers, e.g.
```python
import classifier_tsai
print(classifier_tsai.__doc__)
```

## Installation

To run Feature selection and feature based classification (classifier_featurebased.py`), install via conda/mamba: 
```shell
conda env create -f environment.yaml
conda activate ecg
```

To run classification based on CNNs, install via conda/mamba:
```shell
conda env create -f environment_tsai.yml
conda activate tsai
```
