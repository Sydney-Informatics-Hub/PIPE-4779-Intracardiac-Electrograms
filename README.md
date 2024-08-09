# PIPE-4779-Intracardiac-Electrograms

## Introduction
Ventricular Tachycardia is a life-threatening condition where rapid beating of the lower chambers of the heart can lead to sudden death or fainting. It can be treated with procedures performed by cardiac electrophysiologists called catheter ablations whereby electrical catheters are navigated into the heart chambers and the electrical signals of the heart are mapped in real time. Specific patterns in these electrical signals can be interpreted to identify scar tissue in the bottom chambers of the heart (ventricular scar) which give rise to these dangerous heart rhythms. This scar can be on the inner surface of the heart (easy to identify) but may also be in the middle or outer layers of the heart muscle (which can be missed), leading to failure of catheter ablation. Our program is aimed at improving identification of ventricular scar so as to improve outcomes of patients undergoing catheter ablation.

3D maps of the electrical activation of the heart can be reconstructed during catheter ablation whereby 1000-5000 points per map are collected. At the moment, cardiologists use the amplitude of these electrical signals (also called intracardiac electrograms) to identify scar vs healthy tissue. Also fractionation or splitting or high frequency deflections in these signals can also identify scar. For the moment, these signals are only grossly interpreted (visually).

## Goal

The aim is to improve our understanding of these electrograms and identify features (and hidden features) which may identify hidden middle layer and outer layer scar. To do this we have performed an animal experiment - infarcting 5+ sheep and collecting meticulous tagged electrograms as well as co-registered cardiac MRI and whole heart histology. This way the electrograms can be sorted by the histology (and co-registered at specific points in space). This project will develop and test machine learning models and advanced signal processing tools that aim improve the ability of electrograms to identify ventricular scar.

## Project Partners
A/Prof Saurabh Kumar, Director of Complex Arrhythmia Program, Westmead Hospital / Westmead Applied Research Centre

## Attribution and Acknowledgement
Acknowledgments are an important way for us to demonstrate the value we bring to your research. Your research outcomes are vital for ongoing funding of the Sydney Informatics Hub.

If you make use of this software for your research project, please include the following acknowledgment:

“This research was supported by the Sydney Informatics Hub, a Core Research Facility of the University of Sydney."

## Deployment steps

1. **Clone Repo**. Clone this repository with `git clone https://github.com/Sydney-Informatics-Hub/PIPE-4779-Intracardiac-Electrograms.git` and change directory into it `cd PIPE-4779-Intracardiac-Electrograms`. If this is your first time using git, more information can be found [here](https://www.atlassian.com/git/tutorials/what-is-version-control). SSH keys need to be used for authentication and instructions can be found [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys). 

2. **Load Models**. Pretrained models can be found [here](https://unisyd.sharepoint.com/:f:/r/sites/ComplexArrhythmiaProgram/Shared%20Documents/Project%20Data/Bioinformatics%20EGM%20signal%20analysis/SIH/models/tsai_rawsignal_unipolar?csf=1&web=1&e=HoY3XN) and need to be copied to the `/python/models` subfolder.

3. **Load Data**. A subfolder called `/deploy/data/Export_Analysis` is where all the CARTO data should be kept under when making predictions on scar tissue by depth. The model inference will scan these files during the data injest and create a parquet file in `/deploy/output/preprocessed_rawsignal_unipolar_penta.parquet`that aggregates the data injestion. This file will be overwritten with each run of the data injest process.

4. **Create Environment**. A conda environment called tsai is used to run these scripts. Once [minicoda](https://docs.anaconda.com/miniconda/) is installed, in the `/python` subfolder run `conda env create -f environment_tsai.yml` then activate the environment with `conda activate tsai`. The environment only needs to be created once.

5. **Make Prediction**. Change directory into the python folder with `cd python`. Run the python file with `python inference_pipe.py`. This will run both the data injest and the model inference. Optionally if you want to run the model with only one wavefront type, run the python script with a wavefront parameter. i.e. `python inference_pipe.py --wavefront "RVp"`. Another option is to specify the meta text that will be given to the vtk carto files using the meta argument. One word is expected. i.e. `python inference_pipe.py --meta sheep_18`.The prediction uses the pretrained models in the `/python/models` and makes predictions on the presence of scar tissue at varying depths on the loaded data. Predictions for each wavefront and depth are made. The output of these predictions in 3D space are stored in the folder `deploy/output` as Vtk files. Vtk files named by wavefront and depth (for example: /deploy/output/predictions_clf_AtLeastIntra_RVp_120epochs.vtk). These vtk files can be opened by Carto or [slicer](https://www.slicer.org/).

## Main Data components

1. [Publishable Data ](https://unisyd.sharepoint.com/:f:/r/sites/ComplexArrhythmiaProgram/Shared%20Documents/Project%20Data/Bioinformatics%20EGM%20signal%20analysis/SIH/data/publishable_data?csf=1&web=1&e=fXdf9o) holds the results of the data injest during model training. Phase 1 of the project explored many features including bipolar and unipolar signals, window of interest and raw time horizons, and using different datasets (filtered uses the labels directly versus imputed assumes no-scar where signals exist and labels were blank. This is link holds many files that reflect this exploration. The conclusion of phase1 was that the best performing model used unipolar raw signals with the filtered dataset (no imputation). Hence the file `publishable_model_data_TSAI.parquet` holds the training data (signals and depth labels) that are associated with our best performing model. depth_label is the classification using the at least methodology rather than the independent classification.

2. [TSAI pretrained models](https://unisyd.sharepoint.com/:f:/r/sites/ComplexArrhythmiaProgram/Shared%20Documents/Project%20Data/Bioinformatics%20EGM%20signal%20analysis/SIH/models/tsai_rawsignal_unipolar?csf=1&web=1&e=HoY3XN) by wavefront and depth. The model specifically uses unipolar raw signals during training. 

## Phase1 - Model exporation.

This section describes how the training data was aggregated and how to run the training model. 


**DATA INJEST**

1. aggregating_data.R:  Aggregates data (signals and labels) into a single dataframe given the data structure sheep/labelled (holding the 3 labelled sheets) and sheep/Export_Analysis (holding all other files). Structure expected looks like:

```
├── S12
│   ├── Export_Analysis
│   │   ├── 1-LV_Points_Export.xml
│   │   ├── 1-LV_car.txt
│   │   ├── 10-1-1-1-1-1-1-ReLV RVp Penta.mesh
│   │   ├── 10-1-1-1-1-1-1-ReLV RVp Penta_P100_ECG_Export.txt
│   │   ├── 10-1-1-1-1-1-1-ReLV RVp Penta_P100_MAGNETIC_20_POLE_A_CONNECTOR_Eleclectrode_Positions.txt

... etc etc...

│   ├── Analysis 01_05_2024 17-02-34.xml
│   └── _image.mesh
└── labelled
    ├── cleaned_LVp Penta_car_labelled.xlsx
    ├── cleaned_RVp Penta_car_labelled.xlsx
    └── cleaned_SR Penta_car_labelled.xlsx
```

Note: Files within the S9 Data dump was inconsistently named. This has been accounted for but not that consistent naming is required as the aggregation relies on patters of words which the file names adhere to.


2. post_aggregation.R: aggregates data for all sheep and produces 2 rds files, imputed_aggregate_data.rds and filtered_aggregate_data.rds, used for orange data mining analysis. These files are neccessary for building_features.R and also the /python modelling (point 8)


3. building_features.R: Builds basic features driven by client conversations. Choose either filtered data or imputed (where blanks in histology labels are treated as NoScar)


4. building_tsfeatures.R: builds many features according to the tsfresh package.


3 and 4 are used to save csv files that orange data mining uses in exploring models and metrics.


Some visually driven files include: 

5. EDA.Rmd: Explores data and producing graphs. Relatedly save_plots.R save plots to file

6. prediction_plots.Rmd: Plots of predictions from a model trained in orange data mining. Used to investigate if there are clusters of inaccuracy from within an area of space or from a particular sheep. 

7. 3D.Rmd: Plots of the labelled data (i.e. ground truth) in 3d space

8. /python holds independent modelling based on a definitions that treats 3 layers in an independent way. Models are more sophisticated and have better accuracy (than compared to the orange exploration)


**TRAINING MODEL METHODOLOGY**

The classifier_tsai_published.py is the starting point to training the model given parameters of where the publishable data (see above) is stored and the labels of the ground-truth (using the At Least measure). The conclusion from phase 1 was that the unipolar raw signal yielded the best performing model, particularly in detecting scar tissue in the outer layers (Intra,Epi). A specialised time series model called [tsai](https://timeseriesai.github.io/tsai/) was trained on all the sheep signal data to create the [pretrained models](https://unisyd.sharepoint.com/:f:/r/sites/ComplexArrhythmiaProgram/Shared%20Documents/Project%20Data/Bioinformatics%20EGM%20signal%20analysis/SIH/models/tsai_rawsignal_unipolar?csf=1&web=1&e=HoY3XN). 

This model uses a CNN methodology with the test / train split being 20% 80%. The time series learner [documentation is here](https://timeseriesai.github.io/tsai/tslearner.html) and engaged in the code in the train_model section in classifier_tsai.py.





