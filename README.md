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

## Main Software components

1. building_features.R:  aggregates data into a single dataframe given the data structure sheep/labelled (holding the 3 labelled sheets) and sheep/Export_Analysis (holding all other files). Enables models in orangedatamining to be run.

2. post_aggregation.R: producing 2 rds files imputed_aggregate_data.rds and filtered_aggregate_data.rds used for orange data mining analysis. Produces files neccessary for building_features.R

3. EDA.Rmd: Explores data and producing graphs. 

4. building_features.R Uses above rds files to build features, and prepare data for modelling. Saves model_model_dataimputed.csv and model_datafiltered.csv (Used for orange data mining analysis of models)

5. model.R (to be continued when implementation chosen)


