"""
Preprocess and combine data from multiple sheep, and define the target variables.

The output of this file can be used as input for ML models, 
e.g. in classifier_featurebased.py, or classifier_tsai.py.
 """ 

import os
import numpy as np
import pandas as pd

inpath = '../../../data/generated_4cases'
fname_csv_core = 'NestedData'
animal_labels = ['S9', 'S12', 'S15', 'S17', 'S18', 'S20']
outpath = '../results'
target = 'scar'

def preprocess(outpath = '../results', 
               fname_csv_core = 'NestedData', 
               animal_labels = ['S9', 'S12', 'S15', 'S17', 'S18', 'S20'], 
               inpath = '../../../data/generated',
               fname_out = 'NestedDataAll_clean.csv'):
    """
    The preprocess function reads in all data of list_fnames_csv as dataframe and concatenates them.
    The data is then saved to a csv file.

    Parameters
    ----------
    outpath : str
        Path to the output folder
    fname_csv_core : str
        Core name of the csv files
    animal_labels : list
        List of animal labels
    inpath : str
        Path to the data
    fname_out : str
        Name of the output file
    """
    usecols = [
                'Point_Number',
                'WaveFront',
                'sheep',
                'signal_data',
                'endocardium_scar',
                'intramural_scar',
                'epicardial_scar',
            ]
    df = None
    for label in animal_labels:
            file = os.path.join(inpath, fname_csv_core + label + '.csv')
            # check if file exists
            if not os.path.isfile(file):
                raise FileNotFoundError(f'File {file} not found')
            dfnew = pd.read_csv(file, usecols=usecols)
            # remove nan values
            dfnew = dfnew.dropna()
            # modify point number to avoid duplication
            dfnew['Point_Number'] = dfnew['sheep'] + '_' + dfnew['Point_Number'].astype(str)
            # concatenate dataframes
            # check if 
            if df is None:
                df = dfnew
            else:
                df = pd.concat([df, dfnew], ignore_index=True)
    os.makedirs(outpath, exist_ok=True)
    df.to_csv(os.path.join(outpath, fname_out), index=False)

def preprocess_rawsignal(outpath = '../results', 
               fname_csv_core = 'NestedData', 
               animal_labels = ['S9', 'S12', 'S15', 'S17', 'S18', 'S20'], 
               inpath = '../../../data/generated_4cases',
               fname_out = 'NestedDataAll_rawsignal_unipolar.parquet',
               signal='raw_unipolar'):
    """
    The preprocess function reads in all data of list_fnames_parquet as dataframe and concatenates them.
    The data is then saved to a csv file.

    Parameters
    ----------
    outpath : str
        Path to the output folder
    fname_csv_core : str
        Core name of the csv files
    animal_labels : list
        List of animal labels
    inpath : str
        Path to the data
    fname_out : str
        Name of the output file
    signal: 'signal', 'rawsignal', 'signal_unipolar', 'raw_unipolar',
    """
    usecols = ['Point_Number',
                'WaveFront',
                'sheep',
                signal,
                'endocardium_scar',
                'intramural_scar',
                'epicardial_scar']
    df = None
    for label in animal_labels:
            file = os.path.join(inpath, fname_csv_core + label + '.parquet')
            # check if file exists
            if not os.path.isfile(file):
                raise FileNotFoundError(f'File {file} not found')
            dfnew = pd.read_parquet(file, columns=usecols)
            # explode rawsignal column
            dfnew = dfnew.explode(signal)
            dfnew[signal] = dfnew[signal].apply(lambda x: x['signal_data'] if isinstance(x, dict) and 'signal_data' in x else None)
            # remove nan values
            dfnew = dfnew.dropna()
            # modify point number to avoid duplication
            # rename column signal to 'signal_data'
            dfnew = dfnew.rename(columns={signal: 'signal_data'})
            # convert to int
            dfnew['Point_Number'] = dfnew['Point_Number'].astype(int)
            dfnew['endocardium_scar'] = dfnew['endocardium_scar'].astype(int)
            dfnew['intramural_scar'] = dfnew['intramural_scar'].astype(int)
            dfnew['epicardial_scar'] = dfnew['epicardial_scar'].astype(int)
            dfnew['Point_Number'] = dfnew['sheep'].astype(str) + '_' + dfnew['Point_Number'].astype(str)
            # concatenate dataframes
            # check if 
            if df is None:
                df = dfnew
            else:
                df = pd.concat([df, dfnew], ignore_index=True)
    os.makedirs(outpath, exist_ok=True)
    # save as parquet
    df.to_parquet(os.path.join(outpath, fname_out), index=False)