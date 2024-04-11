# preprocess and combine data

import os
import numpy as np
import pandas as pd


inpath = '../../../data/generated_w_unipolar'
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
               inpath = '../../../data/generated',
               fname_out = 'NestedDataAll_rawsignal_clean.csv'):
    pass