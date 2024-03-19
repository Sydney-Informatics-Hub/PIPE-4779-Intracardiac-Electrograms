"""
Feature extraction and selection for ECG data using the tsfresh library.

Functionality:
--------------
- Load data from a csv file
- Convert the dataframe to tsfresh timeseries format for a given wavefront and target
- Extract features from a timeseries using the tsfresh library
- Select relevant features based on the tsfresh library
- Calculate the relevance of the features
- Generate a dictionary with all relevant features and their description
- Get the parameters for the feature calculators
- Apply feature extraction to a given timeseries

Example how to use:
-------------------
    target = 'scar'
    wavefront = 'SR'
    outpath = 'test'
    inpath = '../../../data/generated'
    fname_csv = 'NestedDataS18.csv'
    fe = FeatureExtraction(inpath, fname_csv, outpath)
    fe.run_wavefront_target(wavefront, target)

    # or for all targets and wavefronts:
    fe.run()

The relevant features can then be used as input for classification models such as Random Forest.

See for tests and examples: test_features.py

Installation and dependencies: environment.yaml

Author: Sebastian Haan
"""

import os
import pandas as pd
import numpy as np
import logging
from tsfresh import extract_features, select_features
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.feature_extraction import settings
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import feature_calculators


class FeatureExtraction:
    """
    Class for feature extraction and selection.

    This class extracts features from a timeseries and selects relevant features based on the tsfresh library.

    The algorithm includes three main steps:
    
    1) Feature extraction: the algorithm characterizes time series with comprehensive and well-established feature mappings 
    and considers additional features describing meta-information.
    
    2) Feature significance testing: each feature vector is individually and independently evaluated 
    with respect to its significance for predicting the target under investigation.
    
    3) Feature selection: The vector of p-values is evaluated on the basis of the Benjamini-Yekutieli procedure
    in order to decide which features to keep. 

    Tsfresh deploys the fresh algorithm (fresh stands for FeatuRe Extraction based on Scalable Hypothesis tests).
    References:
        - http://adsabs.harvard.edu/abs/2016arXiv161007717C
        - https://tsfresh.readthedocs.io/en/latest/index.html

    Args:
        inpath (str): Path to the input csv file
        fname_csv (str): Name of the input csv file
        outpath (str): Path to the output directory
    """
    def __init__(self, inpath, fname_csv, outpath):
        self.inpath = inpath
        self.fname_csv = fname_csv
        self.outpath = outpath
        self.df = None
        self.timeseries = None
        self.y = None
        self.extracted_features = None
        self.impute_features = None
        self.selected_features = None
        self.relevant_features_desc = None
        self.relevance_table = None
        self.fc_parameters = None
        self.target = None
        self.wavefront = None
        os.makedirs(self.outpath, exist_ok=True)
        self.load_data()

    def load_data(self):
        """ Load the data from the csv file and pprepare the dataframe."""
        usecols = [
            'Point_Number',
            'WaveFront',
            'sheep',
            'signal_data',
            'endocardium_scar',
            'intramural_scar',
            'epicardial_scar',
        ]
        # check if file exists
        if not os.path.isfile(os.path.join(self.inpath, self.fname_csv)):
            raise FileNotFoundError(f'File {self.fname_csv} not found in {self.inpath}')
        self.df = pd.read_csv(os.path.join(self.inpath, self.fname_csv), usecols=usecols)
        # remove nan values
        self.df = self.df.dropna()
        # add column "time" to df which starts at 0 and has the same length as "signal_data" for each Point_Number and WaveFront
        self.df['time'] = self.df.groupby(['Point_Number', 'WaveFront']).cumcount()
        # Generate a column 'scar' that is 1 if either of the scar columns is 1, otherwise 0
        self.df['endocardium_scar'] = self.df['endocardium_scar'].astype(int)
        self.df['intramural_scar'] = self.df['intramural_scar'].astype(int)
        self.df['epicardial_scar'] =self. df['epicardial_scar'].astype(int)
        self.df['scar'] = self.df[['endocardium_scar', 'intramural_scar', 'epicardial_scar']].max(axis=1)

    def df_to_ts(self, wavefront, target='scar'):
        """
        Converts the dataframe to a timeseries for the given wavefront

        Args:
            wavefront (str): 'LVp', 'RVp', or 'SR'
            target (str): 'scar' (Default) or 'endocardium_scar', 'intramural_scar', 'epicardial_scar'
        """
        self.wavefront = wavefront
        self.target = target
        self.timeseries = self.df[self.df['WaveFront'] == wavefront][['Point_Number', 'time', 'signal_data']]
        self.y = self.df[self.df['WaveFront'] == self.wavefront][['Point_Number', self.target]].drop_duplicates()
        self.y = self.y.set_index('Point_Number')[self.target]

    def extract_features(self):
        """ Extract features from the timeseries"""
        if self.timeseries is not None:
            self.extracted_features = extract_features(self.timeseries, column_id="Point_Number", column_sort="time")
            print('Number of extracted features: ', len(list(self.extracted_features.columns)))
        else:
            print('No timeseries found')

    def select_relevant_features(self):
        """ Select relevant features from the extracted features and impute missing values."""
        if self.extracted_features is not None:
            self.impute_features = impute(self.extracted_features)
            self.selected_features = select_features(self.impute_features, self.y)
            print(f'Found {len(self.selected_features.columns)} relevant features')
        else:
            print('No extracted features found')

    def calculate_relevance(self):
        """ Calculate the relevance of the features"""
        if self.impute_features is not None:
            self.relevance_table = calculate_relevance_table(self.impute_features, self.y)
            self.relevance_table = self.relevance_table[self.relevance_table['relevant'] == True]
        else:
            print('No imputed features found')
            return None
            

    def generate_feature_dict(self):
        """ Generate a dictionary with all relevant features and their description"""
        # get all functions of feature_calculators
        if self.selected_features is None:
            print('No selected features found')
            return None
            
        all_functions = dir(feature_calculators)
        # get description  for each function
        descriptions = {}
        for function in all_functions:
            if function.startswith('_'):
                continue
            desc = feature_calculators.__dict__[function].__doc__
            if desc:
                try:
                    desc = desc.split('\n')[1]
                    desc = desc.strip()
                except:
                    pass
                descriptions[function] = desc

        # get all features that are relevant
        relevant_features = self.selected_features.columns

        # get descriptions for all relevant features
        self.relevant_features_desc = {}
        for feature in relevant_features:
            feature = feature.split('__')[1]
            if feature in descriptions:
                self.relevant_features_desc[feature] = descriptions[feature]
            else:
                self.relevant_features_desc[feature] = 'No description found'

    def get_fc_parameters(self):
        """ Get the parameters for the feature calculators"""  
        if self.selected_features is not None:
            self.fc_parameters = settings.from_columns(self.selected_features)
        else:
            print('No selected features found')

    def apply_extraction(self, ts):
        """ Apply feature extraction to the given timeseries

        Args:
            ts (pd.DataFrame): Timeseries dataframe with columns 'Point_Number', 'time', and 'signal_data'

        Returns:
            pd.DataFrame: Timeseries dataframe with extracted features
        """
        if (self.fc_parameters is None) and (self.selected_features is not None):
            self.get_fc_parameters()

        if self.fc_parameters is not None:
            #ts = impute(ts)
            parameters = {
            "signal_data": self.fc_parameters["signal_data"]
                }
            features_extracted = extract_features(ts, column_id="Point_Number", column_sort="time", kind_to_fc_parameters=parameters)
            return features_extracted
        else:
            print('No feature calculator parameters found')
            return None 

    def save_results_to_csv(self):
        """
        Save the selected features and relevant features description to csv.
        """
        #self.extracted_features.to_csv(os.path.join(outpath, f'extracted_features_{wavefront}.csv'))
        if (self.selected_features is not None) and (self.relevant_features_desc is not None):
            self.selected_features.to_csv(os.path.join(self.outpath, f'selected_features_{self.target}_{self.wavefront}.csv'))
            relevant_features_desc = pd.DataFrame(self.relevant_features_desc.items(), columns=['feature', 'description'])
            relevant_features_desc.to_csv(os.path.join(self.outpath, f'description_relevant_features_{self.target}_{self.wavefront}.csv'), index=False)
            print(f"Results for wavefront {self.wavefront} and target {self.target} saved to:", self.outpath)
        else:
            print('No relevant features found')
        if self.relevance_table is not None:
            self.relevance_table.to_csv(os.path.join(self.outpath, f'relevance_table_{self.target}_{self.wavefront}.csv'))

    def run_wavefront_target(self, wavefront, target):
        """ Run the feature extraction and selection for the given wavefront and target

        Args:
            wavefront (str): 'LVp', 'RVp', or 'SR'
            target (str): 'scar' (Default) or 'endocardium_scar', 'intramural_scar', 'epicardial_scar'
        """
        self.df_to_ts(wavefront, target)
        self.extract_features()
        self.select_relevant_features()
        self.generate_feature_dict()
        self.calculate_relevance()
        self.save_results_to_csv()
        print(f'Done for wavefront {wavefront} and target {target}')

    def run_target(self, target):
        """ Run the feature extraction and selection for all wavefronts for the given target
        
        Args:
            target (str): 'scar' (Default) or 'endocardium_scar', 'intramural_scar', 'epicardial_scar'
        """
        self.target = target
        for wavefront in ['LVp', 'RVp', 'SR']:
            print(f"Processing features for wavefront {self.wavefront} and target {self.target}...")
            self.run_wavefront_target(wavefront, self.target)

    def run(self):
        """ Run the feature extraction and selection for all targets and wavefronts"""
        for target in ['scar', 'endocardium_scar', 'intramural_scar', 'epicardial_scar']:
            print(f"Processing features for target {target}...")
            self.run_target(target)

