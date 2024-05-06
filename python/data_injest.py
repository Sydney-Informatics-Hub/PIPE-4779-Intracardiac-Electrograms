# Data pipeline for pre-processing intracardiac electrograms for ML inference

import os
import pandas as pd
from pathlib import Path
import re
import time
import argparse

# Settings
deploy_data_path = "../../../data/deploy/data/Export_Analysis"
export_analysis_path =  "../../../data/deploy/data"
catheter_type = "Penta"

class DataIngest:
    """
    Data ingestion pipeline for intracardiac electrograms for ML inference.

    Parameters
    ----------
    deploy_data_path : str
        Path to the directory containing the data to be ingested
    export_analysis_path : str
        Path to the directory where the ingested data will be saved
    catheter_type : str
        Type of catheter used for data collection, default is "Penta"

    Output
    ------
    parquet file : data_injest.parquet (saved to export_analysis_path)
        Ingested data saved as dataframe in parquet format
    """
    def __init__(self, deploy_data_path, export_analysis_path, catheter_type = "Penta"):
        self.deploy_data_path = Path(deploy_data_path)
        self.export_analysis_path = Path(export_analysis_path)
        self.catheter_type = catheter_type
        self.outfname = None
        if not self.deploy_data_path.exists():
            raise FileNotFoundError(f"Directory {self.deploy_data_path} does not exist.")
        os.makedirs(self.export_analysis_path, exist_ok=True)

    def collect_data(self):
        """
        Function to collect data based on Catheter_Type
        """
        template = self.get_template()
        #template['signal_data'] = template.apply(lambda row: get_raw_signal_unipolar_data(row['WaveFront'], catheter_type, row['Point_Number']), axis=1)
        # Apply transformations row-wise
        
        template['signal'] = None
        for index, row in template.iterrows():
            template.at[index, 'signal'] = self.get_raw_signal_unipolar_data(row['WaveFront'], row['Point_Number'])
        
        template = template.astype({'Point_Number': 'int32'})
        print("finished reading signals...")

        wavefronts_collected = template[['WaveFront']].drop_duplicates()

        # Collecting geometry for each wavefront
        all_geometries = pd.concat([self.get_geometry(wf) for wf in wavefronts_collected['WaveFront']], ignore_index=True)

        # Merging geometry info with signal data
        signals = all_geometries.merge(template, on=['Catheter_Type', 'WaveFront', 'Point_Number'], how='right')
        # drop path columns and name filename to path
        signals.drop(columns=['paths'], inplace=True)
        # write parquet outfname
        self.filename_output = f"preprocessed_rawsignal_unipolar_{self.catheter_type.lower()}.parquet"
        # Write to parquet file
        parquet_file = self.export_analysis_path / self.filename_output
        # save as parquet file
        signals.to_parquet(parquet_file, index=False)
        print("Finished.")


    def get_template(self):
        """
        Produces a DataFrame with WaveFront, Catheter_Type, Point_Number combinations 
        given the signals that need to be ingested based on Export analysis folder in deploy_data_path.
        """
        # List all files matching the pattern '_ECG_Export.txt'
        files = [f for f in os.listdir(self.deploy_data_path) if '_ECG_Export.txt' in f]
        
        # Create a DataFrame from the files
        df = pd.DataFrame({'files': files})
        df['paths'] = df['files'].apply(lambda x: os.path.join(self.deploy_data_path, x))
        
        # Extract Point_Number from filenames
        df['Point_Number'] = df['files'].str.extract(r'.*P(\d+).*')[0]
        
        # Split 'files' column to extract 'WaveFront'
        df['WaveFront'] = df['files'].str.split(' ', expand=True)[1]
        
        # Add 'Catheter_Type' to the DataFrame
        df['Catheter_Type'] = self.catheter_type
        
        return df


    def get_raw_signal_unipolar_data(self, wavefront, point_number):
        """
        Function to read raw signal data from the ECG_Export.txt file

        Parameters
        ----------
        wavefront : str
            Wavefront name
        point_number : int
            Point number

        Returns
        -------
        list
            List of raw signal data
        """
        txt_file = self.find_signal_file(wavefront, point_number)
        
        # Read the table content, skipping the first 3 lines
        #tabular_content = pd.read_csv(txt_file, skiprows=3, delim_whitespace=True) 
        tabular_content = pd.read_csv(txt_file, skiprows=3, sep='\s+')

        with open(txt_file, 'r') as file:
            next(file)  # Skip the first line
            gain_line = next(file)  # Get the second line
            channel_line = next(file)  # Get the third line
        raw_ecg_gain_match = re.search(r"Raw ECG to MV \(gain\) = ([0-9.]+)$", gain_line.strip())
        if raw_ecg_gain_match:
            raw_ecg_gain = float(raw_ecg_gain_match.group(1))
        else:
            return None  # Return None if no match is found or gain is not numeric
        
        # Extracting the channel information
        channel_match = re.search(r"Unipolar Mapping Channel=(\w+_\w+)", channel_line.strip())
        if channel_match:
            channel = channel_match.group(1)
        else:
            return None  # Return None if channel information is not found

        column_name = next((col for col in tabular_content.columns if channel in col), None)
        if column_name:
            signal_data = tabular_content[column_name] * raw_ecg_gain
            signal_data.rename("signal_data", inplace=True)
            return list(signal_data.values)
        else:
            return None
        

    def find_signal_file(self, wavefront, point_number):
        """
        Function to find the signal file for a given wavefront and point number

        Parameters
        ----------
        wavefront : str
            Wavefront name
        point_number : int
            Point number

        Returns
        -------
        txt_file : str
            Path to the signal file 
        """
        
        pattern = f".*{wavefront} {self.catheter_type}_P{point_number}_ECG_Export\\.txt$"
        
        # List files in the directory and find the first match
        for filename in os.listdir(self.deploy_data_path):
            if re.match(pattern, filename):
                return self.deploy_data_path / filename
        return None


    def get_geometry(self, wavefront):
        """
        Function to read geometry data from the catheter file

        Parameters
        ----------
        wavefront : str
            Wavefront name

        Returns
        -------
        tabular_content : DataFrame
            DataFrame containing geometry data
        """
        pattern = f".*{wavefront} {self.catheter_type}_car\\.txt$"
        
        # List files in the directory and read the first matching file
        for filename in os.listdir(self.deploy_data_path):
            if re.match(pattern, filename):
                matching_file = self.deploy_data_path / filename
                # Read table, skip the first line, and select specific columns
                #tabular_content = pd.read_csv(matching_file, skiprows=1, header=None, delim_whitespace=True)
                tabular_content = pd.read_csv(matching_file, skiprows=1, header=None, sep='\s+')
                tabular_content = tabular_content[[2, 4, 5, 6]]  # Select columns 3, 5, 6, 7
                tabular_content.columns = ['Point_Number', 'X', 'Y', 'Z']
                tabular_content['WaveFront'] = wavefront
                tabular_content['Catheter_Type'] = self.catheter_type
                return tabular_content
        return None


if __name__ == '__main__':
    # ToDo: add cl arguments for deploy_data_path and export_analysis_path abd wavefront_selected
    now = time.time()
    data_ingest = DataIngest(deploy_data_path, export_analysis_path, catheter_type)
    data_ingest.collect_data()
    print(f'Compute time for processing: {round((time.time() - now)/60,2)} mins')
