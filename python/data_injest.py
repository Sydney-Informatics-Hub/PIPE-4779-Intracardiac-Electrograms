import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import re

# Setting up the path
#deploy_data_path = Path("../../data/deploy/data")
deploy_data_path = Path("../../../data/deploy/data/Export_Analysis")
# Ensures the directory exists
if not deploy_data_path.exists():
    raise FileNotFoundError(f"Directory {deploy_data_path} does not exist.")
export_analysis_path = deploy_data_path

def collect_data(catheter_type):
    """
    Function to collect data based on Catheter_Type
    """
    template = get_template(catheter_type)
    #template['signal_data'] = template.apply(lambda row: get_raw_signal_unipolar_data(row['WaveFront'], catheter_type, row['Point_Number']), axis=1)
     # Apply transformations row-wise
    
    template['signal'] = None
    for index, row in template.iterrows():
        template.at[index, 'signal'] = get_raw_signal_unipolar_data(row['WaveFront'], row['Catheter_Type'], row['Point_Number'])
    
    template = template.astype({'Point_Number': 'int32'})
    print("finished reading signals...")

    wavefronts_collected = template[['WaveFront']].drop_duplicates()

    # Collecting geometry for each wavefront
    all_geometries = pd.concat([get_geometry(wf, catheter_type) for wf in wavefronts_collected['WaveFront']], ignore_index=True)

    # Merging geometry info with signal data
    signals = all_geometries.merge(template, on=['Catheter_Type', 'WaveFront', 'Point_Number'], how='right')
    parquet_file = deploy_data_path / "data_injest.parquet"
    pq.write_table(pa.Table.from_pandas(signals), parquet_file)
    print("finished combing geometries...")
    return template


def get_template(catheter_type):
    """
    Produces a DataFrame with WaveFront, Catheter_Type, Point_Number combinations 
    given the signals that need to be ingested based on Export analysis folder in deploy_data_path.
    """
    # List all files matching the pattern '_ECG_Export.txt'
    files = [f for f in os.listdir(export_analysis_path) if '_ECG_Export.txt' in f]
    
    # Create a DataFrame from the files
    df = pd.DataFrame({'files': files})
    df['paths'] = df['files'].apply(lambda x: os.path.join(export_analysis_path, x))
    
    # Extract Point_Number from filenames
    df['Point_Number'] = df['files'].str.extract(r'.*P(\d+).*')[0]
    
    # Split 'files' column to extract 'WaveFront'
    df['WaveFront'] = df['files'].str.split(' ', expand=True)[1]
    
    # Add 'Catheter_Type' to the DataFrame
    df['Catheter_Type'] = catheter_type
    
    return df


def get_raw_signal_unipolar_data(wavefront, catheter_type, point_number):
    txt_file = find_signal_file(wavefront, catheter_type, point_number)
    
    # Read the table content, skipping the first 3 lines
    #tabular_content = pd.read_csv(txt_file, skiprows=3, delim_whitespace=True) 
    tabular_content = pd.read_csv(txt_file, skiprows=3, sep='\s+')

    # read txt file 
    # Extracting the raw ECG gain from the fourth line of the file
    #with open(txt_file, 'r') as file:
    #    lines = file.readlines()
    #gain_line = next((line for line in lines[:4] if "Raw ECG to MV (gain) =" in line), None)
    #if gain_line:
    #    raw_ecg_gain = float(re.search(r"Raw ECG to MV \(gain\) = ([0-9.]+)", gain_line).group(1))
    #else:
    #    return None  # Return None if gain is not found


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

    # Selecting the relevant channel data and applying the gain factor
    # if channel in tabular_content.columns:
    #     signal = tabular_content[[channel]].apply(lambda x: x * raw_ecg_gain)
    #     signal.rename(columns={channel: 'signal_data'}, inplace=True)
    #     return signal
    # else:
    #     return None
    
    column_name = next((col for col in tabular_content.columns if channel in col), None)
    if column_name:
        signal_data = tabular_content[column_name] * raw_ecg_gain
        signal_data.rename("signal_data", inplace=True)
        return list(signal_data.values)
    else:
        return None

# Example usage
# signal_data = get_raw_signal_unipolar_data('WaveFrontX', 'TypeY', 1)


def find_signal_file(wavefront, catheter_type, point_number):
    
    pattern = f".*{wavefront} {catheter_type}_P{point_number}_ECG_Export\\.txt$"
    
    # List files in the directory and find the first match
    for filename in os.listdir(deploy_data_path):
        if re.match(pattern, filename):
            return deploy_data_path / filename
    return None


def get_geometry(wavefront, catheter_type):
    pattern = f".*{wavefront} {catheter_type}_car\\.txt$"
    
    # List files in the directory and read the first matching file
    for filename in os.listdir(deploy_data_path):
        if re.match(pattern, filename):
            matching_file = deploy_data_path / filename
            # Read table, skip the first line, and select specific columns
            #tabular_content = pd.read_csv(matching_file, skiprows=1, header=None, delim_whitespace=True)
            tabular_content = pd.read_csv(matching_file, skiprows=1, header=None, sep='\s+')
            tabular_content = tabular_content[[2, 4, 5, 6]]  # Select columns 3, 5, 6, 7
            tabular_content.columns = ['Point_Number', 'X', 'Y', 'Z']
            tabular_content['WaveFront'] = wavefront
            tabular_content['Catheter_Type'] = catheter_type
            return tabular_content
    return None

data = collect_data("Penta")
