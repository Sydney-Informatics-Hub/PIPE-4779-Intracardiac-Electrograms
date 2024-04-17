"""

followed by post_aggregation.py
"""

import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import re
import os
import pyarrow.parquet as pq


sheep_names = ["S9" "S17" "S18" "S12" "S20" "S15"]
generated_data_path = Path("data/generated2")
os.makedirs(generated_data_path, exist_ok=True)

def run_all(sheep_names):
    for sheep_name in sheep_names:
        run_sheep(sheep_name)


def run_sheep(sheep_name):
    # Retrieve all signals based on the sheep name
    retrieve_all_signals(sheep_name)

    # Incorporate histology information into the data
    incorporate_histology(sheep_name)

    # Transform the data to long format and save it as CSV
    write_as_long_format(sheep_name)

    # Save the data in Parquet format
    write_as_parquet(sheep_name)


def get_sheep_path(sheep):
    # Creating paths based on the sheep identifier
    generated_data_path = Path("data/generated2")
    labelled_data_path = Path(f"data/{sheep}/labelled")
    main_data_path = Path(f"data/{sheep}/Export_Analysis")
    
    # Returning a dictionary of paths
    return {
        "generated_data_path": generated_data_path,
        "labelled_data_path": labelled_data_path,
        "main_data_path": main_data_path
    }


def load_sheep(sheep_name):
    # Assuming get_sheep_path is a function that sets a global path variable or returns it
    labelled_data_path = get_sheep_path(sheep_name)['labelled_data_path']
    
    # Helper function to process each file
    def process_file(file_path, wavefront):
        df = pd.read_excel(file_path)
        df['Categorical_Label'] = df['Categorical_Label'].apply(lambda x: 'Scar' if x == -1 else 'NoScar')
        df['Catheter_Type'] = 'Penta'
        df = df.iloc[:, 1:]  # Drop the first column
        df['WaveFront'] = wavefront
        return df
    
    # Loading and processing each dataset
    LVpLabelledData = process_file(labelled_data_path / "cleaned_LVp Penta_car_labelled.xlsx", "LVp")
    RVpLabelledData = process_file(labelled_data_path / "cleaned_RVp Penta_car_labelled.xlsx", "RVp")
    SRLabelledData = process_file(labelled_data_path / "cleaned_SR Penta_car_labelled.xlsx", "SR")
    
    # Combining all data into one DataFrame
    LabelledData = pd.concat([LVpLabelledData, RVpLabelledData, SRLabelledData], ignore_index=True)
    LabelledData['sheep'] = sheep_name
    
    return LabelledData


def load_sheep(sheep_name):
    paths = get_sheep_path(sheep_name)
    
    # Helper function to process each file
    def process_file(file_path, wavefront):
        df = pd.read_excel(file_path)
        df['Categorical_Label'] = df['Categorical_Label'].apply(lambda x: 'Scar' if x == -1 else 'NoScar')
        df['Catheter_Type'] = 'Penta'
        df = df.iloc[:, 1:]  # Removes the first column
        df['WaveFront'] = wavefront
        return df
    
    # Paths to the specific files
    LVp_file_path = paths['labelled_data_path'].joinpath("cleaned_LVp Penta_car_labelled.xlsx")
    RVp_file_path = paths['labelled_data_path'].joinpath("cleaned_RVp Penta_car_labelled.xlsx")
    SR_file_path = paths['labelled_data_path'].joinpath("cleaned_SR Penta_car_labelled.xlsx")
    
    # Loading and processing each dataset
    LVpLabelledData = process_file(LVp_file_path, "LVp")
    RVpLabelledData = process_file(RVp_file_path, "RVp")
    SRLabelledData = process_file(SR_file_path, "SR")
    
    # Combining all data into one DataFrame
    LabelledData = pd.concat([LVpLabelledData, RVpLabelledData, SRLabelledData], ignore_index=True)
    LabelledData['sheep'] = sheep_name
    
    return LabelledData



def find_from_woi(WaveFront, Catheter_Type, Point_Number, sheep_name):
    """
    Function to find the 'From' value of the Window of Interest (WOI) for a given point.

    Parameters:
    WaveFront (str): The wavefront of the point.
    Catheter_Type (str): The type of catheter used.
    Point_Number (int): The point number.
    sheep_name (str): The name of the sheep.

    Returns:
    float: The 'From' value of the WOI if found, else None.
    
    """
    file_pattern = f".*{WaveFront}\\s{Catheter_Type}_P{Point_Number}_Point_Export\\.xml$"
    main_data_path = get_sheep_path(sheep_name)['main_data_path'] 
    matching_files = list(main_data_path.glob(file_pattern))
    
    if len(matching_files) == 1:
        xml_data = ET.parse(matching_files[0]).getroot()
        reference_annotation = float(xml_data.find(".//Annotations[@Reference_Annotation]").text)
        woi_from = float(xml_data.find(".//WOI[@From]").text)
        return reference_annotation + woi_from
    else:
        return None  # Python equivalent of R's NULL

def find_to_woi(WaveFront, Catheter_Type, Point_Number, sheep_name):
    """
    Function to find the 'To' value of the Window of Interest (WOI) for a given point.

    Parameters:
    WaveFront (str): The wavefront of the point.
    Catheter_Type (str): The type of catheter used.
    Point_Number (int): The point number.
    sheep_name (str): The name of the sheep.


    Returns:
    float: The 'To' value of the WOI if found, else None.
    
    """
    file_pattern = f".*{WaveFront}\\s{Catheter_Type}_P{Point_Number}_Point_Export\\.xml$"
    main_data_path = get_sheep_path(sheep_name)['main_data_path']  
    matching_files = list(main_data_path.glob(file_pattern))
    
    if len(matching_files) == 1:
        xml_data = ET.parse(matching_files[0]).getroot()
        reference_annotation = float(xml_data.find(".//Annotations[@Reference_Annotation]").text)
        woi_to = float(xml_data.find(".//WOI[@To]").text)
        return reference_annotation + woi_to
    else:
        return None  # Python equivalent of R's NULL
    

def find_window(WaveFront, Catheter_Type, Point_Number, sheep_name):
    """
    Function to find the window of interest for a given point.

    Parameters:
    WaveFront (str): The wavefront of the point.
    Catheter_Type (str): The type of catheter used.
    Point_Number (int): The point number.

    Returns:
    dict: A dictionary containing the 'From' and 'To' values of the window of interest if found, else None.
    
    """
    # Construct the file pattern
    file_pattern = f".*{WaveFront}\\s{Catheter_Type}_P{Point_Number}_Point_Export\\.xml$"
    main_data_path = get_sheep_path(sheep_name)['main_data_path']   # Adjust the path accordingly
    
    # Find files that match the pattern
    matching_files = [f for f in main_data_path.glob('**/*') if re.match(file_pattern, str(f.name))]
    
    # Check if exactly one file matches
    if len(matching_files) == 1:
        # Parse the XML file
        xml_data = ET.parse(matching_files[0]).getroot()
        
        # Extract necessary data from XML using XPath
        reference_annotation = xml_data.find(".//Annotations[@Reference_Annotation]").text
        if reference_annotation is None:
            return None
        else:
            reference_annotation = float(reference_annotation)
        woi_from = xml_data.find(".//WOI[@From]").text
        if woi_from is None:
            return None
        else:
            woi_from = float(woi_from)
        woi_to = float(xml_data.find(".//WOI[@To]").text)
        if woi_to is None:
            return None
        else:
            woi_to = float(woi_to)   
        
        # Calculate window of interest
        woi = {
            'From': reference_annotation + woi_from,
            'To': reference_annotation + woi_to
        }
        return woi
    else:
        return None  # Return None if no file or multiple files found



def find_signal_file(WaveFront, Catheter_Type, Point_Number,sheep_name):
    """
    Function to find the signal file for a given point.

    Parameters:
    WaveFront (str): The wavefront of the point.
    Catheter_Type (str): The type of catheter used.
    Point_Number (int): The point number.

    Returns:
    list: A list of full paths to the matching signal files if found, else an empty list.
    
    """
    # Construct the file pattern
    pattern = f".*{WaveFront} {Catheter_Type}_P{Point_Number}_ECG_Export\\.txt$"
    main_data_path =  get_sheep_path(sheep_name)['main_data_path']  
    
    #print(pattern)
    #print(main_data_path)
    
    # Find files that match the pattern
    matching_files = [f for f in main_data_path.glob('**/*') if re.match(pattern, str(f.name))]
    
    # Returning full paths of the matching files
    return [str(file) for file in matching_files]



def get_signal_data(WaveFront, Catheter_Type, Point_Number, sheep_name):
    """
    Function to retrieve signal data for a given point.

    Parameters:
    WaveFront (str): The wavefront of the point.
    Catheter_Type (str): The type of catheter used.
    Point_Number (int): The point number.
    sheep_name (str): The name of the sheep.

    Returns:
    pd.Series: A pandas Series containing the signal data if found, else None.
    """

    # Find the window of interest
    woi = find_window(WaveFront, Catheter_Type, Point_Number, sheep_name)
    if woi is None:
        return None  # If window of interest is not found, return None

    # Find the signal file
    txt_files = find_signal_file(WaveFront, Catheter_Type, Point_Number, sheep_name)
    if not txt_files:
        return None  # If no signal file is found, return None
    txt_file = txt_files[0]  # Assuming only one file is returned

    # Read the signal data
    tabular_content = pd.read_csv(txt_file, skiprows=3, delim_whitespace=True)

    # Extract the gain value from the file
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    gain_line = next((line for line in lines[:4] if "Raw ECG to MV (gain) =" in line), None)
    if gain_line:
        raw_ecg_gain = float(re.search(r"Raw ECG to MV \(gain\) = ([0-9.]+)", gain_line).group(1))
    else:
        return None  # If gain is not found, return None

    # Extract the channel
    channel_line = next((line for line in lines[:4] if "Bipolar Mapping Channel=" in line), None)
    if channel_line:
        channel = re.search(r"Bipolar Mapping Channel=(\w+-\w+)", channel_line).group(1)
    else:
        return None  # If channel is not found, return None

    # Select the relevant channel and apply the window of interest
    column_name = next((col for col in tabular_content.columns if channel in col), None)
    if column_name:
        signal_data = tabular_content.loc[woi['From']:woi['To'], column_name] * raw_ecg_gain
        signal_data.rename("signal_data", inplace=True)
        return signal_data
    else:
        return None  # If channel column does not exist in the data



def get_signal_unipolar_data(WaveFront, Catheter_Type, Point_Number, sheep_name):
    # Get the window of interest
    woi = find_window(WaveFront, Catheter_Type, Point_Number, sheep_name)
    if woi is None:
        return None  # If no window of interest is found, return None

    # Find the signal file
    txt_files = find_signal_file(WaveFront, Catheter_Type, Point_Number, sheep_name)
    if not txt_files:
        return None  # If no signal file is found, return None
    txt_file = txt_files[0]  # Assuming only one file is returned

    # Read the signal data, skipping the first three rows
    tabular_content = pd.read_csv(txt_file, skiprows=3, delim_whitespace=True)

    # Extract the gain value from the file
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    gain_line = next((line for line in lines[:4] if "Raw ECG to MV (gain) =" in line), None)
    if gain_line:
        raw_ecg_gain = float(re.search(r"Raw ECG to MV \(gain\) = ([0-9.]+)", gain_line).group(1))
    else:
        return None  # If gain is not found, return None

    # Extract the channel information
    channel_line = next((line for line in lines if "Unipolar Mapping Channel=" in line), None)
    if channel_line:
        channel = re.search(r"Unipolar Mapping Channel=(\w+_\w+)", channel_line).group(1)
    else:
        return None  # If channel is not found, return None

    # Find the column that contains the channel, avoiding those with negative signs
    if channel:
        channel_columns = [col for col in tabular_content.columns if channel in col and '-' not in col]
        if not channel_columns:
            return None  # If no columns match the channel, return None
        signal_data = tabular_content[channel_columns[0]][woi['From']:woi['To']] * raw_ecg_gain
        signal_data.rename("signal_data", inplace=True)
        return signal_data

    return None



def get_raw_signal_data(WaveFront, Catheter_Type, Point_Number, sheep_name):
    # Find the signal file
    txt_files = find_signal_file(WaveFront, Catheter_Type, Point_Number, sheep_name)
    if not txt_files:
        return None  # Return None if no signal file is found
    txt_file = txt_files[0]  # Assuming only one file is found

    # Read the table content, skipping the first three rows
    tabular_content = pd.read_csv(txt_file, skiprows=3, delim_whitespace=True)

    # Extract the gain value from the first four lines of the file
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    gain_line = next((line for line in lines[:4] if "Raw ECG to MV (gain) =" in line), None)
    if gain_line:
        raw_ecg_gain = float(re.search(r"Raw ECG to MV \(gain\) = ([0-9.]+)", gain_line).group(1))
    else:
        return None  # Return None if gain is not found

    # Extract the channel information
    channel_line = next((line for line in lines[:4] if "Bipolar Mapping Channel=" in line), None)
    if channel_line:
        channel = re.search(r"Bipolar Mapping Channel=(\w+-\w+)", channel_line).group(1)
    else:
        return None  # Return None if channel is not found

    # Find the appropriate column for the bipolar recordings given the channel and window of interest
    # find column name that includes the channel name in the first characters
    column_name = next((col for col in tabular_content.columns if channel in col), None)
    if column_name:
        signal_data = tabular_content[column_name] * raw_ecg_gain
        signal_data.rename("signal_data", inplace=True)
        return signal_data

    return None


def retrieve_all_signals(sheep_name):
    """
    Function to retrieve all signals for a given sheep and save the data to a CSV file.

    Parameters:
    sheep_name (str): The name of the sheep.    
    """
    # Load the initial labelled data
    LabelledSignalData = load_sheep(sheep_name)
    
    # Apply transformations row-wise
    for index, row in LabelledSignalData.iterrows():
        LabelledSignalData.at[index, 'signal'] = get_signal_data(row['WaveFront'], row['Catheter_Type'], row['Point_Number'], sheep_name)
        LabelledSignalData.at[index, 'rawsignal'] = get_raw_signal_data(row['WaveFront'], row['Catheter_Type'], row['Point_Number'], sheep_name)
        LabelledSignalData.at[index, 'signal_unipolar'] = get_signal_unipolar_data(row['WaveFront'], row['Catheter_Type'], row['Point_Number'], sheep_name)

    #LabelledSignalData['rawsignal'] = LabelledSignalData.apply(
    #lambda row: get_raw_signal_data(row['WaveFront'], row['Catheter_Type'], row['Point_Number'], sheep_name),
    #axis=1
    #)
    
    # Create masks for signals present and not present
    mask_no_signals = LabelledSignalData['signal'].isnull()
    mask_with_signals = ~mask_no_signals
    
    # Handle rows with no signals
    no_signals = LabelledSignalData[mask_no_signals].copy()
    no_signals['From'], no_signals['To'] = 0, 0
    
    # Handle rows with signals
    with_signals = LabelledSignalData[mask_with_signals].copy()
    with_signals['From'] = with_signals.apply(lambda row: find_from_woi(row['WaveFront'], row['Catheter_Type'], row['Point_Number'], sheep_name)[0], axis=1)
    with_signals['To'] = with_signals.apply(lambda row: find_to_woi(row['WaveFront'], row['Catheter_Type'], row['Point_Number'], sheep_name)[1], axis=1)
    
    # Combine the data back together
    LabelledSignalData = pd.concat([with_signals, no_signals])
    
    # Convert all character columns to categorical
    for col in LabelledSignalData.select_dtypes(include=[object]).columns:
        LabelledSignalData[col] = LabelledSignalData[col].astype('category')
    
    # Sort the dataframe as per the requirement
    LabelledSignalData.sort_values(by=['sheep', 'Catheter_Type', 'WaveFront', 'Point_Number'], inplace=True)
    
    # Save the processed data to a file (pickle used as a substitute for RDS)
    output_path = generated_data_path / f"LabelledSignalData{sheep_name}.pkl"
    LabelledSignalData.to_pickle(output_path)


def retrieve_signal(sheep_name, signal_type):
    """
    Function to retrieve all signals for a given sheep and save the data to a CSV file.

    Parameters:
    sheep_name (str): The name of the sheep.   
    signal_type (str): The type of signal to retrieve.
        'signal', 'rawsignal', or 'signal_unipolar' 
    """
    # Load the initial labelled data
    LabelledSignalData = load_sheep(sheep_name)

    if signal_type == 'rawsignal':
        LabelledSignalData['rawsignal'] = LabelledSignalData.apply(
            lambda row: get_raw_signal_data(row['WaveFront'], row['Catheter_Type'], row['Point_Number'], sheep_name),
            axis=1
            )
    elif signal_type == 'signal':
        LabelledSignalData['signal'] = LabelledSignalData.apply(
            lambda row: get_signal_data(row['WaveFront'], row['Catheter_Type'], row['Point_Number'], sheep_name),
            axis=1
            )
    elif signal_type == 'signal_unipolar':
        LabelledSignalData['signal_unipolar'] = LabelledSignalData.apply(
            lambda row: get_signal_unipolar_data(row['WaveFront'], row['Catheter_Type'], row['Point_Number'], sheep_name),
            axis=1
            )
    else:
        raise ValueError("Invalid signal type. Choose from 'signal', 'rawsignal', or 'signal_unipolar'.")

    
    # Create masks for signals present and not present
    if signal_type == 'signal':
        mask_no_signals = LabelledSignalData['signal'].isnull()
        mask_with_signals = ~mask_no_signals
        
        # Handle rows with no signals
        no_signals = LabelledSignalData[mask_no_signals].copy()
        no_signals['From'], no_signals['To'] = 0, 0
        
        # Handle rows with signals
        with_signals = LabelledSignalData[mask_with_signals].copy()
        with_signals['From'] = with_signals.apply(lambda row: find_from_woi(row['WaveFront'], row['Catheter_Type'], row['Point_Number'], sheep_name)[0], axis=1)
        with_signals['To'] = with_signals.apply(lambda row: find_to_woi(row['WaveFront'], row['Catheter_Type'], row['Point_Number'], sheep_name)[1], axis=1)
        
        # Combine the data back together
        LabelledSignalData = pd.concat([with_signals, no_signals])
    
    # Convert all character columns to categorical
    for col in LabelledSignalData.select_dtypes(include=[object]).columns:
        LabelledSignalData[col] = LabelledSignalData[col].astype('category')
    
    # Sort the dataframe as per the requirement
    LabelledSignalData.sort_values(by=['sheep', 'Catheter_Type', 'WaveFront', 'Point_Number'], inplace=True)
    
    # Save the processed data to a file (pickle used as a substitute for RDS)
    output_path = generated_data_path / f"LabelledSignalData{sheep_name}.pkl"
    LabelledSignalData.to_pickle(output_path)



def load_histology():
    """
    Function to load histology data from a CSV file and perform data cleaning.

    Returns:
    pd.DataFrame: A DataFrame containing the cleaned histology data. 
    """
    # Path to the file
    file_path = Path("data/cleaned_histology_all.csv")
    
    # Read the CSV file
    histology_labels = pd.read_csv(file_path)
    
    # Select and rename columns
    histology_labels = histology_labels[[
        'Animal', 'Specimen_ID', 'Endo3_anyscar', 'IM3_anyscar', 'Epi3_anyscar',
        'Endo3__VM', 'IM3_VM', 'Epi3_VM'
    ]]
    histology_labels = histology_labels.rename(columns={
        'Specimen_ID': 'Histology_Biopsy_Label',
        'Endo3_anyscar': 'endocardium_scar',
        'IM3_anyscar': 'intramural_scar',
        'Epi3_anyscar': 'epicardial_scar',
        'Endo3__VM': 'healthy_perc_endo',
        'IM3_VM': 'healthy_perc_intra',
        'Epi3_VM': 'healthy_perc_epi'
    })
    
    # Remove rows with missing values
    histology_labels = histology_labels.dropna()
    
    return histology_labels


def incorporate_histology(sheep_name):
 
    # Load labeled signal data (assuming it's saved as a pickle for Python translation)
    label_signal_path = generated_data_path / f"LabelledSignalData{sheep_name}.pkl"
    LabelledSignalData = pd.read_pickle(label_signal_path)
    
    # Load histology data
    histology = load_histology()
    
    # Merge data
    result = pd.merge(LabelledSignalData, histology, left_on="Histology_Biopsy_Label", right_on="Histology_Biopsy_Label", how="left")
    
    # Save the merged data (using pickle to mimic RDS functionality in Python)
    result_path = generated_data_path / f"NestedData{sheep_name}.pkl"
    result.to_pickle(result_path)


def write_as_long_format(sheep_name):
    # Assuming the data is stored in a Python-readable format, such as a pickle file
    nested_data_path = generated_data_path / f"NestedData{sheep_name}.pkl"
    NestedData = pd.read_pickle(nested_data_path)
    
    # Exploding nested 'signal' data into a long format
    LongData = NestedData.explode('signal')
    
    # Write the data to a CSV file
    csv_file_path = generated_data_path / f"NestedData{sheep_name}.csv"
    LongData.to_csv(csv_file_path, index=False)


def write_as_parquet(sheep_name):
    # Loading data from a Python-readable format
    nested_data_path = generated_data_path / f"NestedData{sheep_name}.pkl"
    NestedData = pd.read_pickle(nested_data_path)
    
    # Writing data to a Parquet file
    parquet_file_path = generated_data_path / f"NestedData{sheep_name}.parquet"
    NestedData.to_parquet(parquet_file_path)


def post_introduction_perc_health(sheep_name):
    # Combine histology with labeled signal data and perform data transformations
    # Not entire workflow so to avoid signal aggregation
    incorporate_histology(sheep_name)
    write_as_long_format(sheep_name)
    write_as_parquet(sheep_name)


if __name__ == "__main__":
    run_all(sheep_names)
    #post_introduction_perc_health(sheep_name)
    print("Data processing completed successfully.")
