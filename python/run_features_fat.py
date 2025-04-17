# Workflow for fat composition prediction with tsai

import pandas as pd
import os
from features_fat import FeatureExtraction
import glob

# path to preprocessed data
inpath =  '../../../data/FAT SUB PROJECT'
target = 'EndoIntra_SCARComposition'
infname_list = ['publishable_model_data_AdiposityElectrogram_master_merged_signal_clean.parquet',
                'publishable_model_data_AdiposityElectrogram_master_merged_raw_unipolar_clean.parquet',
                'publishable_model_data_AdiposityElectrogram_master_merged_signal_unipolar_clean.parquet',
                'publishable_model_data_AdiposityElectrogram_master_merged_rawsignal_clean.parquet']

outpath_names = ['signal', 'raw_unipolar', 'signal_unipolar', 'rawsignal']


for i in range(len(infname_list)):
    # loops over the different signal types and wavefronts
    outpath = os.path.join(f'../../../results/features/fat/{outpath_names[i]}/')
    os.makedirs(outpath, exist_ok=True)
    inpath = inpath
    fname_csv = infname_list[i]
    fe = FeatureExtraction(inpath, fname_csv, outpath)
    fe.run_target(target)

# merge on


def process_and_merge_csv_files(directory_path, master_parquet_path, outfname='merged_data.csv', keep_column=None):
    """
    Process all CSV files that match a specific pattern, rename columns by removing 
    "signal_data" prefix, merge all with a master dataframe, and save the result.
    
    Args:
        directory_path (str): Path to search for CSV files
        master_parquet_path (str): Path to the master parquet file to merge with
        
    Returns:
        pd.DataFrame: The final merged dataframe
    """
    # Load the master dataframe
    master_df = pd.read_parquet(master_parquet_path)

    # drop columns:  'signal','rawsignal','signal_unipolar','signal_data' exce
    list_of_columns = ['signal', 'rawsignal', 'signal_unipolar', 'raw_unipolar']
    for col in list_of_columns:
        # drop columns if they exist
        if col in master_df.columns and col != keep_column:
            master_df = master_df.drop(columns=col)
    #master_df = master_df.drop(columns=['signal', 'rawsignal', 'signal_unipolar', 'raw_unipolar'], errors='ignore')


    
    # Find all matching CSV files
    pattern = os.path.join(directory_path, "selected_features_EndoIntra_SCARComposition_*.csv")
    csv_files = glob.glob(pattern)
    
    # List to store all processed dataframes
    all_dfs = []
    
    # Process each CSV file
    for csv_file in csv_files:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Rename columns by removing "signal_data" prefix
        rename_dict = {}
        for column in df.columns:
            if column.startswith("signal_data"):
                rename_dict[column] = column.replace("signal_data", "")
        
        df = df.rename(columns=rename_dict)
        
        # Add to the list of dataframes
        all_dfs.append(df)
        print(f"Processed: {os.path.basename(csv_file)}")
    
    # Combine all CSV dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs)
        
        # Remove duplicates if needed (in case the same 'id' appears in multiple CSVs)
        combined_df = combined_df.drop_duplicates(subset=['id'])
        
        # Merge with master dataframe on 'id' column
        merged_df = pd.merge(master_df, combined_df, on='id', how='left')

        # check how many valud values in column __fft_coefficient__attr_"imag"__coeff_92
        # merged_df['__fft_coefficient__attr_"imag"__coeff_92'].isna().sum()
        print('Number of valid merges:', merged_df['__maximum'].notna().sum())
        
        # Get the directory of the master file to save the result
        #master_dir = os.path.dirname(master_parquet_path)
        output_path = os.path.join(directory_path, outfname)
        
        # Save the merged dataframe
        merged_df.to_csv(output_path, index=False)
        print(f"Merged data saved to: {output_path}")
        
        return merged_df
    else:
        print("No matching CSV files found.")
        return None


# Example usage:
directory_path = '../../../results/features/fat/signal_unipolar'
master_parquet_path ='../../../data/FAT SUB PROJECT/publishable_model_data_AdiposityElectrogram_master_merged.parquet'

merged_df = process_and_merge_csv_files(directory_path, master_parquet_path, outfname='merged_data.csv', keep_column='signal_unipolar')

# repeat for 'signal', 'rawsignal', 'signal_unipolar', 'raw_unipolar'],


