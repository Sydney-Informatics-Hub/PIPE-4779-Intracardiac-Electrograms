# Exports parque file headers and first 100 rows to CSV and columns as text file

import pandas as pd
import os
import sys

def process_parquet(input_file):
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    # Check if file is a parquet file
    if not input_file.endswith('.parquet'):
        print(f"Error: File '{input_file}' is not a parquet file.")
        sys.exit(1)
    
    # Get the base filename without extension
    base_name = os.path.splitext(input_file)[0]
    
    # Define output filenames
    csv_output = f"{base_name}_head100.csv"
    columns_output = f"{base_name}_columns.txt"

    # add directory to output files, same as input directory
    input_dir = os.path.dirname(input_file)
    csv_output = os.path.join(input_dir, csv_output)
    columns_output = os.path.join(input_dir, columns_output)
    
    try:
        # Read the parquet file
        print(f"Reading parquet file: {input_file}")
        df = pd.read_parquet(input_file)
        
        # Write first 100 rows to CSV
        print(f"Writing first 100 rows to: {csv_output}")
        df.head(100).to_csv(csv_output, index=False)
        
        # Write column names to text file
        print(f"Writing column names to: {columns_output}")
        with open(columns_output, 'w') as f:
            for column in df.columns:
                f.write(f"{column}\n")
            f.write(f"\n")
            f.write(f"Number of columns: {len(df.columns)}\n")
            f.write(f"Number of rows: {len(df)}\n")
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <parquet_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    process_parquet(input_file)