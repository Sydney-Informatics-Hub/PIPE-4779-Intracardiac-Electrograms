import pandas as pd
from data_injest import DataIngest
from inference_pipe import preprocess_data


# Is the python version actually the same as the R data inject version?
# yes - see /r_scripts/testing.R
def test_python_injest():
    data_dir = "../deploy/data/Export_Analysis"
    output_dir = "../deploy/data"
    catheter_type = "Penta"
    
    fname_preprocessed = preprocess_data(data_dir,output_dir,catheter_type)
    #../deploy/data/preprocessed_rawsignal_unipolar_penta.parquet
    

# TODO - pick one model and test rather than ensumble.