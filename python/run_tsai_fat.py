# Workflow for fat composition prediction with tsai

import pandas as pd
import os


### Merging ###
from merge_ecg_fathistology import combine_electro_histology_data

csv_relevant_cols = [
'Sample_number', 'id', 'Histology_Biopsy_Label',
'EndoIntra_SCARComposition', 'Wall_thickness_3', 'TB_Fibrosis%',
'TB_VM%', 'TB_Adip%', 'Endo3_fibrosis', 'Endo3__VM',
'Endo3_Adiposity', 'Endo3_anyscar', 'Endo3_ScarComposition',
'IM3_fibrosis', 'IM3_VM', 'IM3_adiposity', 'IM3_anyscar',
'Intra3_ScarComposition', 'Epi3_fibrosis', 'Epi3_VM',
'Epi3_Adiposity', 'Epi3_anyscar', 'Epi3_ScarComposition',
'LAVA', 'Split_potentials']
merge_cols = ['id', 'Histology_Biopsy_Label']

inpath = '../../../data'
parquet_path = os.path.join(inpath,'publishable_data/publishable_model_data_TSAI.parquet') 
csv_path = os.path.join(inpath,'FAT SUB PROJECT/AdiposityElectrogram_master.csv')  
outpath = os.path.join(inpath, 'FAT SUB PROJECT')

merged_df = combine_electro_histology_data(parquet_path, csv_path, csv_relevant_cols, merge_cols, outpath)

print("\n--- Combined DataFrame Head ---")
print(merged_df.head())

print("\n--- Combined DataFrame Info ---")
merged_df.info()



### Preprocessing to tsai format ###
from classifier_tsai import TSai, preprocess_rawsignal_singlefile, test_tsai


inpath =  '../../../data/FAT SUB PROJECT'

infname_parquet = 'publishable_model_data_AdiposityElectrogram_master_merged.parquet'

# start with unipolar
dfclean, fname_out = preprocess_rawsignal_singlefile(inpath,
                                    infname_parquet, 
                                    signal='raw_unipolar',
                                    target_columns=['EndoIntra_SCARComposition']
            )

# File publishable_model_data_AdiposityElectrogram_master_merged_raw_unipolar_clean.parquet saved to ../../../data/FAT SUB PROJECT

# save head100 to csv
dfclean.head(100).to_csv(os.path.join(inpath, 'publishable_model_data_AdiposityElectrogram_master_merged_raw_unipolar_clean_head100.csv'), index=False)

### test tsai manually first
inpath =  '../../../data/FAT SUB PROJECT'
target = 'EndoIntra_SCARComposition'
wavefront = 'SR'
tsai = TSai(inpath = inpath, 
            fname_csv = 'publishable_model_data_AdiposityElectrogram_master_merged_raw_unipolar_clean.parquet', 
            load_train_data=True, 
            target_type='fat')
X, y = tsai.df_to_ts(wavefront, target)
tsai.train_model(X, y, epochs = 20, balance_classes = True)
path_name = '../results/tsai/fat/raw_unipolar/' + f'_{target}_{wavefront}' 
accuracy, precision, auc, mcc, macro_sensitivity, macro_specificity  = tsai.eval_model(outpath=path_name)


### run all
from classifier_tsai import TSai, preprocess_rawsignal_singlefile, run_all

signal_name_list = ['signal', 'rawsignal', 'signal_unipolar', 'raw_unipolar']
inpath =  '../../../data/FAT SUB PROJECT'
#inpath =  '../../data/'
infname_parquet = 'publishable_model_data_AdiposityElectrogram_master_merged.parquet'
target_list=['EndoIntra_SCARComposition']

for signal_name in signal_name_list:
    _, fname_out = preprocess_rawsignal_singlefile(inpath,
                                        infname_parquet, 
                                        signal=signal_name,
                                        target_columns=['EndoIntra_SCARComposition']
    )

    fname_csv = fname_out
    outpath = f'../results/tsai/fat/{signal_name}/'

    run_all(inpath,
            fname_csv, 
            outpath, 
            target_list=target_list,
            target_type='fat',
            model_signal_name=signal_name,
            method='CNN',
            epochs=150)
    
