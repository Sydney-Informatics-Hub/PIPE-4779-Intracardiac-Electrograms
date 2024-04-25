from classifier_tsai import run_all


def main():
    inpath = '../data/generated/'
    fname_csv = 'publishable_model_data_TSAI.parquet'
    outpath = '../deploy/output'
    target_list = ['NoScar', 'AtLeastEndo', 'AtLeastIntra', 'epiOnly'] 
    run_all(inpath ,fname_csv,outpath,target_list)
    #tsai = TSai(inpath, fname_csv)
    #df = tsai.load_data(inpath, fname_csv)
