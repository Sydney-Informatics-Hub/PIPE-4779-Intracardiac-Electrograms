# Train and test unsupervised classifier using tsai package


from tsai.all import Categorize, TSDatasets, InceptionTime, Learner, accuracy, TSStandardize, ClassificationInterpretation
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd

inpath = '../../../data/generated'
fname_csv = 'NestedDataS18.csv'

def load_data(inpath, fname_csv):
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
    if not os.path.isfile(os.path.join(inpath, fname_csv)):
        raise FileNotFoundError(f'File {fname_csv} not found in {inpath}')
    df = pd.read_csv(os.path.join(inpath, fname_csv), usecols=usecols)
    # remove nan values
    df = df.dropna()
    # add column "time" to df which starts at 0 and has the same length as "signal_data" for each Point_Number and WaveFront
    df['time'] =df.groupby(['Point_Number', 'WaveFront']).cumcount()
    # Generate a column 'scar' that is 1 if either of the scar columns is 1, otherwise 0
    df['endocardium_scar'] = df['endocardium_scar'].astype(int)
    df['intramural_scar'] = df['intramural_scar'].astype(int)
    df['epicardial_scar'] =df['epicardial_scar'].astype(int)
    df['scar'] = df[['endocardium_scar', 'intramural_scar', 'epicardial_scar']].max(axis=1)

def df_to_ts(df, wavefront, target='scar'):
    """
    Converts the dataframe to a timeseries for the given wavefront

    Args:
        wavefront (str): 'LVp', 'RVp', or 'SR'
        target (str): 'scar' (Default) or 'endocardium_scar', 'intramural_scar', 'epicardial_scar'
    """
    dfsel = df[df['WaveFront'] == wavefront][['Point_Number', 'time', 'signal_data', target]]
    npoints_unique = dfsel['Point_Number'].nunique()
    signal = [] #np.zeros((npoints_unique, timeseries['signal_data'].apply(len).max()))
    y = dfsel[['Point_Number', target]].drop_duplicates()
    # get length of signal_data for each point
    signal_length = dfsel.groupby('Point_Number')['signal_data'].apply(len)
    X = np.zeros((len(y), signal_length.max()))
    for i in range(len(y)):
        point = y.iloc[i]['Point_Number']
        # get signal_data
        data = dfsel[dfsel['Point_Number'] == point]['signal_data']
        X[i, :len(data)] = data

    return X.reshape((len(y), 1, -1)), y[target].values

# load data
df = pd.read_csv(os.path.join(inpath, fname))

wavefront = 'SR'
target = 'scar'


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfms  = [None, [Categorize()]]
dsets = TSDatasets(X_train, y_train, tfms=tfms, splits=splits, inplace=True)

# build dataloader to created batches of data
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)

# build learner
model = InceptionTime(dls.vars, dls.c)
learn = Learner(dls, model, metrics=accuracy)
learn.save('stage0')

learn.load('stage0')
learn.lr_find()

learn.fit_one_cycle(25, lr_max=1e-3)
learn.save('stage1')

learn.recorder.plot_metrics()

learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
#learn = load_learner_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')

valid_probas, valid_targets, valid_preds = learn.get_preds(dl=valid_dl, with_decoded=True)

learn.show_results()

learn.show_probas()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# test
test_ds = valid_dl.dataset.add_test(X_test, y_test)# In this case I'll use X and y, but this would be your test data
test_dl = valid_dl.new(test_ds)