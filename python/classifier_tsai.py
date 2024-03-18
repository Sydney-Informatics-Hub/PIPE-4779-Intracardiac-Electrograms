# Train and test unsupervised classifier using tsai package
# use conda environment tsai (Python 3.10)


from tsai.basics import (
    Categorize, 
    TSDatasets, 
    Learner, accuracy, 
    TSStandardize, 
    TSClassification,
    ClassificationInterpretation,
    combine_split_data,
    TSClassifier,
    accuracy,
    ShowGraph
)
from tsai.inference import load_learner
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.signal import resample
import os
import numpy as np
import pandas as pd

inpath = '../../../data/generated'
fname_csv = 'NestedDataS18.csv'

wavefront = 'SR'
target = 'scar'

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
    return df

def df_to_ts(df, wavefront, target='scar'):
    """
    Converts the dataframe to tsai format for a given wavefront and target tissue

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

def resample_data(signal, sample_length):
    """
    Resample the signal to a given sample length
    """
    resampled_signals = []
    for si in signal:
		resampled_signal.append(resample(si, sample_length))
    resampled_signal = np.array(resampled_signal)
    #resampled_signals.append(resampled_signal)
    return resampled_signal



# load data
df = load_data(inpath, fname_csv)
X, y = df_to_ts(df, wavefront=wavefront, target=target)
X_orig = X.copy()
y_orig = y.copy()


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])
X.shape, y.shape, splits
 
tfms  = [None, [TSClassification()]]
batch_tfms = TSStandardize()
clf = TSClassifier(X, y, splits=splits, path='models', arch="InceptionTimePlus", tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy, cbs=ShowGraph())
clf.fit_one_cycle(100, 3e-4)

# save the model
clf.export("clf.pkl")
# load the model
#clf = load_learner("models/clf.pkl")

# inference
#probas, target, preds = clf.get_X_preds(X[splits[1]], y[splits[1]])
probas, _, preds = clf.get_X_preds(X_test)

# convert int array to str preds
#target = np.array([str(t) for t in y_test])
# convert str array to int array
preds = np.array([int(t) for t in preds])

print(classification_report(y_test, preds, target_names=['no scar', 'scar'], output_dict=False))

# convert str preds to int array
#preds = np.array([int(pred) for pred in preds])

# build dataloader to created batches of data

"""
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
"""