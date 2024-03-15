# Train and test unsupervised classifier using tsai package


from tsai.all import Categorize, TSDatasets, InceptionTime, Learner, accuracy, TSStandardize, ClassificationInterpretation
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd

inpath = '../../../data/generated'
fname = 'NestedDataS18.csv'

# load data
df = pd.read_csv(os.path.join(inpath, fname))


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