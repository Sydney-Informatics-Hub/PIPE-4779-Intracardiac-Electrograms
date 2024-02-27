# Random Forest Classifier

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# import the feature selection class
from features import FeatureExtraction

# Settings
inpath = '../../../data/generated'
fname_csv = 'NestedDataS18.csv'
outpath = 'test'
target = 'scar'
wavefront = 'SR'

# extract relevant features
fe = FeatureExtraction(inpath, fname_csv, outpath)
fe.run_wavefront_target(wavefront, target)
X = fe.selected_features
y = fe.y

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# calculate sample weights to rebalance the classes
n_0 = y_train.value_counts()[0]
n_1 = y_train.value_counts()[1]
n_total = n_0 + n_1
sample_weight = y_train.map({0: n_total / (2 * n_0), 1: n_total / (2 * n_1)})

# train random forest classifier
rf = RandomForestClassifier(n_estimators=1000, 
                            max_depth=5, 
                            random_state=42, 
                            warm_start=False,
                            n_jobs=-1, 
                            class_weight='balanced')
rf.fit(X_train, y_train)
#rf.fit(X_train, y_train, sample_weight = sample_weight)

# print classification report
print(classification_report(y_test, rf.predict(X_test)))
#print(classification_report(y_train, rf.predict(X_train)))