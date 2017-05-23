#!/usr/bin/env python
"""
@authors: Kylen Solvik
Date Create: 3/17/17
"""

# Load libraries
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys

# Arguments
in_csv_path = sys.argv[1]
out_csv_path = sys.argv[2]
acut_small = int(sys.argv[3]) # Won't attempt to predict below this. Recommend 1 or 2

# Testing
#in_csv_path = "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/skimage_try2/testprop.csv"
#out_csv_path = "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/skimage_try2/classified.csv"
#acut_small=1

# Set any attributes to exclude for this run
exclude_atts = []

# Load dataset
dataset = pd.read_csv(in_csv_path,header=0)
dataset_acut = dataset.loc[dataset['area'] > acut_small]

# Add any attributes that are all nans to the exclude list
for att in dataset.columns[1:]:
        if sum(np.isfinite(dataset[att])) == 0:
                    exclude_atts.append(att)

for att in exclude_atts:
    del dataset_acut[att]

(ds_y,ds_x) = dataset_acut.shape
print(ds_y,ds_x)

# Convert dataset to array
array = dataset_acut.values
X = array[:,2:ds_x].astype(float)
Y = array[:,1].astype(int)

# Set nans to 0
X = np.nan_to_num(X)

# Scale!
X_scaled = preprocessing.robust_scale(X)
X_scaled_classified = X_scaled[Y > 0]
Y_classified = Y[Y > 0]

# Separate test data
test_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X_scaled_classified, Y_classified, test_size=test_size,
    random_state=seed)

# Test options and evaluation metric
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR.1', LogisticRegression(C=.1)))
models.append(('LR1', LogisticRegression(C=1)))
models.append(('LR10', LogisticRegression(C=10)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM10', SVC(C=10)))
models.append(('SVM1',SVC(C=1)))
models.append(('SVM.1',SVC(C=.1)))
models.append(('RF10',RandomForestClassifier(n_estimators=10)))
models.append(('RF64',RandomForestClassifier(n_estimators=64)))
models.append(('RF100',RandomForestClassifier(n_estimators=100)))
              
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train,
                                                 cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Define random forest
rf = RandomForestClassifier(n_estimators=128)

# # Get learning curve for random forest
# kfold = model_selection.KFold(n_splits=10, random_state=seed)
# t_sizes, t_scores, cv_scores = model_selection.learning_curve(rf, X_train, Y_train, cv=kfold, scoring=scoring,train_sizes=np.array([ 0.1, 0.2, 0.3, 0.4,.5,.6,.7,.8,.9, 1. ]))
# t_scores_mean = np.mean(t_scores,axis=1)
# cv_scores_mean = np.mean(cv_scores,axis=1)
# plt.plot(t_sizes,t_scores_mean,'r--',t_sizes,cv_scores_mean,'b--')
# plt.show()

# Make predictions on test dataset
rf.fit(X_train, Y_train)

# For training accuracy
rf_train_predict = rf.predict(X_train)

print("Training acc = " + str(accuracy_score(Y_train,rf_train_predict)))
# For test accuracy
rf_predictions = rf.predict(X_test)
print("RF CV acc = " + str(accuracy_score(Y_test, rf_predictions)))
print(confusion_matrix(Y_test, rf_predictions))
print(classification_report(Y_test, rf_predictions))


# Run on full dataset
#rf = RandomForestClassifier(n_estimators=200)
#rf.fit(X_scaled_classified,Y_classified)
rf_full_pred = rf.predict(X_scaled)
dataset_out = dataset_acut
dataset_out["rf_pred"] = rf_full_pred
dataset_out.to_csv(out_csv_path,index=False)

