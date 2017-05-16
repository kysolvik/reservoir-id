#!/usr/bin/env python
"""
@authors: Kylen Solvik
Date Create: 3/17/17
"""

# Load libraries
import pandas
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

# Set area cutoff
acut_small = 0
acut_large = 500000

# Set any attributes to exclude for this run

# Load dataset
train_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/att_table_wndwi.csv'
predict_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/tables/att_table.csv'
out_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/tables/att_table_predict.csv'


dataset_train = pandas.read_csv(train_path,header=0)
del dataset_train['ndwi_mean']
del dataset_train['ndwi_max']
del dataset_train['ndwi_min']

dataset_predict = pandas.read_csv(predict_path,header=0)

dataset = dataset_train.append(dataset_predict)

dataset_acut_large = dataset.loc[dataset['obj_area'] < acut_large]
dataset_acut = dataset_acut_large.loc[dataset_acut_large['obj_area'] > acut_small]

(ds_y,ds_x) = dataset_acut.shape
print(ds_y,ds_x)

# Convert dataset to array
array = dataset_acut.values
X = array[:,2:ds_x]
Y = array[:,1]

# Set infs to the max value for float32s
X[np.isinf(X)] = 2.59248034e+15
#X[np.isinf(X)] = (np.finfo(np.float32).max)

# If needed, check for and remove nans
#nan = np.isnan(X).any(axis=1)
#X = X[~nan]
#Y = Y[~nan]

# Scale!
#X_scaled = X
X_scaled = preprocessing.robust_scale(X)
#print np.var(X_scaled,axis=0)
#print np.var(X_scaled,axis=0)
#print np.mean(X_scaled,axis=0)
#print np.min(X_scaled,axis=0)
#print np.max(X_scaled,axis=0)


# Select only classified data
X_scaled_classified = X_scaled[Y > 0]
Y_classified = Y[Y > 0]

# Separate validation data
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_scaled_classified, Y_classified, test_size=validation_size, random_state=seed)

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
#models.append(('SVM5',SVC(C=5)))
models.append(('SVM10', SVC(C=10)))
models.append(('SVM1',SVC(C=1)))
models.append(('SVM.1',SVC(C=.1)))
#models.append(('SVM.01',SVC(C=.01)))
models.append(('RF10',RandomForestClassifier(n_estimators=10)))
#models.append(('RF50',RandomForestClassifier(n_estimators=50)))
models.append(('RF100',RandomForestClassifier(n_estimators=100)))
#models.append(('RF200',RandomForestClassifier(n_estimators=200)))
models.append(('RF300',RandomForestClassifier(n_estimators=300)))
#models.append(('RF400',RandomForestClassifier(n_estimators=400)))
              
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Define random forest
rf = RandomForestClassifier(n_estimators=200)

# Get learning curve for random forest
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
#t_sizes, t_scores, cv_scores = model_selection.learning_curve(rf, X_train, Y_train, cv=kfold, scoring=scoring,train_sizes=np.array([ 0.1, 0.2, 0.3, 0.4,.5,.6,.7,.8,.9, 1. ]))
#t_scores_mean = np.mean(t_scores,axis=1)
#cv_scores_mean = np.mean(cv_scores,axis=1)
#plt.plot(t_sizes,t_scores_mean,'r--',t_sizes,cv_scores_mean,'b--')
#plt.show()

# Make predictions on validation dataset
rf.fit(X_train, Y_train)

# For training accuracy
rf_train_predict = rf.predict(X_train)

print('Training acc = ' + str(accuracy_score(Y_train,rf_train_predict)))
# For validation accuracy
rf_predictions = rf.predict(X_validation)
print('RF CV acc = ' + str(accuracy_score(Y_validation, rf_predictions)))
print(confusion_matrix(Y_validation, rf_predictions))
print(classification_report(Y_validation, rf_predictions))


# Run on full dataset
#rf = RandomForestClassifier(n_estimators=200)
#rf.fit(X_scaled_classified,Y_classified)
rf_full_pred = rf.predict(X_scaled)
dataset_out = dataset_acut
dataset_out['rf_pred'] = rf_full_pred
dataset_out.to_csv(out_path,index=False)

