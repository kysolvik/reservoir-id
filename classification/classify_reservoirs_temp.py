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
acut = 250

# Load dataset
filepath = "/Users/ksolvik/Documents/Research/MarciaWork/data/machineLearning/interDFs/resgeos_wextras.csv"
outpath = "/Users/ksolvik/Documents/Research/MarciaWork/data/machineLearning/outputs/resgeos_predict.csv"
dataset = pandas.read_csv(filepath,header=0)

# Select only data that we have data for.
dataset_classified = dataset.loc[dataset['class'] >0]# & dataset['objectA']>acut]
dataset_classified = dataset_classified.loc[dataset_classified['objectA']>acut]
#print(dataset_classified.head)
(ds_y,ds_x) = dataset_classified.shape
#print(dataset_classified.shape)

# Split-out validation dataset
array = dataset_classified.values
X = array[:,2:ds_x]
Y = array[:,1]-1
#print(X)
# Scale!
X_scaled = preprocessing.scale(X)
#print(X_scaled)
#print(Y)
validation_size = 0.1
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_scaled, Y, test_size=validation_size, random_state=seed)



# Check for NaNs
nans = np.isnan(X_scaled)



# Test options and evaluation metric
seed = 7
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
models.append(('SVM5',SVC(C=5)))
models.append(('SVM10', SVC(C=10)))
models.append(('SVM1',SVC(C=1)))
models.append(('SVM.1',SVC(C=.1)))
models.append(('SVM.01',SVC(C=.01)))
#models.append(('RF10',RandomForestClassifier(n_estimators=10)))
#models.append(('RF50',RandomForestClassifier(n_estimators=50)))
#models.append(('RF100',RandomForestClassifier(n_estimators=100)))
#models.append(('RF200',RandomForestClassifier(n_estimators=200)))
#models.append(('RF300',RandomForestClassifier(n_estimators=300)))
#models.append(('RF400',RandomForestClassifier(n_estimators=400)))
              
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


    
# Define random forest
rf = RandomForestClassifier(n_estimators=100)
lr = LogisticRegression(C=10)

# Get learning curve for random forest
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
#t_sizes, t_scores, cv_scores = model_selection.learning_curve(rf, X_train, Y_train, cv=kfold, scoring=scoring,train_sizes=np.array([ 0.1, 0.2, 0.3, 0.4,.5,.6,.7,.8,.9, 1. ]))
#t_scores_mean = np.mean(t_scores,axis=1)
#cv_scores_mean = np.mean(cv_scores,axis=1)
#plt.plot(t_sizes,t_scores_mean,'r--',t_sizes,cv_scores_mean,'b--')
#plt.show()

# Make predictions on validation dataset
rf.fit(X_train, Y_train)
lr.fit(X_train,Y_train)
# For training accuracy
rf_train_predict = rf.predict(X_train)
lr_train_predict = lr.predict(X_train)
print(rf_train_predict)
print(lr_train_predict)
print(lr_train_predict + rf_train_predict)
both_train_predict = 1*((rf_train_predict + lr_train_predict)>0)
print("Training acc = " + str(accuracy_score(Y_train,both_train_predict)))
# For validation accuracy
rf_predictions = rf.predict(X_validation)
lr_predictions = lr.predict(X_validation)
both_predictions = 1*((rf_predictions + lr_predictions)>0)
print("LR CV acc = " + str(accuracy_score(Y_validation, lr_predictions)))
print("RF CV acc = " + str(accuracy_score(Y_validation, rf_predictions)))
print("Both CV acc = " + str(accuracy_score(Y_validation, both_predictions)))
print(confusion_matrix(Y_validation, both_predictions))
print(classification_report(Y_validation, both_predictions))


# Run on full dataset
X_full = dataset.as_matrix()[:,2:ds_x]
rf_full_pred = rf.predict(X_full)
lr_full_pred = lr.predict(X_full)
both_full_pred = 1*((rf_full_pred + lr_full_pred)>0)
dataset_out = dataset
dataset_out["both_pred"] = (rf_full_pred + lr_full_pred)
print(rf_full_pred + lr_full_pred)
dataset_out["lr_pred"] = lr_full_pred
dataset_out["rf_pred"] = rf_full_pred
dataset_out.to_csv(outpath,index=False)

