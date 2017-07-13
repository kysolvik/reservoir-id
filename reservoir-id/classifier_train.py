#!/usr/bin/env python

"""
Train random forest classifier
Inputs: CSV from build_att_table, small area cutoff
Outputs: Packaged up Random Forest model
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
import sys
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Train Random Forest classifier.')
parser.add_argument('prop_csv',
                    help='Path to attribute table (from build_att_table.py).',
                    type=str)
parser.add_argument('rf_pkl',
                    help='Path to save random forest model as .pkl.',
                    type=str)
parser.add_argument('--ntrees',
                    help='Number of trees for random forest',
                    type=int,default=200)
parser.add_argument('--area_lowbound',
                    help='Lower area bound. All regions <= in size will be ignored',
                    default=2,
                    type=int)
parser.add_argument('--print_imp',
                    help='Print variable importances',
                    action='store_true')
parser.add_argument('--path_prefix',
                    help='To be placed at beginnings of all other path args',
                    type=str,default='')
args = parser.parse_args()

def main():
        # Set any attributes to exclude for this run
        exclude_att_patterns = []

        # Load dataset
        dataset = pd.read_csv(args.path_prefix + args.prop_csv,header=0)
        dataset_acut = dataset.loc[dataset['area'] > args.area_lowbound]

        # Exclude attributes matching user input patterns, or if they are all nans
        exclude_atts = []
        for pattern in exclude_att_patterns:
                col_list = [col for col in dataset_acut.columns if pattern in col]
                exclude_atts.extend(col_list)
                
        for att in dataset.columns[1:]:
                if sum(np.isfinite(dataset[att])) == 0:
                        exclude_atts.append(att)

        for att in list(set(exclude_atts)):
                del dataset_acut[att]
    
        (ds_y,ds_x) = dataset_acut.shape
        print(ds_y,ds_x)
        
        # Convert dataset to array
        feature_names = dataset_acut.columns[2:]
        array = dataset_acut.values
        X = array[:,2:ds_x].astype(float)
        Y = array[:,1].astype(int)

        # Set nans to 0
        X = np.nan_to_num(X)

        # Scale!
        X_scaled = X # preprocessing.scale(X)
        X_scaled_classified = X_scaled[Y > 0]
        Y_classified = Y[Y > 0]

        # Separate test data
        test_size = 0.2
        seed = 5
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
                X_scaled_classified, Y_classified, test_size=test_size,
                random_state=seed)

        # Test options and evaluation metric
        scoring = 'accuracy'

        # Spot Check Algorithms
        models = []
        models.append(('RF10',RandomForestClassifier(n_estimators=10,n_jobs = 4)))
        # models.append(('RF64',RandomForestClassifier(n_estimators=64)))
        # models.append(('RF80',RandomForestClassifier(n_estimators=80)))
        # models.append(('RF100',RandomForestClassifier(n_estimators=100)))
        # models.append(('RF120',RandomForestClassifier(n_estimators=120)))
              
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
        rf = RandomForestClassifier(n_estimators = args.ntrees,criterion="gini",n_jobs = 4)

        # # Get learning curve for random forest
        # kfold = model_selection.KFold(n_splits=10, random_state=seed)
        # t_sizes, t_scores, cv_scores = model_selection.learning_curve(rf, X_train, Y_train, cv=kfold, scoring=scoring,train_sizes=np.array([ 0.1, 0.2, 0.3, 0.4,.5,.6,.7,.8,.9, 1. ]))
        # t_scores_mean = np.mean(t_scores,axis=1)
        # cv_scores_mean = np.mean(cv_scores,axis=1)
        # plt.plot(t_sizes,t_scores_mean,'r--',t_sizes,cv_scores_mean,'b--')
        # plt.show()

        # Make predictions on test dataset
        rf.fit(X_train, Y_train)

        # Get importances
        if args.print_imp:
                importances = rf.feature_importances_
                std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                             axis=0)
                indices = np.argsort(importances)[::-1]

                # Print the feature ranking
                print("Feature ranking:")
                for f in range(X_train.shape[1]):
                        print("%d. %s (%f)" % (f + 1, feature_names[f], importances[indices[f]]))

        
        # For training accuracy
        rf_train_predict = rf.predict(X_train)
        
        print("Training acc = " + str(accuracy_score(Y_train,rf_train_predict)))
        # For test accuracy
        rf_predictions = rf.predict(X_test)
        print("RF CV acc = " + str(accuracy_score(Y_test, rf_predictions)))
        print(confusion_matrix(Y_test, rf_predictions))
        print(classification_report(Y_test, rf_predictions))
                
        # Export classifier trained on full data set
        rf_full = RandomForestClassifier(n_estimators = args.ntrees)
        rf_full.fit(X_scaled_classified,Y_classified)
        joblib.dump(rf, args.path_prefix + args.rf_pkl)
        
if __name__ == '__main__':
        main()
