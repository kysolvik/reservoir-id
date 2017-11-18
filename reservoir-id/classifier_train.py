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
import os
import xgboost as xgb

# Parse arguments
parser = argparse.ArgumentParser(description='Train Random Forest classifier.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

def select_training_obs(full_csv_path):
    """Takes full csv and selects only the training observations.
    Writes out to csv for further use"""
    training_csv_path = full_csv_path.replace('.csv','_trainonly.csv')
    if not os.path.isfile(training_csv_path):
        dataset = pd.read_csv(full_csv_path,header=0)
        training_dataset = dataset.loc[dataset['class'] > 0]
        training_dataset.to_csv(training_csv_path,header=True,index=False)
    return(training_csv_path)

def main():
        # Set any attributes to exclude for this run
        exclude_att_patterns = []

        # Load dataset
        training_csv = select_training_obs(args.path_prefix + args.prop_csv)
        dataset = pd.read_csv(training_csv,header=0)
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
        Y = Y - 1 # Convert from 1s and 2s to 0-1

        # Set nans to 0
        X = np.nan_to_num(X)

        # Separate test data
        test_size = 0.2
        seed = 5
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
                X, Y, test_size=test_size,
                random_state=seed)

        # Convert data to xgboost matrices
        d_train = xgb.DMatrix(X_train,label=Y_train)
        d_test = xgb.DMatrix(X_test,label=Y_test)
        
        # Define classifier
        param = {'max_depth': 5, 'eta': 0.1, 'gamma':2,'silent': 1, 'objective': 'binary:logistic'}
        param['nthread'] = 4
        param['eval_metric'] = 'auc'
        evallist=[(d_test,'eval'),(d_train,'train')]

        num_round=100
        bst = xgb.train(param,d_train,num_round,evallist) 

        # For test accuracy
        # Make predictions on test dataset
        bst_log_preds = bst.predict(d_test)
        bst_preds = np.round(bst_log_preds)
        print("Xgboost Test acc = " + str(accuracy_score(Y_test, bst_preds)))
        print(confusion_matrix(Y_test, bst_preds))
        print(classification_report(Y_test, bst_preds))
                
#         # Export classifier trained on full data set
#         rf_full = RandomForestClassifier(n_estimators = args.ntrees)
#         rf_full.fit(X,Y)
#         joblib.dump(rf, args.path_prefix + args.rf_pkl)
        
if __name__ == '__main__':
        main()
