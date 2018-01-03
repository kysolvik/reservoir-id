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
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *
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
parser.add_argument('xgb_pkl',
                    help='Path to save random forest model as .pkl.',
                    type=str)
parser.add_argument('--area_lowbound',
                    help='Lower area bound. All regions <= in size will be ignored',
                    default=2,
                    type=int)
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
        Y[Y==2] = 0 # Convert from 1s and 2s to 0-1

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
        # d_test = xgb.DMatrix(X_test,label=Y_test)
       
        #----------------------------------------------------------------------
        # Paramater tuning

        # Step 1: Find approximate n_estimators to use
        early_stop_rounds = 40
        n_folds = 5
        xgb_model = xgb.XGBClassifier(
            learning_rate =0.1,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective= 'binary:logistic',
            seed=27)
        xgb_params = xgb_model.get_xgb_params() 
        cvresult = xgb.cv(xgb_params, d_train, 
            num_boost_round=xgb_params['n_estimators'], nfold=n_folds,
            metrics='auc', early_stopping_rounds=early_stop_rounds,
            )
        n_est_best = (cvresult.shape[0] - early_stop_rounds)
        print('Best number of rounds = {}'.format(n_est_best))
        
        # Step 2: Tune hyperparameters
        xgb_model = xgb.XGBClassifier()

        params = {'max_depth': range(5,10,2),
                'learning_rate': [0.1],
                'gamma':[0,0.5,1],
                'silent': [1], 
                'objective': ['binary:logistic'],
                'n_estimators' : [n_est_best],
                'subsample' : [0.7, 0.8,1],
                'min_child_weight' : range(1,4,2),
                'colsample_bytree':[0.7,0.8,1],
                }
        clf = GridSearchCV(xgb_model,params,n_jobs = 1,
                cv = StratifiedKFold(Y_train,
                n_folds=5, shuffle=True),
                scoring = 'roc_auc',
                verbose = 2,
                refit = True)
        clf.fit(X_train,Y_train)

        best_parameters,score,_ = max(clf.grid_scores_,key=lambda x: x[1])
        print('Raw AUC score:',score)
        for param_name in sorted(best_parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        # Step 3: Decrease learning rate and up the # of trees
        #xgb_finalcv = XGBClassifier()
        tuned_params = clf.best_params_
        tuned_params['n_estimators'] = 10000
        tuned_params['learning_rate'] = 0.01
        cvresult = xgb.cv(tuned_params, d_train, 
            num_boost_round=tuned_params['n_estimators'], nfold=n_folds,
            metrics='auc', early_stopping_rounds=early_stop_rounds,
            )

        # Train model with cv results and predict on test set For test accuracy
        n_est_final = int((cvresult.shape[0] - early_stop_rounds) / (1 - 1 / n_folds))
        tuned_params['n_estimators'] = n_est_final
        print(tuned_params)
        xgb_train = xgb.XGBClassifier()
        xgb_train.set_params(**tuned_params)
        xgb_train.fit(X_train,Y_train)
        bst_preds = xgb_train.predict(X_test)
        print("Xgboost Test acc = " + str(accuracy_score(Y_test, bst_preds)))
        print(confusion_matrix(Y_test, bst_preds))
        print(classification_report(Y_test, bst_preds))
        # Export cv classifier
        joblib.dump(cvresult, args.path_prefix + args.xgb_pkl + 'cv')         
                
        # Export classifier trained on full data set
        xgb_full = xgb.XGBClassifier()
        xgb_full.set_params(**tuned_params)
        xgb_full.fit(X,Y)
        joblib.dump(xgb_full, args.path_prefix + args.xgb_pkl) 

if __name__ == '__main__':
        main()
