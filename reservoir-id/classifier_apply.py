#!/usr/bin/env python
"""
Apply classifier exported by classifier_train.py
Inputs: Classifier pkl path, small area cutoff
Outputs: CSV with classified regions
Notes: 
1. Make sure that all columns in the apply csv match the train_csv
2. exclude_att_patterns must match
@authors: Kylen Solvik
Date Create: 5/27/17
"""

# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
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
parser.add_argument('class_csv_out',
                    help='Path for output classified csv',
                    type=str)
parser.add_argument('--area_lowbound',
                    help='Lower area bound. Must match trained model. All regions <= in size will be ignored',
                    default=2,
                    type=int)
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
        array = dataset_acut.values
        X = array[:,2:ds_x].astype(float)
        Y = array[:,1].astype(int)

        # Set nans to 0
        X = np.nan_to_num(X)

        # Scale!
        X_scaled = X #preprocessing.scale(X)

        # Export classifier trained on full data set
        clf = joblib.load(args.path_prefix + args.rf_pkl)
        clf_pred = clf.predict(X_scaled)
        dataset_out = dataset_acut
        dataset_out["clf_pred"] = clf_pred
        print(str(sum(clf_pred == 1)) + " classified as positive")
        print(str(sum(clf_pred == 2)) + " classified as negative")
        dataset_out.to_csv(args.path_prefix + args.class_csv_out,index=False)
        
if __name__ == '__main__':
        main()
