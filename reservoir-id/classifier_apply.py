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

# Arguments
in_csv_path = sys.argv[1]
acut_small = int(sys.argv[2]) # Ignores regions smaller. Recommend 2-4 (pixels)
classifier_pkl = sys.argv[3]
out_csv_path = sys.argv[4]

def main():
        # Set any attributes to exclude for this run
        exclude_att_patterns = []

        # Load dataset
        dataset = pd.read_csv(in_csv_path,header=0)
        dataset_acut = dataset.loc[dataset['area'] > acut_small]

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
        clf = joblib.load(classifier_pkl)
        clf_pred = clf.predict(X_scaled)
        dataset_out = dataset_acut
        dataset_out["clf_pred"] = clf_pred
        print(str(sum(clf_pred == 1)) + " classified as positive")
        print(str(sum(clf_pred == 2)) + " classified as negative")
        dataset_out.to_csv(out_csv_path,index=False)
        
if __name__ == '__main__':
        main()
