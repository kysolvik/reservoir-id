#!/usr/bin/env python
"""
@authors: Kylen Solvik
Date Create: 3/8/17
- Builds attribute table for reservoir classification with shape descriptors
and NDWI values within contour
- Arguments:
1) wat_tif_path: Path to output from grow_shrink.py. Image of opened/closed water objects.
2) ndwi_tif_path: Path to NDWI file 
3) training_csv_dir:
4) prop_csv_outpath: Attribute table for machine learning
5) cont_csv_outpath: csv containing contour IDs and shapes. Needed for plotting 
classification results

"""

import sys
import cv2
import numpy as np
import scipy.misc
import gdal
import pandas as pd
import math
import time
from feature_calc_funcs import *
import gc

#===============================================================================

# Test mode true means that it will only run for training set.
test_mode = True

# Get arguments
wat_tif_dir = sys.argv[1]
wat_pattern = 
ndwi_tif_dir = sys.argv[2]
ndwi_pattern = 
training_csv_dir = sys.argv[3]
prop_csv_outpath = sys.argv[4]
cont_csv_outpath = sys.argv[5]

prop_list_get = ['area','convex_area','eccentricity',
                                  'equivalent_diameter','extent','inertia_tensor',
                                  'inertia_tensor_eigvals','major_axis_length',
                                  'max_intensity','mean_intensity','min_intensity',
                                  'minor_axis_length','moments_normalized','moments_hu',
                                  'orientation','perimeter','solidity',
                                  'weighted_moments_normalized','weighted_moments_hu']
#===============================================================================

def main():

